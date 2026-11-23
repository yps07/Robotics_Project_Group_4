"""Validate the generated arm inverse-dynamics datasets.

Run from the project root with ``python -m dataset.validate_data``.
"""

import json
import os

import mujoco
import numpy as np

from .environment import (
    ARM_JOINT_NAMES,
    GEAR_RATIOS,
    JOINT_LIMITS,
    MAX_ABS_QDOT_FOR_DATA,
    N_ARM,
    OOD_MAX_ABS_QDDOT_FOR_DATA,
    TRAIN_MAX_ABS_QDDOT_FOR_DATA,
    get_arm_state,
    load_dynamics_env,
    move_ball_far,
    set_arm_state,
)
from .preprocess import compute_stats
from analytical_euler_lagrange.analytical_model import AnalyticalArm


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SPLITS = ("train", "val", "test", "ood")
REQUIRED_KEYS = {
    "q",
    "qdot",
    "qddot",
    "tau",
    "action",
    "time_s",
    "episode_id",
    "mode",
    "ncon",
}


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_data(n_dynamics_samples=128, seed=0):
    metadata = _load_json(os.path.join(DATA_DIR, "dataset_metadata.json"))
    physics_env = load_dynamics_env()
    physics = physics_env.physics
    model = physics.model.ptr
    control_dt = physics_env.control_timestep()
    physics_dt = float(model.opt.timestep)

    assert np.isclose(metadata["sample_timestep_s"], control_dt)
    assert np.isclose(metadata["control_timestep_s"], control_dt)
    assert np.isclose(metadata["physics_timestep_s"], physics_dt)
    assert model.nv == 7 and model.nu == N_ARM
    assert int(round(control_dt / physics_dt)) == metadata[
        "physics_substeps_per_sample"
    ]
    assert physics.data.ncon == 0

    loaded = {}
    for split in SPLITS:
        path = os.path.join(DATA_DIR, f"{split}.npz")
        with np.load(path, allow_pickle=False) as archive:
            assert REQUIRED_KEYS.issubset(archive.files), (
                split,
                archive.files,
            )
            data = {key: archive[key] for key in archive.files}
        loaded[split] = data

        n = len(data["q"])
        assert n == metadata["split_sizes"][split]
        for key in ("q", "qdot", "qddot", "tau", "action"):
            assert data[key].shape == (n, N_ARM), (split, key)
            assert np.isfinite(data[key]).all(), (split, key)
        for key in ("time_s", "episode_id", "mode", "ncon"):
            assert data[key].shape == (n,), (split, key)
        assert np.all(data["ncon"] == 0), f"{split} contains contact rows"
        assert np.all(data["q"][:, 0] >= -np.pi - 1e-10)
        assert np.all(data["q"][:, 0] < np.pi + 1e-10)
        assert set(np.unique(data["mode"])).issubset(
            {"random", "sinusoidal", "chirp"}
        )
        assert np.all(np.abs(data["action"]) <= 1.0 + 1e-12)
        assert np.allclose(
            data["tau"],
            data["action"] * GEAR_RATIOS,
            atol=1e-10,
            rtol=0.0,
        ), f"{split}: actuator torque does not match gear * action"

        for joint in range(1, N_ARM):
            low, high = JOINT_LIMITS[joint]
            assert np.all(data["q"][:, joint] >= low + 0.05 - 1e-10)
            assert np.all(data["q"][:, joint] <= high - 0.05 + 1e-10)
        assert np.max(np.abs(data["qdot"])) < MAX_ABS_QDOT_FOR_DATA
        qddot_limit = (
            OOD_MAX_ABS_QDDOT_FOR_DATA
            if split == "ood"
            else TRAIN_MAX_ABS_QDDOT_FOR_DATA
        )
        assert np.max(np.abs(data["qddot"])) < qddot_limit

        # Each episode is contiguous in the file; filtering can skip time
        # samples, so intervals must be positive integer multiples of dt.
        episode_ids = data["episode_id"]
        starts = np.r_[0, np.flatnonzero(np.diff(episode_ids) != 0) + 1]
        ends = np.r_[starts[1:], n]
        assert np.unique(episode_ids).size == len(starts)
        assert len(starts) == metadata["episodes_per_split"][split]
        for start, end in zip(starts, ends):
            t = data["time_s"][start:end]
            assert np.all(np.diff(t) > 0.0)
            if len(t) > 1:
                steps = np.diff(t) / control_dt
                assert np.allclose(steps, np.rint(steps), atol=1e-7)
            assert np.unique(data["mode"][start:end]).size == 1

        print(
            f"{split}: {n} rows, {np.unique(episode_ids).size} episodes, "
            f"ncon=0, torque/action relation OK"
        )

    # Check that the saved normalization is calculated from train only.
    saved_stats = _load_json(os.path.join(DATA_DIR, "stats.json"))
    expected_stats = compute_stats(os.path.join(DATA_DIR, "train.npz"))
    for key in ("X_mean", "X_std", "Y_mean", "Y_std"):
        assert np.allclose(
            saved_stats[key], expected_stats[key], rtol=0.0, atol=1e-12
        ), f"Incorrect training normalization statistic: {key}"

    train_features = np.concatenate(
        (
            loaded["train"]["q"],
            loaded["train"]["qdot"],
            loaded["train"]["qddot"],
        ),
        axis=1,
    )
    ood_features = np.concatenate(
        (
            loaded["ood"]["q"],
            loaded["ood"]["qdot"],
            loaded["ood"]["qddot"],
        ),
        axis=1,
    )
    train_low = train_features.min(axis=0)
    train_high = train_features.max(axis=0)
    outside = (ood_features < train_low) | (ood_features > train_high)
    assert np.all(np.any(outside, axis=1)), (
        "Every OOD sample must be outside at least one empirical training "
        "feature range."
    )
    print(
        "OOD fraction outside training min/max by input feature:",
        np.mean(outside, axis=0),
    )

    # Independently verify the saved inverse-dynamics labels with MuJoCo and
    # the analytical model on representative rows from every split.
    analytical = AnalyticalArm(model, ref_qpos=model.qpos0.copy())
    rng = np.random.default_rng(seed)
    max_mj_error = np.zeros(N_ARM)
    max_analytical_error = np.zeros(N_ARM)
    n_checked = 0
    for split, data in loaded.items():
        count = min(n_dynamics_samples, len(data["q"]))
        indices = rng.choice(len(data["q"]), size=count, replace=False)
        for i in indices:
            physics.reset()
            physics.data.qpos[:] = model.qpos0
            physics.data.qvel[:] = 0.0
            physics.data.qacc[:] = 0.0
            set_arm_state(physics, data["q"][i], data["qdot"][i])
            move_ball_far(physics)
            dofs = [
                model.jnt_dofadr[
                    mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, name
                    )
                ]
                for name in ARM_JOINT_NAMES
            ]
            physics.data.qacc[dofs] = data["qddot"][i]
            mujoco.mj_inverse(model, physics.data.ptr)
            mj_error = np.asarray(
                physics.data.qfrc_inverse[dofs] - data["tau"][i]
            )
            analytical_error = (
                analytical.inverse_dynamics(
                    data["q"][i], data["qdot"][i], data["qddot"][i]
                )
                - data["tau"][i]
            )
            max_mj_error = np.maximum(max_mj_error, np.abs(mj_error))
            max_analytical_error = np.maximum(
                max_analytical_error, np.abs(analytical_error)
            )
            n_checked += 1

    assert np.max(max_mj_error) < 1e-8, max_mj_error
    assert np.max(max_analytical_error) < 1e-8, max_analytical_error
    print(f"MuJoCo/analytical inverse-dynamics checks: {n_checked} rows")
    print("max MuJoCo inverse error (N*m):", max_mj_error)
    print("max analytical model error (N*m):", max_analytical_error)
    print("All dataset validations passed.")


if __name__ == "__main__":
    validate_data()
