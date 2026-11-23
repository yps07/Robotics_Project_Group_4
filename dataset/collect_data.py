"""Collect reproducible, contact-free inverse-dynamics data for the arm.

Each sample contains the four arm coordinates, their velocities and
accelerations, physical actuator torques, normalized controls, and provenance
fields. The model is derived from dm_control's manipulator/bring_ball MJCF;
the gripper is rigidly attached to the wrist and collision geoms are disabled.
"""

import argparse
import json
import os

import numpy as np

try:  # Package invocation: python -m dataset.collect_data
    from .environment import (
        ARM_JOINT_NAMES,
        CONTROL_TIMESTEP_S,
        GEAR_RATIOS,
        JOINT_LIMITS,
        MAX_ABS_QDOT_FOR_DATA,
        N_ARM,
        OOD_MAX_ABS_QDDOT_FOR_DATA,
        TRAIN_MAX_ABS_QDDOT_FOR_DATA,
        get_arm_acc,
        get_arm_state,
        get_arm_torque,
        load_dynamics_env,
        move_ball_far,
        set_arm_state,
    )
    from .preprocess import compute_stats, save_stats
except ImportError:  # Direct invocation from the dataset directory
    from environment import (
        ARM_JOINT_NAMES,
        CONTROL_TIMESTEP_S,
        GEAR_RATIOS,
        JOINT_LIMITS,
        MAX_ABS_QDOT_FOR_DATA,
        N_ARM,
        OOD_MAX_ABS_QDDOT_FOR_DATA,
        TRAIN_MAX_ABS_QDDOT_FOR_DATA,
        get_arm_acc,
        get_arm_state,
        get_arm_torque,
        load_dynamics_env,
        move_ball_far,
        set_arm_state,
    )
    from preprocess import compute_stats, save_stats


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EPISODE_LENGTH = 1000
SPLIT_NAMES = ("train", "val", "test", "ood")
SAMPLE_KEYS = (
    "q",
    "qdot",
    "qddot",
    "tau",
    "action",
    "time_s",
    "episode_id",
    "mode",
    "ncon",
)


def make_action_random(t, rng, scale=0.35):
    del t
    return rng.uniform(-scale, scale, size=N_ARM)


def make_action_sinusoidal(t, rng, freqs, phases, amps):
    del rng
    return np.clip(amps * np.sin(2 * np.pi * freqs * t + phases), -1.0, 1.0)


def make_action_chirp(t, rng, f0, f1, duration, phases, amplitude=0.35):
    del rng
    duration = max(float(duration), np.finfo(float).eps)
    phase = 2 * np.pi * (
        f0 * t + (f1 - f0) * t * t / (2 * duration)
    ) + phases
    return amplitude * np.sin(phase)


def collect_episode(
    env,
    mode,
    n_steps,
    rng,
    freqs=None,
    phases=None,
    amps=None,
    f0=0.1,
    f1=1.0,
    action_scale=0.35,
):
    """Collect one episode; times are in seconds at the control-step rate."""
    physics = env.physics
    dt = float(env.control_timestep())
    env.reset()
    move_ball_far(physics)
    set_arm_state(physics, np.zeros(N_ARM), np.zeros(N_ARM))
    physics.forward()

    rows = {key: [] for key in SAMPLE_KEYS if key not in ("episode_id",)}
    for step in range(n_steps):
        t = step * dt
        if mode == "random":
            action = make_action_random(t, rng, scale=action_scale)
        elif mode == "sinusoidal":
            action = make_action_sinusoidal(t, rng, freqs, phases, amps)
        elif mode == "chirp":
            action = make_action_chirp(
                t, rng, f0, f1, n_steps * dt, phases, action_scale
            )
        else:
            raise ValueError(f"Unknown excitation mode: {mode}")

        env.step(action)
        # Recompute state derivatives at the saved state/control.
        physics.forward()
        if physics.data.ncon != 0:
            raise RuntimeError(
                "The isolated dynamics model produced a contact; refusing to "
                "save a contact-contaminated sample."
            )

        q, qdot = get_arm_state(physics)
        # The root joint is periodic; use one canonical representative.
        q[0] = (q[0] + np.pi) % (2 * np.pi) - np.pi
        qddot = get_arm_acc(physics)
        tau = get_arm_torque(physics)

        rows["q"].append(q)
        rows["qdot"].append(qdot)
        rows["qddot"].append(qddot)
        rows["tau"].append(tau)
        rows["action"].append(np.asarray(action, dtype=np.float64))
        rows["time_s"].append((step + 1) * dt)
        rows["mode"].append(mode)
        rows["ncon"].append(int(physics.data.ncon))

    return {
        key: np.asarray(value)
        for key, value in rows.items()
    }


def _valid_samples(episode, ood, ood_bounds=None):
    q = episode["q"]
    qdot = episode["qdot"]
    qddot = episode["qddot"]
    lower = JOINT_LIMITS[1:, 0] + 0.05
    upper = JOINT_LIMITS[1:, 1] - 0.05
    max_abs_qddot = (
        OOD_MAX_ABS_QDDOT_FOR_DATA
        if ood
        else TRAIN_MAX_ABS_QDDOT_FOR_DATA
    )

    valid = np.all(q[:, 1:] >= lower, axis=1)
    valid &= np.all(q[:, 1:] <= upper, axis=1)
    valid &= np.max(np.abs(qdot), axis=1) < MAX_ABS_QDOT_FOR_DATA
    valid &= np.max(np.abs(qddot), axis=1) < max_abs_qddot
    valid &= episode["ncon"] == 0
    if ood and ood_bounds is not None:
        features = np.concatenate((q, qdot, qddot), axis=1)
        lower, upper = ood_bounds
        valid &= np.any(
            (features < lower[None, :]) | (features > upper[None, :]),
            axis=1,
        )
    return valid


def collect_split(
    n_samples,
    modes,
    rng,
    ood=False,
    ood_bounds=None,
    n_episodes=10,
):
    """Collect a split with samples balanced across independent episodes."""
    if n_samples <= 0 or n_episodes <= 0:
        raise ValueError("n_samples and n_episodes must both be positive.")
    parts = {key: [] for key in SAMPLE_KEYS}
    collected = 0
    episode_id = 0
    samples_per_episode = int(np.ceil(n_samples / n_episodes))
    max_episodes = max(100, 20 * n_episodes)

    while collected < n_samples or episode_id < n_episodes:
        mode = str(rng.choice(modes))
        if mode == "sinusoidal":
            if ood:
                freqs = rng.uniform(0.75, 2.5, size=N_ARM)
            else:
                freqs = rng.uniform(0.1, 1.0, size=N_ARM)
            phases = rng.uniform(0.0, 2 * np.pi, size=N_ARM)
            if ood:
                amps = rng.uniform(0.4, 0.65, size=N_ARM)
            else:
                amps = rng.uniform(0.15, 0.35, size=N_ARM)
            kwargs = dict(freqs=freqs, phases=phases, amps=amps)
        elif mode == "chirp":
            phases = rng.uniform(0.0, 2 * np.pi, size=N_ARM)
            if ood:
                f0 = float(rng.uniform(0.5, 1.0))
                f1 = float(rng.uniform(2.0, 4.0))
            else:
                f0 = float(rng.uniform(0.05, 0.2))
                f1 = float(rng.uniform(0.5, 2.0))
            kwargs = dict(phases=phases, f0=f0, f1=f1)
        else:
            kwargs = {}

        episode_seed = int(rng.integers(0, 10000))
        env = load_dynamics_env(random_seed=episode_seed)
        episode = collect_episode(
            env,
            mode,
            EPISODE_LENGTH,
            rng,
            action_scale=0.65 if ood else 0.35,
            **kwargs,
        )
        valid = _valid_samples(episode, ood, ood_bounds=ood_bounds)
        valid_indices = np.flatnonzero(valid)
        quota = min(samples_per_episode, n_samples - collected)
        n_take = min(quota, len(valid_indices))
        if n_take:
            # Spread each episode's retained rows across the full trajectory
            # rather than taking a correlated prefix.
            chosen = valid_indices[
                np.linspace(0, len(valid_indices) - 1, n_take, dtype=int)
            ]
            for key in episode:
                parts[key].append(episode[key][chosen])
            parts["episode_id"].append(np.full(n_take, episode_id, dtype=np.int32))
            collected += n_take
        episode_id += 1
        if episode_id >= max_episodes and collected < n_samples:
            raise RuntimeError(
                f"Could not collect {n_samples} valid samples after "
                f"{max_episodes} episodes."
            )

    split = {
        key: np.concatenate(values, axis=0)[:n_samples]
        for key, values in parts.items()
    }
    return split


def _metadata(split_sizes, episode_counts):
    return {
        "task": "dm_control.manipulator.bring_ball",
        "data_model": (
            "bring_ball MJCF; four arm joints actuated; gripper hinges "
            "removed so finger bodies are rigidly attached to the wrist; "
            "collision geoms disabled for free-space inverse dynamics"
        ),
        "seed": 42,
        "arm_joints": list(ARM_JOINT_NAMES),
        "raw_state_features": ["q[4]", "qdot[4]", "qddot[4]"],
        "mlp_feature_encoding": (
            "[sin(q_root), cos(q_root), q_shoulder, q_elbow, q_wrist, "
            "qdot[4], qddot[4]]"
        ),
        "mlp_feature_dim": 13,
        "target": "tau[4] (physical actuator generalized forces)",
        "units": {
            "q": "rad",
            "qdot": "rad/s",
            "qddot": "rad/s^2",
            "tau": "N*m",
            "action": "normalized actuator control in [-1, 1]",
            "time_s": "s from the start of each episode",
        },
        "physics_timestep_s": 0.001,
        "control_timestep_s": CONTROL_TIMESTEP_S,
        "sample_timestep_s": CONTROL_TIMESTEP_S,
        "physics_substeps_per_sample": int(
            round(CONTROL_TIMESTEP_S / 0.001)
        ),
        "excitation": ["random", "sinusoidal", "chirp"],
        "excitation_timebase": "control_timestep_s",
        "train_action_amplitude": 0.35,
        "ood_action_amplitude": 0.65,
        "excitation_frequency_hz": {
            "train_sinusoidal": [0.1, 1.0],
            "train_chirp_f0": [0.05, 0.2],
            "train_chirp_f1": [0.5, 2.0],
            "ood_sinusoidal": [0.75, 2.5],
            "ood_chirp_f0": [0.5, 1.0],
            "ood_chirp_f1": [2.0, 4.0],
        },
        "ood_definition": (
            "higher action amplitude/frequencies; every accepted row has at "
            "least one q, qdot, or qddot component outside the empirical "
            "training min/max"
        ),
        "gear_ratios": GEAR_RATIOS.tolist(),
        "contact_handling": (
            "collision geoms disabled; ncon recorded and asserted to be zero"
        ),
        "sample_metadata": [
            "action[4]",
            "time_s",
            "episode_id",
            "mode",
            "ncon",
        ],
        "split_sizes": split_sizes,
        "episodes_per_split": episode_counts,
        "episode_sampling": (
            "accepted samples are spread over each episode; splits target "
            "40 training episodes and 10 episodes per evaluation split"
        ),
        "rejection_filters": {
            "joint_limit_margin_rad": 0.05,
            "max_abs_qdot": MAX_ABS_QDOT_FOR_DATA,
            "max_abs_qddot_train": TRAIN_MAX_ABS_QDDOT_FOR_DATA,
            "max_abs_qddot_ood": OOD_MAX_ABS_QDDOT_FOR_DATA,
            "contacts": 0,
        },
        "measurement_noise": "none",
        "validation_command": "python -m dataset.validate_data",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=20000)
    parser.add_argument("--val", type=int, default=2000)
    parser.add_argument("--test", type=int, default=2000)
    parser.add_argument("--ood", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(DATA_DIR, exist_ok=True)
    modes = ("random", "sinusoidal", "chirp")
    sizes = {
        "train": args.train,
        "val": args.val,
        "test": args.test,
        "ood": args.ood,
    }
    split_data = {}
    episode_counts = {}
    for split_name in SPLIT_NAMES:
        print(f"Collecting {split_name}...")
        ood_bounds = None
        if split_name == "ood":
            train_features = np.concatenate(
                (
                    split_data["train"]["q"],
                    split_data["train"]["qdot"],
                    split_data["train"]["qddot"],
                ),
                axis=1,
            )
            ood_bounds = (train_features.min(axis=0), train_features.max(axis=0))
        split = collect_split(
            sizes[split_name],
            modes,
            rng,
            ood=(split_name == "ood"),
            ood_bounds=ood_bounds,
            n_episodes=40 if split_name == "train" else 10,
        )
        path = os.path.join(DATA_DIR, f"{split_name}.npz")
        np.savez_compressed(path, **split)
        split_data[split_name] = split
        episode_counts[split_name] = int(
            np.unique(split["episode_id"]).size
        )
        print(
            split_name,
            split["q"].shape,
            "episodes=",
            episode_counts[split_name],
        )

    save_stats(
        compute_stats(os.path.join(DATA_DIR, "train.npz")),
        os.path.join(DATA_DIR, "stats.json"),
    )
    metadata = _metadata(sizes, episode_counts)
    metadata["seed"] = args.seed
    with open(os.path.join(DATA_DIR, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved datasets, training-only normalization stats, and metadata.")


if __name__ == "__main__":
    main()
