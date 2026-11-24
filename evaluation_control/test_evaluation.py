"""Correctness tests for Workstream E.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q evaluation_control/test_evaluation.py
"""

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from evaluation_control.common import DATA_DIR, GEAR, ROOT, wrap_angle  # noqa: E402
from evaluation_control.models import (AnalyticalModel, DeLaNModel, MLPModel,  # noqa: E402
                                       TorchModel)


@pytest.fixture(scope="module")
def test_rows():
    d = np.load(os.path.join(DATA_DIR, "test.npz"))
    idx = np.random.default_rng(0).choice(len(d["q"]), 40, replace=False)
    return d["q"][idx], d["qdot"][idx], d["qddot"][idx], d["tau"][idx]


@pytest.fixture(scope="module")
def analytical():
    return AnalyticalModel()


@pytest.fixture(scope="module")
def delan():
    return DeLaNModel(1.0)


@pytest.fixture(scope="module")
def mlp():
    return MLPModel(1.0)


# ---------------------------------------------------------------- shared model interface

def test_analytical_matches_member2_and_dataset(analytical, test_rows):
    q, qd, qdd, tau = test_rows
    ours = analytical.inverse_dynamics(q, qd, qdd)
    member2 = np.stack([analytical.arm.inverse_dynamics(*x) for x in zip(q, qd, qdd)])
    assert np.abs(ours - member2).max() < 1e-12
    assert np.abs(ours - tau).max() < 1e-10


def test_mlp_matches_member3_metric(mlp):
    """The torch re-implementation of the MLP's feature pipeline reproduces Member 3's
    reported 100 % test RMSE."""
    import pandas as pd
    d = np.load(os.path.join(DATA_DIR, "test.npz"))
    rmse = np.sqrt(np.mean((mlp.inverse_dynamics(d["q"], d["qdot"], d["qddot"]) - d["tau"]) ** 2))
    ref = pd.read_csv(os.path.join(ROOT, "black_box_mlp", "results", "data_efficiency_metrics.csv"))
    ref = ref[(ref["fraction"] == 1.0) & (ref["split"] == "test")]["rmse"].iloc[0]
    assert np.isclose(rmse, ref, rtol=1e-4)


def test_single_and_batch_shapes(analytical, delan, mlp, test_rows):
    q, qd, qdd, _ = test_rows
    for m in (analytical, delan, mlp):
        assert m.inverse_dynamics(q[0], qd[0], qdd[0]).shape == (4,)
        assert m.inverse_dynamics(q[:3], qd[:3], qdd[:3]).shape == (3, 4)
        J_qdd, J_qd = m.id_jacobians(q[:3], qd[:3], qdd[:3])
        assert J_qdd.shape == J_qd.shape == (3, 4, 4)


@pytest.mark.parametrize("which", ["analytical", "delan", "mlp"])
def test_jacobians_match_finite_differences(which, analytical, delan, mlp, test_rows):
    m = {"analytical": analytical, "delan": delan, "mlp": mlp}[which]
    q, qd, qdd, _ = (x[:4] for x in test_rows)
    J_qdd, J_qd = m.id_jacobians(q, qd, qdd)
    eye, h = np.eye(4), 1e-6
    fd_qdd = np.stack([(m.inverse_dynamics(q, qd, qdd + h * eye[i]) - m.inverse_dynamics(q, qd, qdd - h * eye[i]))
                       / (2 * h) for i in range(4)], -1)
    fd_qd = np.stack([(m.inverse_dynamics(q, qd + h * eye[i], qdd) - m.inverse_dynamics(q, qd - h * eye[i], qdd))
                      / (2 * h) for i in range(4)], -1)
    assert np.abs(J_qdd - fd_qdd).max() < 1e-6
    assert np.abs(J_qd - fd_qd).max() < 1e-6


def test_delan_structural_jacobian_equals_autograd(delan, test_rows):
    q, qd, qdd, _ = (x[:5] for x in test_rows)
    Js = delan._jac(q, qd, qdd)
    Ja = TorchModel._jac(delan, q, qd, qdd)
    assert np.abs(Js[0] - Ja[0]).max() < 1e-12 and np.abs(Js[1] - Ja[1]).max() < 1e-12


@pytest.mark.parametrize("which", ["analytical", "delan"])
def test_forward_inverse_round_trip(which, analytical, delan, test_rows):
    m = analytical if which == "analytical" else delan
    q, qd, qdd, _ = test_rows
    back = m.forward_dynamics(q, qd, m.inverse_dynamics(q, qd, qdd))
    assert np.abs(back - qdd).max() < 1e-6 * (1 + np.abs(qdd).max())


def test_mlp_newton_inversion(mlp, test_rows):
    """Where the Newton inversion reports convergence, the MLP really maps qdd back to tau."""
    q, qd, _, tau = test_rows
    a = mlp.forward_dynamics(q, qd, tau)
    rel = mlp.last_newton_info["rel_residual"]
    ok = rel < 1e-6
    assert ok.mean() > 0.5
    assert np.abs(mlp.inverse_dynamics(q[ok], qd[ok], a[ok]) - tau[ok]).max() < 1e-5


# ---------------------------------------------------------------- rollouts

@pytest.fixture(scope="module")
def ground_truth():
    from evaluation_control.rollout import generate_ground_truth, rollout_windows
    gt = generate_ground_truth("test", n_episodes=1, seed=123)
    return gt, rollout_windows(gt)[:2]


@pytest.mark.parametrize("substeps,tol", [(10, 1e-4), (1, 5e-3)])
def test_analytical_rollout_matches_mujoco(analytical, ground_truth, substeps, tol):
    """The integrator, run with the exact model, reproduces MuJoCo: this is the floor."""
    from evaluation_control.rollout import evaluate_model, horizon_metrics
    gt, wins = ground_truth
    eq, ev, _, _ = evaluate_model(analytical, gt, wins, substeps=substeps)
    r2 = horizon_metrics(eq, ev)[-1]
    assert r2["frac_diverged"] == 0 and r2["pos_rmse"] < tol


def test_wrap_angle():
    assert np.allclose(wrap_angle(np.array([np.pi + 0.1, -np.pi - 0.1, 0.3])), [-np.pi + 0.1, np.pi - 0.1, 0.3])


# ---------------------------------------------------------------- trajectories and control

@pytest.fixture(scope="module")
def refs():
    from evaluation_control import trajectories as T
    if not os.path.exists(os.path.join(T.RESULTS_DIR, "trajectories_main.npz")):
        pytest.skip("run python -m evaluation_control.trajectories first")
    return T.load("main"), T.load("ood")


def test_trajectory_derivatives_and_constraints(refs, analytical):
    from evaluation_control import trajectories as T
    from evaluation_control.common import CONTROL_DT, train_ranges
    main, ood = refs
    assert len(main["names"]) == 10
    for ref, limit in ((main, 0.7), (ood, 0.9)):
        dq = np.gradient(ref["q"], CONTROL_DT, axis=1)
        dqd = np.gradient(ref["qd"], CONTROL_DT, axis=1)
        inner = slice(2, -2)
        # central differences at dt = 10 ms: O(dt^2) error, so compare relative to the peak
        assert np.abs(dq[:, inner] - ref["qd"][:, inner]).max() < 0.01 * np.abs(ref["qd"]).max()
        assert np.abs(dqd[:, inner] - ref["qdd"][:, inner]).max() < 0.05 * np.abs(ref["qdd"]).max()
        assert np.all(np.abs(ref["q"][..., 1:]) <= T.JOINT_HI[1:] - T.LIMIT_MARGIN)
        assert np.all(np.abs(ref["tau_ref"]) <= limit * GEAR + 1e-9)
    r = train_ranges()
    assert all(T.in_range_fraction(q, qd, qdd, r) == 1.0
               for q, qd, qdd in zip(main["q"], main["qd"], main["qdd"]))
    assert all(f <= 0.8 for f in ood["in_range_frac"])


def test_analytical_computed_torque_tracks(refs, analytical):
    from evaluation_control.computed_torque import ComputedTorqueController
    from evaluation_control.tracking_experiment import run_tracking, tracking_metrics
    main, _ = refs
    one = {k: main[k][:1] for k in ("q", "qd", "qdd", "names")}
    sampled = tracking_metrics(run_tracking(ComputedTorqueController(analytical), one), one)[0]
    classic = tracking_metrics(run_tracking(ComputedTorqueController(analytical, variant="classic"), one), one)[0]
    assert sampled["rmse"] < 3e-3
    assert classic["rmse"] > 5 * sampled["rmse"]          # the sampled-data effect
