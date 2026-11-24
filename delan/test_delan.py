"""Correctness tests for the DeLaN Euler-Lagrange machinery (run: pytest -q test_delan.py)."""

import numpy as np
import pytest
import torch

from delan import DeLaN, DeLaNDynamics, coriolis_from_dM

torch.manual_seed(0)


@pytest.fixture(scope="module")
def analytical():
    from common import load_analytical
    return load_analytical()[0]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return DeLaN(width=32, depth=2).eval()


def test_coriolis_formula_matches_analytical(analytical):
    """Feed Member 2's exact dM/dq into our EL Coriolis formula: must equal AnalyticalArm C(q,qd) qd."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        q, qd = rng.uniform(-2, 2, 4), rng.uniform(-3, 3, 4)
        dM = torch.from_numpy(analytical.dM(q))[None]
        c = coriolis_from_dM(dM, torch.from_numpy(qd)[None])[0].numpy()
        np.testing.assert_allclose(c, analytical.C(q, qd) @ qd, atol=1e-12)


def test_autograd_dM_matches_finite_difference(model):
    q = torch.randn(3, 4)
    dM = model.dM_dq(q)
    h = 1e-6
    for i in range(4):
        e = torch.zeros(4); e[i] = h
        fd = (model.mass_matrix(q + e) - model.mass_matrix(q - e)) / (2 * h)
        torch.testing.assert_close(dM[..., i], fd, atol=1e-8, rtol=1e-5)


def test_gravity_is_gradient_of_potential(model):
    q = torch.randn(3, 4)
    g = model.gravity(q)
    h = 1e-6
    for i in range(4):
        e = torch.zeros(4); e[i] = h
        fd = (model.potential(q + e) - model.potential(q - e)) / (2 * h)
        torch.testing.assert_close(g[:, i], fd, atol=1e-7, rtol=1e-5)


def test_coriolis_matrix_consistent_and_skew(model):
    """c = C qd, and Mdot - 2C is skew-symmetric (passivity property)."""
    q, qd = torch.randn(5, 4), torch.randn(5, 4)
    _, c, _, _ = model.terms(q, qd)
    C = model.coriolis_matrix(q, qd)
    torch.testing.assert_close(torch.einsum("bij,bj->bi", C, qd), c)
    Mdot = torch.einsum("bkji,bi->bkj", model.dM_dq(q), qd)
    N = Mdot - 2 * C
    torch.testing.assert_close(N, -N.transpose(1, 2), atol=1e-10, rtol=0)


def test_forward_inverse_roundtrip(model):
    q, qd, qdd = torch.randn(8, 4), torch.randn(8, 4), torch.randn(8, 4)
    tau = model.inverse_dynamics(q, qd, qdd)
    torch.testing.assert_close(model.forward_dynamics(q, qd, tau), qdd, atol=1e-6, rtol=1e-6)


def test_energy_conserved_without_damping_or_torque():
    """Lagrangian structure => a passive rollout (tau=0, D=0) conserves T+V."""
    torch.manual_seed(1)
    m = DeLaN(width=32, depth=2, learn_damping=False).eval()
    q, qd = torch.randn(1, 4) * 0.5, torch.randn(1, 4) * 0.5
    tau = torch.zeros(1, 4)
    dt = 1e-3
    E0 = m.energy(q, qd).item()

    def f(q, qd):
        return qd, m.forward_dynamics(q, qd, tau)
    with torch.no_grad():
        for _ in range(300):                       # RK4
            k1 = f(q, qd)
            k2 = f(q + dt / 2 * k1[0], qd + dt / 2 * k1[1])
            k3 = f(q + dt / 2 * k2[0], qd + dt / 2 * k2[1])
            k4 = f(q + dt * k3[0], qd + dt * k3[1])
            q = q + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            qd = qd + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    E1 = m.energy(q, qd).item()
    assert abs(E1 - E0) < 1e-6 * max(1.0, abs(E0))


def test_numpy_wrapper_shapes(model):
    dyn = DeLaNDynamics(model)
    q, qd, qdd = np.zeros(4), np.ones(4), np.ones(4)
    assert dyn.M(q).shape == (4, 4)
    assert dyn.G(q).shape == (4,)
    assert dyn.C(q, qd).shape == (4, 4)
    assert dyn.inverse_dynamics(q, qd, qdd).shape == (4,)
    assert dyn.inverse_dynamics(np.zeros((7, 4)), np.ones((7, 4)), np.ones((7, 4))).shape == (7, 4)
    tau = dyn.inverse_dynamics(q, qd, qdd)
    np.testing.assert_allclose(dyn.forward_dynamics(q, qd, tau), qdd, atol=1e-6)
    # single-sample wrapper agrees with the analytical-style decomposition
    np.testing.assert_allclose(tau, dyn.M(q) @ qdd + dyn.C(q, qd) @ qd + dyn.G(q) + dyn.damp * qd,
                               atol=1e-10)


def test_subsets_identical_to_mlp_and_nested():
    """DeLaN trains on exactly the rows black_box_mlp/train_mlp.py uses (same seed, ceil, order)."""
    import math
    import json
    import os
    from common import FRACTIONS, ROOT, subset_indices
    n = 20000
    mlp_order = np.random.default_rng(42).permutation(n)       # train_mlp.py: nested_order
    prev = set()
    for f in FRACTIONS:
        idx = subset_indices(n, f)
        np.testing.assert_array_equal(idx, mlp_order[:max(1, int(math.ceil(n * f)))])
        assert prev <= set(idx.tolist())                         # nested: 10 % ⊂ 30 % ⊂ 50 % ⊂ 100 %
        prev = set(idx.tolist())
        cfg = os.path.join(ROOT, "black_box_mlp", "results", f"{int(round(f * 100)):03d}pct", "config.json")
        if os.path.exists(cfg):
            assert json.load(open(cfg))["training_samples"] == len(idx)


def test_only_official_splits():
    from common import load_split
    for s in ("train", "val", "test", "ood"):
        assert load_split(s)["q"].shape[1] == 4
    with pytest.raises(ValueError):
        load_split("selection")
