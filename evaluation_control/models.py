"""Shared dynamics-model interface for the analytical EL model, the black-box MLP and DeLaN.

Every model exposes the same numpy API (single state ``(4,)`` or batch ``(B, 4)``):

    inverse_dynamics(q, qd, qdd)        -> tau
    id_jacobians(q, qd, qdd)            -> (dtau/dqdd, dtau/dqd), each (..., 4, 4)
    forward_dynamics(q, qd, tau)        -> qdd   (Newton inversion of the inverse dynamics)
    implied_mass_matrix(q)              -> dtau/dqdd at (q, 0, 0)

Only inverse dynamics is needed by the computed-torque controller. Forward dynamics, used
for rollouts, is obtained the same way for every model: by solving ID(q, qd, qdd) = tau for
qdd with Newton's method. For the analytical model and DeLaN, tau is affine in qdd, so one
Newton step is exact. The MLP is only approximately affine in qdd, so its inversion iterates
and reports convergence in ``last_newton_info``.
"""

import os

import numpy as np
import torch
from torch.func import jacrev, vmap

from .common import FRACTIONS, N_DOF, ROOT, import_file, model_label

torch.set_default_dtype(torch.float64)

DELAN_CKPT_DIR = os.path.join(ROOT, "delan", "checkpoints")
MLP_CKPT_DIR = os.path.join(ROOT, "black_box_mlp", "checkpoints")


def _as_batch(*xs):
    xs = [np.asarray(x, dtype=np.float64) for x in xs]
    single = xs[0].ndim == 1
    return [np.atleast_2d(x) for x in xs], single


def _unbatch(out, single):
    return out[0] if single else out


def christoffel_jacobian(dM, qd):
    """d/dqd of c(q, qd) = C(q, qd) qd, given dM[b, k, j, i] = dM_kj / dq_i.

    With Christoffel symbols G_kji = 1/2 (dM_kj/dq_i + dM_ki/dq_j - dM_ji/dq_k),
    c_k = sum_ij G_kji qd_i qd_j, so dc_k/dqd_m = sum_i G_kmi qd_i + sum_j G_kjm qd_j.
    """
    G = 0.5 * (dM + np.swapaxes(dM, 2, 3) - np.transpose(dM, (0, 3, 2, 1)))
    return np.einsum("bkmi,bi->bkm", G, qd) + np.einsum("bkjm,bj->bkm", G, qd)


class DynamicsModel:
    """Base class. Subclasses implement ``_id`` and ``_jac`` on 2-D numpy batches."""

    name = "model"
    kind = "model"
    fraction = None

    def __init__(self):
        self.last_newton_info = {}

    # ---- to implement ----
    def _id(self, q, qd, qdd):
        raise NotImplementedError

    def _jac(self, q, qd, qdd):
        raise NotImplementedError

    # ---- public API ----
    def inverse_dynamics(self, q, qd, qdd):
        (q, qd, qdd), single = _as_batch(q, qd, qdd)
        return _unbatch(self._id(q, qd, qdd), single)

    def id_jacobians(self, q, qd, qdd):
        (q, qd, qdd), single = _as_batch(q, qd, qdd)
        J_qdd, J_qd = self._jac(q, qd, qdd)
        return _unbatch(J_qdd, single), _unbatch(J_qd, single)

    def implied_mass_matrix(self, q):
        (q,), single = _as_batch(q)
        z = np.zeros_like(q)
        return _unbatch(self._jac(q, z, z)[0], single)

    # Newton settings: tau is exactly affine in qdd for the analytical model and DeLaN, so one
    # full step solves it; the MLP is not, and needs iterations with a line search.
    newton_iters = 1
    line_search = False
    multi_start = False

    def forward_dynamics(self, q, qd, tau, qdd0=None, max_iter=None, tol=1e-9):
        """Instantaneous forward dynamics: solve ID(q, qd, qdd) = tau for qdd (Newton).

        This is ill-conditioned for a learned model here: the true wrist inertia is ~3e-5, so
        a 0.01 N*m torque error maps to ~300 rad/s^2. Rollouts therefore use the implicit step
        in ``rollout.py``, whose Jacobian J_qdd + h J_qd is regularized by the damping.
        """
        (q, qd, tau), single = _as_batch(q, qd, tau)
        qdd0 = np.zeros_like(q) if qdd0 is None else np.atleast_2d(np.asarray(qdd0, float))
        def residual(a, idx=None):
            qi, qdi, ti = _rows(idx, q, qd, tau)
            return self._id(qi, qdi, a) - ti

        qdd, info = newton_solve(
            residual, lambda a: self._jac(q, qd, a)[0], qdd0,
            scale=1.0 + np.abs(tau),
            max_iter=max_iter or (1 if not self.line_search else 30),
            tol=tol, line_search=self.line_search)
        self.last_newton_info = info
        return _unbatch(qdd, single)

    def __repr__(self):
        return f"<{type(self).__name__} {self.name}>"


def _solve(J, r):
    """Batched linear solve with a least-squares fallback for singular Jacobians."""
    try:
        return np.linalg.solve(J, r[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.stack([np.linalg.lstsq(Ji, ri, rcond=None)[0] for Ji, ri in zip(J, r)])


def newton_solve(residual, jacobian, x0, scale, max_iter=1, tol=1e-9, line_search=False,
                 max_halvings=6):
    """Batched Newton solve of residual(x) = 0, with x of shape (B, n).

    ``residual(x, idx=None)`` evaluates the rows ``idx`` of the batch (all rows if None);
    ``jacobian(x)`` returns the full (B, n, n) Jacobian.
    With ``line_search`` each sample backtracks (step 1, 1/2, ..., 1/64) until its scaled
    residual norm decreases. This keeps the inversion of the MLP stable where its implied
    Jacobian is indefinite. Returns x and a convergence report based on
    max |residual| / scale.
    """
    x = np.array(x0, dtype=np.float64)
    r = residual(x)
    norm = np.linalg.norm(r / scale, axis=-1)
    it = 0
    for it in range(1, max_iter + 1):
        if line_search and np.all(norm <= tol):
            it -= 1
            break
        dx = -_solve(jacobian(x), r)
        if not line_search:
            x = x + dx
            r = residual(x) if it < max_iter else None
            continue
        alpha = np.ones(len(x))
        pending = np.ones(len(x), bool)
        x_new, r_new, n_new = x.copy(), r.copy(), norm.copy()
        for _ in range(max_halvings + 1):
            idx = np.flatnonzero(pending)
            if idx.size == 0:
                break
            xc = x[idx] + alpha[idx, None] * dx[idx]
            rc = residual(xc, idx)
            nc = np.linalg.norm(rc / scale[idx], axis=-1)
            ok = np.isfinite(nc) & (nc < norm[idx])
            take = idx[ok]
            x_new[take], r_new[take], n_new[take] = xc[ok], rc[ok], nc[ok]
            pending[take] = False
            alpha[idx[~ok]] *= 0.5
        x, r, norm = x_new, r_new, n_new
    if r is None:
        r = residual(x)
    rel = np.abs(r) / scale
    res = float(np.max(rel)) if np.all(np.isfinite(rel)) else np.inf
    return x, {"iterations": it, "max_rel_residual": res, "converged": bool(res <= max(tol, 1e-6)),
               "rel_residual": rel.max(-1)}


def _rows(idx, *arrays):
    return arrays if idx is None else tuple(a[idx] for a in arrays)


# ---------------------------------------------------------------------------
# Analytical Euler-Lagrange model (Member 2)
# ---------------------------------------------------------------------------

class AnalyticalModel(DynamicsModel):
    name = kind = "analytical"

    newton_iters = 2      # tau is affine in qdd but quadratic in qd (implicit step uses both)

    def __init__(self, arm=None):
        super().__init__()
        if arm is None:
            arm = load_analytical_arm()
        self.arm = arm
        self.damp = np.asarray(arm.damp, float)
        self._cache_key, self._cache = None, None

    def _q_terms(self, q):
        """M(q), dM/dq and g(q) for a batch; cached on q, because the implicit integrator and
        Newton evaluate many (qd, qdd) at the same q (AnalyticalArm is per-sample Python)."""
        key = q.tobytes()
        if key != self._cache_key:
            self._cache = (np.stack([self.arm.M(x) for x in q]),
                           np.stack([self.arm.dM(x) for x in q]),
                           np.stack([self.arm.G(x) for x in q]))
            self._cache_key = key
        return self._cache

    def _id(self, q, qd, qdd):
        """Same equation as AnalyticalArm.inverse_dynamics, M qdd + C(q, qd) qd + g + D qd,
        with C built from the same Christoffel symbols of dM."""
        M, dM, G = self._q_terms(q)
        chris = 0.5 * (dM + np.swapaxes(dM, 2, 3) - np.transpose(dM, (0, 3, 2, 1)))
        c = np.einsum("bkji,bi,bj->bk", chris, qd, qd)
        return np.einsum("bij,bj->bi", M, qdd) + c + G + self.damp * qd

    def _jac(self, q, qd, qdd):
        M, dM, _ = self._q_terms(q)
        J_qd = christoffel_jacobian(dM, qd) + np.diag(self.damp)[None]
        return M, J_qd

    def forward_dynamics(self, q, qd, tau, **kw):
        (q, qd, tau), single = _as_batch(q, qd, tau)
        out = np.stack([self.arm.forward_dynamics(*x) for x in zip(q, qd, tau)])
        self.last_newton_info = {"iterations": 1, "max_rel_residual": 0.0, "converged": True}
        return _unbatch(out, single)


def load_analytical_arm():
    """Exactly as Member 4 builds it (``delan/common.py::load_analytical``): the suite model
    with the fingers at qpos0. This reproduces the dataset torques to ~1e-16 N*m."""
    from dm_control import suite
    am = import_file("analytical_model", os.path.join(ROOT, "analytical_euler_lagrange",
                                                       "analytical_model.py"))
    m = suite.load("manipulator", "bring_ball").physics.model.ptr
    return am.AnalyticalArm(m, ref_qpos=m.qpos0.copy())


# ---------------------------------------------------------------------------
# Torch models: DeLaN (Member 4) and the black-box MLP (Member 3)
# ---------------------------------------------------------------------------

class TorchModel(DynamicsModel):
    """Batched torch ``tau = f(q, qd, qdd)``; Jacobians by ``vmap(jacrev(...))``."""

    def _tau_torch(self, q, qd, qdd):
        raise NotImplementedError

    def _single(self, q, qd, qdd):
        return self._tau_torch(q[None], qd[None], qdd[None])[0]

    def _id(self, q, qd, qdd):
        t = [torch.from_numpy(x) for x in (q, qd, qdd)]
        with torch.no_grad():
            return self._tau_torch(*t).numpy()

    def _jac(self, q, qd, qdd):
        t = [torch.from_numpy(x) for x in (q, qd, qdd)]
        J_qd, J_qdd = vmap(jacrev(self._single, argnums=(1, 2)))(*t)
        return J_qdd.detach().numpy(), J_qd.detach().numpy()


class DeLaNModel(TorchModel):
    kind = "delan"
    newton_iters = 2

    def __init__(self, fraction=1.0, seed=42, path=None):
        super().__init__()
        dl = import_file("delan_model", os.path.join(ROOT, "delan", "delan.py"))
        path = path or os.path.join(DELAN_CKPT_DIR, f"delan_f{fraction}_s{seed}.pt")
        self.net, _ = dl.load_model(path)
        self.net.double().eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.fraction = fraction
        self.name = model_label("delan", fraction)

    def _tau_torch(self, q, qd, qdd):
        return self.net.inverse_dynamics(q, qd, qdd)

    def _jac(self, q, qd, qdd):
        """Use DeLaN's structure: dtau/dqdd = M(q), dtau/dqd = D + dc/dqd (Christoffel)."""
        tq = torch.from_numpy(q)
        with torch.no_grad():
            M = self.net.mass_matrix(tq).numpy()
            dM = self.net.dM_dq(tq).numpy()
            D = self.net.damping.numpy()
        return M, christoffel_jacobian(dM, qd) + np.diag(D)[None]

    # extra structure, used by evaluate.py
    def mass_matrix(self, q):
        (q,), single = _as_batch(q)
        with torch.no_grad():
            return _unbatch(self.net.mass_matrix(torch.from_numpy(q)).numpy(), single)


class MLPModel(TorchModel):
    kind = "mlp"
    newton_iters = 10
    line_search = True
    multi_start = True     # the MLP's tau(qdd) is not monotone; see rollout.implicit_euler_step

    def __init__(self, fraction=1.0, path=None):
        super().__init__()
        from black_box_mlp.mlp import MLPInverseDynamics
        path = path or os.path.join(MLP_CKPT_DIR, f"{int(round(fraction * 100)):03d}pct", "best.pt")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        arch = {k: cfg[k] for k in ("input_dim", "output_dim", "hidden_sizes", "activation", "dropout")
                if k in cfg}
        self.net = MLPInverseDynamics(**arch)
        self.net.load_state_dict(ck["model_state_dict"])
        self.net.double().eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        st = ck["metadata"]["normalization_stats"]
        self.x_mean, self.x_std, self.y_mean, self.y_std = (
            torch.tensor(st[k], dtype=torch.float64) for k in ("X_mean", "X_std", "Y_mean", "Y_std"))
        self.fraction = fraction
        self.name = model_label("mlp", fraction)

    def _tau_torch(self, q, qd, qdd):
        # identical to dataset/preprocess.py::encode_features, written in torch for autograd
        x = torch.cat([torch.sin(q[:, :1]), torch.cos(q[:, :1]), q[:, 1:], qd, qdd], dim=-1)
        y = self.net((x - self.x_mean) / self.x_std)
        return y * self.y_std + self.y_mean


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_model(kind, fraction=1.0):
    if kind == "analytical":
        return AnalyticalModel()
    if kind == "delan":
        return DeLaNModel(fraction)
    if kind == "mlp":
        return MLPModel(fraction)
    raise ValueError(f"unknown model kind {kind!r}")


ALL_MODEL_SPECS = [("analytical", None)] + [(k, f) for k in ("mlp", "delan") for f in FRACTIONS]
MAIN_MODEL_SPECS = [("analytical", None), ("mlp", 1.0), ("delan", 1.0)]


def load_models(specs=ALL_MODEL_SPECS):
    analytical = None
    out = []
    for kind, frac in specs:
        if kind == "analytical":
            analytical = analytical or AnalyticalModel()
            out.append(analytical)
        else:
            out.append(load_model(kind, frac))
    return out


assert N_DOF == 4
