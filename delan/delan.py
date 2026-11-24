"""Deep Lagrangian Network (DeLaN) for the 4-joint dm_control manipulator arm.

Model
-----
    L(q)  : lower-triangular, positive diagonal (softplus)   -> M(q) = L L^T + eps*I   (SPD by construction)
    V(q)  : scalar potential energy                           -> g(q) = dV/dq
    D     : learnable diagonal viscous damping, D = softplus(d) >= 0

Euler-Lagrange inverse dynamics (all derivatives via torch.func autodiff):
    tau = M(q) qdd + Mdot qd - 1/2 d/dq (qd^T M qd) + g(q) + D qd
        = M(q) qdd + c(q, qd)                    + g(q) + D qd

The networks see [sin q, cos q] so the unbounded root joint (wraps at +-pi) is handled
correctly. All tensors are float64 by default (M has eigenvalues down to ~1e-5).

Two interfaces:
    DeLaN          - torch nn.Module, batched tensors (B, 4), used for training.
    DeLaNDynamics  - numpy wrapper with the same methods as AnalyticalArm
                     (M, G, C, V, inverse_dynamics, forward_dynamics) for Member 5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacfwd, grad, vmap

N_DOF = 4
torch.set_default_dtype(torch.float64)


def _inv_softplus(x):
    return float(torch.log(torch.expm1(torch.tensor(x))))


def _mlp(n_in, n_out, width, depth, act):
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), act()]
        d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def coriolis_from_dM(dM, qd):
    """c = Mdot qd - 1/2 d/dq (qd^T M qd), with dM[b, k, j, i] = d M_kj / d q_i."""
    Mdot_qd = torch.einsum("bkji,bi,bj->bk", dM, qd, qd)
    dT_dq = 0.5 * torch.einsum("bkji,bk,bj->bi", dM, qd, qd)
    return Mdot_qd - dT_dq


class DeLaN(nn.Module):
    def __init__(self, n_dof=N_DOF, width=128, depth=3, activation="softplus",
                 eps=1e-6, m_scale=0.1, diag_shift=1e-3, damping_init=1.0, learn_damping=True):
        super().__init__()
        act = {"softplus": nn.Softplus, "tanh": nn.Tanh, "silu": nn.SiLU}[activation]
        self.n = n_dof
        n_feat = 2 * n_dof
        self.eps = eps
        self.m_scale = m_scale          # true M is ~1e-2, so keep L outputs small
        self.diag_shift = diag_shift    # strictly positive diagonal of L
        n_off = n_dof * (n_dof - 1) // 2
        self.l_net = _mlp(n_feat, n_dof + n_off, width, depth, act)
        self.v_net = _mlp(n_feat, 1, width, depth, act)
        self.d_raw = nn.Parameter(torch.full((n_dof,), _inv_softplus(damping_init)),
                                  requires_grad=learn_damping)
        # fixed linear map: vector of strictly-lower entries -> flattened (n, n) matrix
        idx = torch.tril_indices(n_dof, n_dof, offset=-1)
        S = torch.zeros(n_dof * n_dof, n_off)
        S[idx[0] * n_dof + idx[1], torch.arange(n_off)] = 1.0
        self.register_buffer("tril_map", S)
        self.config = dict(n_dof=n_dof, width=width, depth=depth, activation=activation, eps=eps,
                           m_scale=m_scale, diag_shift=diag_shift, damping_init=damping_init,
                           learn_damping=learn_damping)

    # ---------- single-sample functions (q: (n,)) used inside vmap ----------
    @staticmethod
    def _features(q):
        return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)

    def _L(self, q):
        out = self.l_net(self._features(q))
        diag = F.softplus(out[: self.n]) + self.diag_shift
        off = (self.tril_map @ out[self.n:]).reshape(self.n, self.n)
        return self.m_scale * (torch.diag_embed(diag) + off)

    def _M(self, q):
        L = self._L(q)
        return L @ L.T + self.eps * torch.eye(self.n, dtype=q.dtype, device=q.device)

    def _V(self, q):
        return self.v_net(self._features(q)).squeeze(-1)

    # ---------- batched quantities (q: (B, n)) ----------
    @property
    def damping(self):
        if not self.config["learn_damping"]:          # ablation: no dissipative term
            return torch.zeros_like(self.d_raw)
        return F.softplus(self.d_raw)

    def mass_matrix(self, q):
        return vmap(self._M)(q)

    def potential(self, q):
        return vmap(self._V)(q)

    def gravity(self, q):
        return vmap(grad(self._V))(q)

    def dM_dq(self, q):
        """dM[b, k, j, i] = d M_kj / d q_i  (same convention as AnalyticalArm.dM)."""
        return vmap(jacfwd(self._M))(q)

    def terms(self, q, qd):
        """Return M, c (Coriolis/centrifugal vector), g, damping force."""
        M = self.mass_matrix(q)
        c = coriolis_from_dM(self.dM_dq(q), qd)
        g = self.gravity(q)
        f_d = self.damping * qd
        return M, c, g, f_d

    def coriolis_matrix(self, q, qd):
        """C(q, qd) with c = C qd, built from Christoffel symbols (matches AnalyticalArm.C)."""
        dM = self.dM_dq(q)
        chris = 0.5 * (dM + dM.permute(0, 1, 3, 2) - dM.permute(0, 3, 2, 1))
        return torch.einsum("bkji,bi->bkj", chris, qd)

    def inverse_dynamics(self, q, qd, qdd):
        M, c, g, f_d = self.terms(q, qd)
        return torch.einsum("bij,bj->bi", M, qdd) + c + g + f_d

    def forward_dynamics(self, q, qd, tau):
        M, c, g, f_d = self.terms(q, qd)
        return torch.linalg.solve(M, (tau - c - g - f_d).unsqueeze(-1)).squeeze(-1)

    def energy(self, q, qd):
        """Total energy T + V (useful for rollout energy-drift checks)."""
        M = self.mass_matrix(q)
        T = 0.5 * torch.einsum("bi,bij,bj->b", qd, M, qd)
        return T + self.potential(q)

    forward = inverse_dynamics


# ---------------------------------------------------------------------------
# Checkpoint I/O and numpy interface (drop-in replacement for AnalyticalArm)
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, extra=None):
    torch.save({"config": model.config, "state_dict": model.state_dict(), **(extra or {})}, path)


def load_model(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = DeLaN(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


class DeLaNDynamics:
    """Numpy wrapper with the AnalyticalArm interface.

    Every method accepts a single state of shape (4,) or a batch of shape (B, 4)
    and returns the matching shape.

        dyn = DeLaNDynamics.from_checkpoint("delan/checkpoints/delan_f1.0_s0.pt")
        tau = dyn.inverse_dynamics(q, qd, qdd)
        qdd = dyn.forward_dynamics(q, qd, tau)
    """

    def __init__(self, model):
        self.model = model.eval().double()

    @classmethod
    def from_checkpoint(cls, path):
        return cls(load_model(path)[0])

    @property
    def damp(self):
        return self.model.damping.detach().numpy()

    def _run(self, fn, *xs):
        import numpy as np
        xs = [np.asarray(x, dtype=np.float64) for x in xs]
        single = xs[0].ndim == 1
        ts = [torch.from_numpy(np.atleast_2d(x)) for x in xs]
        with torch.no_grad():
            out = fn(*ts).numpy()
        return out[0] if single else out

    def M(self, q):
        return self._run(self.model.mass_matrix, q)

    def G(self, q):
        return self._run(self.model.gravity, q)

    def C(self, q, qd):
        return self._run(self.model.coriolis_matrix, q, qd)

    def V(self, q):
        return self._run(self.model.potential, q)

    def energy(self, q, qd):
        return self._run(self.model.energy, q, qd)

    def inverse_dynamics(self, q, qd, qdd):
        return self._run(self.model.inverse_dynamics, q, qd, qdd)

    def forward_dynamics(self, q, qd, tau):
        return self._run(self.model.forward_dynamics, q, qd, tau)
