"""Joint-space computed-torque controllers that use any model from ``models.py``.

Commanded acceleration (outer loop, identical for every model):

    a_cmd = qdd_ref + Kd (qd_ref - qd) + Kp wrap(q_ref - q),   Kp = wn^2, Kd = 2 zeta wn

Inner loop (the model), two variants:

* ``classic``:   tau = ID_model(q, qd, a_cmd)
  The textbook law (Assignment 3). It assumes tau acts continuously. Here the plant is sampled:
  tau is held for 10 ms, and the elbow and wrist are damping-dominated (M/D ~ 1e-3 and 6e-5 s
  << 10 ms). Within one step their velocity relaxes to (tau - g - c)/D, so the inertial term
  M a_cmd changes the velocity by only (M/D) a_cmd instead of dt a_cmd, and feedback is
  almost lost on these joints.

* ``sampled`` (default):   tau = ID_model(q, qd + dt a_cmd, a_cmd)
  The same inverse dynamics, but evaluated at the velocity the joint should reach at the end
  of the hold interval (backward Euler over one control step). For a damping-dominated joint
  this gives qd_{k+1} ~ qd_k + dt a_cmd, which is exactly what the outer loop asks for. For an
  inertia-dominated joint it reduces to the classic law plus a small O(dt) correction.

The torque is clipped to the actuator limits |tau| <= gear = [12, 8, 4, 2] N*m.
"""

import numpy as np

from .common import CONTROL_DT, GEAR, wrap_angle


def gains(wn=15.0, zeta=1.0):
    wn = np.broadcast_to(np.asarray(wn, float), (4,)).copy()
    return wn ** 2, 2 * zeta * wn


class ComputedTorqueController:
    def __init__(self, model, wn=15.0, zeta=1.0, variant="sampled", dt=CONTROL_DT):
        if variant not in ("classic", "sampled"):
            raise ValueError("variant must be 'classic' or 'sampled'")
        self.model = model
        self.Kp, self.Kd = gains(wn, zeta)
        self.variant = variant
        self.dt = dt
        self.name = model.name if variant == "sampled" else f"{model.name} (classic CT)"

    def command(self, q, qd, q_ref, qd_ref, qdd_ref):
        """Batched: all arguments (B, 4). Returns (tau_clipped, tau_unclipped)."""
        e = wrap_angle(q_ref - q)
        a_cmd = qdd_ref + self.Kd * (qd_ref - qd) + self.Kp * e
        v = qd + self.dt * a_cmd if self.variant == "sampled" else qd
        tau = np.atleast_2d(self.model.inverse_dynamics(q, v, a_cmd))
        return np.clip(tau, -GEAR, GEAR), tau


class PDController:
    """Model-free PD reference: tau = Kp e + Kd edot, with gains scaled to each actuator
    (Kp = 2 * gear N*m/rad, Kd = 0.1 * gear N*m*s/rad). No gravity or damping compensation."""

    def __init__(self, kp_scale=2.0, kd_scale=0.1):
        self.Kp, self.Kd = kp_scale * GEAR, kd_scale * GEAR
        self.name = "PD only"
        self.model = None

    def command(self, q, qd, q_ref, qd_ref, qdd_ref):
        tau = self.Kp * wrap_angle(q_ref - q) + self.Kd * (qd_ref - qd)
        return np.clip(tau, -GEAR, GEAR), tau
