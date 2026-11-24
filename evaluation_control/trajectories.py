"""Reference trajectories for computed-torque tracking.

Ten main trajectories (5 s each, sampled at the 10 ms control step):
  * 5 finite Fourier series: 1-3 harmonics of a base frequency in [0.1, 0.3] Hz per joint
  * 5 quintic minimum-jerk point-to-point moves through 4 random waypoints

q_ref, qd_ref and qdd_ref are closed-form. A candidate is accepted only if
  * every limited joint stays at least 0.2 rad inside its MuJoCo range,
  * the analytical inverse dynamics of the reference needs at most 70 % of each actuator's
    torque limit (gear = [12, 8, 4, 2] N*m), so that tracking is not saturation-limited, and
  * (main set) every sample of q, qd, qdd lies inside the empirical training-set range.

A second "ood" set of 4 faster, larger trajectories (Fourier base 0.4-0.56 Hz; min-jerk with
1 s segments through 6 waypoints) must leave the training range for at least
20 % of its samples while staying inside the joint limits and at most 90 % of the torque limits.

Run:  python -m evaluation_control.trajectories
"""

import argparse
import json
import os

import numpy as np

from .common import (CONTROL_DT, FIG_DIR, GEAR, JOINTS, N_DOF, RESULTS_DIR, ensure_dirs,
                     train_ranges, wrap_angle)

DURATION_S = 5.0
LIMIT_MARGIN = 0.2
SEED = 2026
JOINT_HI = np.array([np.inf, 2.792526803190927, 2.792526803190927, 2.443460952792061])


# ---------------------------------------------------------------------------
# Trajectory families (closed form)
# ---------------------------------------------------------------------------

def fourier(t, center, amps, freqs, phases):
    """q_j(t) = c_j + sum_n a_jn [sin(w_jn t + p_jn) - sin(p_jn)]: starts exactly at c."""
    w = 2 * np.pi * freqs                                       # (4, n)
    arg = w[None] * t[:, None, None] + phases[None]             # (T, 4, n)
    q = center + (amps * (np.sin(arg) - np.sin(phases)[None])).sum(-1)
    qd = (amps * w * np.cos(arg)).sum(-1)
    qdd = (-amps * w ** 2 * np.sin(arg)).sum(-1)
    return q, qd, qdd


def min_jerk(t, waypoints, seg_s):
    """Quintic (rest-to-rest) segments between consecutive waypoints, then hold."""
    q = np.empty((len(t), N_DOF)); qd = np.zeros_like(q); qdd = np.zeros_like(q)
    for i, ti in enumerate(t):
        s_idx = min(int(ti // seg_s), len(waypoints) - 2)
        tau = np.clip((ti - s_idx * seg_s) / seg_s, 0.0, 1.0)
        a, b = waypoints[s_idx], waypoints[s_idx + 1]
        d = b - a
        s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        sd = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / seg_s
        sdd = (60 * tau - 180 * tau ** 2 + 120 * tau ** 3) / seg_s ** 2
        if ti > (len(waypoints) - 1) * seg_s:
            s, sd, sdd = 1.0, 0.0, 0.0
        q[i] = a + s * d; qd[i] = sd * d; qdd[i] = sdd * d
    return q, qd, qdd


# ---------------------------------------------------------------------------
# Sampling with feasibility checks
# ---------------------------------------------------------------------------

def _sample_fourier(rng, ood, lo, hi):
    n_h = 3
    base = rng.uniform(0.4, 0.56, N_DOF) if ood else rng.uniform(0.1, 0.3, N_DOF)
    freqs = base[:, None] * np.arange(1, n_h + 1)[None]
    span = (hi - lo) / 2
    scale = (0.3 if ood else 0.22) * span[:, None] / np.arange(1, n_h + 1)[None] ** (1.0 if ood else 1.5)
    amps = scale * rng.uniform(0.3, 1.0, (N_DOF, n_h)) * (rng.random((N_DOF, n_h)) < 0.85)
    phases = rng.uniform(0, 2 * np.pi, (N_DOF, n_h))
    center = rng.uniform(lo + 0.35 * span, hi - 0.35 * span)
    return dict(center=center, amps=amps, freqs=freqs, phases=phases)


def _sample_min_jerk(rng, ood, lo, hi):
    n_wp = 6 if ood else 4
    span = hi - lo
    frac = 0.15
    wps = rng.uniform(lo + frac * span, hi - frac * span, (n_wp, N_DOF))
    seg = (DURATION_S / (n_wp - 1)) if ood else (DURATION_S - 0.5) / (n_wp - 1)
    return dict(waypoints=wps, seg_s=seg)


def _range_box(ranges, ood):
    """Region to draw centers/waypoints from: the central training-q range for the main set,
    the joint limits (minus margin) for the OOD set. The root uses the training range of the
    wrapped angle in both cases."""
    lo, hi = ranges["q"]
    if ood:
        lo = np.where(np.isfinite(JOINT_HI), -JOINT_HI + LIMIT_MARGIN + 0.1, lo)
        hi = np.where(np.isfinite(JOINT_HI), JOINT_HI - LIMIT_MARGIN - 0.1, hi)
    else:
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        lo, hi = mid - 0.85 * half, mid + 0.85 * half
    return lo, hi


def in_range_fraction(q, qd, qdd, ranges):
    """Fraction of samples whose every q (root wrapped), qd, qdd component is inside the
    empirical training min/max."""
    qw = q.copy(); qw[:, 0] = wrap_angle(qw[:, 0])
    inside = np.ones(len(q), bool)
    for x, key in ((qw, "q"), (qd, "qdot"), (qdd, "qddot")):
        lo, hi = ranges[key]
        inside &= np.all((x >= lo) & (x <= hi), axis=1)
    return float(inside.mean())


def check(q, qd, qdd, model, ranges, ood):
    lim = JOINT_HI - LIMIT_MARGIN
    if np.any(np.abs(q[:, 1:]) > lim[1:]):
        return None
    frac = in_range_fraction(q, qd, qdd, ranges)          # cheap checks first
    if (not ood and frac < 1.0) or (ood and frac > 0.8):
        return None
    limit = 0.9 if ood else 0.7
    coarse = model.inverse_dynamics(q[::5], qd[::5], qdd[::5])   # cheap screen (0.2 s)
    if (np.abs(coarse).max(0) / GEAR).max() > limit:
        return None
    tau = model.inverse_dynamics(q, qd, qdd)
    peak = np.abs(tau).max(0) / GEAR
    if peak.max() > limit:
        return None
    return {"tau_ref": tau, "peak_torque_frac": peak, "in_range_frac": frac}


def generate(model, n_fourier=5, n_minjerk=5, ood=False, seed=SEED, max_tries=5000):
    ranges = train_ranges()
    lo, hi = _range_box(ranges, ood)
    rng = np.random.default_rng(seed + (1000 if ood else 0))
    t = np.arange(int(round(DURATION_S / CONTROL_DT)) + 1) * CONTROL_DT
    trajs = []
    for family, n in (("fourier", n_fourier), ("min_jerk", n_minjerk)):
        got, tries = 0, 0
        while got < n:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(f"could not sample {n} feasible {family} trajectories")
            if family == "fourier":
                params = _sample_fourier(rng, ood, lo, hi)
                q, qd, qdd = fourier(t, **params)
            else:
                params = _sample_min_jerk(rng, ood, lo, hi)
                q, qd, qdd = min_jerk(t, **params)
            res = check(q, qd, qdd, model, ranges, ood)
            if res is None:
                continue
            got += 1
            trajs.append({"name": f"{'ood_' if ood else ''}{family}_{got}", "family": family,
                          "q": q, "qd": qd, "qdd": qdd, "tries": tries, **res})
    return t, trajs


def save(t, trajs, path):
    np.savez_compressed(
        path, t=t, names=np.array([tr["name"] for tr in trajs]),
        family=np.array([tr["family"] for tr in trajs]),
        q=np.stack([tr["q"] for tr in trajs]), qd=np.stack([tr["qd"] for tr in trajs]),
        qdd=np.stack([tr["qdd"] for tr in trajs]), tau_ref=np.stack([tr["tau_ref"] for tr in trajs]),
        in_range_frac=np.array([tr["in_range_frac"] for tr in trajs]),
        peak_torque_frac=np.stack([tr["peak_torque_frac"] for tr in trajs]))


def load(which="main"):
    path = os.path.join(RESULTS_DIR, f"trajectories_{which}.npz")
    d = np.load(path)
    return {k: d[k] for k in d.files}


def _plot(t, trajs, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for j, ax in enumerate(axes.flat):
        for tr in trajs:
            ax.plot(t, tr["q"][:, j], lw=1.3, ls="-" if tr["family"] == "fourier" else "--",
                    label=tr["name"])
        ax.set_title(JOINTS[j], fontsize=12); ax.set_ylabel("q_ref (rad)", fontsize=11)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Time (s)", fontsize=11)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(title, fontsize=13); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    from .models import AnalyticalModel
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    ensure_dirs()
    model = AnalyticalModel()
    summary = {}
    for which, ood, nf, nm in (("main", False, 5, 5), ("ood", True, 2, 2)):
        t, trajs = generate(model, nf, nm, ood=ood, seed=args.seed)
        save(t, trajs, os.path.join(RESULTS_DIR, f"trajectories_{which}.npz"))
        _plot(t, trajs, os.path.join(FIG_DIR, f"reference_trajectories_{which}.png"),
              "Reference trajectories" + (" outside the training range" if ood else
                                           " (inside the training range)"))
        summary[which] = [{"name": tr["name"], "in_range_frac": tr["in_range_frac"],
                           "peak_torque_frac": tr["peak_torque_frac"].round(3).tolist(),
                           "max_abs_qd": np.abs(tr["qd"]).max(0).round(3).tolist(),
                           "max_abs_qdd": np.abs(tr["qdd"]).max(0).round(3).tolist(),
                           "tries": tr["tries"]} for tr in trajs]
        for s in summary[which]:
            print(f"{s['name']:18s} in-range {s['in_range_frac']:.2f}  "
                  f"peak |tau|/limit {max(s['peak_torque_frac']):.2f}")
    with open(os.path.join(RESULTS_DIR, "trajectories_summary.json"), "w") as f:
        json.dump({"seed": args.seed, "duration_s": DURATION_S, "dt": CONTROL_DT, **summary}, f,
                  indent=2)


if __name__ == "__main__":
    main()
