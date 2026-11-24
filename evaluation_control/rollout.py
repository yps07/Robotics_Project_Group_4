"""Multi-step forward-dynamics rollouts of every model against MuJoCo.

Ground truth
------------
Fresh, *contiguous* MuJoCo episodes from Member 1's reduced, contact-free arm
(``dataset/environment.py::ArmDynamicsEnv``). They use the same excitation families,
amplitudes and frequency ranges as the dataset (``dataset/collect_data.py``), with new seeds.
The official dataset cannot be used: it keeps only 27-65 subsampled rows per episode.
Steps where a MuJoCo constraint is active (a joint limit; ``nefc > 0``) are excluded,
because none of the models represents limit forces.

Integrator
----------
The wrist is stiff: its damping/inertia ratio is ~0.5 / 3e-5 = 1.6e4 /s, so explicit RK4 is
unstable at any practical step. Every model is integrated with a step that is implicit in
velocity (as MuJoCo's Euler integrator treats damping implicitly), with tau held over each
control step:

    find a such that  ID(q_n, v_n + h a, a) = tau      (Newton, Jacobian J_qdd + h J_qd)
    v_{n+1} = v_n + h a,     q_{n+1} = q_n + h v_{n+1}

This needs only inverse dynamics, so it is identical for all three models. For the MLP it
is the numerical inversion of the inverse-dynamics network.

Step size: h = 10 ms (one step per control interval) by default. With h = 1 ms the first
substep after a torque change needs wrist accelerations of ~tau/(M + hD) ~ 1e3-1e4 rad/s^2.
The dataset never contains those: it is sampled at the end of each 10 ms interval, after the
transient. The MLP, queried there, diverges immediately. At h = 10 ms the implicit solve asks
for the interval-average acceleration, which stays near the training distribution. The
analytical model, integrated the same way, measures the discretization floor of this scheme
(about 1e-3 rad at 1-2 s). ``--substeps 10`` gives h = 1 ms (floor ~2e-5 rad) for the
analytical model and DeLaN.

Run:  python -m evaluation_control.rollout [--quick]
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .common import (CONTROL_DT, FIG_DIR, GEAR, JOINTS, N_DOF, PHYSICS_DT, RESULTS_DIR,
                     ensure_dirs, wrap_angle)
from .models import _rows, newton_solve

HORIZONS_S = (0.5, 1.0, 2.0)
ROLLOUT_S = 2.0
START_EVERY_S = 1.0
EPISODE_S = 10.0
DIVERGED_QD = 50.0
GT_SEEDS = {"test": 5150, "ood": 6160}       # disjoint from Member 1's seed (42)
SUBSTEPS = 1                                  # model integration step h = CONTROL_DT / SUBSTEPS


# ---------------------------------------------------------------------------
# MuJoCo ground truth
# ---------------------------------------------------------------------------

def _excitation(dist, rng):
    """Sample one episode's excitation exactly as dataset/collect_data.py::collect_split."""
    from dataset.collect_data import (make_action_chirp, make_action_random,
                                      make_action_sinusoidal)
    ood = dist == "ood"
    mode = str(rng.choice(("random", "sinusoidal", "chirp")))
    scale = 0.65 if ood else 0.35
    n_steps = int(round(EPISODE_S / CONTROL_DT))
    if mode == "sinusoidal":
        freqs = rng.uniform(0.75, 2.5, N_DOF) if ood else rng.uniform(0.1, 1.0, N_DOF)
        phases = rng.uniform(0, 2 * np.pi, N_DOF)
        amps = rng.uniform(0.4, 0.65, N_DOF) if ood else rng.uniform(0.15, 0.35, N_DOF)
        return mode, lambda t: make_action_sinusoidal(t, rng, freqs, phases, amps)
    if mode == "chirp":
        phases = rng.uniform(0, 2 * np.pi, N_DOF)
        f0 = float(rng.uniform(0.5, 1.0) if ood else rng.uniform(0.05, 0.2))
        f1 = float(rng.uniform(2.0, 4.0) if ood else rng.uniform(0.5, 2.0))
        return mode, lambda t: make_action_chirp(t, rng, f0, f1, n_steps * CONTROL_DT, phases, scale)
    return mode, lambda t: make_action_random(t, rng, scale=scale)


def generate_ground_truth(dist="test", n_episodes=10, seed=None):
    """Contiguous MuJoCo episodes. Arrays have T = 1000 control steps.

    q[k], qd[k] is the state at the start of step k; tau[k] is applied during [k, k+1).
    ``constrained[k]`` is True if a constraint was active at any physics substep of step k.
    """
    from dataset.environment import ArmDynamicsEnv, get_arm_state
    seed = GT_SEEDS[dist] if seed is None else seed
    rng = np.random.default_rng(seed)
    env = ArmDynamicsEnv()
    physics = env.physics
    T = int(round(EPISODE_S / CONTROL_DT))
    out = {k: [] for k in ("q", "qd", "tau", "constrained", "mode")}
    for _ in range(n_episodes):
        mode, act = _excitation(dist, rng)
        env.reset()
        q_ep, qd_ep, tau_ep, con_ep = [], [], [], []
        for k in range(T + 1):
            q, qd = get_arm_state(physics)
            q_ep.append(q); qd_ep.append(qd)
            if k == T:
                break
            a = np.clip(act(k * CONTROL_DT), -1.0, 1.0)
            physics.data.ctrl[:N_DOF] = a
            con = False
            for _ in range(env.n_substeps):
                physics.step()
                con |= physics.data.nefc > 0
            tau_ep.append(GEAR * a); con_ep.append(con)
        out["q"].append(q_ep); out["qd"].append(qd_ep); out["tau"].append(tau_ep)
        out["constrained"].append(con_ep); out["mode"].append(mode)
    gt = {k: np.asarray(v) for k, v in out.items()}
    gt["dist"], gt["seed"] = dist, seed
    return gt


def rollout_windows(gt, horizon_s=ROLLOUT_S, every_s=START_EVERY_S, max_abs_qd=10.0):
    """Start indices (episode, k0) whose whole window is constraint-free and within the
    dataset's |qd| < 10 rad/s filter."""
    H = int(round(horizon_s / CONTROL_DT)); step = int(round(every_s / CONTROL_DT))
    T = gt["tau"].shape[1]
    wins = []
    for e in range(gt["tau"].shape[0]):
        for k0 in range(0, T - H + 1, step):
            if gt["constrained"][e, k0:k0 + H].any():
                continue
            if np.abs(gt["qd"][e, k0:k0 + H + 1]).max() >= max_abs_qd:
                continue
            wins.append((e, k0))
    return wins


# ---------------------------------------------------------------------------
# Model integration
# ---------------------------------------------------------------------------

def implicit_euler_step(model, q, v, tau, h, a0):
    """One step implicit in velocity (see module docstring).

    Solves r(a) = ID(q, v + h a, a) - tau = 0 with ``newton_solve``, using the model's own
    iteration count and line-search setting. Returns q', v', a and the relative residual
    per sample.
    """
    def residual(a, idx=None):
        qi, vi, ti = _rows(idx, q, v, tau)
        return model._id(qi, vi + h * a, a) - ti

    def jacobian(a):
        J_qdd, J_qd = model._jac(q, v + h * a, a)
        return J_qdd + h * J_qd

    scale = 1.0 + np.abs(tau)
    starts = [a0]
    if model.multi_start:
        # The MLP's tau(qdd) along this line is not monotone, so Newton can stall or find a
        # spurious root. Also start from zero and from a damping-only guess
        # a = -r(0) / (h diag(J_qd)), and keep the lowest-residual solution per sample.
        z = np.zeros_like(a0)
        d = np.diagonal(model._jac(q, v, z)[1], axis1=1, axis2=2)
        starts += [z, -residual(z) / (h * np.where(np.abs(d) > 1e-6, d, 1e-6))]
    best_a, best_r = None, None
    for s in starts:
        a, info = newton_solve(residual, jacobian, s, scale=scale,
                               max_iter=model.newton_iters, line_search=model.line_search)
        r = np.where(np.isfinite(info["rel_residual"]), info["rel_residual"], np.inf)
        if best_a is None:
            best_a, best_r = a, r
        else:
            better = r < best_r
            best_a = np.where(better[:, None], a, best_a)
            best_r = np.where(better, r, best_r)
    v_new = v + h * best_a
    return q + h * v_new, v_new, best_a, best_r


def rollout(model, q0, v0, tau_seq, substeps=None):
    """Integrate ``model`` from (q0, v0) under tau_seq (B, K, 4) held per control step.

    Returns q, v of shape (B, K + 1, 4) at the control steps and the largest relative
    inverse-dynamics residual of the implicit solve within every control step (B, K): a
    measure of how well the MLP inversion converged.
    Diverged rollouts (non-finite or |v| > 50 rad/s) are frozen at NaN.
    """
    substeps = substeps or SUBSTEPS
    h = CONTROL_DT / substeps
    q = np.array(q0, float); v = np.array(v0, float)
    B, K, _ = tau_seq.shape
    Q = np.full((B, K + 1, N_DOF), np.nan); V = np.full_like(Q, np.nan)
    R = np.full((B, K), np.nan)
    Q[:, 0], V[:, 0] = q, v
    a = np.zeros_like(q)
    alive = np.ones(B, bool)
    for k in range(K):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        qa, va, aa, ta = q[idx], v[idx], a[idx], tau_seq[idx, k]
        res = np.zeros(idx.size)
        with np.errstate(all="ignore"):
            try:
                for _ in range(substeps):
                    qa, va, aa, r = implicit_euler_step(model, qa, va, ta, h, aa)
                    res = np.maximum(res, r)
            except np.linalg.LinAlgError:
                va = np.full_like(va, np.nan); res[:] = np.nan
        bad = ~np.isfinite(va).all(-1) | (np.abs(va).max(-1) > DIVERGED_QD)
        q[idx], v[idx], a[idx] = qa, va, aa
        good = idx[~bad]
        Q[good, k + 1], V[good, k + 1], R[good, k] = q[good], v[good], res[~bad]
        alive[idx[bad]] = False
    return Q, V, R


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rollout_errors(Q, V, Q_true, V_true):
    """Per-step wrapped position error and velocity error, (B, K + 1, 4)."""
    return wrap_angle(Q - Q_true), V - V_true


def horizon_metrics(eq, ev, horizons=HORIZONS_S):
    """RMSE over (0, H] for every horizon, pooled over rollouts and joints, plus per joint.

    Diverged rollouts (NaN) are counted and excluded from the RMSE; ``frac_diverged``
    reports them so that they are never silently dropped.
    """
    rows = []
    for H in horizons:
        n = int(round(H / CONTROL_DT))
        e = eq[:, 1:n + 1]; f = ev[:, 1:n + 1]
        ok = np.isfinite(e).all(axis=(1, 2))
        row = {"horizon_s": H, "n_rollouts": len(ok), "frac_diverged": float(1 - ok.mean())}
        # median over rollouts of the per-rollout RMSE, a diverged rollout counting as +inf:
        # unlike pos_rmse it cannot be flattered by excluding diverged rollouts
        per = np.where(ok, np.sqrt(np.nanmean(np.where(ok[:, None, None], e, 0) ** 2, axis=(1, 2))),
                       np.inf)
        row["pos_rmse_median_incl_diverged"] = float(np.median(per))
        if ok.any():
            row["pos_rmse"] = float(np.sqrt(np.mean(e[ok] ** 2)))
            row["vel_rmse"] = float(np.sqrt(np.mean(f[ok] ** 2)))
            row["pos_err_at_H"] = float(np.sqrt(np.mean(e[ok, -1] ** 2)))
            for j, name in enumerate(JOINTS):
                row[f"pos_rmse_{name}"] = float(np.sqrt(np.mean(e[ok, :, j] ** 2)))
        rows.append(row)
    return rows


def evaluate_model(model, gt, wins, horizon_s=ROLLOUT_S, substeps=None):
    K = int(round(horizon_s / CONTROL_DT))
    e_idx = np.array([w[0] for w in wins]); k_idx = np.array([w[1] for w in wins])
    steps = k_idx[:, None] + np.arange(K + 1)[None]
    Q_true = gt["q"][e_idx[:, None], steps]; V_true = gt["qd"][e_idx[:, None], steps]
    tau = gt["tau"][e_idx[:, None], steps[:, :-1]]
    t0 = time.perf_counter()
    Q, V, R = rollout(model, Q_true[:, 0], V_true[:, 0], tau, substeps=substeps)
    wall = time.perf_counter() - t0
    eq, ev = rollout_errors(Q, V, Q_true, V_true)
    info = {"wall_s": wall,
            "max_newton_rel_residual": float(np.nanmax(R)) if np.isfinite(R).any() else np.nan,
            "median_newton_rel_residual": float(np.nanmedian(R)) if np.isfinite(R).any() else np.nan}
    return eq, ev, Q, info


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

def _gt_path(dist):
    return os.path.join(RESULTS_DIR, f"rollout_ground_truth_{dist}.npz")


def load_or_make_ground_truth(dist, n_episodes, regenerate=False):
    path = _gt_path(dist)
    if os.path.exists(path) and not regenerate:
        d = np.load(path, allow_pickle=False)
        gt = {k: d[k] for k in d.files}
        if gt["q"].shape[0] == n_episodes:
            return gt
    gt = generate_ground_truth(dist, n_episodes)
    np.savez_compressed(path, **{k: v for k, v in gt.items()})
    return gt


def _plot(curves, dist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, c in curves.items():
        t = np.arange(len(c)) * CONTROL_DT
        ax.semilogy(t[1:], c[1:], label=name, lw=2 if name.endswith("100%") or name == "analytical"
                    else 1, alpha=1 if name.endswith("100%") or name == "analytical" else 0.6)
    ax.set_xlabel("Rollout time (s)", fontsize=12)
    ax.set_ylabel("Position RMSE over joints and rollouts (rad)", fontsize=11)
    ax.set_title(f"Open-loop rollout error vs MuJoCo ({dist} excitation)", fontsize=13)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_example(gt, wins, traces, dist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # first window from a smooth (sinusoidal/chirp) episode that does not start at rest
    smooth = [n for n, (e, k0) in enumerate(wins) if str(gt["mode"][e]) != "random" and k0 > 0]
    i = smooth[0] if smooth else 0
    e, k0 = wins[i]
    K = next(iter(traces.values())).shape[1] - 1
    t = np.arange(K + 1) * CONTROL_DT
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(t, gt["q"][e, k0:k0 + K + 1, j], "k-", lw=2.5, label="MuJoCo")
        for name, Q in traces.items():
            bad = np.flatnonzero(~np.isfinite(Q[i, :, 0]))
            label = name if bad.size == 0 else f"{name} (diverged at {t[bad[0]]:.2f} s)"
            ax.plot(t, Q[i, :, j], "--", lw=1.6, label=label)
        ax.set_title(JOINTS[j], fontsize=12); ax.grid(alpha=0.3)
        ax.set_ylabel("q (rad)", fontsize=11)
    for ax in axes[1]:
        ax.set_xlabel("Time (s)", fontsize=11)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"Example 2 s rollout under recorded torques ({dist}, {gt['mode'][e]} excitation)",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    from .models import ALL_MODEL_SPECS, MAIN_MODEL_SPECS, load_models
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--quick", action="store_true", help="main models only, 3 episodes")
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--substeps", type=int, default=SUBSTEPS,
                   help="integration substeps per 10 ms control step (default 1)")
    args = p.parse_args()
    ensure_dirs()
    n_ep = 3 if args.quick else args.episodes
    models = load_models(MAIN_MODEL_SPECS if args.quick else ALL_MODEL_SPECS)

    rows, curves_all, info_rows = [], {}, []
    for dist in ("test", "ood"):
        gt = load_or_make_ground_truth(dist, n_ep, args.regenerate)
        wins = rollout_windows(gt)
        print(f"[{dist}] {len(wins)} constraint-free 2 s windows from {n_ep} episodes")
        curves, traces = {}, {}
        for m in models:
            eq, ev, Q, info = evaluate_model(m, gt, wins, substeps=args.substeps)
            for r in horizon_metrics(eq, ev):
                rows.append({"model": m.name, "kind": m.kind, "fraction": m.fraction,
                             "dist": dist, **r})
            ok = np.isfinite(eq).all(axis=(1, 2))
            curves[m.name] = np.sqrt(np.mean(eq[ok] ** 2, axis=(0, 2))) if ok.any() \
                else np.full(eq.shape[1], np.nan)
            if m.fraction in (None, 1.0):
                traces[m.name] = Q
            info_rows.append({"model": m.name, "dist": dist, "n_windows": len(wins), **info})
            r1 = [r for r in rows if r["model"] == m.name and r["dist"] == dist][1]
            print(f"  {m.name:14s} 1 s pos RMSE {r1.get('pos_rmse', np.nan):.3e} rad   "
                  f"diverged {r1['frac_diverged']:.0%}   ({info['wall_s']:.1f} s)")
        curves_all[dist] = curves
        _plot(curves, dist, os.path.join(FIG_DIR, f"rollout_error_{dist}.png"))
        _plot_example(gt, wins, traces, dist, os.path.join(FIG_DIR, f"rollout_example_{dist}.png"))

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "rollout_metrics.csv"), index=False)
    pd.DataFrame(info_rows).to_csv(os.path.join(RESULTS_DIR, "rollout_info.csv"), index=False)
    np.savez_compressed(os.path.join(RESULTS_DIR, "rollout_curves.npz"),
                        **{f"{d}|{n}": c for d, cs in curves_all.items() for n, c in cs.items()})
    with open(os.path.join(RESULTS_DIR, "rollout_protocol.json"), "w") as f:
        json.dump({"horizons_s": HORIZONS_S, "rollout_s": ROLLOUT_S, "start_every_s": START_EVERY_S,
                   "episodes_per_dist": n_ep, "seeds": GT_SEEDS, "physics_dt": PHYSICS_DT, "model_step_s": CONTROL_DT / args.substeps,
                   "control_dt": CONTROL_DT,
                   "newton_iters": {m.kind: m.newton_iters for m in models},
                   "line_search": {m.kind: m.line_search for m in models},
                   "integrator": "implicit-in-velocity Euler, tau zero-order hold"}, f, indent=2)
    print("wrote results/rollout_metrics.csv and figures/rollout_*.png")


if __name__ == "__main__":
    main()
