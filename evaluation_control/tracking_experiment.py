"""Closed-loop computed-torque tracking in MuJoCo with the analytical, MLP and DeLaN models.

Plant: Member 1's contact-free arm (``dataset/environment.py::ArmDynamicsEnv``, the
bring_ball MJCF with the gripper rigid), 1 ms physics, 10 ms control with zero-order hold.
Each controller tracks the 10 main reference trajectories (inside the training range) and the
4 OOD trajectories (``trajectories.py``). One MuJoCo env per trajectory is stepped in lock-step,
so the model is called once per control step on a batch of states.

Controllers:
  * computed torque, ``sampled`` variant (default; see computed_torque.py), for every model
    and data fraction
  * computed torque, ``classic`` variant, for the three main models
  * PD only (no model)

Metrics per run (errors wrapped to [-pi, pi)):
  tracking RMSE (overall and per joint), max |e|, velocity RMSE, RMS torque,
  fraction of saturated torque samples, controller wall time per step.

Run:  python -m evaluation_control.tracking_experiment [--quick] [--video] [--offset 0.1]
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from .common import CONTROL_DT, FIG_DIR, GEAR, JOINTS, N_DOF, RESULTS_DIR, ensure_dirs, wrap_angle
from .computed_torque import ComputedTorqueController, PDController
from . import trajectories as trajmod


def run_tracking(controller, ref, offset=0.0, video_env_index=None, video_every=2,
                 render_size=(240, 320)):
    """Track every trajectory in ``ref`` (dict from trajectories.load) with ``controller``.

    Returns logs q, qd (B, T+1, 4), tau (B, T, 4), the unclipped tau, the wall time of every
    control step, and, if requested, rendered frames of one env.
    """
    from dataset.environment import ArmDynamicsEnv, get_arm_state, set_arm_state
    q_ref, qd_ref, qdd_ref = ref["q"], ref["qd"], ref["qdd"]
    B, T1, _ = q_ref.shape
    T = T1 - 1
    envs = [ArmDynamicsEnv() for _ in range(B)]
    for b, env in enumerate(envs):
        env.reset()
        set_arm_state(env.physics, q_ref[b, 0] + offset, qd_ref[b, 0])
        env.physics.forward()
    Q = np.zeros((B, T1, N_DOF)); QD = np.zeros_like(Q)
    TAU = np.zeros((B, T, N_DOF)); TAU_RAW = np.zeros_like(TAU)
    wall = np.zeros(T)
    frames = []
    for k in range(T1):
        for b, env in enumerate(envs):
            Q[b, k], QD[b, k] = get_arm_state(env.physics)
        if video_env_index is not None and k % video_every == 0:
            frames.append(envs[video_env_index].physics.render(*render_size, camera_id=0))
        if k == T:
            break
        t0 = time.perf_counter()
        tau, tau_raw = controller.command(Q[:, k], QD[:, k], q_ref[:, k], qd_ref[:, k], qdd_ref[:, k])
        wall[k] = time.perf_counter() - t0
        if not np.all(np.isfinite(tau)):
            tau = np.where(np.isfinite(tau), tau, 0.0)
        TAU[:, k], TAU_RAW[:, k] = tau, tau_raw
        for b, env in enumerate(envs):
            env.step(np.clip(tau[b] / GEAR, -1.0, 1.0))
    return {"q": Q, "qd": QD, "tau": TAU, "tau_raw": TAU_RAW, "wall": wall, "frames": frames}


def tracking_metrics(log, ref):
    e = wrap_angle(log["q"] - ref["q"])
    ev = log["qd"] - ref["qd"]
    rows = []
    for b in range(e.shape[0]):
        r = {"trajectory": str(ref["names"][b]),
             "rmse": float(np.sqrt(np.mean(e[b] ** 2))),
             "max_abs_err": float(np.abs(e[b]).max()),
             "vel_rmse": float(np.sqrt(np.mean(ev[b] ** 2))),
             "tau_rms": float(np.sqrt(np.mean(log["tau"][b] ** 2))),
             "saturation_frac": float(np.mean(np.abs(log["tau_raw"][b]) > GEAR + 1e-12)),
             "ctrl_ms_per_step_batched": float(1e3 * log["wall"].mean())}
        for j, name in enumerate(JOINTS):
            r[f"rmse_{name}"] = float(np.sqrt(np.mean(e[b, :, j] ** 2)))
        rows.append(r)
    return rows


def build_controllers(quick=False, wn=15.0):
    from .models import ALL_MODEL_SPECS, MAIN_MODEL_SPECS, load_models
    specs = MAIN_MODEL_SPECS if quick else ALL_MODEL_SPECS
    models = load_models(specs)
    ctrls = [ComputedTorqueController(m, wn=wn) for m in models]
    main = [m for m in models if m.fraction in (None, 1.0)]
    ctrls += [ComputedTorqueController(m, wn=wn, variant="classic") for m in main]
    ctrls.append(PDController())
    return ctrls


def _controller_row(c):
    m = c.model
    return {"controller": c.name, "model": m.name if m else "none",
            "kind": m.kind if m else "pd", "fraction": m.fraction if m else None,
            "variant": getattr(c, "variant", "pd")}


def _plot_traces(ref, logs, b, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(ref["q"].shape[1]) * CONTROL_DT
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharex=True)
    for j in range(N_DOF):
        axes[0, j].plot(t, ref["q"][b, :, j], "k-", lw=3, alpha=0.5, label="reference")
        for name, log in logs.items():
            axes[0, j].plot(t, log["q"][b, :, j], lw=1.4, label=name)
            axes[1, j].plot(t, wrap_angle(log["q"][b, :, j] - ref["q"][b, :, j]), lw=1.4, label=name)
        axes[0, j].set_title(JOINTS[j], fontsize=12)
        axes[1, j].set_xlabel("Time (s)", fontsize=11)
        axes[0, j].grid(alpha=0.3); axes[1, j].grid(alpha=0.3)
    axes[0, 0].set_ylabel("q (rad)", fontsize=11)
    axes[1, 0].set_ylabel("tracking error (rad)", fontsize=11)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(title, fontsize=13); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _plot_bars(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, which in zip(axes, ("main", "ood")):
        s = summary[summary["set"] == which].sort_values("rmse_mean")
        ax.barh(s["controller"], s["rmse_mean"], xerr=s["rmse_std"], color="tab:blue", alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Tracking RMSE (rad), mean ± std over trajectories", fontsize=11)
        ax.set_title("Inside the training range (10 trajectories)" if which == "main"
                     else "Outside the training range (4 trajectories)", fontsize=12)
        ax.tick_params(axis="y", labelsize=9); ax.grid(axis="x", which="both", alpha=0.3)
    fig.suptitle("Computed-torque tracking in MuJoCo", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _save_video(ref, ctrls_by_name, path, traj_index=0, fps=50):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    streams = {}
    for name, c in ctrls_by_name.items():
        one = {k: ref[k][traj_index:traj_index + 1] for k in ("q", "qd", "qdd", "names")}
        streams[name] = run_tracking(c, one, video_env_index=0)["frames"]
    n = min(len(f) for f in streams.values())
    with imageio.get_writer(path, fps=fps) as w:
        for i in range(n):
            tiles = []
            for name, fr in streams.items():
                im = Image.fromarray(fr[i]); d = ImageDraw.Draw(im)
                d.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
                d.text((6, 3), f"{name}   t = {i * 2 * CONTROL_DT:4.2f} s", fill=(255, 255, 255))
                tiles.append(np.asarray(im))
            w.append_data(np.concatenate(tiles, axis=1))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--quick", action="store_true", help="main models only")
    p.add_argument("--wn", type=float, default=15.0, help="outer-loop natural frequency (rad/s)")
    p.add_argument("--offset", type=float, default=0.0, help="initial position offset (rad)")
    p.add_argument("--video", action="store_true", help="render demo video (MUJOCO_GL=egl)")
    args = p.parse_args()
    ensure_dirs()
    refs = {w: trajmod.load(w) for w in ("main", "ood")}
    ctrls = build_controllers(args.quick, args.wn)

    rows, traces = [], {}
    for c in ctrls:
        for which, ref in refs.items():
            t0 = time.perf_counter()
            log = run_tracking(c, ref, offset=args.offset)
            for r in tracking_metrics(log, ref):
                rows.append({**_controller_row(c), "set": which, **r})
            traces[(c.name, which)] = log
            mean = np.mean([r["rmse"] for r in rows if r["controller"] == c.name and r["set"] == which])
            print(f"{c.name:28s} {which:4s} mean RMSE {mean:.3e} rad  ({time.perf_counter() - t0:.1f} s)")

    df = pd.DataFrame(rows)
    tag = "" if args.offset == 0 else f"_offset{args.offset:g}"
    df.to_csv(os.path.join(RESULTS_DIR, f"tracking_metrics{tag}.csv"), index=False)
    agg = {"rmse": ["mean", "std", "max"], "max_abs_err": "max", "vel_rmse": "mean",
           "tau_rms": "mean", "saturation_frac": "mean", "ctrl_ms_per_step_batched": "mean",
           **{f"rmse_{n}": "mean" for n in JOINTS}}
    summary = df.groupby(["controller", "model", "kind", "variant", "set"], dropna=False) \
        .agg(agg).reset_index()
    summary.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in summary.columns]
    summary = summary.merge(df[["controller", "fraction"]].drop_duplicates(), on="controller")
    summary.to_csv(os.path.join(RESULTS_DIR, f"tracking_summary{tag}.csv"), index=False)
    np.savez_compressed(os.path.join(RESULTS_DIR, f"tracking_traces{tag}.npz"),
                        **{f"{n}|{w}|{k}": log[k] for (n, w), log in traces.items()
                           for k in ("q", "qd", "tau")})
    _plot_bars(summary, os.path.join(FIG_DIR, f"tracking_rmse{tag}.png"))
    main_names = [c.name for c in ctrls
                  if (c.model is None) or (c.model.fraction in (None, 1.0) and c.variant == "sampled")]
    for which, b in (("main", 0), ("ood", 0)):
        _plot_traces(refs[which], {n: traces[(n, which)] for n in main_names}, b,
                     os.path.join(FIG_DIR, f"tracking_traces_{which}{tag}.png"),
                     f"Computed-torque tracking, trajectory {refs[which]['names'][b]}")
    classic = {c.name: traces[(c.name, "main")] for c in ctrls
               if getattr(c, "variant", "") == "classic"}
    if classic:
        _plot_traces(refs["main"], classic, 0, os.path.join(FIG_DIR, f"tracking_traces_classic{tag}.png"),
                     "Classic computed torque (tau = ID(q, qd, a_cmd)) on the sampled plant")
    if args.video:
        by_name = {c.name: c for c in ctrls if c.name in main_names and c.model is not None}
        _save_video(refs["main"], by_name, os.path.join(RESULTS_DIR, "tracking_demo.mp4"))
        _save_video(refs["ood"], by_name, os.path.join(RESULTS_DIR, "tracking_demo_ood.mp4"))
        print("wrote results/tracking_demo.mp4 and tracking_demo_ood.mp4")
    print(f"wrote results/tracking_metrics{tag}.csv, tracking_summary{tag}.csv and figures/tracking_*{tag}.png")


if __name__ == "__main__":
    main()
