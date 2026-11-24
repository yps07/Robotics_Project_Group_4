"""Deck-specific figures, generated from the project's saved results (run from the repo root
with the project venv and MUJOCO_GL=egl). Colours match the deck palette."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = sys.argv[1]
FONTS = sys.argv[2]
for f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"):
    font_manager.fontManager.addfont(os.path.join(FONTS, "liberation-fonts-ttf-2.1.5", f))
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 15, "axes.titlesize": 17,
                     "axes.labelsize": 15, "legend.fontsize": 13, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "8A96A3",
                     "xtick.color": "4A5868", "ytick.color": "4A5868", "axes.labelcolor": "1B2430"})
COL = {"analytical": "#4A5868", "delan@100%": "#1F8A8A", "mlp@100%": "#E0892E", "PD only": "#A7B1BC",
       "MuJoCo": "#1B2430"}
LAB = {"analytical": "Analytical", "delan@100%": "DeLaN", "mlp@100%": "MLP", "PD only": "PD only"}

sys.path.insert(0, os.getcwd())
from evaluation_control.common import CONTROL_DT, wrap_angle  # noqa: E402
from evaluation_control import trajectories as T  # noqa: E402

# 1. tracking on an OOD trajectory: error vs time, root and elbow
tr = np.load("evaluation_control/results/tracking_traces.npz")
ood = T.load("ood")
t = np.arange(ood["q"].shape[1]) * CONTROL_DT
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
for ax, j, name in ((axes[0], 0, "Root joint"), (axes[1], 2, "Elbow joint")):
    for c in ("PD only", "mlp@100%", "delan@100%", "analytical"):
        q = tr[f"{c}|ood|q"][0, :, j]
        ax.plot(t, wrap_angle(q - ood["q"][0, :, j]), color=COL[c], lw=2.6 if c != "PD only" else 2,
                label=LAB[c])
    ax.set_title(name, loc="left", fontweight="bold")
    ax.set_xlabel("Time (s)"); ax.grid(alpha=0.25)
axes[0].set_ylabel("Tracking error (rad)")
axes[0].legend(ncol=2, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "tracking_ood.png"), dpi=200); plt.close(fig)

# 2. example rollout against MuJoCo (chirp window from the test ground truth)
from evaluation_control.models import load_models, MAIN_MODEL_SPECS  # noqa: E402
from evaluation_control import rollout as R  # noqa: E402
d = np.load("evaluation_control/results/rollout_ground_truth_test.npz")
gt = {k: d[k] for k in d.files}
wins = R.rollout_windows(gt)
smooth = [w for w in wins if str(gt["mode"][w[0]]) != "random" and w[1] > 0]
win = [smooth[0]]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
traces = {}
for m in load_models(MAIN_MODEL_SPECS):
    _, _, Q, _ = R.evaluate_model(m, gt, win)
    traces[m.name] = Q[0]
e, k0 = win[0]
tt = np.arange(201) * CONTROL_DT
for ax, j, name in ((axes[0], 0, "Root joint"), (axes[1], 2, "Elbow joint")):
    ax.plot(tt, gt["q"][e, k0:k0 + 201, j], color=COL["MuJoCo"], lw=4, alpha=0.35, label="MuJoCo")
    for n in ("analytical", "mlp@100%", "delan@100%"):
        ax.plot(tt, traces[n][:, j], "--" if n != "analytical" else ":", color=COL[n], lw=2.4, label=LAB[n])
    ax.set_title(name, loc="left", fontweight="bold"); ax.set_xlabel("Time (s)"); ax.grid(alpha=0.25)
axes[0].set_ylabel("Joint angle (rad)")
axes[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "rollout_example.png"), dpi=200); plt.close(fig)

# 3. hero render of the manipulator task
from dm_control import suite  # noqa: E402
env = suite.load("manipulator", "bring_ball", task_kwargs={"random": 3})
env.reset()
for _ in range(40):
    env.step(np.array([0.25, -0.3, 0.4, 0.3, 0.0]))
img = env.physics.render(480, 640, camera_id=0)
plt.imsave(os.path.join(OUT, "hero.png"), img)

# 4. arm model render (reduced data model) in a reference pose
from dataset.environment import ArmDynamicsEnv, set_arm_state  # noqa: E402
ae = ArmDynamicsEnv()
set_arm_state(ae.physics, np.array([0.6, -0.9, 1.3, 0.7]), np.zeros(4)); ae.physics.forward()
plt.imsave(os.path.join(OUT, "arm.png"), ae.physics.render(480, 640, camera_id=0))
print("figures written to", OUT)
