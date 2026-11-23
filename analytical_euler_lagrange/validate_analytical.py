import numpy as np, mujoco, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dm_control import suite
from analytical_model import AnalyticalArm, ARM
import pandas as pd

env = suite.load("manipulator", "bring_ball")
m, d = env.physics.model.ptr, env.physics.data.ptr
arm = AnalyticalArm(m, ref_qpos=m.qpos0.copy())
dofs, jid = arm.dofs, arm.jid

rng = np.random.default_rng(0)
lo = np.array([m.jnt_range[j][0] if m.jnt_limited[j] else -np.pi for j in jid])
hi = np.array([m.jnt_range[j][1] if m.jnt_limited[j] else np.pi for j in jid])
mid, half = (lo + hi) / 2, 0.45 * (hi - lo)          # stay well inside limits

tau_mj, tau_an, kept = [], [], 0
for _ in range(5000):
    q = mid + rng.uniform(-1, 1, 4) * half
    qd = rng.uniform(-3, 3, 4)
    qdd = rng.uniform(-10, 10, 4)
    d.qpos[:] = m.qpos0; d.qvel[:] = 0; d.qacc[:] = 0     # fingers fixed at reference
    d.qpos[dofs] = q; d.qvel[dofs] = qd; d.qacc[dofs] = qdd
    mujoco.mj_inverse(m, d)
    if d.ncon > 0:                                         # skip contact states
        continue
    tau_mj.append(d.qfrc_inverse[dofs].copy())
    tau_an.append(arm.inverse_dynamics(q, qd, qdd))
    kept += 1
    if kept >= 1000: break

tau_mj, tau_an = np.array(tau_mj), np.array(tau_an)
err = tau_an - tau_mj
print(f"samples kept: {kept}")
print("max |err| per joint:", np.abs(err).max(0))
print("RMS err per joint  :", np.sqrt((err**2).mean(0)))

# term-by-term checks (version-proof): at qvel=0, tau(qacc) = M*qacc + G
def mj_tau(q, qd, qdd):
    d.qpos[:] = m.qpos0; d.qvel[:] = 0; d.qacc[:] = 0
    d.qpos[dofs] = q; d.qvel[dofs] = qd; d.qacc[dofs] = qdd
    mujoco.mj_inverse(m, d)
    return d.qfrc_inverse[dofs].copy()

q = mid.copy()
G_mj = mj_tau(q, np.zeros(4), np.zeros(4))
M_mj = np.zeros((4, 4))
for i in range(4):
    e = np.zeros(4); e[i] = 1.0
    M_mj[:, i] = mj_tau(q, np.zeros(4), e) - G_mj
print("M  max err:", np.abs(arm.M(q) - M_mj).max())
print("G  max err:", np.abs(arm.G(q) - G_mj).max())

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    ax[i].scatter(tau_mj[:, i], tau_an[:, i], s=4)
    lim = [tau_mj[:, i].min(), tau_mj[:, i].max()]
    ax[i].plot(lim, lim, "r--")
    ax[i].set_title(ARM[i]); ax[i].set_xlabel("MuJoCo torque (N·m)"); ax[i].set_ylabel("Analytical torque (N·m)")
plt.tight_layout(); plt.savefig("analytical_validation.png", dpi=150)

df = pd.DataFrame({"joint": ARM,
                   "max_abs_err": np.abs(err).max(0),
                   "rms_err": np.sqrt((err**2).mean(0))})
df.to_csv("analytical_validation_errors.csv", index=False)
print(df)
