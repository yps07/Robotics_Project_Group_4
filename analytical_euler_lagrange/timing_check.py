import time
import numpy as np
from dm_control import suite
from analytical_model import AnalyticalArm

m = suite.load("manipulator", "bring_ball").physics.model.ptr
arm = AnalyticalArm(m, ref_qpos=m.qpos0.copy())

q, qd, qdd = np.zeros(4), np.zeros(4), np.zeros(4)
arm.inverse_dynamics(q, qd, qdd)          # warm-up call

t = time.time()
for _ in range(100):
    arm.inverse_dynamics(q, qd, qdd)
print("inverse_dynamics:", (time.time() - t) / 100, "s per call")

tau = np.zeros(4)
t = time.time()
for _ in range(100):
    arm.forward_dynamics(q, qd, tau)
print("forward_dynamics:", (time.time() - t) / 100, "s per call")
