import mujoco, pandas as pd
from dm_control import suite
from analytical_model import AnalyticalArm, ARM

m = suite.load("manipulator", "bring_ball").physics.model.ptr
arm = AnalyticalArm(m, ref_qpos=m.qpos0.copy())

bodies = []
for o, mass, com, I in arm.bodies:
    bodies.append({"owner_joint": ARM[o], "mass_kg": mass,
                   "com_x": com[0].item(), "com_y": com[1].item(), "com_z": com[2].item(),
                   "Iyy": I[1, 1].item()})
pd.DataFrame(bodies).to_csv("body_parameters.csv", index=False)

joints = [{"joint": n, "axis": m.jnt_axis[j].tolist(), "damping": m.dof_damping[m.jnt_dofadr[j]],
           "armature": m.dof_armature[m.jnt_dofadr[j]],
           "range_lo": m.jnt_range[j][0], "range_hi": m.jnt_range[j][1],
           "gear": m.actuator_gear[i][0]}
          for i, (n, j) in enumerate(zip(ARM, arm.jid))]
pd.DataFrame(joints).to_csv("joint_parameters.csv", index=False)
print(pd.DataFrame(bodies)); print(pd.DataFrame(joints))
