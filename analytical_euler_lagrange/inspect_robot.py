import mujoco
from dm_control import suite

env = suite.load("manipulator", "bring_ball")
m = env.physics.model.ptr
print("gravity:", m.opt.gravity, " nq:", m.nq, " nv:", m.nv, " nu:", m.nu)

print("\nJOINTS: name, type, dofadr, axis, range(limited), damping, armature, frictionloss")
for j in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    d = m.jnt_dofadr[j]
    print(f"{j:2d} {name:16s} type={m.jnt_type[j]} dof={d:2d} axis={m.jnt_axis[j]} "
          f"range={m.jnt_range[j]}({m.jnt_limited[j]}) damp={m.dof_damping[d]:.4g} "
          f"arm={m.dof_armature[d]:.4g} fric={m.dof_frictionloss[d]:.4g}")

print("\nBODIES: name, parent, mass")
for b in range(m.nbody):
    print(f"{b:2d} {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b):16s} "
          f"parent={m.body_parentid[b]:2d} mass={m.body_mass[b]:.4g}")

print("\nACTUATORS: name, gear")
for a in range(m.nu):
    print(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a), m.actuator_gear[a][0])
