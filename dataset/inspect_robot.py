"""Inspect the manipulator MuJoCo model and dump joint/actuator/physics info."""

import os
import json
import numpy as np
from dm_control import suite
try:
    from .environment import (
        ARM_JOINTS,
        ARM_ACTUATORS,
        GEAR_RATIOS,
        JOINT_LIMITS,
    )
except ImportError:  # Support running this file directly from dataset/.
    from environment import (
        ARM_JOINTS,
        ARM_ACTUATORS,
        GEAR_RATIOS,
        JOINT_LIMITS,
    )

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def inspect():
    env = suite.load("manipulator", "bring_ball", task_kwargs={"random": 0})
    physics = env.physics
    model = physics.model
    info = {}
    info["nq"] = int(model.nq)
    info["nv"] = int(model.nv)
    info["nu"] = int(model.nu)
    info["timestep"] = float(model.opt.timestep)
    info["control_timestep"] = float(env.control_timestep())
    info["arm_joints"] = []
    for i in ARM_JOINTS:
        info["arm_joints"].append({
            "index": int(i),
            "name": model.joint(i).name,
            "type": int(model.jnt_type[i]),
            "qposadr": int(model.jnt_qposadr[i]),
            "dofadr": int(model.jnt_dofadr[i]),
            "range": (
                [float(x) for x in model.jnt_range[i]]
                if model.jnt_limited[i]
                else None
            ),
            "limited": bool(model.jnt_limited[i]),
            "damping": float(model.dof_damping[i]),
            "armature": float(model.dof_armature[i]),
            "frictionloss": float(model.dof_frictionloss[i]),
        })
    info["arm_actuators"] = []
    for i in ARM_ACTUATORS:
        info["arm_actuators"].append({
            "index": int(i),
            "name": model.actuator(i).name,
            "gear": [float(x) for x in model.actuator_gear[i]],
            "ctrlrange": [float(x) for x in model.actuator_ctrlrange[i]],
        })
    info["gear_ratios"] = [float(x) for x in GEAR_RATIOS]
    info["joint_limits"] = [
        [float(x) if np.isfinite(x) else None for x in limits]
        for limits in JOINT_LIMITS
    ]

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "robot_info.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2, allow_nan=False)
    print(json.dumps(info, indent=2))
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    inspect()
