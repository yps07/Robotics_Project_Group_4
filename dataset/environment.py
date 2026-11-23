"""MuJoCo / dm_control environment helpers for the manipulator (bring_ball) task."""

import os
import numpy as np
from dm_control import suite

# Arm joint / actuator indices (0-based) inside the full model
ARM_JOINTS = [0, 1, 2, 3]          # arm_root, arm_shoulder, arm_elbow, arm_wrist
ARM_ACTUATORS = [0, 1, 2, 3]       # root, shoulder, elbow, wrist
GRIPPER_ACTUATOR = 4
BALL_QPOS_INDICES = [8, 9, 10]     # ball_x, ball_z, ball_y
BALL_QVEL_INDICES = [8, 9, 10]

GEAR_RATIOS = np.array([12.0, 8.0, 4.0, 2.0])
N_ARM = 4

# Joint limits for the 4 arm joints (from model inspection)
JOINT_LIMITS = np.array([
    [-np.inf, np.inf],            # arm_root (unlimited hinge)
    [-2.7925268, 2.7925268],     # arm_shoulder
    [-2.7925268, 2.7925268],     # arm_elbow
    [-2.44346095, 2.44346095],   # arm_wrist
])
# Velocity limits used for data generation (rad/s)
VEL_LIMITS = np.array([3.0, 3.0, 3.0, 3.0])


def load_env(task="bring_ball", random_seed=0):
    env = suite.load("manipulator", task, task_kwargs={"random": random_seed})
    return env


def get_arm_state(physics):
    """Return (q, qdot) for the 4 arm joints."""
    q = np.array([physics.data.qpos[i] for i in ARM_JOINTS], dtype=np.float64)
    qd = np.array([physics.data.qvel[i] for i in ARM_JOINTS], dtype=np.float64)
    return q, qd


def get_arm_torque(physics):
    """Return tau for the 4 arm joints from qfrc_actuator."""
    tau = np.array([physics.data.qfrc_actuator[i] for i in ARM_JOINTS], dtype=np.float64)
    return tau


def get_arm_acc(physics):
    """Return qddot for the 4 arm joints."""
    acc = np.array([physics.data.qacc[i] for i in ARM_JOINTS], dtype=np.float64)
    return acc


def move_ball_far(physics, distance=10.0):
    """Move the ball out of the arm's workspace to avoid contacts during data collection."""
    for idx in BALL_QPOS_INDICES:
        physics.data.qpos[idx] = distance
    for idx in BALL_QVEL_INDICES:
        physics.data.qvel[idx] = 0.0


def set_arm_state(physics, q, qd):
    for k, i in enumerate(ARM_JOINTS):
        physics.data.qpos[i] = q[k]
        physics.data.qvel[i] = qd[k]


def action_from_torque(tau):
    """Convert desired torque to action in [-1,1] using gear ratios (clipped)."""
    act = tau / GEAR_RATIOS
    return np.clip(act, -1.0, 1.0)
