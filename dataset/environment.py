"""Helpers for the standard suite task and an isolated arm-dynamics model.

The data-collection model is derived from the suite's ``manipulator/bring_ball``
MJCF. Its four gripper hinges are removed (the finger bodies remain rigidly
attached to the wrist), and collision geoms are disabled. This isolates the
four actuated arm coordinates while retaining their MuJoCo inertial parameters.
The ordinary suite environment remains available through :func:`load_env` for
inspection and later closed-loop experiments.
"""

import numpy as np
import mujoco
from dm_control import suite
from dm_control.suite import manipulator
from lxml import etree

ARM_JOINT_NAMES = (
    "arm_root",
    "arm_shoulder",
    "arm_elbow",
    "arm_wrist",
)
GRIPPER_JOINT_NAMES = ("finger", "fingertip", "thumb", "thumbtip")
BALL_JOINT_NAMES = ("ball_x", "ball_z", "ball_y")

# These indices are valid for the standard suite model, and also for the
# reduced data model's first four arm joints.
ARM_JOINTS = [0, 1, 2, 3]
ARM_ACTUATORS = [0, 1, 2, 3]
GRIPPER_ACTUATOR = 4
GEAR_RATIOS = np.array([12.0, 8.0, 4.0, 2.0])
N_ARM = 4
CONTROL_TIMESTEP_S = 0.01

# Joint limits from the MuJoCo model. arm_root is an unlimited hinge.
JOINT_LIMITS = np.array([
    [-np.inf, np.inf],
    [-2.792526803190927, 2.792526803190927],
    [-2.792526803190927, 2.792526803190927],
    [-2.443460952792061, 2.443460952792061],
])
# MuJoCo does not impose arm velocity limits; this is only a dataset filter.
MAX_ABS_QDOT_FOR_DATA = 10.0
TRAIN_MAX_ABS_QDDOT_FOR_DATA = 1200.0
OOD_MAX_ABS_QDDOT_FOR_DATA = 1800.0


def _joint_ids(physics, names):
    model = physics.model.ptr
    ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in names
    ]
    if any(joint_id < 0 for joint_id in ids):
        raise ValueError(f"Model is missing one or more joints: {names}")
    return ids


def load_env(task="bring_ball", random_seed=0):
    """Load the ordinary dm_control manipulator task."""
    return suite.load("manipulator", task, task_kwargs={"random": random_seed})


class ArmDynamicsEnv:
    """Small control-step wrapper around a contact-free, four-DOF arm model."""

    def __init__(self, random_seed=0, control_timestep=CONTROL_TIMESTEP_S):
        del random_seed  # This reduced model has no randomized task state.
        xml_string, assets = manipulator.make_model(use_peg=False, insert=False)
        root = etree.fromstring(xml_string)

        # Preserve finger bodies and their inertias, but make them rigid parts
        # of the wrist by removing their hinge joints.
        for joint in list(root.iter("joint")):
            if joint.get("name") in GRIPPER_JOINT_NAMES:
                joint.getparent().remove(joint)

        # The grasp motor/tendons only actuate the removed gripper coordinates.
        for actuator in list(root.iter()):
            if (actuator.get("name") == "grasp"
                    and actuator.getparent() is not None
                    and actuator.getparent().tag == "actuator"):
                actuator.getparent().remove(actuator)
        for section_name in ("tendon", "equality"):
            section = root.find(section_name)
            if section is not None:
                root.remove(section)

        # This dataset is for free-space rigid-body dynamics. Disable all
        # collision pairs, including self/floor contacts, not just the ball.
        for geom in root.iter("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

        reduced_xml = etree.tostring(root, encoding="unicode")
        self.physics = manipulator.Physics.from_xml_string(reduced_xml, assets)
        self._control_timestep = float(control_timestep)
        physics_timestep = float(self.physics.model.opt.timestep)
        n_substeps = round(self._control_timestep / physics_timestep)
        if n_substeps < 1 or not np.isclose(
            n_substeps * physics_timestep, self._control_timestep
        ):
            raise ValueError(
                "Control timestep must be a positive integer multiple of the "
                f"MuJoCo timestep ({physics_timestep:g} s)."
            )
        self.n_substeps = int(n_substeps)
        self.time = 0.0
        self.reset()

    def control_timestep(self):
        return self._control_timestep

    def reset(self):
        """Reset to a repeatable arm pose with the ball outside the workspace."""
        self.physics.reset()
        data = self.physics.data
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.ctrl[:] = 0.0
        set_arm_state(self.physics, np.zeros(N_ARM), np.zeros(N_ARM))
        move_ball_far(self.physics)
        self.physics.forward()
        self.time = 0.0
        if data.ncon:
            raise RuntimeError("Collision-disabled arm model unexpectedly has contacts.")

    def step(self, action):
        """Apply a normalized four-motor action for one control interval."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (N_ARM,):
            raise ValueError(f"Expected a ({N_ARM},) action, got {action.shape}.")
        if not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.0 + 1e-12):
            raise ValueError("Actions must be finite and lie in [-1, 1].")
        self.physics.data.ctrl[:N_ARM] = action
        self.physics.step(nstep=self.n_substeps)
        self.physics.forward()
        self.time += self._control_timestep
        if self.physics.data.ncon:
            raise RuntimeError("Collision-disabled arm model unexpectedly has contacts.")
        return self.physics.data


def load_dynamics_env(random_seed=0):
    """Load the bring_ball-derived, fixed-gripper dynamics environment."""
    return ArmDynamicsEnv(random_seed=random_seed)


def get_arm_state(physics):
    """Return arm ``(q, qdot)`` in the declared joint order."""
    joint_ids = _joint_ids(physics, ARM_JOINT_NAMES)
    model = physics.model.ptr
    qpos_ids = [model.jnt_qposadr[joint_id] for joint_id in joint_ids]
    dof_ids = [model.jnt_dofadr[joint_id] for joint_id in joint_ids]
    return (
        np.asarray(physics.data.qpos[qpos_ids], dtype=np.float64).copy(),
        np.asarray(physics.data.qvel[dof_ids], dtype=np.float64).copy(),
    )


def get_arm_torque(physics):
    """Return the four arm actuator forces in generalized-coordinate order."""
    model = physics.model.ptr
    dof_ids = [
        model.jnt_dofadr[joint_id]
        for joint_id in _joint_ids(physics, ARM_JOINT_NAMES)
    ]
    return np.asarray(physics.data.qfrc_actuator[dof_ids], dtype=np.float64).copy()


def get_arm_acc(physics):
    """Return arm generalized accelerations in the declared joint order."""
    model = physics.model.ptr
    dof_ids = [
        model.jnt_dofadr[joint_id]
        for joint_id in _joint_ids(physics, ARM_JOINT_NAMES)
    ]
    return np.asarray(physics.data.qacc[dof_ids], dtype=np.float64).copy()


def move_ball_far(physics, distance=10.0):
    """Move the ball away by joint name (works in both standard/reduced models)."""
    model = physics.model.ptr
    joint_ids = _joint_ids(physics, BALL_JOINT_NAMES)
    for joint_id in joint_ids:
        qpos_id = model.jnt_qposadr[joint_id]
        dof_id = model.jnt_dofadr[joint_id]
        physics.data.qpos[qpos_id] = distance
        physics.data.qvel[dof_id] = 0.0


def set_arm_state(physics, q, qd):
    """Set the four arm coordinates without assuming their array offsets."""
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    if q.shape != (N_ARM,) or qd.shape != (N_ARM,):
        raise ValueError(f"Expected q and qd with shape ({N_ARM},).")
    model = physics.model.ptr
    joint_ids = _joint_ids(physics, ARM_JOINT_NAMES)
    for k, joint_id in enumerate(joint_ids):
        physics.data.qpos[model.jnt_qposadr[joint_id]] = q[k]
        physics.data.qvel[model.jnt_dofadr[joint_id]] = qd[k]


def action_from_torque(tau):
    """Convert desired arm torque to normalized actuator controls."""
    return np.clip(np.asarray(tau) / GEAR_RATIOS, -1.0, 1.0)
