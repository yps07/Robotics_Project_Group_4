"""Collect inverse-dynamics datasets for the manipulator arm.

Each sample: X = [q(4), qdot(4), qddot(4)] (12), y = tau(4).
Also stores q, qdot, qddot, tau separately.
"""

import os
import argparse
import json
import numpy as np
from dm_control import suite
from environment import (
    ARM_JOINTS, ARM_ACTUATORS, GEAR_RATIOS, JOINT_LIMITS, VEL_LIMITS,
    load_env, get_arm_state, get_arm_torque, get_arm_acc, move_ball_far,
    set_arm_state,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def make_action_random(t, rng, scale=0.35):
    return rng.uniform(-scale, scale, size=4)


def make_action_sinusoidal(t, rng, freqs, phases, amps):
    a = amps * np.sin(2 * np.pi * freqs * t + phases)
    return np.clip(a, -1.0, 1.0)


def make_action_chirp(t, rng, f0, f1, T, phases, amplitude=0.35):
    f = f0 + (f1 - f0) * (t / T)
    a = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * T)) + phases)
    return amplitude * np.clip(a, -1.0, 1.0)


def collect_episode(env, mode, T, rng, freqs=None, phases=None, amps=None, f0=0.1, f1=1.0, action_scale=0.35):
    """Collect T samples from one episode. Returns arrays (T,4) for q,qd,qdd,tau."""
    physics = env.physics
    env.reset()
    move_ball_far(physics)
    # initial state
    q0 = np.zeros(4)
    qd0 = np.zeros(4)
    set_arm_state(physics, q0, qd0)
    physics.forward()

    qs, qds, qdds, taus = [], [], [], []
    for step in range(T):
        t = step * 0.001  # timestep
        if mode == "random":
            a_arm = make_action_random(t, rng, scale=action_scale)
        elif mode == "sinusoidal":
            a_arm = make_action_sinusoidal(t, rng, freqs, phases, amps)
        elif mode == "chirp":
            a_arm = make_action_chirp(t, rng, f0, f1, T * 0.001, phases, action_scale)
        else:
            raise ValueError(mode)
        # full action including gripper (keep gripper neutral)
        action = np.zeros(5)
        action[:4] = a_arm
        action[4] = 0.0

        # step
        ts = env.step(action)
        # move ball far to avoid contacts, recompute dynamics
        move_ball_far(physics)
        physics.forward()

        q, qd = get_arm_state(physics)
        # Hinge angles are periodic. Store the root angle canonically so it
        # cannot drift to many revolutions during a long rollout.
        q[0] = (q[0] + np.pi) % (2 * np.pi) - np.pi
        tau = get_arm_torque(physics)
        qdd = get_arm_acc(physics)

        qs.append(q.copy())
        qds.append(qd.copy())
        qdds.append(qdd.copy())
        taus.append(tau.copy())

    return (np.array(qs), np.array(qds), np.array(qdds), np.array(taus))


def collect_split(n_samples, modes, rng, ood=False):
    """Collect ~n_samples across modes."""
    qs_all, qds_all, qdds_all, taus_all = [], [], [], []
    collected = 0
    episode_len = 1000
    while collected < n_samples:
        mode = rng.choice(modes)
        if mode == "sinusoidal":
            freqs = rng.uniform(0.1, 1.0, size=4)
            phases = rng.uniform(0, 2 * np.pi, size=4)
            # OOD uses stronger excitation, but remains a valid simulator
            # rollout. It must never be fabricated by scaling inputs only.
            amp = 0.65 if ood else 0.35
            amps = rng.uniform(0.15, amp, size=4)
            kw = dict(freqs=freqs, phases=phases, amps=amps)
        elif mode == "chirp":
            phases = rng.uniform(0, 2 * np.pi, size=4)
            kw = dict(phases=phases, f0=rng.uniform(0.05, 0.2), f1=rng.uniform(0.5, 2.0))
        else:
            kw = {}
        env = load_env(random_seed=int(rng.integers(0, 10000)))
        # Collect extra samples because boundary/impulse samples are rejected
        # below rather than silently entering the learning set.
        T = episode_len
        q, qd, qdd, tau = collect_episode(
            env, mode, T, rng, action_scale=(0.65 if ood else 0.35), **kw
        )
        margin = 0.05
        valid = np.all(q[:, 1:] >= (JOINT_LIMITS[1:, 0] + margin), axis=1)
        valid &= np.all(q[:, 1:] <= (JOINT_LIMITS[1:, 1] - margin), axis=1)
        valid &= np.max(np.abs(qd), axis=1) < 10.0
        valid &= np.max(np.abs(qdd), axis=1) < (1800.0 if ood else 1200.0)
        q, qd, qdd, tau = q[valid], qd[valid], qdd[valid], tau[valid]
        qs_all.append(q)
        qds_all.append(qd)
        qdds_all.append(qdd)
        taus_all.append(tau)
        collected += len(q)
    q = np.concatenate(qs_all, axis=0)[:n_samples]
    qd = np.concatenate(qds_all, axis=0)[:n_samples]
    qdd = np.concatenate(qdds_all, axis=0)[:n_samples]
    tau = np.concatenate(taus_all, axis=0)[:n_samples]
    return q, qd, qdd, tau


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=20000)
    parser.add_argument("--val", type=int, default=2000)
    parser.add_argument("--test", type=int, default=2000)
    parser.add_argument("--ood", type=int, default=2000)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    os.makedirs(DATA_DIR, exist_ok=True)

    modes = ["random", "sinusoidal", "chirp"]

    print("Collecting train...")
    q, qd, qdd, tau = collect_split(args.train, modes, rng, ood=False)
    np.savez(os.path.join(DATA_DIR, "train.npz"), q=q, qdot=qd, qddot=qdd, tau=tau)
    print("train", q.shape)

    print("Collecting val...")
    q, qd, qdd, tau = collect_split(args.val, modes, rng, ood=False)
    np.savez(os.path.join(DATA_DIR, "val.npz"), q=q, qdot=qd, qddot=qdd, tau=tau)
    print("val", q.shape)

    print("Collecting test...")
    q, qd, qdd, tau = collect_split(args.test, modes, rng, ood=False)
    np.savez(os.path.join(DATA_DIR, "test.npz"), q=q, qdot=qd, qddot=qdd, tau=tau)
    print("test", q.shape)

    print("Collecting OOD...")
    q, qd, qdd, tau = collect_split(args.ood, modes, rng, ood=True)
    np.savez(os.path.join(DATA_DIR, "ood.npz"), q=q, qdot=qd, qddot=qdd, tau=tau)
    print("ood", q.shape)

    print("Done.")
    metadata = {
        "task": "dm_control.manipulator.bring_ball",
        "seed": 42,
        "features": ["q[4]", "qdot[4]", "qddot[4]"],
        "target": "tau[4]",
        "units": {"q": "rad", "qdot": "rad/s", "qddot": "rad/s^2", "tau": "N*m"},
        "timestep_s": 0.001,
        "excitation": ["random", "sinusoidal", "chirp"],
        "train_action_amplitude": 0.35,
        "ood_action_amplitude": 0.65,
        "contact_handling": "ball moved outside workspace before forward dynamics",
        "rejection_filters": {
            "joint_limit_margin_rad": 0.05,
            "max_abs_qdot": 10.0,
            "max_abs_qddot_train": 1200.0,
            "max_abs_qddot_ood": 1800.0,
        },
        "validation_command": "python -m src.validate_data",
    }
    with open(os.path.join(DATA_DIR, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
