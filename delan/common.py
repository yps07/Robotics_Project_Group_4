"""Shared paths, data loading and training subsets for the DeLaN workstream.

Only the official shared dataset in ../dataset/data/ is used:
    train.npz  training            val.npz  early stopping + hyper-parameter selection
    test.npz   final evaluation    ood.npz  out-of-distribution evaluation
"""

import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "dataset", "data")
ANALYTICAL_DIR = os.path.join(ROOT, "analytical_euler_lagrange")
CKPT_DIR = os.path.join(HERE, "checkpoints")
LOG_DIR = os.path.join(HERE, "runs")
RESULTS_DIR = os.path.join(HERE, "results")

FRACTIONS = [0.1, 0.3, 0.5, 1.0]
SUBSET_SEED = 42             # identical to black_box_mlp (train_mlp.py / tune_mlp.py --seed 42)
SEEDS = [42]                 # same single seed as the MLP
JOINTS = ["arm_root", "arm_shoulder", "arm_elbow", "arm_wrist"]
SPLITS = ("train", "val", "test", "ood")


def load_split(split):
    """Raw physical-unit tensors from the official dataset.

    DeLaN must see physical q, qdot, qddot: z-scoring them (as the MLP loader does)
    would break the Euler-Lagrange structure tau = M(q) qdd + c(q, qd) + g(q) + D qd.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    d = np.load(os.path.join(DATA_DIR, f"{split}.npz"))
    return {k: torch.from_numpy(d[k].astype(np.float64)) for k in ("q", "qdot", "qddot", "tau")}


def tau_std():
    """Per-joint training torque std (stats.json). Dividing errors by it gives exactly the
    MLP's training loss (MSE on normalised torques), so validation losses are comparable."""
    with open(os.path.join(DATA_DIR, "stats.json")) as f:
        return torch.tensor(json.load(f)["Y_std"], dtype=torch.float64)


def subset_indices(n, fraction, seed=SUBSET_SEED):
    """Nested training subset, exactly as black_box_mlp/train_mlp.py builds it:

        nested_order = np.random.default_rng(42).permutation(n)
        subset       = nested_order[:max(1, ceil(n * fraction))]

    So every 10 % row is also in the 30 %, 50 % and 100 % sets, and DeLaN and the MLP are
    trained on *identical* samples at every fraction.
    """
    n_train = max(1, int(math.ceil(n * fraction)))
    return np.random.default_rng(seed).permutation(n)[:n_train]


def ckpt_path(fraction, seed):
    return os.path.join(CKPT_DIR, f"delan_f{fraction}_s{seed}.pt")


def load_analytical():
    """Member 2's analytical Euler-Lagrange model. It reproduces the official dataset's tau to
    ~1e-16 N*m, so it is exact ground truth for M(q), g(q) and the damping D."""
    if ANALYTICAL_DIR not in sys.path:
        sys.path.insert(0, ANALYTICAL_DIR)
    from analytical_model import AnalyticalArm
    from dm_control import suite
    m = suite.load("manipulator", "bring_ball").physics.model.ptr
    return AnalyticalArm(m, ref_qpos=m.qpos0.copy()), m


def joint_limits(model):
    from mujoco import mj_name2id, mjtObj
    lo, hi = [], []
    for name in JOINTS:
        j = mj_name2id(model, mjtObj.mjOBJ_JOINT, name)
        if model.jnt_limited[j]:
            lo.append(model.jnt_range[j][0]); hi.append(model.jnt_range[j][1])
        else:
            lo.append(-np.pi); hi.append(np.pi)
    return np.array(lo), np.array(hi)


def regression_metrics(y_true, y_pred):
    """Member 3's metric function (black_box_mlp/training_utils.py), so DeLaN and the MLP
    are scored by literally the same code."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from black_box_mlp.training_utils import regression_metrics as _rm
    return _rm(y_true, y_pred, joint_names=JOINTS)
