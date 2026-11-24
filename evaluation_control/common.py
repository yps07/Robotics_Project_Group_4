"""Paths, constants and small helpers shared by every Workstream E script."""

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "dataset", "data")
RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

JOINTS = ["arm_root", "arm_shoulder", "arm_elbow", "arm_wrist"]
N_DOF = 4
GEAR = np.array([12.0, 8.0, 4.0, 2.0])          # |tau| limits in N*m (ctrl in [-1, 1])
CONTROL_DT = 0.01                                 # dm_control control step
PHYSICS_DT = 0.001                                # MuJoCo physics step
FRACTIONS = [0.1, 0.3, 0.5, 1.0]


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def wrap_angle(x):
    """Wrap to [-pi, pi). Applied to every angle error (the root joint is unlimited)."""
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def import_file(module_name, path):
    """Import a module by file path under a unique name.

    Member 4's ``delan/delan.py`` shares its name with the ``delan/`` folder, and
    Member 2's code is a loose script folder, so neither is importable as a package.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def train_ranges():
    """Empirical min/max of q, qdot, qddot over the official training split."""
    d = np.load(os.path.join(DATA_DIR, "train.npz"))
    return {k: (d[k].min(0), d[k].max(0)) for k in ("q", "qdot", "qddot")}


def model_label(kind, fraction=None):
    if kind in ("analytical", "pd"):
        return kind
    return f"{kind}@{int(round(fraction * 100))}%"
