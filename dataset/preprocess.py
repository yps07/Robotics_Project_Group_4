"""Normalization utilities for inverse-dynamics datasets."""

import os
import json
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def compute_stats(npz_path):
    d = np.load(npz_path)
    q = d["q"]; qd = d["qdot"]; qdd = d["qddot"]; tau = d["tau"]
    X = np.concatenate([q, qd, qdd], axis=1)  # (N,12)
    Y = tau  # (N,4)
    stats = {
        "X_mean": X.mean(axis=0).tolist(),
        "X_std": (X.std(axis=0) + 1e-8).tolist(),
        "Y_mean": Y.mean(axis=0).tolist(),
        "Y_std": (Y.std(axis=0) + 1e-8).tolist(),
    }
    return stats


def save_stats(stats, path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {path}")


def load_stats(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "stats.json")
    with open(path) as f:
        s = json.load(f)
    s["X_mean"] = np.array(s["X_mean"])
    s["X_std"] = np.array(s["X_std"])
    s["Y_mean"] = np.array(s["Y_mean"])
    s["Y_std"] = np.array(s["Y_std"])
    return s


def normalize(X, Y, stats):
    Xn = (X - stats["X_mean"]) / stats["X_std"]
    Yn = (Y - stats["Y_mean"]) / stats["Y_std"]
    return Xn, Yn


def denormalize_y(Yn, stats):
    return Yn * stats["Y_std"] + stats["Y_mean"]


if __name__ == "__main__":
    stats = compute_stats(os.path.join(DATA_DIR, "train.npz"))
    save_stats(stats)
    print("X_mean", stats["X_mean"])
    print("X_std", stats["X_std"])
    print("Y_mean", stats["Y_mean"])
    print("Y_std", stats["Y_std"])
