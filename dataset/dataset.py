"""PyTorch datasets for inverse dynamics."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import load_stats, normalize

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class InverseDynamicsDataset(Dataset):
    def __init__(self, split="train", stats=None, normalize_data=True):
        path = os.path.join(DATA_DIR, f"{split}.npz")
        d = np.load(path)
        q = d["q"]; qd = d["qdot"]; qdd = d["qddot"]; tau = d["tau"]
        X = np.concatenate([q, qd, qdd], axis=1).astype(np.float32)
        Y = tau.astype(np.float32)
        if stats is None:
            stats = load_stats()
        self.stats = stats
        if normalize_data:
            X, Y = normalize(X, Y, stats)
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))
        self.q = torch.from_numpy(q.astype(np.float32))
        self.qdot = torch.from_numpy(qd.astype(np.float32))
        self.qddot = torch.from_numpy(qdd.astype(np.float32))
        self.tau = torch.from_numpy(tau.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def make_loaders(batch_size=256, normalize_data=True):
    stats = load_stats()
    train = InverseDynamicsDataset("train", stats, normalize_data)
    val = InverseDynamicsDataset("val", stats, normalize_data)
    test = InverseDynamicsDataset("test", stats, normalize_data)
    ood = InverseDynamicsDataset("ood", stats, normalize_data)
    return {
        "train": DataLoader(train, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test, batch_size=batch_size, shuffle=False),
        "ood": DataLoader(ood, batch_size=batch_size, shuffle=False),
    }, stats
