"""Validation-only hyper-parameter sweep for DeLaN (mirrors black_box_mlp/tune_mlp.py).

Protocol (same as the MLP): 30 % nested training subset, seed 42, every candidate ranked
ONLY by its best validation loss (MSE on std-normalised torques). test/ood are never used.

    python tune_delan.py --jobs 4      ->  checkpoints/tuning/*.pt, results/tuning_results.csv

The winner is copied into train_delan.DEFAULTS by hand (and recorded in the README).
"""

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from common import CKPT_DIR, RESULTS_DIR

FRACTION, SEED = 0.3, 42
GRID = {
    "size": [(64, 2), (128, 2), (128, 3), (256, 3)],     # (width, hidden layers) of each sub-network
    "activation": ["tanh", "softplus"],                  # smooth: 2nd derivatives are needed
    "weight_decay": [1e-5, 1e-4],
}


def _name(c):
    return f"w{c['width']}_d{c['depth']}_{c['activation']}_wd{c['weight_decay']:g}"


def _run(cfg_threads):
    cfg, threads = cfg_threads
    import torch
    torch.set_num_threads(threads)
    from train_delan import train
    out = os.path.join(CKPT_DIR, "tuning", _name(cfg) + ".pt")
    _, best_val = train(fraction=FRACTION, seed=SEED, out=out, verbose=False, **cfg)
    ck = torch.load(out, weights_only=False)
    print(f"{_name(cfg):32s} best val {best_val:.4e}  epoch {ck['best_epoch']}", flush=True)
    return dict(name=_name(cfg), **cfg, fraction=FRACTION, seed=SEED, n_train=ck["hparams"]["n_train"],
                best_epoch=ck["best_epoch"], epochs_run=ck["hparams"]["epochs_run"],
                best_val_loss_normalized=best_val,
                learned_damping=[round(d, 4) for d in ck["history"][ck["best_epoch"]]["damping"]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--threads", type=int, default=4, help="torch threads per job")
    a = p.parse_args()
    cfgs = [dict(width=w, depth=d, activation=act, weight_decay=wd)
            for (w, d), act, wd in itertools.product(GRID["size"], GRID["activation"], GRID["weight_decay"])]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(_run, [(c, a.threads) for c in cfgs]))
    df = pd.DataFrame(rows).sort_values("best_val_loss_normalized")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "tuning_results.csv"), index=False)
    print(df.drop(columns=["fraction", "seed"]).to_string(index=False))
    print("\nselected:", df.iloc[0]["name"])


if __name__ == "__main__":
    main()
