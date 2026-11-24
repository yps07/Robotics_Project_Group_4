"""Positive-definiteness evaluation of the DeLaN mass matrix.

    pytest -q test_spd.py                 # SPD tests on untrained + every trained checkpoint
    python test_spd.py                    # writes results/spd_results.csv for every checkpoint

M(q) = L L^T + eps*I is SPD by construction; this is a numerical sanity check of that
structural guarantee on the test set, the OOD set and 10,000 uniformly random configurations
inside the joint limits:
    min eigenvalue > 0, symmetry error, condition number,
    and the same statistics for the analytical M(q) for reference.
"""

import glob
import os

import numpy as np
import pytest
import torch

from common import CKPT_DIR, RESULTS_DIR, load_analytical, load_split, joint_limits
from delan import DeLaN, load_model


def sample_sets(lo, hi, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    sets = {
        "test": load_split("test")["q"],
        "ood": load_split("ood")["q"],
        "uniform_in_limits": torch.from_numpy(rng.uniform(lo, hi, (n, 4))),
    }
    return sets


def spd_stats(M):
    """M: (B, n, n) tensor or array."""
    M = torch.as_tensor(M)
    sym_err = (M - M.transpose(1, 2)).abs().amax().item()
    ev = torch.linalg.eigvalsh(0.5 * (M + M.transpose(1, 2)))
    cond = ev[:, -1] / ev[:, 0]
    chol_ok = torch.linalg.cholesky_ex(M).info.eq(0).float().mean().item()
    return {
        "n": len(M),
        "frac_pd": (ev[:, 0] > 0).double().mean().item(),
        "cholesky_success": chol_ok,
        "min_eig": ev[:, 0].min().item(),
        "median_min_eig": ev[:, 0].median().item(),
        "max_eig": ev[:, -1].max().item(),
        "median_cond": cond.median().item(),
        "max_cond": cond.max().item(),
        "max_sym_err": sym_err,
    }


def evaluate_checkpoint(path, sets):
    model, ck = load_model(path)
    rows = []
    with torch.no_grad():
        for name, q in sets.items():
            r = spd_stats(model.mass_matrix(q))
            r.update(model=os.path.basename(path), set=name,
                     fraction=ck["hparams"]["fraction"], seed=ck["hparams"]["seed"])
            rows.append(r)
    return rows


# ------------------------------- pytest ---------------------------------

@pytest.fixture(scope="module")
def sets():
    _, mj = load_analytical()
    lo, hi = joint_limits(mj)
    return sample_sets(lo, hi, n=2000)


def test_untrained_model_is_spd(sets):
    torch.manual_seed(0)
    m = DeLaN()
    with torch.no_grad():
        for q in sets.values():
            s = spd_stats(m.mass_matrix(q))
            assert s["frac_pd"] == 1.0 and s["cholesky_success"] == 1.0
            assert s["max_sym_err"] < 1e-12


def test_extreme_inputs_still_spd():
    """Even with huge weights / inputs, M = L L^T + eps I stays SPD."""
    torch.manual_seed(0)
    m = DeLaN()
    with torch.no_grad():
        for p in m.l_net.parameters():
            p.mul_(50)
        q = torch.randn(1000, 4) * 100
        assert (torch.linalg.eigvalsh(m.mass_matrix(q))[:, 0] > 0).all()


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(CKPT_DIR, "delan_*.pt"))))
def test_trained_checkpoint_is_spd(path, sets):
    for r in evaluate_checkpoint(path, sets):
        assert r["frac_pd"] == 1.0, r
        assert r["cholesky_success"] == 1.0, r


# ------------------------------- script ---------------------------------

def main():
    import pandas as pd
    arm, mj = load_analytical()
    lo, hi = joint_limits(mj)
    sets = sample_sets(lo, hi)
    rows = []
    for name, q in sets.items():
        M = np.stack([arm.M(x) for x in q.numpy()])
        r = spd_stats(torch.from_numpy(M))
        r.update(model="analytical", set=name, fraction=np.nan, seed=np.nan)
        rows.append(r)
    for path in sorted(glob.glob(os.path.join(CKPT_DIR, "delan_*.pt"))):
        rows += evaluate_checkpoint(path, sets)
    df = pd.DataFrame(rows)
    cols = ["model", "fraction", "seed", "set", "n", "frac_pd", "cholesky_success", "min_eig",
            "median_min_eig", "max_eig", "median_cond", "max_cond", "max_sym_err"]
    df = df[cols]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "spd_results.csv")
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(f"\nsaved {out}")
    n_bad = (df["frac_pd"] < 1).sum()
    print("ALL SPD" if n_bad == 0 else f"WARNING: {n_bad} rows with non-PD matrices")


if __name__ == "__main__":
    main()
