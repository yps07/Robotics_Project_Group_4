"""Evaluate the trained DeLaN checkpoints on the official test and OOD splits.

    python evaluate_delan.py

Torque metrics use Member 3's regression_metrics (black_box_mlp/training_utils.py) and the
training subsets are identical to the MLP's, so Member 5 can compare the models directly.
The cross-model comparison itself is left to Member 5's common evaluation.

Writes to results/:
    torque_errors.csv        per fraction x split: RMSE / MAE / R2 overall and per joint
    physics_errors.csv       learned M(q), g(q), D vs the exact analytical model
    ablation.csv             DeLaN with vs without the damping term (100 % data)
    data_efficiency.png      torque RMSE vs training fraction, test + OOD
    physics_vs_data.png      M / g / D recovery error vs training fraction
    torque_scatter.png       predicted vs true torque, 100 % model, test + OOD
    physics_comparison.png   learned vs analytical M entries, g, damping (test split)
    training_curves.png      validation loss per run
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import CKPT_DIR, JOINTS, RESULTS_DIR, load_analytical, load_split, regression_metrics
from delan import load_model
from train_delan import predict

SPLITS = ["test", "ood"]
FULL_CKPT = os.path.join(CKPT_DIR, "delan_f1.0_s42.pt")
ABLATION_CKPT = os.path.join(CKPT_DIR, "ablation", "delan_no_damping_f1.0_s42.pt")


def torque_row(model_name, fraction, split, true, pred):
    m = regression_metrics(true, pred)
    row = dict(model=model_name, fraction=fraction, split=split,
               mse=m["mse"], rmse=m["rmse"], mae=m["mae"], r2=m["r2"])
    for j, n in enumerate(JOINTS):
        row[f"rmse_{n}"] = m["rmse_per_joint"][j]
        row[f"r2_{n}"] = m["r2_per_joint"][j]
    return row


def analytical_truth(arm, data):
    out = {}
    for s, d in data.items():
        q, qd, qdd = d["q"].numpy(), d["qdot"].numpy(), d["qddot"].numpy()
        out[s] = {"tau": np.stack([arm.inverse_dynamics(*x) for x in zip(q, qd, qdd)]),
                  "M": np.stack([arm.M(x) for x in q]),
                  "G": np.stack([arm.G(x) for x in q])}
    return out


@torch.no_grad()
def outputs(model, d):
    return {"tau": predict(model, d).numpy(), "M": model.mass_matrix(d["q"]).numpy(),
            "G": model.gravity(d["q"]).numpy()}


def physics_row(fraction, split, out, truth, D, D_true):
    Mt, Ml = truth["M"], out["M"]
    rel = np.linalg.norm(Ml - Mt, axis=(1, 2)) / np.linalg.norm(Mt, axis=(1, 2))
    row = dict(fraction=fraction, split=split,
               M_rel_fro_mean=rel.mean(), M_rel_fro_median=np.median(rel), M_rel_fro_p95=np.percentile(rel, 95),
               g_rmse=np.sqrt(((out["G"] - truth["G"]) ** 2).mean()),
               g_rel=np.linalg.norm(out["G"] - truth["G"]) / np.linalg.norm(truth["G"]),
               D_max_rel_err=np.max(np.abs(D - D_true) / D_true),
               min_eig_M=np.linalg.eigvalsh(Ml)[:, 0].min())
    row.update({f"D_{n}": v for n, v in zip(JOINTS, D)})
    return row


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(CKPT_DIR, "delan_f*_s42.pt")))
    if not paths:
        raise SystemExit(f"no checkpoints in {CKPT_DIR}: run train_delan.py --all first")
    data = {s: load_split(s) for s in SPLITS}
    arm, _ = load_analytical()
    truth = analytical_truth(arm, data)

    torque_rows, phys_rows, ckpts, outs = [], [], {}, {}
    for s in SPLITS:   # analytical reference (exact on this dataset: ~1e-16)
        torque_rows.append(torque_row("analytical", np.nan, s, data[s]["tau"].numpy(), truth[s]["tau"]))
    for p in paths:
        model, ck = load_model(p)
        f = ck["hparams"]["fraction"]
        ckpts[f] = ck
        D = model.damping.detach().numpy()
        for s in SPLITS:
            o = outputs(model, data[s])
            outs[(f, s)] = o
            torque_rows.append(torque_row("delan", f, s, data[s]["tau"].numpy(), o["tau"]))
            phys_rows.append(physics_row(f, s, o, truth[s], D, arm.damp))
    tq = pd.DataFrame(torque_rows)
    tq.to_csv(os.path.join(RESULTS_DIR, "torque_errors.csv"), index=False)
    ph = pd.DataFrame(phys_rows)
    ph.to_csv(os.path.join(RESULTS_DIR, "physics_errors.csv"), index=False)

    abl_rows = []
    if os.path.exists(ABLATION_CKPT) and os.path.exists(FULL_CKPT):
        for label, path in [("DeLaN (with damping)", FULL_CKPT), ("DeLaN without damping", ABLATION_CKPT)]:
            model, ck = load_model(path)
            for s in SPLITS:
                o = outputs(model, data[s])
                r = torque_row(label, 1.0, s, data[s]["tau"].numpy(), o["tau"])
                pr = physics_row(1.0, s, o, truth[s], model.damping.detach().numpy(), arm.damp)
                r.update(best_val_loss_normalized=ck["best_val"], M_rel_fro_mean=pr["M_rel_fro_mean"],
                         g_rmse=pr["g_rmse"])
                abl_rows.append(r)
        pd.DataFrame(abl_rows).to_csv(os.path.join(RESULTS_DIR, "ablation.csv"), index=False)

    # ---------- console summary ----------
    fmt = lambda x: f"{x:.4g}"  # noqa: E731
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("=== Torque prediction (N*m) ===")
        cols = ["model", "fraction", "split", "rmse", "mae", "r2"] + [f"rmse_{n}" for n in JOINTS]
        print(tq[cols].to_string(index=False, float_format=fmt))
        print("\n=== Physics recovery ===")
        pc = ["fraction", "split", "M_rel_fro_mean", "g_rmse", "g_rel", "D_max_rel_err", "min_eig_M"] + \
             [f"D_{n}" for n in JOINTS]
        print(ph[pc].to_string(index=False, float_format=fmt))
        print("true damping:", arm.damp)
        if abl_rows:
            print("\n=== Ablation (100 %) ===")
            print(pd.DataFrame(abl_rows)[["model", "split", "rmse", "r2", "best_val_loss_normalized",
                                          "M_rel_fro_mean", "g_rmse"]].to_string(index=False, float_format=fmt))

    # ---------- plots ----------
    dl = tq[tq.model == "delan"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for s, c in zip(SPLITS, ["tab:blue", "tab:orange"]):
        e = dl[dl.split == s].sort_values("fraction")
        ax.plot(e.fraction * 100, e.rmse, marker="o", color=c, label=s)
    ax.set_yscale("log"); ax.set_xticks([10, 30, 50, 100]); ax.grid(alpha=0.3, which="both")
    ax.set_xlabel("training data used (%)"); ax.set_ylabel("torque RMSE (N·m)")
    ax.set_title("DeLaN data efficiency"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "data_efficiency.png"), dpi=150); plt.close(fig)

    pt = ph[ph.split == "test"].sort_values("fraction")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, col, lab in [(axes[0], "M_rel_fro_mean", "mean relative error of M(q)"),
                         (axes[1], "g_rel", "relative error of g(q)"),
                         (axes[2], "D_max_rel_err", "max relative error of D")]:
        ax.plot(pt.fraction * 100, pt[col], marker="o")
        ax.set_yscale("log"); ax.set_xticks([10, 30, 50, 100]); ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("training data used (%)"); ax.set_title(lab)
    fig.suptitle("Physical quantities recovered by DeLaN vs the exact analytical model (test split)")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "physics_vs_data.png"), dpi=150); plt.close(fig)

    best = 1.0 if 1.0 in ckpts else max(ckpts)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for r, s in enumerate(SPLITS):
        true, pred = data[s]["tau"].numpy(), outs[(best, s)]["tau"]
        for j in range(4):
            ax = axes[r, j]
            ax.scatter(true[:, j], pred[:, j], s=3, alpha=0.4)
            lim = [true[:, j].min(), true[:, j].max()]
            ax.plot(lim, lim, "k--", lw=1)
            rmse = np.sqrt(((pred[:, j] - true[:, j]) ** 2).mean())
            ax.set_title(f"{JOINTS[j]} ({s})  RMSE {rmse:.3g}")
            ax.set_xlabel("true τ (N·m)"); ax.set_ylabel("DeLaN τ (N·m)")
    fig.suptitle(f"DeLaN predicted vs true torque ({int(best * 100)} % training data)")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "torque_scatter.png"), dpi=150); plt.close(fig)

    o, Mt, Gt = outs[(best, "test")], truth["test"]["M"], truth["test"]["G"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for i in range(4):
        for k in range(i, 4):
            axes[0].scatter(Mt[:, i, k], o["M"][:, i, k], s=3, alpha=0.4, label=f"M{i}{k}")
    lim = [Mt.min(), Mt.max()]
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set_xlabel("analytical M_ij"); axes[0].set_ylabel("DeLaN M_ij")
    axes[0].set_title("Mass-matrix entries"); axes[0].legend(fontsize=6, ncol=2, markerscale=3)
    for j in range(4):
        axes[1].scatter(Gt[:, j], o["G"][:, j], s=3, alpha=0.4, label=JOINTS[j])
    lim = [Gt.min(), Gt.max()]
    axes[1].plot(lim, lim, "k--", lw=1)
    axes[1].set_xlabel("analytical g(q) (N·m)"); axes[1].set_ylabel("DeLaN g(q) (N·m)")
    axes[1].set_title("Gravity torque"); axes[1].legend(fontsize=8, markerscale=3)
    x, w = np.arange(4), 0.38
    Dl = ph[(ph.fraction == best) & (ph.split == "test")][[f"D_{n}" for n in JOINTS]].values[0]
    axes[2].bar(x - w / 2, arm.damp, w, label="MuJoCo (true)")
    axes[2].bar(x + w / 2, Dl, w, label="DeLaN (learned)")
    axes[2].set_xticks(x, JOINTS, rotation=15); axes[2].set_ylabel("damping (N·m·s/rad)")
    axes[2].set_title("Joint damping"); axes[2].legend()
    fig.suptitle(f"Learned physics vs analytical model, test split ({int(best * 100)} % training data)")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "physics_comparison.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for f, ck in sorted(ckpts.items()):
        ax.plot([e["val"] for e in ck["history"]], lw=1.2, label=f"{int(f * 100)} %")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("validation loss (normalised MSE)")
    ax.legend(title="training data"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150); plt.close(fig)
    print(f"\nresults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
