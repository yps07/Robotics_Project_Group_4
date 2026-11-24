"""Combined evaluation of the analytical model, the black-box MLP and DeLaN.

Run after ``rollout.py`` and ``tracking_experiment.py``. Produces ``results/summary.md`` with:

  1. one-step torque error on the official test and OOD splits (all models and data fractions;
     recomputed with Member 3's ``regression_metrics`` and cross-checked against the CSVs of
     Members 3 and 4)
  2. rollout RMSE at 0.5 / 1 / 2 s (from rollout_metrics.csv)
  3. computed-torque tracking RMSE (from tracking_summary.csv)
  4. data efficiency (10 / 30 / 50 / 100 %) for one-step, rollout and tracking
  5. positive-definiteness of M(q): DeLaN's M = L L^T + eps I against the MLP's *implied*
     mass matrix dtau/dqdd, and their error against the analytical M(q)
  6. computational cost per call

Run:  python -m evaluation_control.evaluate
"""

import os
import time

import numpy as np
import pandas as pd

from .common import DATA_DIR, FIG_DIR, FRACTIONS, JOINTS, RESULTS_DIR, ROOT, ensure_dirs

N_UNIFORM = 10000
UNIFORM_SEED = 7


def _split(name):
    d = np.load(os.path.join(DATA_DIR, f"{name}.npz"))
    return d["q"], d["qdot"], d["qddot"], d["tau"]


# ---------------------------------------------------------------------------
# 1. one-step torque error
# ---------------------------------------------------------------------------

def one_step_table(models):
    from black_box_mlp.training_utils import regression_metrics
    rows = []
    for split in ("test", "ood"):
        q, qd, qdd, tau = _split(split)
        for m in models:
            pred = m.inverse_dynamics(q, qd, qdd)
            r = regression_metrics(tau, pred, joint_names=JOINTS)
            rows.append({"model": m.name, "kind": m.kind, "fraction": m.fraction, "split": split,
                         "rmse": r["rmse"], "mae": r["mae"], "r2": r["r2"],
                         **{f"rmse_{j}": v for j, v in zip(JOINTS, r["rmse_per_joint"])}})
    df = pd.DataFrame(rows)
    _cross_check(df)
    return df


def _cross_check(df):
    """The recomputed RMSEs must match what Members 3 and 4 reported."""
    mlp = pd.read_csv(os.path.join(ROOT, "black_box_mlp", "results", "data_efficiency_metrics.csv"))
    dl = pd.read_csv(os.path.join(ROOT, "delan", "results", "torque_errors.csv"))
    for _, r in df.iterrows():
        if r["kind"] == "mlp":
            ref = mlp[(np.isclose(mlp["fraction"], r["fraction"])) & (mlp["split"] == r["split"])]
        elif r["kind"] == "delan":
            ref = dl[(dl["model"] == "delan") & np.isclose(dl["fraction"].astype(float), r["fraction"])
                     & (dl["split"] == r["split"])]
        else:
            continue
        if len(ref) and not np.isclose(ref["rmse"].iloc[0], r["rmse"], rtol=1e-3):
            print(f"WARNING: {r['model']} {r['split']} RMSE {r['rmse']:.5f} differs from the "
                  f"workstream CSV ({ref['rmse'].iloc[0]:.5f})")


# ---------------------------------------------------------------------------
# 5. mass matrix: positive-definiteness and accuracy
# ---------------------------------------------------------------------------

def _uniform_configs(n=N_UNIFORM, seed=UNIFORM_SEED):
    from dataset.environment import JOINT_LIMITS
    lo = np.where(np.isfinite(JOINT_LIMITS[:, 0]), JOINT_LIMITS[:, 0], -np.pi)
    hi = np.where(np.isfinite(JOINT_LIMITS[:, 1]), JOINT_LIMITS[:, 1], np.pi)
    return np.random.default_rng(seed).uniform(lo, hi, (n, 4))


def _spd_stats(M):
    sym = 0.5 * (M + np.swapaxes(M, 1, 2))
    ev = np.linalg.eigvalsh(sym)
    asym = np.linalg.norm(M - np.swapaxes(M, 1, 2), axis=(1, 2)) / np.linalg.norm(M, axis=(1, 2))
    pd_ = ev.min(1) > 0
    cond = np.where(pd_, ev.max(1) / np.where(pd_, ev.min(1), 1), np.inf)
    return {"frac_pd": float(pd_.mean()), "min_eig": float(ev.min()),
            "median_min_eig": float(np.median(ev.min(1))),
            "median_cond_pd": float(np.median(cond[pd_])) if pd_.any() else np.nan,
            "max_rel_asym": float(asym.max())}


def mass_matrix_table(models, analytical):
    """For DeLaN the mass matrix is its M(q). The MLP has none, so its implied mass matrix
    dtau/dqdd is used, evaluated at each sample's own (q, qd, qdd) on test/OOD (where the MLP
    is not exactly affine in qdd), and at (q, 0, 0) on uniform configurations."""
    sets = {}
    for split in ("test", "ood"):
        q, qd, qdd, _ = _split(split)
        sets[split] = (q, qd, qdd)
    qu = _uniform_configs()
    sets["uniform_in_limits"] = (qu, np.zeros_like(qu), np.zeros_like(qu))
    M_true = {k: analytical.implied_mass_matrix(v[0]) for k, v in sets.items()}
    rows = []
    for m in models:
        for name, (q, qd, qdd) in sets.items():
            M = m.id_jacobians(q, qd, qdd)[0]
            err = np.linalg.norm(M - M_true[name], axis=(1, 2)) / np.linalg.norm(M_true[name], axis=(1, 2))
            rows.append({"model": m.name, "kind": m.kind, "fraction": m.fraction, "set": name,
                         "n": len(q), **_spd_stats(M), "M_rel_fro_err_mean": float(err.mean()),
                         "M_rel_fro_err_median": float(np.median(err))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. computational cost
# ---------------------------------------------------------------------------

def timing_table(models, n_rep=50):
    from .computed_torque import ComputedTorqueController
    from .rollout import implicit_euler_step
    q, qd, qdd, tau = (x[:1] for x in _split("test"))
    rows = []
    for m in models:
        ctrl = ComputedTorqueController(m)
        out = {}
        for label, fn in (
                ("inverse_dynamics_ms", lambda: m.inverse_dynamics(q + 1e-9 * np.random.rand(1, 4), qd, qdd)),
                ("implicit_step_ms", lambda: implicit_euler_step(m, q + 1e-9 * np.random.rand(1, 4), qd,
                                                                 tau, 0.01, np.zeros((1, 4)))),
                ("ct_command_ms", lambda: ctrl.command(q + 1e-9 * np.random.rand(1, 4), qd, q, qd, qdd))):
            fn()
            t0 = time.perf_counter()
            for _ in range(n_rep):
                fn()
            out[label] = 1e3 * (time.perf_counter() - t0) / n_rep
        rows.append({"model": m.name, **out})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_data_efficiency(one_step, rollout, tracking, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pct = [100 * f for f in FRACTIONS]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    panels = [
        (axes[0], "One-step torque RMSE (N·m)", lambda k, s: one_step.query("kind == @k and split == @s")
         .sort_values("fraction")["rmse"].values, one_step.query("kind == 'analytical'")),
        (axes[1], "Rollout position RMSE at 1 s (rad)",
         lambda k, s: _frac_vals(rollout.query("kind == @k and dist == @s and horizon_s == 1.0"),
                                 "pos_rmse_median_incl_diverged"), None),
        (axes[2], "Tracking RMSE (rad)", lambda k, s: _frac_vals(
            tracking.query("kind == @k and variant == 'sampled' and set == @s"), "rmse_mean"), None),
    ]
    split_of = {0: ("test", "ood"), 1: ("test", "ood"), 2: ("main", "ood")}
    styles = {"mlp": ("tab:orange", "s"), "delan": ("tab:blue", "o")}
    for i, (ax, title, getter, _) in enumerate(panels):
        for kind, (c, mk) in styles.items():
            for s, ls in zip(split_of[i], ("-", "--")):
                y = getter(kind, s)
                if len(y) == len(pct):
                    lab = f"{'MLP' if kind == 'mlp' else 'DeLaN'} ({'in-range' if s in ('test', 'main') else 'OOD'})"
                    ax.plot(pct, y, ls, color=c, marker=mk, lw=2, label=lab)
        ax.set_yscale("log"); ax.set_xlabel("Training data used (%)", fontsize=11)
        ax.set_title(title, fontsize=12); ax.grid(True, which="both", alpha=0.3)
        ax.set_xticks(pct)
    ax = axes[3]
    for kind, (c, mk) in styles.items():
        for s, ls in zip(("test", "ood"), ("-", "--")):
            y = _frac_vals(rollout.query("kind == @kind and dist == @s and horizon_s == 2.0"), "frac_diverged")
            if len(y) == len(pct):
                ax.plot(pct, 100 * np.asarray(y), ls, color=c, marker=mk, lw=2,
                        label=f"{'MLP' if kind == 'mlp' else 'DeLaN'} ({'in-range' if s == 'test' else 'OOD'})")
    ax.set_xlabel("Training data used (%)", fontsize=11); ax.set_xticks(pct)
    ax.set_title("Rollouts diverged within 2 s (%)", fontsize=12); ax.grid(alpha=0.3)
    ax.set_ylim(-5, 105)
    axes[0].legend(fontsize=9)
    fig.suptitle("Data efficiency: MLP vs DeLaN (median over rollouts; diverged rollouts count as ∞)",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _frac_vals(df, col):
    return df.sort_values("fraction")[col].values


def _plot_mass(mass, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = mass[mass["set"] == "uniform_in_limits"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, col, title in ((axes[0], "frac_pd", "Fraction of positive-definite M(q)"),
                           (axes[1], "M_rel_fro_err_median", "Median relative error of M(q) vs analytical")):
        for kind, c in (("mlp", "tab:orange"), ("delan", "tab:blue")):
            d = s[s["kind"] == kind].sort_values("fraction")
            ax.plot(100 * d["fraction"], d[col], "-o", color=c, lw=2,
                    label="MLP (implied ∂τ/∂q̈)" if kind == "mlp" else "DeLaN (L Lᵀ + εI)")
        ax.set_xlabel("Training data used (%)", fontsize=11); ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3); ax.set_xticks([10, 30, 50, 100])
    axes[1].set_yscale("log")
    axes[0].set_ylim(-0.05, 1.05); axes[0].legend(fontsize=10)
    fig.suptitle(f"Mass matrix on {N_UNIFORM} uniform configurations within the joint limits", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _md(df, floatfmt=".3g"):
    """Markdown table without the optional ``tabulate`` dependency."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "–" if np.isnan(v) else format(v, floatfmt)
        return "–" if v is None else str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_summary(one_step, rollout, tracking, mass, timing, path):
    main_names = ["analytical", "mlp@100%", "delan@100%"]
    L = ["# Workstream E: combined evaluation\n",
         "Generated by `python -m evaluation_control.evaluate`. All numbers come from the CSV files "
         "in this folder.\n"]

    L.append("## 1. One-step torque error (official test and OOD splits)\n")
    t = one_step.pivot_table(index="model", columns="split", values=["rmse", "r2"]).reset_index()
    t.columns = ["model"] + [f"{a}_{b}" for a, b in t.columns[1:]]
    L.append(_md(t.sort_values("rmse_test"), ".4g") + "\n")

    if rollout is not None:
        L.append("## 2. Rollout error against MuJoCo (2 s rollouts under recorded torques)\n")
        L.append("`pos_rmse` excludes diverged rollouts; `median` is the median per-rollout RMSE "
                 "with diverged rollouts counted as ∞ (so it is `inf` when more than half diverge).\n")
        for dist in ("test", "ood"):
            r = rollout[(rollout["dist"] == dist)].pivot_table(
                index="model", columns="horizon_s",
                values=["pos_rmse", "pos_rmse_median_incl_diverged", "frac_diverged"]).reset_index()
            r.columns = ["model"] + [f"{ {'pos_rmse': 'rmse', 'pos_rmse_median_incl_diverged': 'median', 'frac_diverged': 'diverged'}[a]}@{b:g}s"
                                     for a, b in r.columns[1:]]
            L.append(f"**{'In-distribution' if dist == 'test' else 'OOD'} excitation**\n")
            L.append(_md(r) + "\n")

    if tracking is not None:
        L.append("## 3. Computed-torque tracking in MuJoCo\n")
        cols = ["controller", "set", "rmse_mean", "rmse_std", "rmse_max", "max_abs_err_max",
                "saturation_frac_mean"] + [f"rmse_{j}_mean" for j in JOINTS]
        tr = tracking[cols].sort_values(["set", "rmse_mean"])
        L.append(_md(tr) + "\n")

    L.append("## 4. Mass matrix M(q): positive-definiteness and accuracy\n")
    L.append("DeLaN: its M(q) = L Lᵀ + εI. MLP: the implied mass matrix ∂τ/∂q̈ (it has no M). "
             "Analytical: Member 2's M(q).\n")
    L.append(_md(mass[["model", "set", "n", "frac_pd", "median_min_eig", "median_cond_pd",
                       "max_rel_asym", "M_rel_fro_err_median"]]) + "\n")

    L.append("## 5. Computational cost (single state, CPU)\n")
    L.append(_md(timing) + "\n")
    with open(path, "w") as f:
        f.write("\n".join(L))
    del main_names


def _read(name):
    p = os.path.join(RESULTS_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    from .models import ALL_MODEL_SPECS, MAIN_MODEL_SPECS, load_models
    ensure_dirs()
    models = load_models(ALL_MODEL_SPECS)
    analytical = models[0]

    print("1. one-step torque error ...")
    one_step = one_step_table(models)
    one_step.to_csv(os.path.join(RESULTS_DIR, "one_step_errors.csv"), index=False)

    print("5. mass matrix ...")
    mass = mass_matrix_table(models, analytical)
    mass.to_csv(os.path.join(RESULTS_DIR, "mass_matrix_pd.csv"), index=False)

    print("6. timing ...")
    timing = timing_table(load_models(MAIN_MODEL_SPECS))
    timing.to_csv(os.path.join(RESULTS_DIR, "timing.csv"), index=False)

    rollout, tracking = _read("rollout_metrics.csv"), _read("tracking_summary.csv")
    if rollout is not None and tracking is not None:
        _plot_data_efficiency(one_step, rollout, tracking, os.path.join(FIG_DIR, "data_efficiency.png"))
    _plot_mass(mass, os.path.join(FIG_DIR, "mass_matrix_pd.png"))
    write_summary(one_step, rollout, tracking, mass, timing, os.path.join(RESULTS_DIR, "summary.md"))
    print("wrote results/summary.md, one_step_errors.csv, mass_matrix_pd.csv, timing.csv, figures/")


if __name__ == "__main__":
    main()
