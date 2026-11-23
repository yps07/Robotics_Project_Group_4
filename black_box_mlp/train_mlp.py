"""Train/evaluate the black-box MLP, including data-efficiency experiments."""

import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Permit both `python -m black_box_mlp.train_mlp` from the repository root
# and direct execution of this file.
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import make_loaders
from dataset.environment import ARM_JOINT_NAMES
try:
    from .mlp import MLPInverseDynamics
    from .training_utils import (
        TrainingConfig,
        evaluate_regression,
        fit,
        resolve_device,
        seed_everything,
    )
except ImportError:  # Direct script execution from black_box_mlp/.
    from mlp import MLPInverseDynamics
    from training_utils import (
        TrainingConfig,
        evaluate_regression,
        fit,
        resolve_device,
        seed_everything,
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument(
        "--activation",
        choices=("relu", "tanh", "silu", "gelu"),
        default="tanh",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument(
        "--output-dir", default=str(PACKAGE_DIR / "results")
    )
    parser.add_argument(
        "--checkpoint-dir", default=str(PACKAGE_DIR / "checkpoints")
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive")
    if not args.fractions or any(not 0.0 < f <= 1.0 for f in args.fractions):
        parser.error("--fractions must all lie in (0, 1]")
    if args.dropout < 0.0 or args.dropout >= 1.0:
        parser.error("--dropout must lie in [0, 1)")
    if args.torch_threads < 0:
        parser.error("--torch-threads cannot be negative")
    args.fractions = sorted(set(args.fractions))
    return args


def _flatten_metrics(metrics, row, prefix=""):
    for key in ("mse", "rmse", "mae", "r2"):
        row[f"{prefix}{key}"] = metrics[key]
    for metric_name in ("mse", "rmse", "mae", "r2"):
        for joint, value in zip(
            metrics["joint_names"], metrics[f"{metric_name}_per_joint"]
        ):
            row[f"{prefix}{metric_name}_{joint}"] = value


def _save_csv(path, rows):
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _save_data_efficiency_plot(path, records, metric, ylabel):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    styles = {
        "test": {"marker": "o", "color": "#2166ac"},
        "ood": {"marker": "s", "color": "#b2182b"},
    }
    for split in ("test", "ood"):
        points = sorted(
            (item["fraction"], item[split][metric])
            for item in records
        )
        ax.plot(
            [100 * point[0] for point in points],
            [point[1] for point in points],
            label=split.upper(),
            linewidth=2,
            markersize=6,
            **styles[split],
        )
    ax.set_xlabel("Training data used (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"MLP data efficiency: {metric.upper()}")
    ax.set_xticks([100 * fraction for fraction in sorted(
        {item["fraction"] for item in records}
    )])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_experiments(args):
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    output_root = Path(args.output_dir)
    checkpoint_root = Path(args.checkpoint_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    loaders, stats = make_loaders(
        batch_size=args.batch_size,
        normalize_data=True,
    )
    train_dataset = loaders["train"].dataset
    input_dim = int(train_dataset.X.shape[1])
    output_dim = int(train_dataset.Y.shape[1])
    total_training_samples = len(train_dataset)
    environment_record = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "resolved_device": str(device),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
    }
    with (output_root / "run_environment.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(environment_record, f, indent=2)

    # The subsets are nested: the 10% rows are also present in 30%, 50%, and
    # 100%. This makes the data-efficiency comparison less seed-dependent.
    subset_rng = np.random.default_rng(args.seed)
    nested_order = subset_rng.permutation(total_training_samples)
    stats_for_checkpoint = {
        key: np.asarray(stats[key]).tolist()
        for key in ("X_mean", "X_std", "Y_mean", "Y_std")
    }
    rows = []
    records = []

    for fraction in args.fractions:
        n_train = max(1, int(math.ceil(total_training_samples * fraction)))
        run_name = f"{int(round(fraction * 100)):03d}pct"
        run_dir = output_root / run_name
        ckpt_dir = checkpoint_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(args.seed)
        subset = Subset(train_dataset, nested_order[:n_train].tolist())
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        train_loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        model_config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_sizes": args.hidden_sizes,
            "activation": args.activation,
            "dropout": args.dropout,
        }
        model = MLPInverseDynamics(**model_config)
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        fit_config = TrainingConfig(
            max_epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            grad_clip_norm=(
                args.grad_clip_norm if args.grad_clip_norm > 0 else None
            ),
            device=str(device),
        )
        run_config = {
            "seed": args.seed,
            "fraction": fraction,
            "training_samples": n_train,
            "total_training_samples": total_training_samples,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": str(device),
            "model": model_config,
        }
        with (run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)

        if not args.quiet:
            print(
                f"\n=== {run_name}: {n_train}/{total_training_samples} "
                f"training samples, device={device} ==="
            )
        fit_result = fit(
            model,
            train_loader,
            loaders["val"],
            optimizer,
            config=fit_config,
            output_dir=run_dir,
            checkpoint_dir=ckpt_dir,
            run_config=run_config,
            checkpoint_metadata={
                "normalization_stats": stats_for_checkpoint,
                "feature_encoding": (
                    "sin(q_root), cos(q_root), q_shoulder, q_elbow, "
                    "q_wrist, qdot[4], qddot[4]"
                ),
            },
            verbose=not args.quiet,
        )

        record = {
            "fraction": fraction,
            "training_samples": n_train,
            "best_epoch": fit_result.best_epoch,
            "best_val_loss_normalized": fit_result.best_val_loss,
            "epochs_run": fit_result.epochs_run,
            "test": evaluate_regression(
                model,
                loaders["test"],
                stats,
                device=device,
                joint_names=ARM_JOINT_NAMES,
            ),
            "ood": evaluate_regression(
                model,
                loaders["ood"],
                stats,
                device=device,
                joint_names=ARM_JOINT_NAMES,
            ),
        }
        records.append(record)
        for split in ("test", "ood"):
            row = {
                "fraction": fraction,
                "training_percent": 100.0 * fraction,
                "training_samples": n_train,
                "split": split,
                "best_epoch": fit_result.best_epoch,
                "best_val_loss_normalized": fit_result.best_val_loss,
            }
            _flatten_metrics(record[split], row)
            rows.append(row)
            if not args.quiet:
                print(
                    f"{split.upper():4s} RMSE={record[split]['rmse']:.6g} "
                    f"MAE={record[split]['mae']:.6g} "
                    f"R2={record[split]['r2']:.6g}"
                )

    _save_csv(output_root / "data_efficiency_metrics.csv", rows)
    with (output_root / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "environment": environment_record,
                "runs": records,
            },
            f,
            indent=2,
            allow_nan=False,
        )
    _save_data_efficiency_plot(
        output_root / "data_efficiency_rmse.png",
        records,
        "rmse",
        "Torque RMSE (N·m)",
    )
    _save_data_efficiency_plot(
        output_root / "data_efficiency_mae.png",
        records,
        "mae",
        "Torque MAE (N·m)",
    )
    print(f"\nSaved metrics and plots under {output_root}.")
    print(f"Saved checkpoints under {checkpoint_root}.")
    return records


def main():
    run_experiments(_parse_args())


if __name__ == "__main__":
    main()
