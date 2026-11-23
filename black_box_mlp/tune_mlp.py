"""Validation-only MLP architecture/hyperparameter comparison."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Permit both `python -m black_box_mlp.tune_mlp` from the repository root
# and direct execution of this file.
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import make_loaders
try:
    from .mlp import MLPInverseDynamics
    from .training_utils import (
        TrainingConfig,
        fit,
        resolve_device,
        seed_everything,
    )
except ImportError:  # Direct script execution from black_box_mlp/.
    from mlp import MLPInverseDynamics
    from training_utils import (
        TrainingConfig,
        fit,
        resolve_device,
        seed_everything,
    )


CONFIGS = (
    {
        "name": "tanh_256_256_128_wd1e-5",
        "hidden_sizes": [256, 256, 128],
        "activation": "tanh",
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
    },
    {
        "name": "tanh_128_128_wd1e-5",
        "hidden_sizes": [128, 128],
        "activation": "tanh",
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
    },
    {
        "name": "relu_256_256_128_wd1e-5",
        "hidden_sizes": [256, 256, 128],
        "activation": "relu",
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
    },
    {
        "name": "tanh_256_256_128_lr5e-4",
        "hidden_sizes": [256, 256, 128],
        "activation": "tanh",
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
    },
    {
        "name": "tanh_256_256_128_wd1e-4",
        "hidden_sizes": [256, 256, 128],
        "activation": "tanh",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
    },
)


def run_sweep(
    *,
    seed=42,
    fraction=0.3,
    epochs=120,
    patience=15,
    batch_size=256,
    device="auto",
    output_dir=None,
    verbose=False,
):
    device = resolve_device(device)
    loaders, _ = make_loaders(batch_size=batch_size, normalize_data=True)
    train_dataset = loaders["train"].dataset
    n_samples = max(1, int(np.ceil(len(train_dataset) * fraction)))
    order = np.random.default_rng(seed).permutation(len(train_dataset))[:n_samples]
    subset = Subset(train_dataset, order.tolist())

    output_path = (
        Path(output_dir)
        if output_dir is not None
        else PACKAGE_DIR / "results" / "hyperparameter_sweep"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in CONFIGS:
        seed_everything(seed)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        model = MLPInverseDynamics(
            input_dim=train_dataset.X.shape[1],
            output_dim=train_dataset.Y.shape[1],
            hidden_sizes=config["hidden_sizes"],
            activation=config["activation"],
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        run_config = {
            **config,
            "seed": seed,
            "fraction": fraction,
            "training_samples": n_samples,
            "batch_size": batch_size,
            "device": str(device),
        }
        result = fit(
            model,
            train_loader,
            loaders["val"],
            optimizer,
            config=TrainingConfig(
                max_epochs=epochs,
                patience=patience,
                device=str(device),
            ),
            output_dir=output_path / config["name"],
            run_config=run_config,
            verbose=verbose,
        )
        row = {
            **run_config,
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "best_val_loss_normalized": result.best_val_loss,
        }
        rows.append(row)
        print(
            config["name"],
            f"val_mse={result.best_val_loss:.8g}",
            f"epoch={result.best_epoch}",
        )

    csv_path = output_path / "sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = min(rows, key=lambda row: row["best_val_loss_normalized"])
    print("Selected by validation loss:", best["name"])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        default=str(PACKAGE_DIR / "results" / "hyperparameter_sweep"),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_sweep(**vars(args))


if __name__ == "__main__":
    main()
