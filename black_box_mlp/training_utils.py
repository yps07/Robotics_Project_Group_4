"""Reusable PyTorch training, checkpointing, and regression metrics.

The default step is ordinary supervised regression. Physics-informed models
can supply ``step_fn(model, batch) -> loss`` (or ``(loss, metrics)``) to reuse
the same optimizer loop, early stopping, checkpointing, and logging. Set
``validation_requires_grad=True`` for validation steps that use autograd.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


BatchStep = Callable[[nn.Module, Any], Any]


@dataclass
class TrainingConfig:
    max_epochs: int = 120
    patience: int = 15
    min_delta: float = 1e-6
    grad_clip_norm: float | None = 10.0
    device: str = "auto"


@dataclass
class FitResult:
    best_epoch: int
    best_val_loss: float
    epochs_run: int
    stopped_early: bool
    history: list[dict[str, float]]


class EarlyStopping:
    """Stop after ``patience`` epochs without a ``min_delta`` improvement."""

    def __init__(self, patience=15, min_delta=1e-6):
        if patience < 1:
            raise ValueError("patience must be at least 1.")
        if min_delta < 0:
            raise ValueError("min_delta must be nonnegative.")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = math.inf
        self.bad_epochs = 0

    def update(self, metric):
        metric = float(metric)
        if not math.isfinite(metric):
            raise ValueError(f"Early-stopping metric must be finite, got {metric}.")
        if metric < self.best - self.min_delta:
            self.best = metric
            self.bad_epochs = 0
            return True, False
        self.bad_epochs += 1
        return False, self.bad_epochs >= self.patience


def resolve_device(device="auto"):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return resolved


def seed_everything(seed=42):
    """Seed Python, NumPy, and PyTorch RNGs."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def supervised_mse_step(model, batch):
    """Default normalized-torque regression step."""
    features, target = batch
    prediction = model(features)
    loss = F.mse_loss(prediction, target)
    return loss, {"mse": loss.detach()}


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _batch_size(batch):
    if torch.is_tensor(batch):
        return int(batch.shape[0])
    if isinstance(batch, (tuple, list)) and batch:
        return _batch_size(batch[0])
    if isinstance(batch, dict) and batch:
        return _batch_size(next(iter(batch.values())))
    raise ValueError("Could not infer batch size from batch.")


def _normalize_step_result(result):
    if torch.is_tensor(result):
        loss, metrics = result, {}
    elif isinstance(result, tuple) and len(result) == 2:
        loss, metrics = result
    else:
        raise TypeError(
            "step_fn must return a scalar loss or a (loss, metrics) pair."
        )
    if not torch.is_tensor(loss) or loss.ndim != 0:
        raise ValueError("step_fn loss must be a scalar torch.Tensor.")
    if metrics is None:
        metrics = {}
    if not isinstance(metrics, dict):
        raise TypeError("step_fn metrics must be a dictionary.")
    return loss, metrics


def _metric_float(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("Logged tensor metrics must be scalar.")
        value = value.detach().cpu().item()
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Training metric must be finite, got {value}.")
    return value


def _run_epoch(
    model,
    loader,
    device,
    step_fn,
    optimizer=None,
    grad_clip_norm=None,
    validation_requires_grad=False,
):
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0

    grad_context = torch.enable_grad if (
        training or validation_requires_grad
    ) else torch.no_grad
    with grad_context():
        for batch in loader:
            batch = _move_to_device(batch, device)
            batch_count = _batch_size(batch)
            if training:
                optimizer.zero_grad(set_to_none=True)

            loss, metrics = _normalize_step_result(step_fn(model, batch))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss encountered.")

            if training:
                loss.backward()
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            values = {"loss": loss, **metrics}
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + (
                    _metric_float(value) * batch_count
                )
            count += batch_count

    if count == 0:
        raise ValueError("Cannot train/evaluate on an empty DataLoader.")
    return {name: value / count for name, value in totals.items()}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def save_checkpoint(
    path,
    model,
    optimizer=None,
    *,
    epoch=None,
    metric=None,
    config=None,
    metadata=None,
):
    """Atomically write a portable model/optimizer checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "epoch": epoch,
        "metric": metric,
        "config": _json_safe(config or {}),
        "metadata": _json_safe(metadata or {}),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(
        path, map_location=map_location, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    *,
    config: TrainingConfig | None = None,
    step_fn: BatchStep | None = None,
    validation_step_fn: BatchStep | None = None,
    validation_requires_grad=False,
    output_dir=None,
    checkpoint_dir=None,
    run_config=None,
    checkpoint_metadata=None,
    scheduler=None,
    verbose=True,
):
    """Train with validation-based early stopping and reusable logging.

    ``step_fn`` may implement custom physics-informed losses. It receives the
    model and device-moved batch, and returns a scalar loss or ``(loss, dict)``.
    """
    config = config or TrainingConfig()
    if config.max_epochs < 1:
        raise ValueError("max_epochs must be at least 1.")
    if config.patience < 1:
        raise ValueError("patience must be at least 1.")
    device = resolve_device(config.device)
    model.to(device)
    step_fn = step_fn or supervised_mse_step
    validation_step_fn = validation_step_fn or step_fn
    early_stopping = EarlyStopping(config.patience, config.min_delta)
    history = []
    best_state = None
    best_epoch = 0
    started = time.perf_counter()

    output_path = Path(output_dir) if output_dir is not None else None
    checkpoint_path = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else output_path
    )
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        history_path = output_path / "history.jsonl"
        summary_path = output_path / "summary.json"
        with history_path.open("w", encoding="utf-8"):
            pass
    else:
        history_path = summary_path = None
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        best_checkpoint = checkpoint_path / "best.pt"
        last_checkpoint = checkpoint_path / "last.pt"
    else:
        best_checkpoint = last_checkpoint = None

    stopped_early = False
    for epoch in range(1, config.max_epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            step_fn,
            optimizer=optimizer,
            grad_clip_norm=config.grad_clip_norm,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device,
            validation_step_fn,
            validation_requires_grad=validation_requires_grad,
        )
        val_loss = val_metrics["loss"]
        if scheduler is not None:
            try:
                scheduler.step(val_loss)
            except TypeError:
                scheduler.step()

        improved, should_stop = early_stopping.update(val_loss)
        if improved:
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if best_checkpoint is not None:
                save_checkpoint(
                    best_checkpoint,
                    model,
                    optimizer,
                    epoch=epoch,
                    metric=val_loss,
                    config={**asdict(config), **(run_config or {})},
                    metadata=checkpoint_metadata,
                )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_s": time.perf_counter() - started,
        }
        history.append(record)
        if history_path is not None:
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_json_safe(record), allow_nan=False) + "\n")

        if verbose:
            print(
                f"epoch {epoch:03d}/{config.max_epochs} "
                f"train={train_metrics['loss']:.6g} "
                f"val={val_loss:.6g} "
                f"lr={optimizer.param_groups[0]['lr']:.3g}"
            )

        if should_stop:
            stopped_early = True
            break

    if best_state is None:
        raise RuntimeError("Training produced no finite validation checkpoint.")

    if last_checkpoint is not None:
        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            epoch=epoch,
            metric=val_metrics["loss"],
            config={**asdict(config), **(run_config or {})},
            metadata=checkpoint_metadata,
        )
    model.load_state_dict(best_state)

    result = FitResult(
        best_epoch=best_epoch,
        best_val_loss=early_stopping.best,
        epochs_run=epoch,
        stopped_early=stopped_early,
        history=history,
    )
    if summary_path is not None:
        summary = {
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "epochs_run": result.epochs_run,
            "stopped_early": result.stopped_early,
            "config": {**asdict(config), **(run_config or {})},
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, indent=2, allow_nan=False)
    return result


def regression_metrics(y_true, y_pred, joint_names=None):
    """Compute torque regression metrics in physical units."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError(
            f"Expected matching (N, joints) arrays, got "
            f"{y_true.shape} and {y_pred.shape}."
        )
    error = y_pred - y_true
    mse_joint = np.mean(error**2, axis=0)
    mae_joint = np.mean(np.abs(error), axis=0)
    variance = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    residual = np.sum(error**2, axis=0)
    r2_joint = np.where(variance > 0, 1.0 - residual / variance, np.nan)
    names = list(joint_names or [f"joint_{i}" for i in range(y_true.shape[1])])
    if len(names) != y_true.shape[1]:
        raise ValueError("joint_names length must match output dimension.")
    return {
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(
            1.0 - np.sum(error**2) / np.sum((y_true - y_true.mean()) ** 2)
        ),
        "mse_per_joint": mse_joint.tolist(),
        "rmse_per_joint": np.sqrt(mse_joint).tolist(),
        "mae_per_joint": mae_joint.tolist(),
        "r2_per_joint": r2_joint.tolist(),
        "joint_names": names,
    }


@torch.no_grad()
def predict(model, loader, device="auto"):
    """Collect normalized predictions and targets from a loader."""
    device = resolve_device(device)
    model.to(device)
    was_training = model.training
    model.eval()
    predictions, targets = [], []
    try:
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            predictions.append(model(features).cpu().numpy())
            targets.append(target.numpy())
    finally:
        model.train(was_training)
    if not predictions:
        raise ValueError("Cannot predict on an empty DataLoader.")
    return np.concatenate(predictions), np.concatenate(targets)


def evaluate_regression(
    model,
    loader,
    stats,
    *,
    device="auto",
    joint_names=None,
):
    """Evaluate normalized model outputs in physical torque units."""
    pred_norm, target_norm = predict(model, loader, device=device)
    y_std = np.asarray(stats["Y_std"], dtype=np.float64)
    y_mean = np.asarray(stats["Y_mean"], dtype=np.float64)
    prediction = pred_norm * y_std + y_mean
    target = target_norm * y_std + y_mean
    return regression_metrics(target, prediction, joint_names=joint_names)
