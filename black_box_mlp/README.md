# Workstream C — Black-box MLP baseline

This folder contains the MLP model, reusable training utilities, hyperparameter
sweep, training script, checkpoints, and results for Workstream C. It consumes
the validated datasets and PyTorch loader in the repository's `dataset/`
package.

## Install dependencies

Use the **single project-level requirements file** from the repository root:

```bash
python -m pip install -r requirements.txt
```

The validated environment is Python 3.11.9.

## Reproduce

Run from the repository root:

```bash
python -m black_box_mlp.tune_mlp
python -m black_box_mlp.train_mlp
```

The tuning run compares candidate architectures and optimizer settings using
only the validation split. The training run then trains nested 10%, 30%, 50%,
and 100% subsets and evaluates the selected model on the test and OOD splits.
To invoke a script directly, use `python black_box_mlp/train_mlp.py` or
`python black_box_mlp/tune_mlp.py` from the project root.

## Files and outputs

- `mlp.py`: configurable feed-forward inverse-dynamics regressor.
- `training_utils.py`: generic training loop, autograd-capable batch-step
  callback, early stopping, logging, checkpointing, and regression metrics.
- `tune_mlp.py`: validation-only architecture/hyperparameter sweep.
- `train_mlp.py`: data-efficiency training and test/OOD evaluation.
- `checkpoints/`: best and last checkpoint for each data fraction.
- `results/`: sweep, per-epoch logs, configurations, metrics, and plots.

The MLP receives 13 normalized features: `sin(q_root)`, `cos(q_root)`, the
other three positions, four velocities, and four accelerations. Its four
outputs are normalized physical actuator torques. Raw dataset states and
normalization statistics remain in `../dataset/data/`.

The selected model is 13–256–256–128–4 with `tanh` hidden activations, AdamW
(`lr=1e-3`, `weight_decay=1e-4`), MSE loss, batch size 256, gradient clipping
at 10, seed 42, and early stopping (maximum 120 epochs, patience 15).
See [`results/data_efficiency_metrics.csv`](results/data_efficiency_metrics.csv)
for overall and per-joint physical-unit metrics.

## Results

| Training data | Test RMSE (N·m) | OOD RMSE (N·m) |
| ---: | ---: | ---: |
| 10% | 0.06901 | 0.51116 |
| 30% | 0.04963 | 0.42329 |
| 50% | 0.04449 | 0.43192 |
| 100% | 0.04397 | 0.44971 |

Test metrics improve with more training data; OOD RMSE is lowest at 30%, so
generalization does not improve monotonically with training-set size. The
validation-only architecture sweep and its selected settings are recorded in
`results/hyperparameter_sweep/sweep.csv`.
