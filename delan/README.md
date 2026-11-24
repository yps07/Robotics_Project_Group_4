# Workstream D — Deep Lagrangian Network

A physics-informed inverse-dynamics model for the 4 arm joints of the `dm_control` `manipulator/bring_ball` arm. It is trained and evaluated **only on the official shared dataset** in `../dataset/data/` (`train`, `val`, `test`, `ood`). It uses **the same training subsets, seed and metric code as the black-box MLP** (`black_box_mlp/`), so the models can be compared directly. The cross-model comparison is left to the common model evaluation.

## 1. Model

```
τ = M(q) q̈  +  c(q, q̇)  +  g(q)  +  D q̇
```

| Component | Implementation (`delan.py`) |
|---|---|
| Input features | `[sin q, cos q]`: the root joint is unlimited and wraps at ±π |
| Mass matrix | MLP → lower-triangular `L(q)` with a softplus (positive) diagonal; `M(q) = L Lᵀ + εI`, ε = 1e-6. **SPD by construction.** |
| Potential energy | separate MLP → scalar `V(q)`; gravity `g(q) = ∂V/∂q` by autograd |
| Coriolis / centrifugal | `c = Ṁq̇ − ½ ∂(q̇ᵀMq̇)/∂q`, with `∂M/∂q` from `torch.func.jacfwd` (nothing derived by hand) |
| Damping / friction | learnable diagonal viscous damping `D = softplus(d) ≥ 0` |
| Forward dynamics | `q̈ = M⁻¹(τ − c − g − Dq̇)`: a linear solve, no numerical inversion needed |

**Damping/friction decision.** The arm has viscous joint damping (2.0, 1.5, 1.0, 0.5 N·m·s/rad) and no armature or Coulomb friction (`analytical_euler_lagrange/joint_parameters.csv`). A Lagrangian is conservative, so it cannot represent dissipation; a learnable, non-negative viscous term is the physically correct minimal extension. Damping is also the largest term in the data (§5), and the ablation in §4.4 shows the model fails without it.

**Units.** DeLaN uses raw physical `q, q̇, q̈`. The MLP's z-scored features can't be used, because normalizing `q̈` and `q` would break the Euler–Lagrange structure. The true `M(q)` is small (eigenvalues 3e-5 … 0.027), so `L` is scaled by 0.1 and computations run in float64.

**Loss.** MSE on torques divided per joint by the training torque std (`stats.json`). This is the same normalized MSE the MLP trains on.

## 2. Experimental protocol

| | Setting |
|---|---|
| Training subsets | nested 10 / 30 / 50 / 100 % = 2000 / 6000 / 10000 / 20000 rows, `default_rng(42).permutation(20000)[:ceil(f·N)]`: **the same rows as `black_box_mlp/train_mlp.py`** (checked by `test_delan.py::test_subsets_identical_to_mlp_and_nested`) |
| Seed | 42, the same as the MLP |
| Early stopping and model selection | `val.npz` only |
| Hyper-parameter sweep | 30 % subset, seed 42, ranked by val loss (`tune_delan.py`, same protocol as `tune_mlp.py`) |
| Final evaluation | `test.npz` and `ood.npz`, never used for any decision |
| Metrics | `regression_metrics`, imported from `black_box_mlp/training_utils.py` (the same function the MLP is scored with) |
| Optimizer | Adam, lr 1e-3 with cosine decay over ≤ 300 epochs, batch 256, gradient clipping at 10, early stopping with patience 40, best-val weights kept |

**Damping learning rate.** The 4 damping scalars use a 10× learning rate with no weight decay. At lr 1e-3 they move about 1e-3 per step, and the 10 % subset gets only 2,400 steps in 300 epochs. Without the 10× rate, the 10 % run's root damping stalled at 1.73 (true value 2.0) while the cosine schedule decayed. Its val loss stopped at 3.0e-2 and its test RMSE was 0.19 N·m, a convergence failure rather than a data-efficiency result. The 30–100 % runs were unaffected. With the 10× rate every fraction converges (§4).

## 3. Hyper-parameter selection (`results/tuning_results.csv`)

Grid of 16 configs: size {64×2, 128×2, 128×3, 256×3} × activation {tanh, softplus} × weight decay {1e-5, 1e-4}. Only smooth activations are used, because second derivatives of the networks enter the dynamics.

| Config (width × layers, activation, wd) | best val loss (normalized MSE) |
|---|---|
| **128×3, tanh, 1e-5 (selected)** | **8.2e-5** |
| 64×2, tanh, 1e-5 | 1.1e-4 |
| 128×2, tanh, 1e-5 | 1.3e-4 |
| 256×3, tanh, 1e-5 | 1.6e-4 |
| best softplus (64×2, 1e-5) | 2.7e-4 |
| 3 deep softplus configs | ≈ 2.1e-2 (stuck in a local minimum) |

**Is early stopping cutting training short?** The 100 % run stops early (best epoch 14). As a check, using val only, it was continued for the full 300-epoch schedule. Train loss kept falling (2.9e-5 → 1.3e-5) while val loss rose (8.2e-5 → 1.3e-4). That is real overfitting, so the early-stopped weights are the right ones.

## 4. Results (`results/`, seed 42)

### 4.1 Torque prediction (`torque_errors.csv`, `data_efficiency.png`, `torque_scatter.png`)

| Training data | test RMSE (N·m) | test R² | OOD RMSE (N·m) | OOD R² |
|---|---|---|---|---|
| 10 % | 0.0144 | 0.99992 | 0.0261 | 0.99994 |
| 30 % | 0.0140 | 0.99992 | 0.0249 | 0.99994 |
| 50 % | 0.0140 | 0.99992 | 0.0259 | 0.99994 |
| 100 % | 0.0147 | 0.99992 | 0.0250 | 0.99994 |
| *analytical (exact)* | *1.7e-16* | *1* | *3.1e-16* | *1* |

Per joint at 100 %, test / OOD (N·m): root 0.017 / 0.043, shoulder 0.020 / 0.019, elbow 0.011 / 0.016, wrist 0.009 / 0.005.

The OOD split (larger amplitudes and frequencies; every row lies outside the training range in at least one of q, q̇ or q̈) only roughly doubles the error. The learned structure (SPD inertia, gravity from a potential, linear damping) extrapolates.

The analytical model is exact on this dataset (error about 1e-16), because the data come from the same rigid-body model with the gripper locked. It is used as the ground truth in §4.2.

### 4.2 Physics recovered (compared with the analytical model; `physics_errors.csv`, `physics_comparison.png`, `physics_vs_data.png`)

| Training data | damping D (max rel. error) | gravity g(q) (rel. error, test) | M(q) (mean rel. Frobenius error, test) |
|---|---|---|---|
| 10 % | 0.13 % | 6.6 % | 33 % |
| 30 % | 0.12 % | 6.6 % | 34 % |
| 50 % | 0.04 % | 6.4 % | 31 % |
| 100 % | 0.10 % | 6.8 % | 33 % |

Learned damping at 100 % is [1.999, 1.500, 1.000, 0.500], against the true [2, 1.5, 1, 0.5]. Gravity RMSE is 0.014 N·m, against true gravity torques up to 0.62 N·m. The inertia is recovered well for the root and shoulder entries. The small elbow/wrist entries are over-estimated; see §5.

### 4.3 Positive-definiteness (`test_spd.py`, `spd_results.csv`)

`M = LLᵀ + εI` is SPD by construction; this is a numerical sanity check of that structural guarantee. Each of the 4 checkpoints was evaluated on test, OOD and 10 000 uniform configurations inside the joint limits: 56 000 mass matrices in total. **100 % were positive-definite, with 100 % Cholesky success and exact symmetry.** The minimum eigenvalue is 1.3e-3, and the maximum condition number is 14 (the true M has a condition number of about 900).

### 4.4 Ablation: no damping term (`ablation.csv`, 100 % data)

| Model | val loss | test RMSE | test R² | OOD RMSE |
|---|---|---|---|---|
| DeLaN (with D) | 8.2e-5 | 0.0147 | 0.99992 | 0.0250 |
| DeLaN without D | 0.94 | 1.49 | 0.14 | 3.08 |

A purely conservative Lagrangian can't fit the data at all. The damping term is essential, not a tuning detail.

## 5. Interpretation

1. **Damping dominates the torque.** RMS torque contributions in the training data:

   | Joint | D q̇ | M q̈ | g(q) | c(q, q̇) |
   |---|---|---|---|---|
   | root | 2.37 | 0.61 | 0.37 | 0.012 |
   | wrist | 0.40 | 0.026 | 0.010 | 0.001 |

   The wrist and elbow inertia, and all Coriolis terms, move the torque by about 0.01–0.03 N·m, the same order as the model error. These entries are weakly excited by the data, so they're only weakly identifiable. That is why M's relative error is about 33 % even though torque prediction is very accurate. It is a limit of the excitation, not of the model.

2. **The data-efficiency curve is flat.** The nested subsets sample rows at random, so even the 10 % subset contains **all 40 training episodes** (27–65 samples each). More data adds density along the same trajectories, not new regions of state space. DeLaN's structure needs only a few dozen samples per trajectory to reach its accuracy limit.

3. **Note for control.** The learned M is better-conditioned than the true one: the small wrist/elbow inertia is over-estimated. Gravity and damping, the dominant feed-forward terms, are near exact. Any effect on wrist tracking should show up in the computed-torque tracking experiments.

## 6. Files

| File | Purpose |
|---|---|
| `delan.py` | `DeLaN` (torch, batched) and `DeLaNDynamics` (numpy, same API as `AnalyticalArm`) |
| `common.py` | official-split loading, MLP-identical nested subsets, paths, shared metric function |
| `train_delan.py` | training with early stopping on val, checkpoints, TensorBoard logs |
| `tune_delan.py` | validation-only hyper-parameter sweep → `results/tuning_results.csv` |
| `evaluate_delan.py` | test/OOD metrics, physics recovery, ablation, plots |
| `test_spd.py` | SPD tests (pytest) and SPD report (script) |
| `test_delan.py` | correctness tests: EL formula vs analytical model, autograd vs finite differences, `Ṁ−2C` skew-symmetry, energy conservation, forward/inverse round trip, subset identity with the MLP, official splits only |
| `run_experiments.sh` | full pipeline |
| `checkpoints/` | `delan_f{0.1,0.3,0.5,1.0}_s42.pt`, `ablation/`, `tuning/` |
| `runs/` | per-run logs, TensorBoard events, history JSON |
| `results/` | CSV tables and PNG plots listed above |

## 7. Reproducing

From `delan/` (CPU is enough; about 40 min including the sweep):

```bash
bash run_experiments.sh                    # sweep, then 4 fractions + ablation, tests, SPD, evaluation
bash run_experiments.sh --skip-tuning      # same, without the 16-config sweep
python train_delan.py --fraction 0.1       # a single run
python evaluate_delan.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q test_delan.py test_spd.py   # 15 tests
```

Set `PY=/path/to/python` for `run_experiments.sh` if `python` isn't the project environment. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is needed only when a sourced ROS installation injects pytest plugins.

## 8. Interface for rollouts and computed-torque control

```python
import sys; sys.path.insert(0, "delan")
from delan import DeLaNDynamics
dyn = DeLaNDynamics.from_checkpoint("delan/checkpoints/delan_f1.0_s42.pt")

dyn.M(q)                          # (4,4) SPD mass matrix
dyn.C(q, qd)                      # (4,4) Coriolis matrix, c = C @ qd
dyn.G(q)                          # (4,)  gravity torque
dyn.damp                          # (4,)  learned viscous damping
dyn.inverse_dynamics(q, qd, qdd)  # (4,)  tau
dyn.forward_dynamics(q, qd, tau)  # (4,)  qdd (direct solve, no numerical inversion)
dyn.energy(q, qd)                 #       T + V
```

These are the same method names and conventions as `AnalyticalArm`; batches of shape `(B, 4)` are also accepted.
Computed-torque law: `tau = dyn.M(q) @ (qdd_ref + Kd·ė + Kp·e) + dyn.C(q, qd) @ qd + dyn.G(q) + dyn.damp * qd`.
