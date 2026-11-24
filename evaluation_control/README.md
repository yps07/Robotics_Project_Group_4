# Workstream E — Dynamics evaluation and control

This workstream puts the three models into the same experiments:
- the analytical Euler–Lagrange model (Member 2)
- the black-box MLP (Member 3)
- DeLaN (Member 4)

It provides one shared model interface, multi-step forward-dynamics rollouts against MuJoCo, and computed-torque tracking of 10 reference trajectories in MuJoCo. It also combines everything into the comparison tables and figures (`results/summary.md`).

Nothing in Workstreams A–D is modified. Their models are loaded from their own checkpoints and code.

## 1. Shared model interface (`models.py`)

| Method | Meaning |
|---|---|
| `inverse_dynamics(q, qd, qdd)` | τ, for a single state `(4,)` or a batch `(B, 4)` |
| `id_jacobians(q, qd, qdd)` | `(∂τ/∂q̈, ∂τ/∂q̇)`, each `(…, 4, 4)` |
| `forward_dynamics(q, qd, tau)` | q̈, by Newton inversion of the inverse dynamics |
| `implied_mass_matrix(q)` | `∂τ/∂q̈` at `(q, 0, 0)` |

| Adapter | Source | Jacobians |
|---|---|---|
| `AnalyticalModel` | `AnalyticalArm`, built exactly as in `delan/common.py` | `M(q)`; `D + ∂(C q̇)/∂q̇` from the Christoffel symbols. M, ∂M/∂q and g are cached per q. |
| `DeLaNModel` | `delan/checkpoints/delan_f{f}_s42.pt` | DeLaN's own `M(q)`, `∂M/∂q` and `D`. Checked against autograd to 1e-16. |
| `MLPModel` | `black_box_mlp/checkpoints/{pct}/best.pt`, with its stored normalisation statistics | autograd through Member 3's feature encoding, rewritten in torch |

`load_models(ALL_MODEL_SPECS)` returns 9 models: the analytical model, plus the MLP and DeLaN at each of 10, 30, 50 and 100 %.

The computed-torque controller uses only `inverse_dynamics`, so the three models plug in unchanged.

## 2. Rollouts (`rollout.py`)

**Ground truth**
- The ground truth is 10 fresh, contiguous 10 s MuJoCo episodes per distribution, from Member 1's contact-free arm. They use the same excitation families, amplitudes and frequencies as the dataset, with new seeds (5150 for test, 6160 for OOD).
- The official dataset can't be used, because it keeps only 27–65 subsampled rows per episode.
- Windows of 2 s start every 1 s. A window is kept only if no MuJoCo constraint (a joint limit) is active and |q̇| < 10 rad/s, the dataset's filter. No model represents limit forces.

**Integrator**
- The wrist is stiff: damping/inertia ≈ 0.5 / 3·10⁻⁵ ≈ 1.6·10⁴ s⁻¹. Explicit RK4 is therefore unstable.
- Each model is stepped implicitly in velocity, as MuJoCo's Euler integrator treats damping:

  find a with ID(qₙ, vₙ + h·a, a) = τ  →  vₙ₊₁ = vₙ + h·a,  qₙ₊₁ = qₙ + h·vₙ₊₁

  The Newton Jacobian is ∂τ/∂q̈ + h·∂τ/∂q̇. This needs only inverse dynamics, so the scheme is identical for all three models.
- The step is h = 10 ms, with τ held over each control step.
- With h = 1 ms, the first substep after each torque change asks the models for wrist accelerations of about 10³–10⁴ rad/s². The dataset never contains those, because it is sampled at the end of each 10 ms interval. The MLP diverges immediately there.
- The analytical model, integrated by the same scheme, gives the discretisation floor: about 1·10⁻³ rad at 1–2 s. It is about 2·10⁻⁵ rad with `--substeps 10`.

**Inverting the inverse-dynamics MLP**
- For the analytical model and DeLaN, τ is affine in q̈ and M is positive-definite, so 2 Newton iterations converge.
- The MLP's implied mass matrix ∂τ/∂q̈ is generally **not positive-definite**, and τ(q̈) along the implicit line is not monotone. Plain Newton stalls or finds spurious roots.
- The MLP therefore gets the most robust inversion we found. It uses Newton with a per-sample backtracking line search, run from three starting points: the warm start, zero, and a damping-only guess a = −r(0) / (h·diag ∂τ/∂q̇). The lowest-residual solution is kept.
- The worst Newton residual within each step is recorded, and a rollout counts as **diverged** if it becomes non-finite or |q̇| > 50 rad/s.
- `pos_rmse` excludes diverged rollouts. `pos_rmse_median_incl_diverged` counts them as ∞, so excluding them can't flatter a model.

**Metrics** are the wrapped position RMSE and the velocity RMSE over (0, H] for H = 0.5, 1 and 2 s, both pooled and per joint, plus the error at H.

## 3. Reference trajectories (`trajectories.py`)

Each trajectory lasts 5 s, sampled at 10 ms, with closed-form q, q̇ and q̈. The seed is 2026.

| Set | Trajectories | Constraints |
|---|---|---|
| main (10) | 5 Fourier series (3 harmonics of a 0.1–0.3 Hz base) and 5 quintic minimum-jerk moves through 4 waypoints | ≥ 0.2 rad inside the joint limits; analytical τ_ref ≤ 70 % of the gear limits; **100 % of samples inside the training range** of q, q̇ and q̈ |
| ood (4) | 2 faster Fourier series (0.4–0.56 Hz base) and 2 minimum-jerk moves with 1 s segments through 6 waypoints | ≥ 0.2 rad inside the limits; τ_ref ≤ 90 % of the limits; ≤ 80 % of samples inside the training range (actual: 8–67 %) |

## 4. Computed-torque control (`computed_torque.py`, `tracking_experiment.py`)

- The outer loop is a_cmd = q̈_ref + K_d(q̇_ref − q̇) + K_p·wrap(q_ref − q), with ω_n = 15 rad/s and ζ = 1.
- The control rate is 100 Hz with zero-order hold, the dm_control rate. Torque is clipped to ±gear.

| Variant | Law |
|---|---|
| `sampled` (default) | τ = ID(q, q̇ + Δt·a_cmd, a_cmd) |
| `classic` | τ = ID(q, q̇, a_cmd) |
| PD only | τ = 2·gear·e + 0.1·gear·ė (no model) |

**Why the sampled variant is needed**
- The elbow and wrist are damping-dominated: M/D ≈ 10⁻³ s and 6·10⁻⁵ s, against a 10 ms hold.
- Within one hold their velocity settles at (τ − g − c)/D. Under the classic law the inertial term M·a_cmd therefore changes the velocity by only (M/D)·a_cmd, not Δt·a_cmd. Feedback on these joints is almost lost.
- Evaluating the *same* inverse dynamics at the velocity the joint should reach by the end of the step restores it. For an inertia-dominated joint the sampled law reduces to the classic one.
- The classic variant is still run for the three main models, to show the effect.

The experiment runs every model at every data fraction, the classic variant and PD only, on all 14 trajectories. It also has a robustness run with a 0.1 rad initial error (`--offset 0.1`).

## 5. Results

The full tables are in [`results/summary.md`](results/summary.md).

All runs use seed 42 checkpoints and the 100 % models unless a fraction is given.

**One-step torque RMSE** (N·m; the official splits, recomputed and cross-checked against Members 3 and 4)

| | test | OOD |
|---|---|---|
| analytical | 1.7e-16 | 3.1e-16 |
| DeLaN | 0.0147 | 0.0250 |
| MLP | 0.0440 | 0.450 |

**Rollout position RMSE against MuJoCo** (rad, 90 windows of 2 s per distribution; *median* counts diverged rollouts as ∞)

| | in-dist. median @0.5 / 1 / 2 s | diverged @2 s | OOD median @0.5 / 1 / 2 s | diverged @2 s |
|---|---|---|---|---|
| analytical (integration floor) | 2.1e-4 / 3.3e-4 / 3.7e-4 | 0 % | 1.2e-3 / 1.1e-3 / 1.2e-3 | 0 % |
| DeLaN | 5.3e-3 / 6.6e-3 / 7.9e-3 | 0 % | 9.3e-3 / 9.8e-3 / 1.0e-2 | 0 % |
| MLP | 2.2e-2 / 2.9e-2 / 5.1e-2 | 31 % | ∞ | 92 % |

**Computed-torque tracking RMSE** (rad; mean over trajectories, `sampled` law)

| | 10 main trajectories | 4 OOD trajectories | main, 0.1 rad initial error |
|---|---|---|---|
| analytical | 1.0e-3 | 5.9e-3 | 1.27e-2 |
| DeLaN | 3.7e-3 | 1.0e-2 | 1.33e-2 |
| MLP | 5.3e-3 | 2.0e-1 | 1.42e-2 |
| PD only | 5.1e-2 | 1.3e-1 | 5.3e-2 |
| classic CT: analytical / DeLaN / MLP | 0.26 / 0.017 / 0.34 | 0.60 / 0.056 / 0.65 | |

**Data efficiency** (`figures/data_efficiency.png`)

| | 10 % | 30 % | 50 % | 100 % |
|---|---|---|---|---|
| DeLaN tracking, main | 3.5e-3 | 3.6e-3 | 3.6e-3 | 3.7e-3 |
| MLP tracking, main | 6.6e-2 | 8.1e-3 | 6.9e-3 | 5.3e-3 |
| DeLaN rollouts diverged | 0 % | 0 % | 0 % | 0 % |
| MLP rollouts diverged (in-dist.) | 68 % | 41 % | 43 % | 31 % |

**Mass matrix** (10 000 uniform configurations within the limits)

| | positive-definite | median relative error to analytical M |
|---|---|---|
| DeLaN, all fractions | 100 % | 0.60–0.64 |
| MLP, implied ∂τ/∂q̈ | 0.1–1.8 % | 2.7–3.8 |

On the test states the MLP's implied mass matrix is positive-definite at only 0–10.6 % of them, and it is also strongly non-symmetric (relative asymmetry about 1.4).

**Findings**

1. **Physics structure is what makes the learned model usable as a simulator.**
   - DeLaN's positive-definite M and explicit damping give a well-posed inversion. None of its 720 rollouts (90 windows × 2 distributions × 4 fractions) diverged, at any data fraction or in either distribution.
   - The MLP's one-step torque error is only 3× worse than DeLaN's in-range. However, its implied inertia is indefinite and non-symmetric, so its inversion has no unique, well-behaved root. 31–68 % of its in-range rollouts and 92–100 % of its OOD rollouts diverge.
2. **Computed torque needs only inverse dynamics, so the MLP tracks well inside the training range.** It reaches 5.3e-3 rad, against 3.7e-3 for DeLaN.
3. **Outside the training range the MLP's tracking degrades 38×, to 0.20 rad**, which is worse than PD with no model. DeLaN degrades only 2.7×, to 1.0e-2 rad, matching its 1.7× one-step OOD degradation.
4. **DeLaN is data-efficient; the MLP is not.** DeLaN's closed-loop error is flat from 10 % to 100 % data, as its one-step error was. At 10 % data the MLP tracks 19× worse than DeLaN (6.6e-2 rad).
5. **The sampled computed-torque law matters on this plant.** It is needed because the elbow and wrist are damping-dominated (M/D ≪ 10 ms).
   - Under the classic law even the exact model tracks at 0.26 rad, worse than PD. Its error grows from root to wrist as M/D shrinks: 0.07 → 0.20 → 0.25 → 0.40 rad.
   - DeLaN under the classic law does far better (0.017 rad), because it over-estimates the tiny elbow and wrist inertias (Member 4 §5). This is a lucky side effect, not accuracy: under the sampled law the exact model is best.
6. **The gap between DeLaN and analytical** (about 3.7× in tracking, about 20× in rollout) comes from the 0.015–0.025 N·m torque error and the weakly identified elbow and wrist inertia. Gravity and damping are near-exact.
7. **Cost per state, on CPU:**

   | | inverse dynamics | implicit rollout step | controller step |
   |---|---|---|---|
   | MLP | 0.1 ms | 12 ms (multi-start Newton) | 0.1 ms |
   | analytical | 1.9 ms | 1.9 ms | 1.8 ms |
   | DeLaN | 4.4 ms | 21 ms | 3.4 ms |

   All three are well within the 10 ms control period for computed torque.

## 6. Reproducing

From the repository root, with the project environment:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash evaluation_control/run_all.sh --video          # everything, ~15–20 min on a laptop CPU
```

Individual steps, and the figures each produces in `results/figures/`:

| Command | Produces |
|---|---|
| `python -m evaluation_control.trajectories` | `trajectories_{main,ood}.npz`, `reference_trajectories_*.png` |
| `python -m evaluation_control.rollout` (`--quick` for main models only, `--substeps 10` for h = 1 ms) | `rollout_metrics.csv`, `rollout_error_{test,ood}.png`, `rollout_example_{test,ood}.png` |
| `python -m evaluation_control.tracking_experiment [--video] [--offset 0.1] [--quick]` | `tracking_metrics.csv`, `tracking_summary.csv`, `tracking_rmse.png`, `tracking_traces_*.png`, `tracking_demo.mp4`, `tracking_demo_ood.mp4` |
| `python -m evaluation_control.evaluate` | `summary.md`, `one_step_errors.csv`, `mass_matrix_pd.csv`, `timing.csv`, `data_efficiency.png`, `mass_matrix_pd.png` |
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q evaluation_control/test_evaluation.py` | 15 tests (below) |

Set `MUJOCO_GL=egl` on a machine without a display; `run_all.sh` does this. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is only needed when a sourced ROS installation injects pytest plugins.

**Tests**
- the analytical adapter equals Member 2's model and the dataset (1e-10)
- the MLP adapter reproduces Member 3's test RMSE
- the Jacobians of all models match finite differences
- DeLaN's structural Jacobian equals autograd
- the forward/inverse round trip
- the MLP Newton inversion
- the analytical rollout matches MuJoCo: < 1e-4 rad at h = 1 ms, < 5e-3 rad at h = 10 ms
- the trajectory constraints hold
- analytical computed torque tracks below 3e-3 rad, and the classic law is more than 5× worse

## 7. Files

| File | Purpose |
|---|---|
| `common.py` | paths, constants, angle wrapping, training ranges |
| `models.py` | shared interface, the three adapters, batched Newton solver |
| `rollout.py` | MuJoCo ground truth, implicit integrator, rollout metrics |
| `trajectories.py` | reference trajectories with feasibility and training-range checks |
| `computed_torque.py` | computed-torque (`sampled`/`classic`) and PD controllers |
| `tracking_experiment.py` | closed-loop experiments in MuJoCo, metrics, plots, demo video |
| `evaluate.py` | combined one-step, rollout, tracking, data-efficiency, mass-matrix and timing tables |
| `test_evaluation.py` | tests |
| `run_all.sh` | full pipeline |
| `results/` | CSV/NPZ results, `summary.md`, `figures/`, `tracking_demo{,_ood}.mp4` |
