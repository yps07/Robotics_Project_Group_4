#!/usr/bin/env bash
# Workstream E full pipeline. Run from anywhere; uses $PY (default: the repo's .venv).
#   bash evaluation_control/run_all.sh            # everything (~15 min on a laptop CPU)
#   bash evaluation_control/run_all.sh --video    # also render results/tracking_demo.mp4
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
VIDEO=""
[[ "${1:-}" == "--video" ]] && VIDEO="--video"

$PY -m evaluation_control.trajectories                 # results/trajectories_*.npz, figures/reference_trajectories_*.png
$PY -m evaluation_control.rollout                      # results/rollout_metrics.csv, figures/rollout_*.png
$PY -m evaluation_control.tracking_experiment $VIDEO   # results/tracking_*.csv, figures/tracking_*.png
$PY -m evaluation_control.tracking_experiment --offset 0.1   # robustness: 0.1 rad initial error
$PY -m evaluation_control.evaluate                     # results/summary.md, figures/data_efficiency.png, mass_matrix_pd.png
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest -q evaluation_control/test_evaluation.py
