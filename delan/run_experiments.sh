#!/usr/bin/env bash
# Member 4 (DeLaN) full pipeline on the official dataset in ../dataset/data/.
# Usage (from delan/):  PY=/path/to/python bash run_experiments.sh [--skip-tuning]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p runs checkpoints/ablation

# 1. validation-only hyper-parameter sweep (30 % subset, seed 42; mirrors tune_mlp.py)
if [ "${1:-}" != "--skip-tuning" ]; then
  $PY -W ignore tune_delan.py --jobs 4 --threads 4 | tee runs/tuning.log
fi

# 2. data-efficiency runs: 10/30/50/100 % (MLP-identical nested subsets), seed 42
for f in 1.0 0.5 0.3 0.1; do
  $PY -W ignore train_delan.py --fraction "$f" --seed 42 --threads 3 > "runs/train_f${f}_s42.log" 2>&1 &
done
# 3. ablation: no damping term, 100 % data, seed 42
$PY -W ignore train_delan.py --fraction 1.0 --seed 42 --no-damping --threads 3 \
    --out checkpoints/ablation/delan_no_damping_f1.0_s42.pt > runs/ablation.log 2>&1 &
wait

# 4. tests, SPD report, evaluation
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest -q -p no:cacheprovider -W ignore test_delan.py test_spd.py
$PY -W ignore test_spd.py
$PY -W ignore evaluate_delan.py
