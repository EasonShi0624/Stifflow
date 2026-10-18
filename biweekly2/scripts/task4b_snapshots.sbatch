#!/bin/bash
# Task 4b: monitor learning-curve snapshots for Alternative A and control.
# Usage: sbatch --array=0-6 ... task4b_snapshots.sbatch generate|evaluate
#
# Index map (formal 250k runs, seed 42):
#   0: alternative_a 25000   1: alternative_a 50000   2: alternative_a 100000
#   3: alternative_a 175000  4: control 50000         5: control 100000
#   6: control 175000
# (A/control at 25k where missing and both at 250k already exist.)
set -euo pipefail
export STIFFLOW_WORKSPACE="${STIFFLOW_WORKSPACE:-/gpfsnyu/scratch/ys6132/stifflow}"
export ETFLOW_ENV="${ETFLOW_ENV:-/gpfsnyu/home/ys6132/anaconda3/envs/etflow}"
export STIFFLOW_BIWEEKLY_ROOT="$STIFFLOW_WORKSPACE/runs/biweekly2-20261019"
export STIFFLOW_EVAL_ROOT="$STIFFLOW_WORKSPACE/runs/formal-eval-20261018"
export STIFFLOW_RUN_ROOT="$STIFFLOW_WORKSPACE/runs/formal-4090d-staged"
export STIFFLOW_REFERENCE_ROOT="$STIFFLOW_WORKSPACE/cache/qm9-v3-references-4090d"

mode="${1:?Pass generate or evaluate}"
index="${SLURM_ARRAY_TASK_ID:-0}"
[[ "$index" =~ ^[0-6]$ ]] || { echo "Array index must be 0 through 6" >&2; exit 2; }

methods=(alternative_a alternative_a alternative_a alternative_a control control control)
steps=(25000 50000 100000 175000 50000 100000 175000)
method="${methods[$index]}"
step="${steps[$index]}"
seed=42
base="$STIFFLOW_BIWEEKLY_ROOT/learning-curves/$method-seed$seed-step$(printf '%06d' "$step")"
mkdir -p "$STIFFLOW_BIWEEKLY_ROOT/logs" "$base"

source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"

if [[ "$mode" == generate ]]; then
  python -m stifflow.formal generate \
    --config "$STIFFLOW_WORKSPACE/stifflow-v1/configs/qm9-formal-4090d.yaml" \
    --checkpoint "$STIFFLOW_RUN_ROOT/$method-seed$seed/snapshots/step-$(printf '%08d' "$step").ckpt" \
    --references "$STIFFLOW_REFERENCE_ROOT/val/monitor.json" \
    --stage monitor \
    --output "$base/run"
elif [[ "$mode" == evaluate ]]; then
  python -m stifflow.formal evaluate \
    --manifest "$base/run/generated/manifest.json" \
    --output "$base/evaluation.json"
else
  echo "Unknown mode: $mode" >&2
  exit 2
fi
