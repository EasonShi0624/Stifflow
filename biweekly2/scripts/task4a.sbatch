#!/bin/bash
# Task 4a: stiffness-signal quantification + label-audit aggregation (CPU).
set -euo pipefail
export STIFFLOW_WORKSPACE="${STIFFLOW_WORKSPACE:-/gpfsnyu/scratch/ys6132/stifflow}"
export ETFLOW_ENV="${ETFLOW_ENV:-/gpfsnyu/home/ys6132/anaconda3/envs/etflow}"
source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"
export STIFFLOW_BIWEEKLY_ROOT="$STIFFLOW_WORKSPACE/runs/biweekly2-20261019"
mkdir -p "$STIFFLOW_BIWEEKLY_ROOT/results"

python "$STIFFLOW_BIWEEKLY_ROOT/scripts/task4a_stiffness_signal.py"
