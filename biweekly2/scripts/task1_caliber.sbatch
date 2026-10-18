#!/bin/bash
# Task 1: evaluation-caliber audit + pooled 3-seed paired bootstrap.
# CPU-only; reads existing formal-eval artifacts, writes new summaries.
set -euo pipefail
export STIFFLOW_WORKSPACE="${STIFFLOW_WORKSPACE:-/gpfsnyu/scratch/ys6132/stifflow}"
export ETFLOW_ENV="${ETFLOW_ENV:-/gpfsnyu/home/ys6132/anaconda3/envs/etflow}"
source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"
export STIFFLOW_BIWEEKLY_ROOT="$STIFFLOW_WORKSPACE/runs/biweekly2-20261019"
mkdir -p "$STIFFLOW_BIWEEKLY_ROOT/results"

python "$STIFFLOW_BIWEEKLY_ROOT/scripts/task1_caliber.py"
