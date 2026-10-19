#!/bin/bash
# Full-budget (250k) aligned_linear training run for one seed.
# Usage: sbatch ... train_250k.sbatch <seed>
set -euo pipefail
export STIFFLOW_WORKSPACE="${STIFFLOW_WORKSPACE:-/gpfsnyu/scratch/ys6132/stifflow}"
export ETFLOW_ENV="${ETFLOW_ENV:-/gpfsnyu/home/ys6132/anaconda3/envs/etflow}"
export STIFFLOW_AL250K_ROOT="${STIFFLOW_AL250K_ROOT:-$STIFFLOW_WORKSPACE/runs/aligned-linear-250k}"

seed="${1:?seed (42|43|44)}"
[[ "$seed" =~ ^(42|43|44)$ ]] || { echo "seed must be 42, 43 or 44" >&2; exit 2; }

ROOT="$STIFFLOW_AL250K_ROOT/train/aligned_linear-seed$seed"
CACHE="$STIFFLOW_WORKSPACE/cache/qm9-v3-packed-strict"
STAGE="/tmp/stifflow-packed-v3-aligned_linear-seed$seed"
TMPDIR_JOB="/tmp/stifflow-al250k-tmp-$SLURM_JOB_ID"
mkdir -p "$ROOT"
if [[ -e "$STAGE" || -e "$TMPDIR_JOB" ]]; then
  echo "Staging path already exists: $STAGE / $TMPDIR_JOB" >&2
  exit 1
fi
cleanup() { rm -rf "$STAGE" "$TMPDIR_JOB"; }
trap cleanup EXIT

source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"

python "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc/stage_packed_cache.py" \
  --src-root "$CACHE" --dst-root "$STAGE" --splits train val \
  --verify-samples 4 --min-free-gib 120
export TMPDIR="$TMPDIR_JOB"
mkdir -p "$TMPDIR"

python -m stifflow.preflight --gpu --warmup \
  --output "$ROOT/preflight-$SLURM_JOB_ID.json"

resume_args=()
if [[ -e "$ROOT/checkpoints/latest.ckpt" ]]; then
  resume_args+=(--resume "$ROOT/checkpoints/latest.ckpt")
fi

python -m stifflow.training \
  --config "$STIFFLOW_AL250K_ROOT/configs/qm9-full-budget-250k.yaml" \
  --profile full_budget --method aligned_linear --seed "$seed" \
  --train-labels "$CACHE/train/manifest.json" \
  --val-labels "$CACHE/val/manifest.json" \
  --packed-cache-root "$STAGE" \
  --checkpoint-dir "$ROOT/checkpoints" \
  --output "$ROOT/run-$SLURM_JOB_ID" \
  "${resume_args[@]}"
