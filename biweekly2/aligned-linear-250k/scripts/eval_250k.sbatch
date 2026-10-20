#!/bin/bash
# Full-budget aligned_linear evaluation: monitor -> test -> val (frozen protocols).
# Usage: sbatch ... eval_250k.sbatch <seed>
set -euo pipefail
export STIFFLOW_WORKSPACE="${STIFFLOW_WORKSPACE:-/gpfsnyu/scratch/ys6132/stifflow}"
export ETFLOW_ENV="${ETFLOW_ENV:-/gpfsnyu/home/ys6132/anaconda3/envs/etflow}"
export STIFFLOW_AL250K_ROOT="${STIFFLOW_AL250K_ROOT:-$STIFFLOW_WORKSPACE/runs/aligned-linear-250k}"

seed="${1:?seed (42|43|44)}"
[[ "$seed" =~ ^(42|43|44)$ ]] || { echo "seed must be 42, 43 or 44" >&2; exit 2; }

ROOT="$STIFFLOW_AL250K_ROOT/eval/aligned_linear-seed$seed"
CKPT="$STIFFLOW_AL250K_ROOT/train/aligned_linear-seed$seed/checkpoints/step-00250000.ckpt"
[[ -f "$CKPT" ]] || { echo "Missing 250k checkpoint: $CKPT" >&2; exit 2; }
[[ -e "$ROOT" ]] && { echo "Existing evaluation output is protected: $ROOT" >&2; exit 1; }
mkdir -p "$ROOT"

source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"

REFS="$STIFFLOW_WORKSPACE/cache/qm9-v3-references-4090d"

for stage in monitor test val; do
  case "$stage" in
    monitor) manifest="$REFS/val/monitor.json" ;;
    test)    manifest="$REFS/test/manifest.json" ;;
    val)     manifest="$REFS/val/manifest.json" ;;
  esac
  out="$ROOT/$stage"
  # generate creates the output directory; it refuses to write into an existing one.
  python -m stifflow_benchmark.generate "$manifest" \
    --checkpoint "$CKPT" --steps 50 --seed 42 \
    --output "$out/generated"
  python -m stifflow_benchmark.evaluate "$out/generated/manifest.json" \
    --allow-disconnected --output "$out/evaluation.json"
  echo "done: $stage"
done
