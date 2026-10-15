#!/bin/bash
# Formal StifFlow training with the frozen packed label cache staged onto
# node-local NVMe (avoids GPFS per-worker full-cache verification and shard-open
# latency). Staged content is byte-identical; only manifest paths are rewritten.
# Supply --array=0-8%N, account/partition/GPU/CPU/memory/walltime via sbatch.
#SBATCH --job-name=stifflow-formal
set -euo pipefail
: "${STIFFLOW_WORKSPACE:?Set actual SHHPC workspace}"
: "${STIFFLOW_ACCEPTANCE:?Set the matching torch_runtime acceptance json}"

index="${SLURM_ARRAY_TASK_ID:-0}"
[[ "$index" =~ ^[0-8]$ ]] || { echo "Array index must be 0 through 8" >&2; exit 2; }
methods=(precision alternative_a control)
seeds=(42 43 44)
method="${methods[$((index / 3))]}"
seed="${seeds[$((index % 3))]}"
echo "=== node $(hostname)  run=$method-seed$seed  index=$index job=${SLURM_JOB_ID} ==="

# --- pick node-local block storage (never GPFS / tmpfs) ---
STAGE_BASE=""; BEST=0
for path in "${SLURM_TMPDIR:-}" /tmp /local /scratch /nvme; do
  [[ -n "$path" && -d "$path" ]] || continue
  out=$(df -B1 --output=fstype,avail "$path" 2>/dev/null | tail -1 || true)
  fstype=$(printf '%s' "$out" | awk '{print $1}')
  avail=$(printf '%s' "$out" | awk '{print $2}')
  case "$fstype" in
    gpfs|tmpfs|lustre|nfs*|cifs|fuse*) continue;;
    xfs|ext4|ext3|ext2|btrfs|f2fs) ;;
    *) continue;;
  esac
  [[ -n "$avail" ]] || continue
  if (( avail > BEST )); then BEST=$avail; STAGE_BASE=$path; fi
done
if (( BEST < 120*1024*1024*1024 )); then
  echo "NO_LOCAL_DISK best=$((BEST/1024/1024/1024))G need=120G" >&2; exit 3
fi
echo "staging base $STAGE_BASE ($((BEST/1024/1024/1024))G free)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
source "$REPO_DIR/scripts/shhpc_env.sh"

# Deterministic per-run staging path so that a requeue/resume re-creates an
# identical manifest and the checkpoint owner signature still matches.
STAGE="$STAGE_BASE/stifflow-packed-v3-$method-seed$seed"
cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT
export TMPDIR="$STAGE_BASE/stifflow-tmp-$method-seed$seed"; mkdir -p "$TMPDIR"

if [[ -f "$STAGE/.staged" ]]; then
  echo "reusing staged cache at $STAGE"
else
  rm -rf "$STAGE"
  echo "staging packed cache to $STAGE"
  time python "$SCRIPT_DIR/stage_packed_cache.py" \
    --src-root "$STIFFLOW_WORKSPACE/cache/qm9-v3-packed-strict" \
    --dst-root "$STAGE" --splits train val --verify-samples 4
  touch "$STAGE/.staged"
fi
TRAIN="$STAGE/train/manifest.json"; VAL="$STAGE/val/manifest.json"
echo "staged_train_manifest_sha256=$(sha256sum "$TRAIN" | awk '{print $1}')"
echo "staged_val_manifest_sha256=$(sha256sum "$VAL" | awk '{print $1}')"
df -h "$STAGE_BASE" | tail -1

base="$STIFFLOW_RUN_ROOT/$method-seed$seed"
mkdir -p "$base"
python -m stifflow.preflight --gpu --warmup --output "$base/preflight-$SLURM_JOB_ID.json"
args=(--config "${STIFFLOW_FORMAL_CONFIG:-configs/qm9-formal-4090d.yaml}" --profile formal --method "$method" --seed "$seed"
      --train-labels "$TRAIN" --val-labels "$VAL"
      --acceptance "$STIFFLOW_ACCEPTANCE" --checkpoint-dir "$base/checkpoints"
      --output "$base/segments/$SLURM_JOB_ID")
if [[ -e "$base/checkpoints/latest.ckpt" ]]; then args+=(--resume "$base/checkpoints/latest.ckpt"); fi
python -m stifflow.training "${args[@]}"
