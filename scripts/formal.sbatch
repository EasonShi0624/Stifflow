#!/bin/bash
#SBATCH --job-name=stifflow-formal
# Supply account/partition/GPU/CPU/memory/walltime and --array=0-8%N to sbatch.
set -euo pipefail
: "${STIFFLOW_WORKSPACE:?Set actual SHHPC workspace}"
source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"
if [[ "${STIFFLOW_STAGE_CACHE:-0}" == 1 ]]; then
  source_root="${STIFFLOW_PACKED_SOURCE_ROOT:-$STIFFLOW_WORKSPACE/cache/qm9-v3-packed-strict}"
  STIFFLOW_TRAIN_LABELS="$source_root/train/manifest.json"
  STIFFLOW_VAL_LABELS="$source_root/val/manifest.json"
fi
: "${STIFFLOW_TRAIN_LABELS:?Set canonical merged training manifest}"
: "${STIFFLOW_VAL_LABELS:?Set canonical merged validation manifest}"
: "${STIFFLOW_ACCEPTANCE:?Set the matching acceptance workflow.json}"
: "${STIFFLOW_RUN_ROOT:?Set external formal run root}"
index="${SLURM_ARRAY_TASK_ID:-0}"
[[ "$index" =~ ^[0-8]$ ]] || { echo "Array index must be 0 through 8" >&2; exit 2; }
methods=(precision alternative_a control)
seeds=(42 43 44)
method="${methods[$((index / 3))]}"
seed="${seeds[$((index % 3))]}"
base="$STIFFLOW_RUN_ROOT/$method-seed$seed"
mkdir -p "$base"
exec 9>"$base/.run.lock"
flock -n 9 || { echo "Another job owns $base" >&2; exit 4; }
if [[ -e "$base/checkpoints/owner.json" && ! -e "$base/checkpoints/latest.ckpt" ]]; then
  echo "Checkpoint owner has no latest checkpoint; inspect this run before restarting" >&2; exit 5
fi
attempt="${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}-$(python -c 'import uuid; print(uuid.uuid4().hex)')"
private=""
cleanup() {
  if [[ -n "$private" && -d "$private" && ! -L "$private" && -f "$private/.owner" ]] &&
     [[ "$(cat "$private/.owner")" == "$attempt" ]]; then
    rm -rf -- "$private"
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ "${STIFFLOW_STAGE_CACHE:-0}" == 1 ]]; then
  stage_base=""; best=0
  for path in "${SLURM_TMPDIR:-}" /tmp /local /scratch /nvme; do
    [[ -n "$path" && -d "$path" && -w "$path" ]] || continue
    info=$(df -B1 --output=fstype,avail "$path" 2>/dev/null | tail -1 || true)
    read -r fstype avail <<< "$info"
    case "$fstype" in xfs|ext4|ext3|ext2|btrfs|f2fs) ;; *) continue;; esac
    [[ "$avail" =~ ^[0-9]+$ ]] || continue
    if (( avail > best )); then best=$avail; stage_base=$path; fi
  done
  if (( best < 120*1024*1024*1024 )); then
    echo "No writable local block storage with 120 GiB free" >&2; exit 3
  fi
  private=$(mktemp -d "$stage_base/stifflow-$UID-${SLURM_JOB_ID}-XXXXXXXX")
  stage_device=$(stat -c %d "$stage_base")
  printf '%s\n' "$attempt" > "$private/.owner"
  export TMPDIR="$private/tmp"
  mkdir "$TMPDIR"
  python scripts/shhpc/stage_packed_cache.py --src-root "$source_root" \
    --dst-root "$private/cache" --splits train val --verify-samples 4 \
    --space-lock "/tmp/stifflow-stage-$UID-$stage_device.lock"
  STIFFLOW_PACKED_CACHE_ROOT="$private/cache"
fi
echo "attempt=$attempt method=$method seed=$seed packed_root=${STIFFLOW_PACKED_CACHE_ROOT:-canonical}"
python -m stifflow.preflight --gpu --warmup --output "$base/preflight-$attempt.json"
args=(--config "${STIFFLOW_FORMAL_CONFIG:-configs/qm9-formal.yaml}" --profile formal --method "$method" --seed "$seed"
      --train-labels "$STIFFLOW_TRAIN_LABELS" --val-labels "$STIFFLOW_VAL_LABELS"
      --acceptance "$STIFFLOW_ACCEPTANCE" --checkpoint-dir "$base/checkpoints"
      --output "$base/segments/$attempt")
if [[ -n "${STIFFLOW_IO_COMPATIBILITY:-}" ]]; then args+=(--io-compatibility "$STIFFLOW_IO_COMPATIBILITY"); fi
if [[ -n "${STIFFLOW_PACKED_CACHE_ROOT:-}" ]]; then args+=(--packed-cache-root "$STIFFLOW_PACKED_CACHE_ROOT"); fi
if [[ -e "$base/checkpoints/latest.ckpt" ]]; then args+=(--resume "$base/checkpoints/latest.ckpt"); fi
python -m stifflow.training "${args[@]}"
