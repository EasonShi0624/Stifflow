#!/bin/bash
#SBATCH --job-name=stifflow-formal
# Supply account/partition/GPU/CPU/memory/walltime and --array=0-8%N to sbatch.
set -euo pipefail
: "${STIFFLOW_WORKSPACE:?Set actual SHHPC workspace}"
source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"
: "${STIFFLOW_TRAIN_LABELS:?Set merged training manifest}"
: "${STIFFLOW_VAL_LABELS:?Set merged validation manifest}"
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
python -m stifflow.preflight --gpu --warmup --output "$base/preflight-$SLURM_JOB_ID.json"
args=(--config "${STIFFLOW_FORMAL_CONFIG:-configs/qm9-formal.yaml}" --profile formal --method "$method" --seed "$seed"
      --train-labels "$STIFFLOW_TRAIN_LABELS" --val-labels "$STIFFLOW_VAL_LABELS"
      --acceptance "$STIFFLOW_ACCEPTANCE" --checkpoint-dir "$base/checkpoints"
      --output "$base/segments/$SLURM_JOB_ID")
if [[ -e "$base/checkpoints/latest.ckpt" ]]; then args+=(--resume "$base/checkpoints/latest.ckpt"); fi
python -m stifflow.training "${args[@]}"
