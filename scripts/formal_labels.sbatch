#!/bin/bash
#SBATCH --job-name=stifflow-full-labels
# Pass account, CPU partition, memory, walltime and array bounds to sbatch.
set -euo pipefail
: "${STIFFLOW_WORKSPACE:?Set actual SHHPC workspace}"
source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"
: "${STIFFLOW_FROZEN_DATA:?Set audited data manifest}"
: "${STIFFLOW_LABEL_ROOT:?Set an external output root}"
: "${STIFFLOW_LABEL_SPLIT:?Set train or val}"
: "${SLURM_ARRAY_TASK_ID:?Submit as a CPU job array}"
output="$STIFFLOW_LABEL_ROOT/$STIFFLOW_LABEL_SPLIT/shard-$SLURM_ARRAY_TASK_ID"
args=(build --manifest "$STIFFLOW_FROZEN_DATA" --config configs/qm9-formal.yaml
      --split "$STIFFLOW_LABEL_SPLIT" --shard-index "$SLURM_ARRAY_TASK_ID" --output "$output")
if [[ -e "$output/owner.json" ]]; then args+=(--resume); fi
if [[ -n "${STIFFLOW_REUSE_LABELS:-}" ]]; then args+=(--reuse "$STIFFLOW_REUSE_LABELS"); fi
python -m stifflow.formal_cache "${args[@]}"
