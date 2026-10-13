#!/bin/bash
#SBATCH --job-name=stifflow-formal-accept
# Pass SHHPC resources to sbatch, including one GPU and at least four CPUs.
set -euo pipefail
: "${STIFFLOW_WORKSPACE:?Set actual SHHPC workspace}"
source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"
: "${STIFFLOW_TRAIN_LABELS:?Set merged training manifest}"
: "${STIFFLOW_VAL_LABELS:?Set merged validation manifest}"
: "${STIFFLOW_ACCEPTANCE_ROOT:?Set a fresh external acceptance directory}"
python -m stifflow.formal acceptance --config configs/qm9-formal.yaml \
    --train-labels "$STIFFLOW_TRAIN_LABELS" --val-labels "$STIFFLOW_VAL_LABELS" \
    --output "$STIFFLOW_ACCEPTANCE_ROOT"
