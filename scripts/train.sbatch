#!/bin/bash
#SBATCH --job-name=stifflow-train
#SBATCH --account=torch_pr_1034_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
: "${STIFFLOW_WORKSPACE:?Export workspace path}"
: "${ETFLOW_ENV:?Export existing etflow environment prefix}"
: "${STIFFLOW_TRAIN_LABELS:?Export complete training label manifest}"
: "${STIFFLOW_VAL_LABELS:?Export complete validation label manifest}"
source "$(dirname "$(dirname "$ETFLOW_ENV")")/etc/profile.d/conda.sh"
conda activate "$ETFLOW_ENV"
export PYTHONPATH="$STIFFLOW_WORKSPACE/stifflow-v1:$STIFFLOW_WORKSPACE/stifflow-v1-alternative-A:$STIFFLOW_WORKSPACE/stifflow-v1-benchmark"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="$STIFFLOW_WORKSPACE/cache/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
cd "$STIFFLOW_WORKSPACE"
python -m stifflow.training --train-labels "$STIFFLOW_TRAIN_LABELS" \
    --val-labels "$STIFFLOW_VAL_LABELS" --output "$STIFFLOW_WORKSPACE/runs/train-$SLURM_JOB_ID" "$@"
