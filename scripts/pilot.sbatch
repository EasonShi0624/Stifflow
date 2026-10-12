#!/bin/bash
#SBATCH --job-name=stifflow-pilot
#SBATCH --account=wjg3
#SBATCH --partition=rooster,chem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
: "${STIFFLOW_WORKSPACE:?Export the absolute workspace path}"
: "${STIFFLOW_TRAIN_LABELS:?Export complete official training labels}"
: "${STIFFLOW_VAL_LABELS:?Export complete official validation labels}"
export ETFLOW_ENV="${ETFLOW_ENV:-$HOME/anaconda3/envs/etflow}"
set +u
source "$(dirname "$(dirname "$ETFLOW_ENV")")/etc/profile.d/conda.sh"
conda activate "$ETFLOW_ENV"
set -u
export PYTHONPATH="$STIFFLOW_WORKSPACE/stifflow-v1:$STIFFLOW_WORKSPACE/stifflow-v1-alternative-A:$STIFFLOW_WORKSPACE/stifflow-v1-benchmark"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="$STIFFLOW_WORKSPACE/cache/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
cd "$STIFFLOW_WORKSPACE"
python -m stifflow.pilot --train-labels "$STIFFLOW_TRAIN_LABELS" --val-labels "$STIFFLOW_VAL_LABELS" \
    --output "$STIFFLOW_WORKSPACE/runs/pilot-$SLURM_JOB_ID" "$@"
