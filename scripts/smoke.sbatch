#!/bin/bash
#SBATCH --job-name=etflow-smoke
#SBATCH --account=wjg3
#SBATCH --partition=rooster,chem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
: "${STIFFLOW_WORKSPACE:?Export the absolute workspace path before sbatch}"
export ETFLOW_ENV="${ETFLOW_ENV:-$HOME/anaconda3/envs/etflow}"
source "$(dirname "$(dirname "$ETFLOW_ENV")")/etc/profile.d/conda.sh"
conda activate "$ETFLOW_ENV"
export PYTHONPATH="$STIFFLOW_WORKSPACE/stifflow-v1:$STIFFLOW_WORKSPACE/stifflow-v1-alternative-A:$STIFFLOW_WORKSPACE/stifflow-v1-benchmark"
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$STIFFLOW_WORKSPACE/cache/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
cd "$STIFFLOW_WORKSPACE"
python -m stifflow.smoke \
    --cache-dir "$STIFFLOW_WORKSPACE/cache/etflow-smoke-$SLURM_JOB_ID" \
    --output-dir "$STIFFLOW_WORKSPACE/runs/etflow-smoke-$SLURM_JOB_ID"
