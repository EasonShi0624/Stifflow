#!/bin/bash
#SBATCH --job-name=stifflow-labels
#SBATCH --account=wjg3
#SBATCH --partition=parallel,debug,chem,rooster
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
: "${STIFFLOW_WORKSPACE:?Export the absolute workspace path before sbatch}"
: "${STIFFLOW_LABEL_MANIFEST:?Export the frozen split manifest path}"
: "${STIFFLOW_LABEL_OUTPUT:?Export a fresh label output directory}"
export ETFLOW_ENV="${ETFLOW_ENV:-$HOME/anaconda3/envs/etflow}"
set +u
source "$(dirname "$(dirname "$ETFLOW_ENV")")/etc/profile.d/conda.sh"
conda activate "$ETFLOW_ENV"
set -u
export PYTHONPATH="$STIFFLOW_WORKSPACE/stifflow-v1:$STIFFLOW_WORKSPACE/stifflow-v1-alternative-A:$STIFFLOW_WORKSPACE/stifflow-v1-benchmark"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$STIFFLOW_WORKSPACE/cache/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
cd "$STIFFLOW_WORKSPACE"
python -m stifflow.stiffness --manifest "$STIFFLOW_LABEL_MANIFEST" \
    --output "$STIFFLOW_LABEL_OUTPUT" "$@"
