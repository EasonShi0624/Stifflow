#!/bin/bash
#SBATCH --job-name=geom_attr
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=4:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_attribution_%x_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_attribution_%x_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs

export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general

: "${STIFFLOW_ATTRIBUTION_SPLIT_TYPE:?Set STIFFLOW_ATTRIBUTION_SPLIT_TYPE to random or scaffold}"
: "${STIFFLOW_ATTRIBUTION_MODE:?Set STIFFLOW_ATTRIBUTION_MODE to isotropic, harmonic_prior, torsion_only, or stiffness_structured}"
: "${STIFFLOW_ATTRIBUTION_SEED:?Set STIFFLOW_ATTRIBUTION_SEED to 0, 1, or 2}"

source scripts/activate_env.sh

python scripts/run_geom_full_database_attribution_formal_training_single.py \
  --split-type "${STIFFLOW_ATTRIBUTION_SPLIT_TYPE}" \
  --mode "${STIFFLOW_ATTRIBUTION_MODE}" \
  --seed "${STIFFLOW_ATTRIBUTION_SEED}" \
  --device cuda \
  --require-cuda
