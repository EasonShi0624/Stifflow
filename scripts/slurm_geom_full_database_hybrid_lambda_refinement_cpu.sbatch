#!/bin/bash
#SBATCH --job-name=geom_hyb_lambda_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_hybrid_lambda_refinement_cpu_%x_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_hybrid_lambda_refinement_cpu_%x_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs

: "${STIFFLOW_LAMBDA_CONFIG:?Set STIFFLOW_LAMBDA_CONFIG to one lambda-refinement config path.}"

export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general

source scripts/activate_env.sh

python scripts/run_geom_full_database_formal_training_config_single.py \
  --config "${STIFFLOW_LAMBDA_CONFIG}" \
  --split-type random \
  --seed 0 \
  --device cpu \
  --no-require-cuda
