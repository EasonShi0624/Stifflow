#!/bin/bash
#SBATCH --job-name=geom_final_opt_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_final_optimized_harmonic_vs_hybrid_cpu_%x_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_final_optimized_harmonic_vs_hybrid_cpu_%x_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs

: "${STIFFLOW_FINAL_CONFIG:?Set STIFFLOW_FINAL_CONFIG to one final optimized config path.}"
: "${STIFFLOW_FINAL_SPLIT_TYPE:?Set STIFFLOW_FINAL_SPLIT_TYPE to random or scaffold.}"
: "${STIFFLOW_FINAL_SEED:?Set STIFFLOW_FINAL_SEED to 0, 1, or 2.}"

export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general

source scripts/activate_env.sh

python scripts/run_geom_full_database_formal_training_config_single.py \
  --config "${STIFFLOW_FINAL_CONFIG}" \
  --split-type "${STIFFLOW_FINAL_SPLIT_TYPE}" \
  --seed "${STIFFLOW_FINAL_SEED}" \
  --device cpu \
  --no-require-cuda
