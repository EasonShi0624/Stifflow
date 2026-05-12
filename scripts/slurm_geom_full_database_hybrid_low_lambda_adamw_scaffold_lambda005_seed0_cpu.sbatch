#!/bin/bash
#SBATCH --job-name=geom_ll_scaf_l005
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_hybrid_low_lambda_adamw_scaffold_lambda005_seed0_cpu_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_hybrid_low_lambda_adamw_scaffold_lambda005_seed0_cpu_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs
export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general
source scripts/activate_env.sh

python scripts/run_geom_full_database_formal_training_config_single.py \
  --config configs/train/final_optimized_low_lambda/geom_full_database_scaffold_hybrid_harmonic_stiffness_v1_lambda005_adamw_lr1e3.yaml \
  --split-type scaffold \
  --seed 0 \
  --device cpu \
  --no-require-cuda
