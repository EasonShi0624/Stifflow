#!/bin/bash
#SBATCH --job-name=hlam_rand_l0025_s2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_hybrid_lambda_adamw_random_lambda0025_seed2_cpu_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_hybrid_lambda_adamw_random_lambda0025_seed2_cpu_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs
export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general
source scripts/activate_env.sh

python scripts/run_geom_full_database_formal_training_config_single.py \
  --config configs/train/hybrid_lambda_adamw_multiseed/geom_full_database_random_hybrid_harmonic_stiffness_v1_lambda0025_seed2_adamw_lr1e3.yaml \
  --split-type random \
  --seed 2 \
  --device cpu \
  --no-require-cuda
