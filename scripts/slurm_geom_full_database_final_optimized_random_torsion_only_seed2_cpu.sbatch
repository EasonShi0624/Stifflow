#!/bin/bash
#SBATCH --job-name=geom_finopt_rand_tors_s2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_final_optimized_random_torsion_only_seed2_cpu_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_final_optimized_random_torsion_only_seed2_cpu_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs
export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general
source scripts/activate_env.sh

python scripts/run_geom_full_database_formal_training_config_single.py \
  --config configs/train/final_optimized/geom_full_database_random_torsion_only_adamw_lr1e3.yaml \
  --split-type random \
  --seed 2 \
  --device cpu \
  --no-require-cuda
