#!/bin/bash
#SBATCH --job-name=geom_attr_s_t_s2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=4:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_attribution_scaffold_torsion_only_seed2_gpu_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_attribution_scaffold_torsion_only_seed2_gpu_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs

export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general

source scripts/activate_env.sh

python scripts/run_geom_full_database_attribution_formal_training_single.py \
  --split-type scaffold \
  --mode torsion_only \
  --seed 2 \
  --device cuda \
  --require-cuda
