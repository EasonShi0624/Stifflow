#!/bin/bash
#SBATCH --job-name=hyb_hs_l500_s2_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_scaffold_hybrid_harmonic_stiffness_v1_lambda500_seed2_cpu_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_scaffold_hybrid_harmonic_stiffness_v1_lambda500_seed2_cpu_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs

export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general

source scripts/activate_env.sh

python scripts/run_geom_full_database_hybrid_harmonic_stiffness_pilot_single.py \
  --config configs/train/geom_full_database_scaffold_formal_hybrid_harmonic_stiffness_v1_lambda500.yaml \
  --split-type scaffold \
  --seed 2 \
  --device cpu \
  --no-require-cuda
