#!/bin/bash
#SBATCH --job-name=hyb_hs_l025_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_hybrid_harmonic_stiffness_v1_lambda025_cpu_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_hybrid_harmonic_stiffness_v1_lambda025_cpu_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs

export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general

source scripts/activate_env.sh

python scripts/run_geom_full_database_hybrid_harmonic_stiffness_pilot_single.py \
  --config configs/train/geom_full_database_formal_hybrid_harmonic_stiffness_v1_lambda025.yaml \
  --split-type random \
  --seed 0 \
  --device cpu \
  --no-require-cuda
