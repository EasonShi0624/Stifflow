#!/bin/bash
#SBATCH --job-name=geom_xtb_plan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=0:30:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_xtb_subset_plan_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_xtb_subset_plan_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs experiments/results reports/tables

export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general

source scripts/activate_env.sh

python scripts/plan_geom_full_database_xtb_subset.py \
  --samples-per-stratum "${STIFFLOW_XTB_SAMPLES_PER_STRATUM:-2}"
