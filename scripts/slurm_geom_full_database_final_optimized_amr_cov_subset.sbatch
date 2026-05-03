#!/bin/bash
#SBATCH --job-name=geom_finopt_amr_cov
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_final_optimized_amr_cov_subset_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_final_optimized_amr_cov_subset_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs
export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general

source scripts/activate_env.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python scripts/evaluate_geom_full_database_final_optimized_amr_cov_subset.py \
  --plan-path experiments/results/geom_full_database_final_optimized_amr_cov_subset_plan.json \
  --summary-path experiments/results/geom_full_database_final_optimized_amr_cov_subset_summary.json \
  --report-path reports/tables/geom_full_database_final_optimized_amr_cov_subset_summary.md \
  --per-molecule-jsonl-path experiments/results/geom_full_database_final_optimized_amr_cov_subset_per_molecule.jsonl \
  --device cpu \
  --delta 0.75 \
  --progress-every 25
