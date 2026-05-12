#!/bin/bash
#SBATCH --job-name=geom_lambda_cov
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_hybrid_lambda_cov_amr_selection_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_hybrid_lambda_cov_amr_selection_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs
export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general

source scripts/activate_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python scripts/evaluate_geom_full_database_hybrid_lambda_cov_amr_selection.py \
  --plan-path experiments/results/geom_full_database_hybrid_lambda_cov_amr_selection_plan.json \
  --summary-path experiments/results/geom_full_database_hybrid_lambda_cov_amr_selection_summary.json \
  --report-path reports/tables/geom_full_database_hybrid_lambda_cov_amr_selection_summary.md \
  --per-molecule-jsonl-path experiments/results/geom_full_database_hybrid_lambda_cov_amr_selection_per_molecule.jsonl \
  --device cpu \
  --progress-every 20
