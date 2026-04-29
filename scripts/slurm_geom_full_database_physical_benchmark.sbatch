#!/bin/bash
#SBATCH --job-name=geom_phys_bench
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_physical_benchmark_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_physical_benchmark_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs experiments/results reports/tables

export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general

source scripts/activate_env.sh

ARGS=(
  --run-matrix
  --output-dir "${STIFFLOW_PHYSICAL_OUTPUT_DIR:-experiments/results/geom_full_database_physical_benchmark}"
  --summary-json "${STIFFLOW_PHYSICAL_SUMMARY_JSON:-experiments/results/geom_full_database_physical_benchmark_summary.json}"
  --summary-md "${STIFFLOW_PHYSICAL_SUMMARY_MD:-reports/tables/geom_full_database_physical_benchmark_summary.md}"
  --device cpu
)

if [[ -n "${STIFFLOW_PHYSICAL_MAX_MOLECULES:-}" ]]; then
  ARGS+=(--max-molecules-per-result "${STIFFLOW_PHYSICAL_MAX_MOLECULES}")
else
  ARGS+=(--allow-full-database)
fi

python scripts/evaluate_geom_full_database_physical_benchmark.py "${ARGS[@]}"
