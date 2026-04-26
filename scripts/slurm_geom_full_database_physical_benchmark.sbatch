#!/bin/bash
#SBATCH --job-name=geom_phys_bench
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --time=8:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_physical_benchmark_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_physical_benchmark_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs experiments/results reports/tables

export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general

source scripts/activate_env.sh

: "${STIFFLOW_PHYSICAL_RESULT_PATH:?Set STIFFLOW_PHYSICAL_RESULT_PATH to one completed result JSON.}"
: "${STIFFLOW_PHYSICAL_OUTPUT_JSON:?Set STIFFLOW_PHYSICAL_OUTPUT_JSON to the diagnostics JSON path.}"

ARGS=(
  --run
  --result-path "${STIFFLOW_PHYSICAL_RESULT_PATH}"
  --output-json "${STIFFLOW_PHYSICAL_OUTPUT_JSON}"
  --device cpu
)

if [[ -n "${STIFFLOW_PHYSICAL_MAX_MOLECULES:-}" ]]; then
  ARGS+=(--max-molecules "${STIFFLOW_PHYSICAL_MAX_MOLECULES}")
elif [[ "${STIFFLOW_PHYSICAL_ALLOW_FULL_DATABASE:-0}" == "1" ]]; then
  ARGS+=(--allow-full-database)
else
  echo "Refusing to run without STIFFLOW_PHYSICAL_MAX_MOLECULES or STIFFLOW_PHYSICAL_ALLOW_FULL_DATABASE=1." >&2
  exit 2
fi

python scripts/evaluate_geom_full_database_physical_benchmark.py "${ARGS[@]}"
