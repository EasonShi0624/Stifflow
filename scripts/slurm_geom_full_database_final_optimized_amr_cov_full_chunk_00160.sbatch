#!/bin/bash
#SBATCH --job-name=geom_amr_00160
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_final_optimized_amr_cov_full_chunk_00160_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_final_optimized_amr_cov_full_chunk_00160_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs experiments/results/geom_full_database_final_optimized_amr_cov_full_chunks
export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general

source scripts/activate_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Standalone single-job chunk. Array directives are intentionally absent.
python scripts/evaluate_geom_full_database_final_optimized_amr_cov_full_chunk.py \
  --plan-path experiments/results/geom_full_database_final_optimized_amr_cov_full_plan.json \
  --task-id 160 \
  --device cpu \
  --progress-every 128

# Wrapper: scripts/slurm_geom_full_database_final_optimized_amr_cov_full_chunk_00160.sbatch
# Split/partition/method/seed/chunk: random val isotropic seed1 chunk10
