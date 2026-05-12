#!/bin/bash
#SBATCH --job-name=covamr_01044
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_final_cov_amr_complete_valtest_chunk_01044_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_final_cov_amr_complete_valtest_chunk_01044_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs experiments/results/geom_full_database_final_cov_amr_complete_valtest_chunks
export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general

source scripts/activate_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python scripts/evaluate_geom_full_database_final_cov_amr_complete_valtest_chunk.py \
  --plan-path experiments/results/geom_full_database_final_cov_amr_complete_valtest_plan.json \
  --task-id 1044 \
  --device cpu \
  --progress-every 128

# Wrapper: scripts/slurm_geom_full_database_final_cov_amr_complete_valtest_chunk_01044.sbatch
# Scope: scaffold val torsion_only seed0 chunk9
