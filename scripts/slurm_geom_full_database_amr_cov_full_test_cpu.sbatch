#!/bin/bash
#SBATCH --job-name=geom_amr_cov_full
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_amr_cov_full_test_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_amr_cov_full_test_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs

source scripts/activate_env.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python scripts/evaluate_geom_full_database_amr_cov.py \
  --max-molecules-per-split 0 \
  --allow-full-test-split \
  --chunk-size 128 \
  --progress-every 500 \
  --generated-conformers-per-mol 16 \
  --max-reference-conformers-per-mol 40 \
  --delta 0.75 \
  --device cpu

python scripts/make_full_database_presentation_figures.py
