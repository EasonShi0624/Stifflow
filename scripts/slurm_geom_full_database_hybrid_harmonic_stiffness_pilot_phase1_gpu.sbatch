#!/bin/bash
#SBATCH --job-name=geom_hybrid_p1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=experiments/logs/slurm_geom_full_database_hybrid_pilot_phase1_%A_%a.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_hybrid_pilot_phase1_%A_%a.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs

export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general

CONFIGS=(
  "configs/train/geom_full_database_formal_hybrid_harmonic_stiffness_v1_lambda010.yaml"
  "configs/train/geom_full_database_formal_hybrid_harmonic_stiffness_v1_lambda025.yaml"
  "configs/train/geom_full_database_formal_hybrid_harmonic_stiffness_v1_lambda050.yaml"
  "configs/train/geom_full_database_formal_harmonic_stiffness_residual_v1_lambda010.yaml"
  "configs/train/geom_full_database_formal_harmonic_stiffness_residual_v1_lambda025.yaml"
  "configs/train/geom_full_database_formal_harmonic_stiffness_residual_v1_lambda050.yaml"
)

CONFIG_PATH="${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"

source scripts/activate_env.sh

python scripts/run_geom_full_database_hybrid_harmonic_stiffness_pilot_single.py \
  --config "${CONFIG_PATH}" \
  --split-type random \
  --seed 0 \
  --device cuda \
  --require-cuda
