#!/bin/bash
#SBATCH --job-name=geom_full_iso_s1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=4:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_formal_training_isotropic_seed1_gpu_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_formal_training_isotropic_seed1_gpu_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs

export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general

source scripts/activate_env.sh

python scripts/run_geom_full_database_formal_training_single.py --mode isotropic --seed 1 --device cuda --require-cuda
