#!/bin/bash
#SBATCH --job-name=geom_formal10k_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96GB
#SBATCH --time=48:00:00
#SBATCH --output=experiments/logs/slurm_geom_formal_10000_training_multiseed_%j.out
#SBATCH --error=experiments/logs/slurm_geom_formal_10000_training_multiseed_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs

export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general

source scripts/activate_env.sh

python scripts/run_data_scale_check_10000_formal_training_multiseed.py
