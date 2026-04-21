#!/bin/bash
#SBATCH --job-name=geom_dscale10k_ckpt_ms
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=experiments/logs/slurm_geom_data_scale_check_10000_multiseed_%j.out
#SBATCH --error=experiments/logs/slurm_geom_data_scale_check_10000_multiseed_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs

source scripts/activate_env.sh

python scripts/run_data_scale_check_10000_checkpoint_sampling_multiseed.py
