#!/bin/bash
#SBATCH --job-name=geom_full_scaf_split
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=2:00:00
#SBATCH --output=experiments/logs/slurm_geom_full_database_scaffold_split_%j.out
#SBATCH --error=experiments/logs/slurm_geom_full_database_scaffold_split_%j.err
#SBATCH --account=torch_pr_872_general

set -euo pipefail

cd /scratch/ys6132/project/0_stiff_flow_matching
mkdir -p experiments/logs

export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general

source scripts/activate_env.sh

python scripts/prepare_geom_full_database_scaffold_split.py
