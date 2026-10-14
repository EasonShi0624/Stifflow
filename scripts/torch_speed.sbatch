#!/bin/bash
#SBATCH --job-name=stifflow-speed
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@180
# Supply account, GPU constraint, one GPU, CPU and memory; Slurm selects the partition.
set -euo pipefail
export STIFFLOW_SPEED_START_EPOCH="$(date +%s)"
: "${STIFFLOW_WORKSPACE:?Set Torch workspace}"
: "${STIFFLOW_SPEED_ROOT:?Set external speed-run root}"
: "${STIFFLOW_SPEED_CACHE:?Set external small-cache root}"
: "${STIFFLOW_EXPECTED_COMMIT:?Pin the benchmark code SHA}"
source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"
[[ "$(git rev-parse HEAD)" == "$STIFFLOW_EXPECTED_COMMIT" ]] || { echo "Wrong code SHA" >&2; exit 2; }
git diff --exit-code --quiet
git diff --cached --exit-code --quiet
export MPLCONFIGDIR="$STIFFLOW_WORKSPACE/cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
exec python -u -m stifflow.speed_benchmark \
  --split-manifest "${STIFFLOW_SPEED_SPLIT:-$STIFFLOW_WORKSPACE/cache/qm9-split-manifest.json}" \
  --cache "$STIFFLOW_SPEED_CACHE/$SLURM_JOB_ID" \
  --output "$STIFFLOW_SPEED_ROOT/$SLURM_JOB_ID"
