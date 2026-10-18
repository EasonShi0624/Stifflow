#!/bin/bash
# Private node-local staging; resource requests are supplied by SHHPC via sbatch.
#SBATCH --job-name=stifflow-formal
set -euo pipefail
: "${STIFFLOW_WORKSPACE:?Set actual SHHPC workspace}"
export STIFFLOW_STAGE_CACHE=1
export STIFFLOW_FORMAL_CONFIG="${STIFFLOW_FORMAL_CONFIG:-configs/qm9-formal-4090d.yaml}"
bash "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/formal.sbatch"
