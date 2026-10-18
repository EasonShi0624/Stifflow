#!/bin/bash
# Stage D: 64-molecule ensembles chirality/mirror audit for the four candidates.
set -euo pipefail
export STIFFLOW_WORKSPACE="${STIFFLOW_WORKSPACE:-/gpfsnyu/scratch/ys6132/stifflow}"
export ETFLOW_ENV="${ETFLOW_ENV:-/gpfsnyu/home/ys6132/anaconda3/envs/etflow}"
source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"
export STIFFLOW_RESEARCH_OUTPUT="$STIFFLOW_WORKSPACE/runs/research-20261018"
mkdir -p "$STIFFLOW_RESEARCH_OUTPUT/compare"

for method in a_rot_null a_trace_null linear aligned_linear; do
  out="$STIFFLOW_RESEARCH_OUTPUT/compare/ensembles-${method}-25k.json"
  [[ -e "$out" ]] && { echo "existing: $out"; continue; }
  python -m stifflow_benchmark.research_diagnostics ensembles \
    "$STIFFLOW_RESEARCH_OUTPUT/generated/${method}-25k/generated/manifest.json" \
    --limit 64 \
    --data-manifest "$STIFFLOW_WORKSPACE/cache/qm9-v3-freeze-strict/manifest.json" \
    --output "$out"
done
