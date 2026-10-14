#!/bin/bash
# Supply GPU/CPU/memory and walltime; batch optimization allows one GPU for 55m.
#SBATCH --job-name=stifflow-runtime
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@90
set -euo pipefail
export STIFFLOW_SPEED_START_EPOCH="$(date +%s)"
: "${STIFFLOW_WORKSPACE:?Set workspace}"
: "${STIFFLOW_EXPECTED_COMMIT:?Pin code}"
: "${STIFFLOW_RUNTIME_DEVICES:?Set 1 or 2}"
: "${STIFFLOW_RUNTIME_OUTPUT:?Set fresh output}"
: "${STIFFLOW_RUNTIME_MANIFEST:?Set existing small cache}"
: "${STIFFLOW_RUNTIME_MAXIMUM:?Set audited 29-atom cache}"
source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"
[[ "$(git rev-parse HEAD)" == "$STIFFLOW_EXPECTED_COMMIT" ]]
git diff --exit-code --quiet
git diff --cached --exit-code --quiet
export TMPDIR="${SLURM_TMPDIR:-/tmp}/stifflow-runtime-$SLURM_JOB_ID"
export MPLCONFIGDIR="$STIFFLOW_WORKSPACE/cache/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
args=(--devices "$STIFFLOW_RUNTIME_DEVICES" --manifest "$STIFFLOW_RUNTIME_MANIFEST"
      --stress-manifest "$STIFFLOW_RUNTIME_MAXIMUM" --output "$STIFFLOW_RUNTIME_OUTPUT")
if [[ -n "${STIFFLOW_RUNTIME_CONFIG:-}" ]]; then
  args+=(--config "$STIFFLOW_RUNTIME_CONFIG")
fi
if [[ "${STIFFLOW_RUNTIME_OPTIMIZE_BATCH:-0}" == 1 ]]; then
  [[ "$STIFFLOW_RUNTIME_DEVICES" == 1 ]] || { echo "Batch optimization requires one GPU" >&2; exit 2; }
  : "${STIFFLOW_RUNTIME_SECONDS:?Set the approved optimization deadline}"
  args+=(--optimize-batch)
fi
if [[ "$STIFFLOW_RUNTIME_DEVICES" == 2 ]]; then
  : "${STIFFLOW_RUNTIME_SINGLE_REPORT:?Set matching single-card report}"
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT="$((20000 + SLURM_JOB_ID % 30000))"
  args+=(--max-seconds "${STIFFLOW_RUNTIME_SECONDS:-990}" --single-report "$STIFFLOW_RUNTIME_SINGLE_REPORT")
  exec srun --ntasks=2 --ntasks-per-node=2 --kill-on-bad-exit=1 python -u -m stifflow.runtime_benchmark "${args[@]}"
elif [[ "$STIFFLOW_RUNTIME_DEVICES" == 1 ]]; then
  args+=(--max-seconds "${STIFFLOW_RUNTIME_SECONDS:-630}")
  exec python -u -m stifflow.runtime_benchmark "${args[@]}"
else
  echo "Only 1/2 GPUs are approved" >&2
  exit 2
fi
