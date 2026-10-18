#!/bin/bash
#SBATCH --job-name=stifflow-research
# Supply SHHPC account/partition/resources explicitly when submitting.
# CPU: research_clock, diagnostics, compare, evaluate. GPU: training, sampling, generate.
set -euo pipefail
: "${STIFFLOW_WORKSPACE:?Set the actual SHHPC workspace}"
source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"
task="${1:?Pass a research task name}"
shift
case "$task" in
  train) module=stifflow.training ;;
  clock) module=stifflow.research_clock ;;
  paths) module=stifflow.research_path_statistics ;;
  diagnostics) module=stifflow_benchmark.research_diagnostics ;;
  sampling) module=stifflow_benchmark.research_sampling ;;
  generate) module=stifflow_benchmark.generate ;;
  evaluate) module=stifflow_benchmark.evaluate ;;
  compare) module=stifflow_benchmark.research_compare ;;
  *) echo "Unsupported research task: $task" >&2; exit 2 ;;
esac
python -m "$module" "$@"
