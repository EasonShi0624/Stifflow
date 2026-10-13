#!/bin/bash
#SBATCH --job-name=stifflow-formal-task
# CPU allocation: metadata/merge/references/evaluate/summarize. GPU: generate.
set -euo pipefail
: "${STIFFLOW_WORKSPACE:?Set actual SHHPC workspace}"
source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"
module="${1:?Pass freeze_data, formal_cache, or formal}"
shift
case "$module" in freeze_data|formal_cache|formal) ;; *) echo "Unsupported task module" >&2; exit 2;; esac
python -m "stifflow.$module" "$@"
