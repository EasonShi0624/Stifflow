#!/bin/bash
#SBATCH --job-name=stifflow-locality
# Supply actual SHHPC account/partition/GPU/CPU/memory/time at submission.
set -euo pipefail
method="${1:?Use a_delayed or a_delayed_rot_null}"
case "$method" in
  a_delayed|a_delayed_rot_null) ;;
  *) echo "Only the two registered locality runs are supported" >&2; exit 2 ;;
esac
: "${STIFFLOW_WORKSPACE:?Set the actual SHHPC workspace}"
: "${STIFFLOW_TRAIN_LABELS:?Set canonical train manifest}"
: "${STIFFLOW_VAL_LABELS:?Set canonical validation manifest}"
: "${STIFFLOW_LOCALITY_REPORT:?Set the frozen train locality report.json}"
: "${STIFFLOW_RUN_ROOT:?Set an external persistent campaign directory}"
source "$STIFFLOW_WORKSPACE/stifflow-v1/scripts/shhpc_env.sh"
python - <<'PY'
import os
from pathlib import Path
from stifflow.locality import load_locality_report
start, sha = load_locality_report(
    Path(os.environ['STIFFLOW_LOCALITY_REPORT']),
    Path(os.environ['STIFFLOW_TRAIN_LABELS']), {'sigma0': 1.0, 'rho': 1.0})
print({'frozen_stiffness_start': start, 'locality_report_sha256': sha})
PY
run="$STIFFLOW_RUN_ROOT/$method"
python - "$run" <<'PY'
import sys
from stifflow.smoke import external_directory
external_directory(sys.argv[1])
PY
mkdir -p "$STIFFLOW_RUN_ROOT"
mkdir "$run" # Own a fresh method directory; never overwrite/restart a campaign.
common=(--config configs/qm9-research-25k.yaml --profile research --seed 42
  --method "$method" --train-labels "$STIFFLOW_TRAIN_LABELS"
  --val-labels "$STIFFLOW_VAL_LABELS" --locality-report "$STIFFLOW_LOCALITY_REPORT")
if [[ -n "${STIFFLOW_PACKED_CACHE_ROOT:-}" ]]; then
  common+=(--packed-cache-root "$STIFFLOW_PACKED_CACHE_ROOT")
fi
python -m stifflow.training "${common[@]}" --stop-after 40 \
  --checkpoint-dir "$run/checkpoints" --output "$run/checks/continuous"
python -m stifflow.training "${common[@]}" --stop-after 20 \
  --output "$run/checks/first20"
python -m stifflow.training "${common[@]}" --stop-after 40 \
  --resume "$run/checks/first20/final.ckpt" --output "$run/checks/resumed40"
python -m stifflow.verify_resume "$run/checks/continuous/final.ckpt" \
  "$run/checks/resumed40/final.ckpt" --output "$run/checks/resume.json"
# Continue the accepted trajectory; keep the original 250k-step LR schedule.
python -m stifflow.training "${common[@]}" --resume "$run/checks/continuous/final.ckpt" \
  --checkpoint-dir "$run/checkpoints" --output "$run/train25k"
