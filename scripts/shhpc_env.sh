#!/bin/bash
# Source only inside an allocation; use the existing environment and worktrees.
: "${SLURM_JOB_ID:?Run inside an SHHPC allocation}"
: "${STIFFLOW_WORKSPACE:?Set the actual workspace path}"
: "${ETFLOW_ENV:?Set the existing etflow environment path}"
set +u
source "$(dirname "$(dirname "$ETFLOW_ENV")")/etc/profile.d/conda.sh"
conda activate "$ETFLOW_ENV"
set -u
export PYTHONPATH="$STIFFLOW_WORKSPACE/stifflow-v1:$STIFFLOW_WORKSPACE/stifflow-v1-alternative-A:$STIFFLOW_WORKSPACE/stifflow-v1-benchmark"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="$STIFFLOW_WORKSPACE/cache/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
cd "$STIFFLOW_WORKSPACE/stifflow-v1"
python - <<'PY'
import importlib.util, os
from pathlib import Path
root = Path(os.environ['STIFFLOW_WORKSPACE']).resolve()
for package, directory in [('stifflow','stifflow-v1'), ('stifflow_alt_a','stifflow-v1-alternative-A'),
                           ('stifflow_benchmark','stifflow-v1-benchmark'), ('etflow','ETFlow-reference')]:
    spec = importlib.util.find_spec(package)
    if spec is None or not Path(spec.origin).resolve().is_relative_to(root / directory):
        raise SystemExit(f'Wrong import path: {package}: {None if spec is None else spec.origin}')
PY
