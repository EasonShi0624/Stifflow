#!/usr/bin/env bash
# Minimal helper for activating the expected conda environment.
# This script is intentionally conservative because shell initialization
# differs across HPC systems.

set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Load your shell/conda setup first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stifflow

echo "Activated conda environment: stifflow"
python -V
which python
