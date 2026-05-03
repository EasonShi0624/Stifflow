#!/usr/bin/env python
"""Run bounded AMR/COV evaluation from existing GEOM full-database checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.amr_cov import main  # noqa: E402


if __name__ == "__main__":
    main()
