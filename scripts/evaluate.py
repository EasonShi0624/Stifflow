#!/usr/bin/env python
"""Re-run the smoke evaluation path from a config.

This is deliberately equivalent to ``scripts/train.py --dry-run`` for now:
there are no model-generated samples yet, so evaluation validates source-vs-
target smoke metrics and writes machine-readable outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.smoke import run_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a smoke-test YAML config.")
    args = parser.parse_args()
    paths = run_smoke(args.config, dry_run=True)
    print(f"results: {paths['results']}")


if __name__ == "__main__":
    main()
