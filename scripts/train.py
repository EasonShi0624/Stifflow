#!/usr/bin/env python
"""Validate the tiny smoke-test flow-matching path.

This entry point intentionally does not run neural training yet. With
``--dry-run`` it validates config/data/split/corruption wiring and writes the
manifest/results contract expected by later real runs.
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the smoke path without claiming model training was executed.",
    )
    args = parser.parse_args()

    paths = run_smoke(args.config, dry_run=args.dry_run)
    print(f"manifest: {paths['manifest']}")
    print(f"results: {paths['results']}")
    print(f"log: {paths['log']}")


if __name__ == "__main__":
    main()
