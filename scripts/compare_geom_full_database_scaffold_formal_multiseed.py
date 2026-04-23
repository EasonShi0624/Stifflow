#!/usr/bin/env python
"""Aggregate full-database GEOM-Drugs scaffold-split formal result JSONs across seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.formal_full_database_compare import (
    DEFAULT_SCAFFOLD_MULTI_SEED_REPORT_PATH,
    DEFAULT_SCAFFOLD_MULTI_SEED_RESULT_PATHS,
    DEFAULT_SCAFFOLD_MULTI_SEED_SUMMARY_PATH,
    write_formal_full_database_scaffold_multiseed_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        dest="results",
        help="Input result JSON. Pass six times to override the default scaffold seed0/1/2 isotropic/stiffness paths.",
    )
    parser.add_argument(
        "--summary",
        default=str(DEFAULT_SCAFFOLD_MULTI_SEED_SUMMARY_PATH),
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_SCAFFOLD_MULTI_SEED_REPORT_PATH),
        help="Output markdown report path.",
    )
    args = parser.parse_args()

    result_paths = [Path(path) for path in args.results] if args.results else DEFAULT_SCAFFOLD_MULTI_SEED_RESULT_PATHS
    paths = write_formal_full_database_scaffold_multiseed_summary(
        result_paths=result_paths,
        summary_path=args.summary,
        report_path=args.report,
        expected_seeds=(0, 1, 2),
    )
    print(f"summary: {paths['summary']}")
    print(f"report: {paths['report']}")


if __name__ == "__main__":
    main()
