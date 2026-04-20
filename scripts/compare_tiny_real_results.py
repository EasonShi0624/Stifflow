#!/usr/bin/env python
"""Write a tiny real-data dry-run comparison summary from existing results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.tiny_real_compare import DEFAULT_RESULT_PATHS, write_tiny_real_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        nargs="+",
        default=[str(path) for path in DEFAULT_RESULT_PATHS],
        help="Tiny real dry-run result JSON files to summarize.",
    )
    parser.add_argument(
        "--summary",
        default="experiments/results/geom_tiny_real_dry_run_summary.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--report",
        default="reports/tables/geom_tiny_real_dry_run_summary.md",
        help="Output markdown report path.",
    )
    args = parser.parse_args()

    paths = write_tiny_real_comparison(
        result_paths=[Path(path) for path in args.results],
        summary_path=args.summary,
        report_path=args.report,
    )
    print(f"summary: {paths['summary']}")
    print(f"report: {paths['report']}")


if __name__ == "__main__":
    main()
