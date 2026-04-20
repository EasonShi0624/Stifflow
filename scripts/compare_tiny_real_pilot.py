#!/usr/bin/env python
"""Write a bounded tiny real-data pilot comparison from existing results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.tiny_real_pilot_compare import (
    DEFAULT_RESULT_PATHS,
    MICRO_PILOT_RESULT_PATHS,
    write_tiny_real_pilot_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        nargs="+",
        default=None,
        help="Bounded tiny real pilot result JSON files to summarize.",
    )
    parser.add_argument(
        "--micro-pilot",
        action="store_true",
        help="Summarize the bounded six-molecule micro-pilot outputs.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Output markdown report path.",
    )
    args = parser.parse_args()
    comparison_scope = "micro_pilot" if args.micro_pilot else "tiny_real_pilot"
    default_results = MICRO_PILOT_RESULT_PATHS if args.micro_pilot else DEFAULT_RESULT_PATHS
    result_paths = [Path(path) for path in (args.results or [str(path) for path in default_results])]

    paths = write_tiny_real_pilot_comparison(
        result_paths=result_paths,
        summary_path=args.summary,
        report_path=args.report,
        comparison_scope=comparison_scope,
    )
    print(f"summary: {paths['summary']}")
    print(f"report: {paths['report']}")


if __name__ == "__main__":
    main()
