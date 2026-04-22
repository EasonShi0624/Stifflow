#!/usr/bin/env python
"""Summarize convergence diagnostics for existing bounded 10000-molecule steps200 pilots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.data_scale_convergence_diagnostics import (
    DEFAULT_REPORT_PATH,
    DEFAULT_RESULT_PATHS,
    DEFAULT_SUMMARY_PATH,
    write_data_scale_check_convergence_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        dest="results",
        help="Existing steps200 result JSON. Pass six times to override the default bounded 10000-molecule paths.",
    )
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH), help="Output JSON summary path.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Output markdown report path.")
    args = parser.parse_args()

    result_paths = [Path(path) for path in args.results] if args.results else DEFAULT_RESULT_PATHS
    paths = write_data_scale_check_convergence_diagnostics(
        result_paths=result_paths,
        summary_path=args.summary,
        report_path=args.report,
    )
    print(f"summary: {paths['summary']}")
    print(f"report: {paths['report']}")


if __name__ == "__main__":
    main()
