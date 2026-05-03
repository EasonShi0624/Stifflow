#!/usr/bin/env python
"""Summarize final optimized full-database all-method outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.final_optimized_all_methods import (  # noqa: E402
    DEFAULT_FINAL_OPTIMIZED_ALL_METHODS_REPORT_PATH,
    DEFAULT_FINAL_OPTIMIZED_ALL_METHODS_SUMMARY_PATH,
    write_final_optimized_all_methods_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="experiments/results")
    parser.add_argument("--summary-path", default=str(DEFAULT_FINAL_OPTIMIZED_ALL_METHODS_SUMMARY_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_FINAL_OPTIMIZED_ALL_METHODS_REPORT_PATH))
    parser.add_argument(
        "--require-results",
        action="store_true",
        help="Fail if any expected final optimized all-method result JSON is missing.",
    )
    args = parser.parse_args()

    outputs = write_final_optimized_all_methods_summary(
        results_root=args.results_root,
        summary_path=args.summary_path,
        report_path=args.report_path,
        require_results=args.require_results,
    )
    print(f"summary: {outputs['summary']}")
    print(f"report: {outputs['report']}")
    print("benchmark_result: false")
    print(
        "note: Reporting-only final optimized all-method summary; no training, sampling, scheduler submission, benchmark jobs, raw-data changes, or environment changes."
    )


if __name__ == "__main__":
    main()
