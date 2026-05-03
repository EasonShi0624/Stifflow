#!/usr/bin/env python
"""Summarize fixed lambda-5 hybrid harmonic/stiffness multiseed outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.hybrid_harmonic_stiffness_lambda500_multiseed import (  # noqa: E402
    DEFAULT_MASTER_REPORT_PATH,
    DEFAULT_MASTER_SUMMARY_PATH,
    DEFAULT_RANDOM_REPORT_PATH,
    DEFAULT_RANDOM_SUMMARY_PATH,
    DEFAULT_SCAFFOLD_REPORT_PATH,
    DEFAULT_SCAFFOLD_SUMMARY_PATH,
    write_hybrid_lambda500_multiseed_summaries,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="experiments/results")
    parser.add_argument("--random-summary-path", default=str(DEFAULT_RANDOM_SUMMARY_PATH))
    parser.add_argument("--random-report-path", default=str(DEFAULT_RANDOM_REPORT_PATH))
    parser.add_argument("--scaffold-summary-path", default=str(DEFAULT_SCAFFOLD_SUMMARY_PATH))
    parser.add_argument("--scaffold-report-path", default=str(DEFAULT_SCAFFOLD_REPORT_PATH))
    parser.add_argument("--master-summary-path", default=str(DEFAULT_MASTER_SUMMARY_PATH))
    parser.add_argument("--master-report-path", default=str(DEFAULT_MASTER_REPORT_PATH))
    parser.add_argument(
        "--require-hybrid-results",
        action="store_true",
        help="Fail if any expected fixed lambda-5 hybrid result is missing.",
    )
    args = parser.parse_args()

    paths = write_hybrid_lambda500_multiseed_summaries(
        results_root=args.results_root,
        random_summary_path=args.random_summary_path,
        random_report_path=args.random_report_path,
        scaffold_summary_path=args.scaffold_summary_path,
        scaffold_report_path=args.scaffold_report_path,
        master_summary_path=args.master_summary_path,
        master_report_path=args.master_report_path,
        require_hybrid_results=args.require_hybrid_results,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    print("benchmark_result: false")
    print(
        "note: Reporting-only fixed lambda-5 hybrid harmonic/stiffness multiseed summaries; no training, sampling, jobs, GPUs, raw-data changes, physical benchmark jobs, xTB, or environment changes."
    )


if __name__ == "__main__":
    main()
