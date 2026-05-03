#!/usr/bin/env python
"""Summarize completed optimizer/LR and lambda-refinement pilot outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.formal_optimization_phase_summary import (  # noqa: E402
    DEFAULT_DECISION_REPORT_PATH,
    DEFAULT_DECISION_SUMMARY_PATH,
    DEFAULT_DIAGNOSTICS_PATH,
    DEFAULT_LAMBDA_REPORT_PATH,
    DEFAULT_LAMBDA_SUMMARY_PATH,
    DEFAULT_OPTIMIZER_REPORT_PATH,
    DEFAULT_OPTIMIZER_SUMMARY_PATH,
    write_optimization_phase_summaries,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="experiments/results")
    parser.add_argument("--diagnostics-path", default=str(DEFAULT_DIAGNOSTICS_PATH))
    parser.add_argument("--optimizer-summary-path", default=str(DEFAULT_OPTIMIZER_SUMMARY_PATH))
    parser.add_argument("--optimizer-report-path", default=str(DEFAULT_OPTIMIZER_REPORT_PATH))
    parser.add_argument("--lambda-summary-path", default=str(DEFAULT_LAMBDA_SUMMARY_PATH))
    parser.add_argument("--lambda-report-path", default=str(DEFAULT_LAMBDA_REPORT_PATH))
    parser.add_argument("--decision-summary-path", default=str(DEFAULT_DECISION_SUMMARY_PATH))
    parser.add_argument("--decision-report-path", default=str(DEFAULT_DECISION_REPORT_PATH))
    args = parser.parse_args()

    outputs = write_optimization_phase_summaries(
        results_root=args.results_root,
        diagnostics_path=args.diagnostics_path,
        optimizer_summary_path=args.optimizer_summary_path,
        optimizer_report_path=args.optimizer_report_path,
        lambda_summary_path=args.lambda_summary_path,
        lambda_report_path=args.lambda_report_path,
        decision_summary_path=args.decision_summary_path,
        decision_report_path=args.decision_report_path,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    print("benchmark_result: false")
    print(
        "note: Reporting-only optimization-prep summary; no training, sampling, scheduler submission, benchmark jobs, raw-data changes, or environment changes."
    )


if __name__ == "__main__":
    main()
