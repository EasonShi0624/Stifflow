#!/usr/bin/env python
"""Write optimization diagnostics from completed full-database formal outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.formal_optimization_diagnostics import (  # noqa: E402
    DEFAULT_OPTIMIZATION_DIAGNOSTICS_PATH,
    DEFAULT_OPTIMIZATION_DIAGNOSTICS_REPORT_PATH,
    write_optimization_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="experiments/results")
    parser.add_argument("--summary-path", default=str(DEFAULT_OPTIMIZATION_DIAGNOSTICS_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_OPTIMIZATION_DIAGNOSTICS_REPORT_PATH))
    parser.add_argument(
        "--require-results",
        action="store_true",
        help="Fail if any expected random/scaffold full-database method/seed result is missing.",
    )
    args = parser.parse_args()

    outputs = write_optimization_diagnostics(
        results_root=args.results_root,
        summary_path=args.summary_path,
        report_path=args.report_path,
        require_results=args.require_results,
    )
    print(f"summary: {outputs['summary']}")
    print(f"report: {outputs['report']}")
    print("benchmark_result: false")
    print(
        "note: Reporting-only optimization diagnostics; no training, sampling, scheduler submission, benchmark jobs, physical benchmark jobs, xTB, raw-data changes, or environment changes."
    )


if __name__ == "__main__":
    main()
