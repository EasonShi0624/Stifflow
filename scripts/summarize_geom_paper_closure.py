#!/usr/bin/env python
"""Generate reporting-only scientific closure summaries from existing full-database outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.paper_closure import (  # noqa: E402
    DEFAULT_CLAIM_REPORT_PATH,
    DEFAULT_CLAIM_SUMMARY_PATH,
    DEFAULT_CORE_RESULTS_REPORT_PATH,
    DEFAULT_CORE_RESULTS_SUMMARY_PATH,
    DEFAULT_PHYSICAL_REPORT_PATH,
    DEFAULT_PHYSICAL_SUMMARY_PATH,
    DEFAULT_ROTATABLE_REPORT_PATH,
    DEFAULT_ROTATABLE_SUMMARY_PATH,
    write_paper_closure_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-summary-path", default=str(DEFAULT_CORE_RESULTS_SUMMARY_PATH))
    parser.add_argument("--core-report-path", default=str(DEFAULT_CORE_RESULTS_REPORT_PATH))
    parser.add_argument("--physical-summary-path", default=str(DEFAULT_PHYSICAL_SUMMARY_PATH))
    parser.add_argument("--physical-report-path", default=str(DEFAULT_PHYSICAL_REPORT_PATH))
    parser.add_argument("--rotatable-summary-path", default=str(DEFAULT_ROTATABLE_SUMMARY_PATH))
    parser.add_argument("--rotatable-report-path", default=str(DEFAULT_ROTATABLE_REPORT_PATH))
    parser.add_argument("--claim-summary-path", default=str(DEFAULT_CLAIM_SUMMARY_PATH))
    parser.add_argument("--claim-report-path", default=str(DEFAULT_CLAIM_REPORT_PATH))
    args = parser.parse_args()

    outputs = write_paper_closure_outputs(
        core_summary_path=args.core_summary_path,
        core_report_path=args.core_report_path,
        physical_summary_path=args.physical_summary_path,
        physical_report_path=args.physical_report_path,
        rotatable_summary_path=args.rotatable_summary_path,
        rotatable_report_path=args.rotatable_report_path,
        claim_summary_path=args.claim_summary_path,
        claim_report_path=args.claim_report_path,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
    print("benchmark_result: false")
    print("note: Reporting-only closure summaries from existing outputs; no training, sampling, jobs, raw-data changes, or environment changes.")


if __name__ == "__main__":
    main()
