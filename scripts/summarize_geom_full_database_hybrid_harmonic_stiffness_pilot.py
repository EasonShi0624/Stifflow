#!/usr/bin/env python
"""Summarize the full-database hybrid harmonic + stiffness pilot outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.hybrid_harmonic_stiffness_pilot import (  # noqa: E402
    DEFAULT_HYBRID_PILOT_REPORT_PATH,
    DEFAULT_HYBRID_PILOT_SUMMARY_PATH,
    write_hybrid_harmonic_stiffness_pilot_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        dest="result_paths",
        default=None,
        help="Input result JSON. May be repeated. Defaults to existing random seed0 baseline plus expected hybrid pilot paths.",
    )
    parser.add_argument("--results-root", default="experiments/results")
    parser.add_argument("--summary-path", default=str(DEFAULT_HYBRID_PILOT_SUMMARY_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_HYBRID_PILOT_REPORT_PATH))
    parser.add_argument("--split-type", choices=["random", "scaffold"], default="random")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument(
        "--require-hybrid-results",
        action="store_true",
        help="Fail if any expected hybrid lambda result is missing.",
    )
    args = parser.parse_args()

    paths = write_hybrid_harmonic_stiffness_pilot_summary(
        result_paths=args.result_paths,
        results_root=args.results_root,
        summary_path=args.summary_path,
        report_path=args.report_path,
        split_type=args.split_type,
        seed=args.seed,
        require_hybrid_results=args.require_hybrid_results,
    )
    print(f"summary: {paths['summary']}")
    print(f"report: {paths['report']}")
    print("benchmark_result: false")
    print(
        "note: Reporting-only hybrid harmonic/stiffness pilot summary; no training, sampling, jobs, GPUs, raw-data changes, physical benchmark jobs, or environment changes."
    )


if __name__ == "__main__":
    main()
