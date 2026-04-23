#!/usr/bin/env python
"""Summarize completed full-database A/B/D/E attribution formal outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.formal_attribution import (  # noqa: E402
    DEFAULT_ATTRIBUTION_REPORT_PATH,
    DEFAULT_ATTRIBUTION_SUMMARY_PATH,
    write_formal_attribution_phase_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        dest="result_paths",
        default=None,
        help="Input result JSON. May be repeated. Defaults to the prepared random/scaffold A/B/D/E seed 0/1/2 matrix.",
    )
    parser.add_argument("--summary-path", default=str(DEFAULT_ATTRIBUTION_SUMMARY_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_ATTRIBUTION_REPORT_PATH))
    parser.add_argument(
        "--split-type",
        action="append",
        dest="split_types",
        choices=["random", "scaffold"],
        default=None,
        help="Expected split type. May be repeated. Defaults to random and scaffold.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Expected seed. May be repeated. Defaults to 0, 1, and 2.",
    )
    args = parser.parse_args()

    paths = write_formal_attribution_phase_summary(
        result_paths=args.result_paths,
        summary_path=args.summary_path,
        report_path=args.report_path,
        expected_split_types=tuple(args.split_types or ["random", "scaffold"]),
        expected_seeds=tuple(args.seeds or [0, 1, 2]),
    )
    print(f"summary: {paths['summary']}")
    print(f"report: {paths['report']}")
    print("benchmark_result: false")
    print("note: Reporting-only full-database attribution-phase summary; no training, sampling, jobs, GPUs, raw-data changes, or environment changes.")


if __name__ == "__main__":
    main()
