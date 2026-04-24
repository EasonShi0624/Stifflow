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
    DEFAULT_ATTRIBUTION_RANDOM_REPORT_PATH,
    DEFAULT_ATTRIBUTION_RANDOM_SUMMARY_PATH,
    DEFAULT_ATTRIBUTION_REPORT_PATH,
    DEFAULT_ATTRIBUTION_SCAFFOLD_REPORT_PATH,
    DEFAULT_ATTRIBUTION_SCAFFOLD_SUMMARY_PATH,
    DEFAULT_ATTRIBUTION_SUMMARY_PATH,
    write_formal_attribution_phase_outputs,
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
    parser.add_argument(
        "--summary-path",
        dest="master_summary_path",
        default=str(DEFAULT_ATTRIBUTION_SUMMARY_PATH),
        help="Cross-split master summary JSON output path.",
    )
    parser.add_argument(
        "--report-path",
        dest="master_report_path",
        default=str(DEFAULT_ATTRIBUTION_REPORT_PATH),
        help="Cross-split master markdown report output path.",
    )
    parser.add_argument("--random-summary-path", default=str(DEFAULT_ATTRIBUTION_RANDOM_SUMMARY_PATH))
    parser.add_argument("--random-report-path", default=str(DEFAULT_ATTRIBUTION_RANDOM_REPORT_PATH))
    parser.add_argument("--scaffold-summary-path", default=str(DEFAULT_ATTRIBUTION_SCAFFOLD_SUMMARY_PATH))
    parser.add_argument("--scaffold-report-path", default=str(DEFAULT_ATTRIBUTION_SCAFFOLD_REPORT_PATH))
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

    paths = write_formal_attribution_phase_outputs(
        result_paths=args.result_paths,
        master_summary_path=args.master_summary_path,
        master_report_path=args.master_report_path,
        random_summary_path=args.random_summary_path,
        random_report_path=args.random_report_path,
        scaffold_summary_path=args.scaffold_summary_path,
        scaffold_report_path=args.scaffold_report_path,
        expected_split_types=tuple(args.split_types or ["random", "scaffold"]),
        expected_seeds=tuple(args.seeds or [0, 1, 2]),
    )
    for key in [
        "random_summary",
        "random_report",
        "scaffold_summary",
        "scaffold_report",
        "master_summary",
        "master_report",
    ]:
        if key in paths:
            print(f"{key}: {paths[key]}")
    print("benchmark_result: false")
    print(
        "note: Reporting-only full-database attribution-phase summaries; no training, sampling, jobs, GPUs, raw-data changes, or environment changes."
    )


if __name__ == "__main__":
    main()
