#!/usr/bin/env python
"""Run bounded six-molecule micro-pilot training across a tiny seed set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.tiny_real_pilot_compare import write_micro_pilot_multiseed_summary
from src.train.tiny_real_pilot import DEFAULT_MULTI_SEED_SET, MICRO_PILOT_CONFIG_PATHS, run_micro_pilot_multiseed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        type=int,
        help="Seed to include. Pass more than once; defaults to the fixed tiny seed set 0,1.",
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Micro-pilot config path. Pass twice to override the default isotropic/stiffness configs.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output-root override for per-seed manifest/results/log files.",
    )
    parser.add_argument(
        "--summary",
        default="experiments/results/geom_micro_pilot_multiseed_summary.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--report",
        default="reports/tables/geom_micro_pilot_multiseed_summary.md",
        help="Output markdown report path.",
    )
    args = parser.parse_args()

    seed_set = tuple(args.seeds) if args.seeds else DEFAULT_MULTI_SEED_SET
    config_paths = tuple(Path(path) for path in args.configs) if args.configs else MICRO_PILOT_CONFIG_PATHS
    outputs = run_micro_pilot_multiseed(
        seed_set=seed_set,
        config_paths=config_paths,
        output_root=args.output_root,
    )
    result_paths = []
    for mode in sorted(outputs):
        for seed in sorted(outputs[mode]):
            paths = outputs[mode][seed]
            result_paths.append(paths["results"])
            print(f"{mode}_seed{seed}_manifest: {paths['manifest']}")
            print(f"{mode}_seed{seed}_results: {paths['results']}")
            print(f"{mode}_seed{seed}_log: {paths['log']}")

    summary_paths = write_micro_pilot_multiseed_summary(
        result_paths=result_paths,
        summary_path=args.summary,
        report_path=args.report,
        expected_seeds=seed_set,
    )
    print(f"summary: {summary_paths['summary']}")
    print(f"report: {summary_paths['report']}")


if __name__ == "__main__":
    main()
