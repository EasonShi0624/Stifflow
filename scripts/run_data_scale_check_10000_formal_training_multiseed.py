#!/usr/bin/env python
"""Run first formal bounded 10000-molecule training over the fixed seed set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.formal_10000_training import (
    FORMAL_10000_REPORT_PATH,
    FORMAL_10000_SEED_SET,
    FORMAL_10000_SUMMARY_PATH,
    run_formal_10000_training_multiseed,
    write_formal_10000_training_multiseed_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output-root override for per-seed manifest/results/log/checkpoint files.",
    )
    parser.add_argument("--summary", default=str(FORMAL_10000_SUMMARY_PATH), help="Output JSON summary path.")
    parser.add_argument("--report", default=str(FORMAL_10000_REPORT_PATH), help="Output markdown report path.")
    args = parser.parse_args()

    outputs = run_formal_10000_training_multiseed(output_root=args.output_root)
    result_paths = []
    for mode in sorted(outputs):
        for seed in sorted(outputs[mode]):
            paths = outputs[mode][seed]
            result_paths.append(paths["results"])
            print(f"{mode}_seed{seed}_manifest: {paths['manifest']}")
            print(f"{mode}_seed{seed}_results: {paths['results']}")
            print(f"{mode}_seed{seed}_log: {paths['log']}")
            print(f"{mode}_seed{seed}_best_checkpoint: {paths['best_checkpoint']}")
            print(f"{mode}_seed{seed}_latest_checkpoint: {paths['latest_checkpoint']}")

    summary_paths = write_formal_10000_training_multiseed_summary(
        result_paths=result_paths,
        summary_path=args.summary,
        report_path=args.report,
        expected_seeds=FORMAL_10000_SEED_SET,
    )
    print(f"summary: {summary_paths['summary']}")
    print(f"report: {summary_paths['report']}")


if __name__ == "__main__":
    main()
