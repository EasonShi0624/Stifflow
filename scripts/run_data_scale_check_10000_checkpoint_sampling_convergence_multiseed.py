#!/usr/bin/env python
"""Run bounded 10000-molecule checkpoint/sampling convergence pilots across the fixed seed set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.tiny_real_checkpoint_sampling import (
    DATA_SCALE_CHECK_10000_CONVERGENCE_MAX_STEPS,
    DATA_SCALE_CHECK_10000_MAX_MOLECULES,
    DEFAULT_DATA_SCALE_CHECK_10000_CONVERGENCE_MULTI_SEED_REPORT_PATH,
    DEFAULT_DATA_SCALE_CHECK_10000_CONVERGENCE_MULTI_SEED_SUMMARY_PATH,
    DEFAULT_DATA_SCALE_CHECK_CHECKPOINT_SEED_SET,
    run_data_scale_check_10000_convergence_checkpoint_sampling_multiseed,
    write_data_scale_check_checkpoint_sampling_multiseed_summary,
)


SUMMARY_NAME = "geom_data_scale_check_ckpt_sample_multiseed_summary_mols10000_steps200_convergence"
REPORT_TITLE = "Bounded 10000-Molecule Checkpoint/Sampling Convergence Multi-Seed Summary"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output-root override for per-seed manifest/results/log/checkpoint files.",
    )
    parser.add_argument(
        "--summary",
        default=str(DEFAULT_DATA_SCALE_CHECK_10000_CONVERGENCE_MULTI_SEED_SUMMARY_PATH),
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_DATA_SCALE_CHECK_10000_CONVERGENCE_MULTI_SEED_REPORT_PATH),
        help="Output markdown report path.",
    )
    args = parser.parse_args()

    outputs = run_data_scale_check_10000_convergence_checkpoint_sampling_multiseed(
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
            print(f"{mode}_seed{seed}_checkpoint: {paths['checkpoint']}")

    summary_paths = write_data_scale_check_checkpoint_sampling_multiseed_summary(
        result_paths=result_paths,
        summary_path=args.summary,
        report_path=args.report,
        expected_seeds=DEFAULT_DATA_SCALE_CHECK_CHECKPOINT_SEED_SET,
        max_molecules=DATA_SCALE_CHECK_10000_MAX_MOLECULES,
        max_steps=DATA_SCALE_CHECK_10000_CONVERGENCE_MAX_STEPS,
        summary_name=SUMMARY_NAME,
        report_title=REPORT_TITLE,
    )
    print(f"summary: {summary_paths['summary']}")
    print(f"report: {summary_paths['report']}")


if __name__ == "__main__":
    main()
