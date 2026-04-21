#!/usr/bin/env python
"""Run bounded 10000-molecule checkpoint/sampling pilots across a tiny seed set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.tiny_real_checkpoint_sampling import (
    DATA_SCALE_CHECK_10000_CONFIG_PATHS,
    DATA_SCALE_CHECK_10000_MAX_MOLECULES,
    DEFAULT_DATA_SCALE_CHECK_10000_MULTI_SEED_REPORT_PATH,
    DEFAULT_DATA_SCALE_CHECK_10000_MULTI_SEED_SUMMARY_PATH,
    DEFAULT_DATA_SCALE_CHECK_CHECKPOINT_SEED_SET,
    DEFAULT_MICRO_PILOT_MAX_STEPS,
    DEFAULT_SAMPLE_STEP_SIZE,
    run_data_scale_check_10000_checkpoint_sampling_multiseed,
    write_data_scale_check_checkpoint_sampling_multiseed_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        type=int,
        help="Seed to include. Pass more than once; defaults to 0,1,2.",
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Data-scale-check 10000 config path. Pass twice to override the default isotropic/stiffness configs.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output-root override for per-seed manifest/results/log/checkpoint files.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MICRO_PILOT_MAX_STEPS,
        help="Bounded CPU optimizer-step count, capped at five for this checkpoint/sampling pilot.",
    )
    parser.add_argument(
        "--sample-step-size",
        type=float,
        default=DEFAULT_SAMPLE_STEP_SIZE,
        help="Single bounded coordinate update scale.",
    )
    parser.add_argument(
        "--summary",
        default=str(DEFAULT_DATA_SCALE_CHECK_10000_MULTI_SEED_SUMMARY_PATH),
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_DATA_SCALE_CHECK_10000_MULTI_SEED_REPORT_PATH),
        help="Output markdown report path.",
    )
    args = parser.parse_args()

    seed_set = tuple(args.seeds) if args.seeds else DEFAULT_DATA_SCALE_CHECK_CHECKPOINT_SEED_SET
    config_paths = tuple(Path(path) for path in args.configs) if args.configs else DATA_SCALE_CHECK_10000_CONFIG_PATHS
    outputs = run_data_scale_check_10000_checkpoint_sampling_multiseed(
        seed_set=seed_set,
        config_paths=config_paths,
        output_root=args.output_root,
        max_steps=args.max_steps,
        sample_step_size=args.sample_step_size,
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
        expected_seeds=seed_set,
        max_molecules=DATA_SCALE_CHECK_10000_MAX_MOLECULES,
        max_steps=args.max_steps,
    )
    print(f"summary: {summary_paths['summary']}")
    print(f"report: {summary_paths['report']}")


if __name__ == "__main__":
    main()
