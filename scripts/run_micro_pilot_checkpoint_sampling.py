#!/usr/bin/env python
"""Run bounded six-molecule micro-pilot checkpoint-and-sampling pilots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.tiny_real_checkpoint_sampling import (
    DEFAULT_MICRO_PILOT_MAX_STEPS,
    DEFAULT_SAMPLE_STEP_SIZE,
    MICRO_PILOT_CONFIG_PATHS,
    run_micro_pilot_checkpoint_sampling_configs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Micro-pilot config path. Pass twice to override the default isotropic/stiffness configs.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output-root override for manifest/results/log/checkpoint files.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Single fixed micro-pilot seed.")
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
    args = parser.parse_args()

    config_paths = tuple(Path(path) for path in args.configs) if args.configs else MICRO_PILOT_CONFIG_PATHS
    outputs = run_micro_pilot_checkpoint_sampling_configs(
        config_paths=config_paths,
        output_root=args.output_root,
        seed=args.seed,
        max_steps=args.max_steps,
        sample_step_size=args.sample_step_size,
    )
    for mode in sorted(outputs):
        paths = outputs[mode]
        print(f"{mode}_manifest: {paths['manifest']}")
        print(f"{mode}_results: {paths['results']}")
        print(f"{mode}_log: {paths['log']}")
        print(f"{mode}_checkpoint: {paths['checkpoint']}")


if __name__ == "__main__":
    main()
