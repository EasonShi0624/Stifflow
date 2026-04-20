#!/usr/bin/env python
"""Run bounded tiny real-data CPU pilot training for both source modes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.tiny_real_pilot import DEFAULT_CONFIG_PATHS, run_tiny_real_pilot_configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help=(
            "Pilot config path. Pass twice to override the default isotropic and "
            "stiffness-structured pilot configs."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output-root override for manifest/results/log files.",
    )
    args = parser.parse_args()

    config_paths = tuple(Path(path) for path in args.configs) if args.configs else DEFAULT_CONFIG_PATHS
    outputs = run_tiny_real_pilot_configs(config_paths=config_paths, output_root=args.output_root)
    for mode, paths in outputs.items():
        print(f"{mode}_manifest: {paths['manifest']}")
        print(f"{mode}_results: {paths['results']}")
        print(f"{mode}_log: {paths['log']}")


if __name__ == "__main__":
    main()
