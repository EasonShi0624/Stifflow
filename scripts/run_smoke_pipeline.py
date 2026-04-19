#!/usr/bin/env python
"""Run the two synthetic smoke configs as one local CPU-only pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.pipeline import DEFAULT_SMOKE_CONFIGS, run_smoke_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[str(path) for path in DEFAULT_SMOKE_CONFIGS],
        help="Smoke configs to run locally. Defaults to isotropic and stiffness smoke configs.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root override. Defaults to the first config's experiment.output_root.",
    )
    parser.add_argument(
        "--pipeline-name",
        default="smoke_pair_synthetic_seed0",
        help="Name used for aggregate pipeline manifest/result/log files.",
    )
    args = parser.parse_args()

    paths = run_smoke_pipeline(
        config_paths=[Path(path) for path in args.configs],
        output_root=args.output_root,
        pipeline_name=args.pipeline_name,
    )
    print(f"pipeline_manifest: {paths['manifest']}")
    print(f"pipeline_results: {paths['results']}")
    print(f"pipeline_log: {paths['log']}")


if __name__ == "__main__":
    main()
