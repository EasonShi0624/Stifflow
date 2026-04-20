#!/usr/bin/env python
"""Run tiny real-data CPU single-step training smoke tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.tiny_real_single_step import run_tiny_real_single_step_smoke_tests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-artifact",
        default="data/processed/geom_drugs_subset/tiny_real_dry_run/molecules.json",
        help="Tiny real GEOM processed-output dry-run artifact.",
    )
    parser.add_argument(
        "--split",
        default="data/splits/geom_drugs_tiny_real_dry_run/random_seed0/split.json",
        help="Tiny real GEOM split dry-run artifact.",
    )
    parser.add_argument("--output-root", default="experiments", help="Output root for manifest/results/logs.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic single-step smoke seed.")
    parser.add_argument("--max-molecules", type=int, default=3, help="Maximum tiny real molecules to use.")
    parser.add_argument("--learning-rate", type=float, default=1.0e-2, help="Single optimizer-step learning rate.")
    args = parser.parse_args()

    outputs = run_tiny_real_single_step_smoke_tests(
        processed_artifact_path=args.processed_artifact,
        split_path=args.split,
        output_root=args.output_root,
        seed=args.seed,
        max_molecules=args.max_molecules,
        learning_rate=args.learning_rate,
    )
    for mode, paths in outputs.items():
        print(f"{mode}_manifest: {paths['manifest']}")
        print(f"{mode}_results: {paths['results']}")
        print(f"{mode}_log: {paths['log']}")


if __name__ == "__main__":
    main()
