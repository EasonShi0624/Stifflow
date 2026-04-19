#!/usr/bin/env python
"""Run RDKit ETKDGv3 + MMFF on the tiny real GEOM dry-run artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.rdkit_etkdg import run_rdkit_tiny_real_dry_run


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
    parser.add_argument(
        "--run-name",
        default="geom_tiny_real_rdkit_etkdg_mmff_seed0",
        help="Name used for manifest/result/log files.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic RDKit random seed.")
    parser.add_argument("--max-molecules", type=int, default=3, help="Maximum tiny real molecules to run.")
    args = parser.parse_args()

    paths = run_rdkit_tiny_real_dry_run(
        processed_artifact_path=args.processed_artifact,
        split_path=args.split,
        output_root=args.output_root,
        run_name=args.run_name,
        seed=args.seed,
        max_molecules=args.max_molecules,
    )
    print(f"manifest: {paths['manifest']}")
    print(f"results: {paths['results']}")
    print(f"log: {paths['log']}")


if __name__ == "__main__":
    main()
