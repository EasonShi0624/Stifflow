#!/usr/bin/env python
"""Write a tiny non-benchmark split dry-run for the GEOM processed artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.geom_split_dry_run import run_geom_tiny_split_dry_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-artifact",
        default="data/processed/geom_drugs_subset/tiny_real_dry_run/molecules.json",
        help="Tiny GEOM processed-output dry-run molecules.json path.",
    )
    parser.add_argument(
        "--split-root",
        default="data/splits/geom_drugs_tiny_real_dry_run",
        help="Output root for tiny split dry-run artifacts.",
    )
    parser.add_argument("--split-type", default="random", choices=["random"], help="Tiny split dry-run type.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic split seed.")
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=3,
        help="Maximum molecules to read from the tiny processed artifact.",
    )
    args = parser.parse_args()

    paths = run_geom_tiny_split_dry_run(
        processed_artifact_path=args.processed_artifact,
        split_root=args.split_root,
        split_type=args.split_type,
        seed=args.seed,
        max_molecules=args.max_molecules,
    )
    print(f"split: {paths['split']}")
    print(f"manifest: {paths['manifest']}")


if __name__ == "__main__":
    main()
