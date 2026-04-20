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
        default=None,
        help="Tiny GEOM processed-output dry-run molecules.json path.",
    )
    parser.add_argument(
        "--split-root",
        default=None,
        help="Output root for tiny split dry-run artifacts.",
    )
    parser.add_argument("--split-type", default="random", choices=["random"], help="Tiny split dry-run type.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic split seed.")
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=None,
        help="Maximum molecules to read from the tiny processed artifact.",
    )
    parser.add_argument(
        "--micro-pilot",
        action="store_true",
        help="Use the bounded micro-pilot processed artifact and split root defaults.",
    )
    parser.add_argument(
        "--data-scale-check",
        action="store_true",
        help="Use the bounded 30-molecule data-scale-check processed artifact and split root defaults.",
    )
    args = parser.parse_args()
    if args.data_scale_check:
        processed_artifact = args.processed_artifact or "data/processed/geom_drugs_subset/data_scale_check/molecules.json"
        split_root = args.split_root or "data/splits/geom_drugs_data_scale_check"
        max_molecules = args.max_molecules if args.max_molecules is not None else 30
        split_scope = "data_scale_check_processed_dry_run"
        hard_max_molecules = 30
    elif args.micro_pilot:
        processed_artifact = args.processed_artifact or "data/processed/geom_drugs_subset/micro_pilot/molecules.json"
        split_root = args.split_root or "data/splits/geom_drugs_micro_pilot"
        max_molecules = args.max_molecules if args.max_molecules is not None else 6
        split_scope = "micro_pilot_processed_dry_run"
        hard_max_molecules = 6
    else:
        processed_artifact = args.processed_artifact or "data/processed/geom_drugs_subset/tiny_real_dry_run/molecules.json"
        split_root = args.split_root or "data/splits/geom_drugs_tiny_real_dry_run"
        max_molecules = args.max_molecules if args.max_molecules is not None else 3
        split_scope = None
        hard_max_molecules = 6

    paths = run_geom_tiny_split_dry_run(
        processed_artifact_path=processed_artifact,
        split_root=split_root,
        split_type=args.split_type,
        seed=args.seed,
        max_molecules=max_molecules,
        split_scope=split_scope,
        hard_max_molecules=hard_max_molecules,
    )
    print(f"split: {paths['split']}")
    print(f"manifest: {paths['manifest']}")


if __name__ == "__main__":
    main()
