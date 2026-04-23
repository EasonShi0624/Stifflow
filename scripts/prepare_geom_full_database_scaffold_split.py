#!/usr/bin/env python
"""Prepare or reuse the cached full-database GEOM-Drugs scaffold split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.geom_full_database import (
    FULL_DATABASE_ARTIFACT_PATH,
    FULL_DATABASE_SPLIT_ROOT,
    prepare_geom_full_database_scaffold_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-artifact",
        default=str(FULL_DATABASE_ARTIFACT_PATH),
        help="Cached full-database processed artifact to read.",
    )
    parser.add_argument(
        "--split-root",
        default=str(FULL_DATABASE_SPLIT_ROOT),
        help="Root under which scaffold_seed<seed>/split.json is written.",
    )
    parser.add_argument("--seed", type=int, default=0, choices=[0], help="Prepared scaffold split seed.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing cached scaffold split.")
    args = parser.parse_args()

    paths = prepare_geom_full_database_scaffold_split(
        processed_artifact_path=args.processed_artifact,
        split_root=args.split_root,
        seed=args.seed,
        force=args.force,
    )
    print(f"processed_artifact: {args.processed_artifact}")
    print(f"split: {paths['split']}")
    print(f"manifest: {paths['manifest']}")
    print(f"reused_existing: {paths['reused_existing']}")
    print(f"counts: {paths['counts']}")
    if "scaffold_group_count" in paths:
        print(f"scaffold_group_count: {paths['scaffold_group_count']}")
    print(f"stereo_fallback_molecule_count: {paths.get('stereo_fallback_molecule_count', 0)}")
    print(f"skipped_molecule_count: {paths.get('skipped_molecule_count', 0)}")
    print("benchmark_result: false")
    print("note: Cached full-database scaffold split only; no training or benchmark result.")


if __name__ == "__main__":
    main()
