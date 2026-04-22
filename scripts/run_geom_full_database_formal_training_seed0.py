#!/usr/bin/env python
"""Prepare and run first full-database GEOM-Drugs formal training at seed 0."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.geom_full_database import (
    FULL_DATABASE_PROCESSED_ROOT,
    FULL_DATABASE_SPLIT_ROOT,
    prepare_geom_full_database_artifact,
    prepare_geom_full_database_split,
)
from src.train.formal_full_database_training import run_formal_full_database_training_configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw/geom_drugs", help="GEOM-Drugs raw root.")
    parser.add_argument(
        "--processed-root",
        default=str(FULL_DATABASE_PROCESSED_ROOT),
        help="Processed full-database one-conformer artifact root.",
    )
    parser.add_argument(
        "--split-root",
        default=str(FULL_DATABASE_SPLIT_ROOT),
        help="Full-database split root.",
    )
    parser.add_argument(
        "--output-root",
        default="experiments",
        help="Output root for manifest/results/log/checkpoint files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Single fixed full-database formal training seed. Must remain 0 for this prepared path.",
    )
    parser.add_argument(
        "--max-pickle-bytes",
        type=int,
        default=0,
        help="Optional raw pickle byte cap. A value <= 0 means no per-pickle byte cap.",
    )
    parser.add_argument(
        "--max-processed-atoms",
        type=int,
        default=512,
        help="Skip molecules with more atoms than this extractor bound.",
    )
    args = parser.parse_args()

    if args.seed != 0:
        raise SystemExit("The first full-database formal training path is prepared only for seed 0.")
    if Path(args.processed_root) != FULL_DATABASE_PROCESSED_ROOT:
        raise SystemExit(f"processed-root must remain {FULL_DATABASE_PROCESSED_ROOT} for the frozen config path.")
    if Path(args.split_root) != FULL_DATABASE_SPLIT_ROOT:
        raise SystemExit(f"split-root must remain {FULL_DATABASE_SPLIT_ROOT} for the frozen config path.")

    max_pickle_bytes = None if args.max_pickle_bytes <= 0 else int(args.max_pickle_bytes)
    processed = prepare_geom_full_database_artifact(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        max_conformers_per_mol=1,
        max_pickle_bytes=max_pickle_bytes,
        max_atoms=args.max_processed_atoms,
        force=False,
    )
    split = prepare_geom_full_database_split(
        processed_artifact_path=Path(args.processed_root) / "molecules.json",
        split_root=args.split_root,
        seed=args.seed,
        force=False,
    )
    outputs = run_formal_full_database_training_configs(output_root=args.output_root, seed=args.seed)

    print(f"processed_artifact: {processed['artifact']}")
    print(f"processed_manifest: {processed['manifest']}")
    print(f"processed_reused_existing: {processed['reused_existing']}")
    print(f"processed_molecule_count: {processed['molecule_count']}")
    print(f"source_pickle_count: {processed['source_pickle_count']}")
    print(f"split: {split['split']}")
    print(f"split_manifest: {split['manifest']}")
    print(f"split_reused_existing: {split['reused_existing']}")
    print(f"split_counts: {split['counts']}")
    for mode in sorted(outputs):
        paths = outputs[mode]
        print(f"{mode}_manifest: {paths['manifest']}")
        print(f"{mode}_results: {paths['results']}")
        print(f"{mode}_log: {paths['log']}")
        print(f"{mode}_best_checkpoint: {paths['best_checkpoint']}")
        print(f"{mode}_latest_checkpoint: {paths['latest_checkpoint']}")


if __name__ == "__main__":
    main()
