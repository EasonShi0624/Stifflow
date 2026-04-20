#!/usr/bin/env python
"""Read-only GEOM-Drugs preflight and preprocessing dry-run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.geom_preflight import run_geom_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Run the bounded preflight / dry-run only.")
    parser.add_argument("--raw-root", default="data/raw/geom_drugs", help="Expected GEOM-Drugs raw root.")
    parser.add_argument(
        "--processed-root",
        default="data/processed/geom_drugs_subset",
        help="Future processed-output root reported in the dry-run plan.",
    )
    parser.add_argument(
        "--manifest",
        default="experiments/manifests/geom_drugs_preflight.json",
        help="Output preflight manifest path.",
    )
    parser.add_argument(
        "--report",
        default="reports/tables/geom_drugs_preflight.md",
        help="Output markdown preflight report path.",
    )
    parser.add_argument(
        "--max-scan-entries",
        type=int,
        default=200,
        help="Maximum filesystem entries to inspect under raw root.",
    )
    parser.add_argument(
        "--max-dry-run-molecules",
        type=int,
        default=2,
        help="Maximum GEOM-Drugs pickle records to read for metadata-only dry-run.",
    )
    parser.add_argument(
        "--max-dry-run-conformers-per-mol",
        type=int,
        default=3,
        help="Maximum conformer metadata records to summarize per sampled molecule.",
    )
    parser.add_argument(
        "--max-pickle-bytes",
        type=int,
        default=1_000_000,
        help="Skip metadata dry-run pickle files larger than this many bytes.",
    )
    parser.add_argument(
        "--write-tiny-processed-output",
        action="store_true",
        help="Opt in to writing a tiny non-benchmark processed-output dry-run artifact.",
    )
    parser.add_argument(
        "--write-micro-pilot-output",
        action="store_true",
        help=(
            "Opt in to writing a strictly bounded six-molecule micro-pilot processed "
            "dry-run artifact under the processed root."
        ),
    )
    parser.add_argument(
        "--tiny-processed-tag",
        default="tiny_real_dry_run",
        help="Subdirectory name under processed root for the tiny processed-output dry-run.",
    )
    parser.add_argument(
        "--max-processed-molecules",
        type=int,
        default=3,
        help="Maximum real GEOM molecule records to write in the tiny processed-output dry-run.",
    )
    parser.add_argument(
        "--max-processed-conformers-per-mol",
        type=int,
        default=1,
        help="Maximum conformers per molecule to write in the tiny processed-output dry-run.",
    )
    parser.add_argument(
        "--max-processed-atoms",
        type=int,
        default=128,
        help="Skip tiny processed-output molecules with more atoms than this bound.",
    )
    args = parser.parse_args()

    if not args.dry_run:
        raise SystemExit(
            "Real GEOM-Drugs preprocessing is not implemented yet. "
            "Run with --dry-run for the read-only raw-data preflight."
        )

    write_tiny_processed_output = args.write_tiny_processed_output or args.write_micro_pilot_output
    tiny_processed_tag = args.tiny_processed_tag
    max_processed_molecules = args.max_processed_molecules
    if args.write_micro_pilot_output:
        if tiny_processed_tag == "tiny_real_dry_run":
            tiny_processed_tag = "micro_pilot"
        if max_processed_molecules == 3:
            max_processed_molecules = 6

    paths = run_geom_preflight(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        manifest_path=args.manifest,
        report_path=args.report,
        max_scan_entries=args.max_scan_entries,
        max_dry_run_molecules=args.max_dry_run_molecules,
        max_dry_run_conformers_per_mol=args.max_dry_run_conformers_per_mol,
        max_pickle_bytes=args.max_pickle_bytes,
        write_tiny_processed_output=write_tiny_processed_output,
        write_micro_pilot_output=args.write_micro_pilot_output,
        tiny_processed_tag=tiny_processed_tag,
        max_processed_molecules=max_processed_molecules,
        max_processed_conformers_per_mol=args.max_processed_conformers_per_mol,
        max_processed_atoms=args.max_processed_atoms,
    )
    print(f"manifest: {paths['manifest']}")
    print(f"report: {paths['report']}")
    if "processed_artifact" in paths:
        print(f"processed_artifact: {paths['processed_artifact']}")
        print(f"processed_manifest: {paths['processed_manifest']}")


if __name__ == "__main__":
    main()
