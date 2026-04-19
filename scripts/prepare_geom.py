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
    parser.add_argument("--dry-run", action="store_true", help="Run the read-only preflight only.")
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
    args = parser.parse_args()

    if not args.dry_run:
        raise SystemExit(
            "Real GEOM-Drugs preprocessing is not implemented yet. "
            "Run with --dry-run for the read-only raw-data preflight."
        )

    paths = run_geom_preflight(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        manifest_path=args.manifest,
        report_path=args.report,
        max_scan_entries=args.max_scan_entries,
    )
    print(f"manifest: {paths['manifest']}")
    print(f"report: {paths['report']}")


if __name__ == "__main__":
    main()
