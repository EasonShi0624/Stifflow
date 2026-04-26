#!/usr/bin/env python
"""Backfill sampled physical-validity and rotatable diagnostics for full-database formal A/E outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.formal_diagnostic_backfill import (  # noqa: E402
    backfill_full_database_formal_diagnostics,
    default_backfill_result_paths,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-path",
        action="append",
        default=None,
        help="Specific full-database isotropic/stiffness result JSON to backfill. Repeat to pass multiple paths.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for evaluation-only checkpoint reload + sampling. Defaults to cpu.",
    )
    parser.add_argument(
        "--skip-report-refresh",
        action="store_true",
        help="Only backfill the selected result JSONs. Do not regenerate attribution/closure summaries.",
    )
    args = parser.parse_args()

    result_paths = args.result_path or [str(path) for path in default_backfill_result_paths()]
    outputs = backfill_full_database_formal_diagnostics(
        result_paths=result_paths,
        device=args.device,
        refresh_reports=not args.skip_report_refresh,
    )
    for row in outputs["refreshed_results"]:
        print(
            f"result={row['result_path']} updated={row['updated']} "
            f"physical_keys={row['status_after']['physical_key_count']}/{row['status_after']['sample_count']} "
            f"rotatable_numeric={row['status_after']['rotatable_numeric_count']}/{row['status_after']['sample_count']}"
        )
    if outputs["report_refresh_executed"]:
        for section in ("attribution_outputs", "closure_outputs"):
            for key, value in outputs[section].items():
                print(f"{section}.{key}: {value}")
    print("benchmark_result: false")
    print("note: Evaluation-only checkpoint reload + sampling backfill from existing full-database outputs; no training rerun.")


if __name__ == "__main__":
    main()
