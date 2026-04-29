#!/usr/bin/env python
"""Prepare or run gated physical diagnostics for GEOM full-database outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.physical_benchmark import (  # noqa: E402
    DEFAULT_PHYSICAL_OUTPUT_DIR,
    DEFAULT_PHYSICAL_SUMMARY_JSON,
    DEFAULT_PHYSICAL_SUMMARY_MD,
    DEFAULT_PLAN_JSON,
    DEFAULT_PLAN_MD,
    evaluate_existing_result_physical_subset,
    evaluate_full_database_physical_matrix,
    write_physical_benchmark_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-json",
        default=str(DEFAULT_PLAN_JSON),
        help="Path for the non-benchmark physical benchmark plan JSON.",
    )
    parser.add_argument(
        "--plan-md",
        default=str(DEFAULT_PLAN_MD),
        help="Path for the non-benchmark physical benchmark plan markdown report.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run a bounded evaluation-only diagnostic pass for one result JSON.",
    )
    parser.add_argument(
        "--run-matrix",
        action="store_true",
        help="Run the 24-result full-database physical diagnostic matrix.",
    )
    parser.add_argument(
        "--result-path",
        help="Existing full-database result JSON to evaluate when --run is set.",
    )
    parser.add_argument(
        "--output-json",
        help="Output JSON for --run diagnostics. Required with --run.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_PHYSICAL_OUTPUT_DIR),
        help="Output directory for per-run JSONs when --run-matrix is set.",
    )
    parser.add_argument(
        "--summary-json",
        default=str(DEFAULT_PHYSICAL_SUMMARY_JSON),
        help="Summary JSON path when --run-matrix is set.",
    )
    parser.add_argument(
        "--summary-md",
        default=str(DEFAULT_PHYSICAL_SUMMARY_MD),
        help="Summary markdown path when --run-matrix is set.",
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=None,
        help="Bound the number of molecules evaluated from the selected result. Required unless --allow-full-database.",
    )
    parser.add_argument(
        "--max-molecules-per-result",
        type=int,
        default=None,
        help="Bound the number of molecules evaluated per result when --run-matrix is set.",
    )
    parser.add_argument(
        "--allow-full-database",
        action="store_true",
        help="Allow evaluating the full selected result. Intended for approved scheduler runs only.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for evaluation-only checkpoint reload. Defaults to CPU.",
    )
    parser.add_argument(
        "--skip-posebusters",
        action="store_true",
        help="Use only RDKit/fallback intramolecular checks.",
    )
    parser.add_argument(
        "--skip-forcefields",
        action="store_true",
        help="Skip MMFF/UFF diagnostics.",
    )
    parser.add_argument(
        "--save-per-molecule",
        action="store_true",
        help="Save per-molecule diagnostic rows. Avoid this for large full-database runs unless needed.",
    )
    args = parser.parse_args()

    outputs = write_physical_benchmark_plan(args.plan_json, args.plan_md)
    print(f"plan_json: {outputs['json']}")
    print(f"plan_markdown: {outputs['markdown']}")
    print("plan_status: written")

    if args.run and args.run_matrix:
        raise SystemExit("Use only one of --run or --run-matrix")

    if not args.run and not args.run_matrix:
        print("evaluation_status: not_run")
        print("note: default action is plan generation only; no training, preprocessing, sampling pass, job, or xTB run.")
        return

    if args.run_matrix:
        if args.max_molecules_per_result is None and not args.allow_full_database:
            raise SystemExit("--run-matrix requires --max-molecules-per-result unless --allow-full-database is set")
        result = evaluate_full_database_physical_matrix(
            output_dir=args.output_dir,
            summary_json_path=args.summary_json,
            summary_markdown_path=args.summary_md,
            max_molecules_per_result=args.max_molecules_per_result,
            device=args.device,
            include_posebusters=not args.skip_posebusters,
            include_forcefields=not args.skip_forcefields,
            allow_full_database=args.allow_full_database,
        )
        print("evaluation_status: completed")
        print(f"per_run_output_dir: {args.output_dir}")
        print(f"summary_json: {args.summary_json}")
        print(f"summary_markdown: {args.summary_md}")
        print(f"run_count: {result['run_count']}")
        print("xtb_execution_status: not_run")
        print("benchmark_result: false")
        print("note: matrix evaluation-only checkpoint reload; no training or preprocessing was run.")
        return

    if not args.result_path or not args.output_json:
        raise SystemExit("--run requires --result-path and --output-json")
    if args.max_molecules is None and not args.allow_full_database:
        raise SystemExit("--run requires --max-molecules unless --allow-full-database is explicitly set")

    result = evaluate_existing_result_physical_subset(
        result_path=args.result_path,
        output_path=args.output_json,
        max_molecules=args.max_molecules,
        device=args.device,
        include_posebusters=not args.skip_posebusters,
        include_forcefields=not args.skip_forcefields,
        allow_full_database=args.allow_full_database,
        save_per_molecule=args.save_per_molecule,
    )
    print(f"evaluation_status: completed")
    print(f"diagnostics_json: {args.output_json}")
    print(f"sample_count: {result['sample_count']}")
    print("benchmark_result: false")
    print("note: evaluation-only checkpoint reload; no training or preprocessing was run.")


if __name__ == "__main__":
    main()
