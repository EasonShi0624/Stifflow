#!/usr/bin/env python
"""Run bounded 10000-molecule data-scale-check checkpoint/sampling pilots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.geom_preflight import DATA_SCALE_CHECK_10000_HARD_MAX_MOLECULES, run_geom_preflight
from src.data.geom_split_dry_run import run_geom_tiny_split_dry_run
from src.train.tiny_real_checkpoint_sampling import (
    DATA_SCALE_CHECK_10000_CONFIG_PATHS,
    DATA_SCALE_CHECK_10000_MAX_MOLECULES,
    DEFAULT_MICRO_PILOT_MAX_STEPS,
    DEFAULT_SAMPLE_STEP_SIZE,
    run_data_scale_check_10000_checkpoint_sampling_configs,
    write_data_scale_check_checkpoint_sampling_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw/geom_drugs", help="GEOM-Drugs raw root.")
    parser.add_argument(
        "--processed-root",
        default="data/processed/geom_drugs_subset",
        help="Processed dry-run root.",
    )
    parser.add_argument(
        "--processed-artifact",
        default="data/processed/geom_drugs_subset/data_scale_check_10000/molecules.json",
        help="Bounded 10000-molecule data-scale-check processed artifact path.",
    )
    parser.add_argument(
        "--split-root",
        default="data/splits/geom_drugs_data_scale_check_10000",
        help="Bounded 10000-molecule data-scale-check split root.",
    )
    parser.add_argument(
        "--output-root",
        default="experiments",
        help="Output root for manifest/results/log/checkpoint files.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Single fixed data-scale-check seed.")
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=DATA_SCALE_CHECK_10000_MAX_MOLECULES,
        help="Must remain exactly at the bounded 10000-molecule data-scale-check cap.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MICRO_PILOT_MAX_STEPS,
        help="Bounded CPU optimizer-step count, capped at five for this checkpoint/sampling pilot.",
    )
    parser.add_argument(
        "--sample-step-size",
        type=float,
        default=DEFAULT_SAMPLE_STEP_SIZE,
        help="Single bounded coordinate update scale.",
    )
    parser.add_argument(
        "--max-pickle-bytes",
        type=int,
        default=1_000_000,
        help="Skip raw pickle files larger than this bound while building the dry-run artifact.",
    )
    parser.add_argument(
        "--max-processed-atoms",
        type=int,
        default=128,
        help="Skip processed dry-run molecules with more atoms than this bound.",
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Data-scale-check 10000 config path. Pass twice to override the default isotropic/stiffness configs.",
    )
    args = parser.parse_args()

    if args.max_molecules != DATA_SCALE_CHECK_10000_MAX_MOLECULES:
        raise SystemExit("The bounded 10000-molecule data-scale-check command must use exactly 10000 molecules.")

    preflight_paths = run_geom_preflight(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        manifest_path="experiments/manifests/geom_drugs_data_scale_check_10000_preflight.json",
        report_path="reports/tables/geom_drugs_data_scale_check_10000_preflight.md",
        max_dry_run_molecules=2,
        max_dry_run_conformers_per_mol=3,
        max_pickle_bytes=args.max_pickle_bytes,
        write_tiny_processed_output=True,
        write_data_scale_check_10000_output=True,
        tiny_processed_tag="data_scale_check_10000",
        max_processed_molecules=args.max_molecules,
        max_processed_conformers_per_mol=1,
        max_processed_atoms=args.max_processed_atoms,
    )
    split_paths = run_geom_tiny_split_dry_run(
        processed_artifact_path=args.processed_artifact,
        split_root=args.split_root,
        split_type="random",
        seed=args.seed,
        max_molecules=args.max_molecules,
        split_scope="data_scale_check_10000_processed_dry_run",
        hard_max_molecules=DATA_SCALE_CHECK_10000_HARD_MAX_MOLECULES,
    )
    config_paths = tuple(Path(path) for path in args.configs) if args.configs else DATA_SCALE_CHECK_10000_CONFIG_PATHS
    outputs = run_data_scale_check_10000_checkpoint_sampling_configs(
        config_paths=config_paths,
        output_root=args.output_root,
        seed=args.seed,
        max_steps=args.max_steps,
        sample_step_size=args.sample_step_size,
    )
    result_paths = [outputs[mode]["results"] for mode in sorted(outputs)]
    comparison = write_data_scale_check_checkpoint_sampling_comparison(
        result_paths=result_paths,
        output_root=args.output_root,
        seed=args.seed,
        max_molecules=args.max_molecules,
        max_steps=args.max_steps,
    )

    print(f"processed_artifact: {preflight_paths['processed_artifact']}")
    print(f"processed_manifest: {preflight_paths['processed_manifest']}")
    print(f"split: {split_paths['split']}")
    print(f"split_manifest: {split_paths['manifest']}")
    for mode in sorted(outputs):
        paths = outputs[mode]
        print(f"{mode}_manifest: {paths['manifest']}")
        print(f"{mode}_results: {paths['results']}")
        print(f"{mode}_log: {paths['log']}")
        print(f"{mode}_checkpoint: {paths['checkpoint']}")
    print(f"comparison_manifest: {comparison['manifest']}")
    print(f"comparison_results: {comparison['results']}")
    print(f"comparison_log: {comparison['log']}")


if __name__ == "__main__":
    main()
