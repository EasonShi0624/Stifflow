#!/usr/bin/env python
"""Run one cached-input full-database attribution formal training job."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.geom_full_database import (  # noqa: E402
    FULL_DATABASE_ARTIFACT_PATH,
    FULL_DATABASE_SCAFFOLD_SPLIT_PATH,
    FULL_DATABASE_SPLIT_PATH,
)
from src.train.formal_full_database_training import (  # noqa: E402
    FORMAL_FULL_DATABASE_ATTRIBUTION_MODES,
    FORMAL_FULL_DATABASE_SEED_SET,
    run_formal_full_database_training,
)
from src.utils.config import load_config  # noqa: E402


CONFIG_BY_SPLIT_AND_MODE = {
    ("random", "isotropic"): Path("configs/train/geom_full_database_formal_isotropic.yaml"),
    ("random", "harmonic_prior"): Path("configs/train/geom_full_database_formal_harmonic.yaml"),
    ("random", "torsion_only"): Path("configs/train/geom_full_database_formal_torsion_only.yaml"),
    ("random", "stiffness_structured"): Path("configs/train/geom_full_database_formal_stiffness.yaml"),
    ("scaffold", "isotropic"): Path("configs/train/geom_full_database_scaffold_formal_isotropic.yaml"),
    ("scaffold", "harmonic_prior"): Path("configs/train/geom_full_database_scaffold_formal_harmonic.yaml"),
    ("scaffold", "torsion_only"): Path("configs/train/geom_full_database_scaffold_formal_torsion_only.yaml"),
    ("scaffold", "stiffness_structured"): Path("configs/train/geom_full_database_scaffold_formal_stiffness.yaml"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-type", choices=["random", "scaffold"], required=True)
    parser.add_argument("--mode", choices=FORMAL_FULL_DATABASE_ATTRIBUTION_MODES, required=True)
    parser.add_argument("--seed", type=int, choices=FORMAL_FULL_DATABASE_SEED_SET, required=True)
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for model execution. The prepared SLURM wrapper uses cuda.",
    )
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail loudly if CUDA is unavailable. Enabled by default for GPU SLURM jobs.",
    )
    args = parser.parse_args()

    if not FULL_DATABASE_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Cached full-database processed artifact is required: {FULL_DATABASE_ARTIFACT_PATH}")
    split_path = FULL_DATABASE_SPLIT_PATH if args.split_type == "random" else FULL_DATABASE_SCAFFOLD_SPLIT_PATH
    if not split_path.exists():
        raise FileNotFoundError(f"Cached full-database {args.split_type} split is required: {split_path}")

    config_path = CONFIG_BY_SPLIT_AND_MODE[(args.split_type, args.mode)]
    config = copy.deepcopy(load_config(config_path))
    config["experiment"]["seed"] = int(args.seed)
    config["experiment"]["output_root"] = str(args.output_root)
    config["data"]["processed_root"] = str(FULL_DATABASE_ARTIFACT_PATH.parent)
    config["data"]["split_root"] = str(split_path.parent)
    config["data"]["processed_artifact_path"] = str(FULL_DATABASE_ARTIFACT_PATH)
    config["data"]["split_path"] = str(split_path)
    config["data"]["split_type"] = str(args.split_type)
    config["eval"]["compute_physical_validity_diagnostics"] = True
    config["eval"]["compute_rotatable_bond_stratification"] = True
    config["eval"]["compute_rotatable_bond_count_from_smiles"] = True
    config["train"]["device"] = str(args.device)
    config["train"]["require_cuda"] = bool(args.require_cuda)

    outputs = run_formal_full_database_training(config=config, config_path=config_path)
    print(f"split_type: {args.split_type}")
    print(f"mode: {args.mode}")
    print(f"seed: {args.seed}")
    print(f"processed_artifact: {FULL_DATABASE_ARTIFACT_PATH}")
    print(f"split: {split_path}")
    print(f"manifest: {outputs['manifest']}")
    print(f"results: {outputs['results']}")
    print(f"log: {outputs['log']}")
    print(f"best_checkpoint: {outputs['best_checkpoint']}")
    print(f"latest_checkpoint: {outputs['latest_checkpoint']}")
    print("benchmark_result: false")
    print("note: Full-database attribution formal training path; not benchmark results.")


if __name__ == "__main__":
    main()
