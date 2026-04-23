#!/usr/bin/env python
"""Run one full-database scaffold-split GEOM-Drugs formal training job."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.geom_full_database import FULL_DATABASE_ARTIFACT_PATH, FULL_DATABASE_SCAFFOLD_SPLIT_PATH
from src.train.formal_full_database_training import run_formal_full_database_training
from src.utils.config import load_config


CONFIG_BY_MODE = {
    "isotropic": Path("configs/train/geom_full_database_scaffold_formal_isotropic.yaml"),
    "stiffness_structured": Path("configs/train/geom_full_database_scaffold_formal_stiffness.yaml"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(CONFIG_BY_MODE), required=True, help="Formal source-corruption mode.")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0, help="Prepared scaffold training seed.")
    parser.add_argument("--output-root", default="experiments", help="Manifest/results/log/checkpoint output root.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for model execution. The prepared SLURM jobs use cuda.",
    )
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail loudly if CUDA is unavailable. Enabled by default for GPU SLURM jobs.",
    )
    args = parser.parse_args()

    if not FULL_DATABASE_ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Cached full-database processed artifact is required before scaffold training: {FULL_DATABASE_ARTIFACT_PATH}"
        )
    if not FULL_DATABASE_SCAFFOLD_SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Cached full-database scaffold split is required before scaffold training: {FULL_DATABASE_SCAFFOLD_SPLIT_PATH}"
        )

    config_path = CONFIG_BY_MODE[args.mode]
    config = copy.deepcopy(load_config(config_path))
    config["experiment"]["seed"] = int(args.seed)
    config["experiment"]["output_root"] = str(args.output_root)
    config["data"]["processed_root"] = str(FULL_DATABASE_ARTIFACT_PATH.parent)
    config["data"]["split_root"] = str(FULL_DATABASE_SCAFFOLD_SPLIT_PATH.parent)
    config["data"]["processed_artifact_path"] = str(FULL_DATABASE_ARTIFACT_PATH)
    config["data"]["split_path"] = str(FULL_DATABASE_SCAFFOLD_SPLIT_PATH)
    config["data"]["split_type"] = "scaffold"
    config["train"]["device"] = str(args.device)
    config["train"]["require_cuda"] = bool(args.require_cuda)

    outputs = run_formal_full_database_training(config=config, config_path=config_path)
    print(f"mode: {args.mode}")
    print(f"seed: {args.seed}")
    print(f"processed_artifact: {FULL_DATABASE_ARTIFACT_PATH}")
    print(f"scaffold_split: {FULL_DATABASE_SCAFFOLD_SPLIT_PATH}")
    print(f"manifest: {outputs['manifest']}")
    print(f"results: {outputs['results']}")
    print(f"log: {outputs['log']}")
    print(f"best_checkpoint: {outputs['best_checkpoint']}")
    print(f"latest_checkpoint: {outputs['latest_checkpoint']}")
    print("benchmark_result: false")
    print("note: Full-database scaffold-split formal training path; not benchmark results.")


if __name__ == "__main__":
    main()
