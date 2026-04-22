#!/usr/bin/env python
"""Run one full-database GEOM-Drugs formal training job for one mode/seed."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.geom_full_database import prepare_geom_full_database_artifact, prepare_geom_full_database_split
from src.train.formal_full_database_training import run_formal_full_database_training
from src.utils.config import load_config


CONFIG_BY_MODE = {
    "isotropic": Path("configs/train/geom_full_database_formal_isotropic.yaml"),
    "stiffness_structured": Path("configs/train/geom_full_database_formal_stiffness.yaml"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(CONFIG_BY_MODE), required=True, help="Formal source-corruption mode.")
    parser.add_argument("--seed", type=int, choices=[1, 2], required=True, help="Prepared full-database seed.")
    parser.add_argument("--raw-root", default="data/raw/geom_drugs", help="GEOM-Drugs raw root.")
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

    run_label = _run_label(args.mode, args.seed)
    processed_root = Path("data/processed/geom_drugs_full_database_runs") / run_label
    split_root = Path("data/splits/geom_drugs_full_database_runs") / run_label
    processed_artifact = processed_root / "molecules.json"
    split_path = split_root / "random_seed0" / "split.json"
    max_pickle_bytes = None if args.max_pickle_bytes <= 0 else int(args.max_pickle_bytes)

    processed = prepare_geom_full_database_artifact(
        raw_root=args.raw_root,
        processed_root=processed_root,
        max_conformers_per_mol=1,
        max_pickle_bytes=max_pickle_bytes,
        max_atoms=args.max_processed_atoms,
        force=True,
    )
    split = prepare_geom_full_database_split(
        processed_artifact_path=processed_artifact,
        split_root=split_root,
        seed=0,
        force=True,
    )

    config = copy.deepcopy(load_config(CONFIG_BY_MODE[args.mode]))
    config["experiment"]["seed"] = int(args.seed)
    config["experiment"]["output_root"] = str(args.output_root)
    config["data"]["processed_root"] = str(processed_root)
    config["data"]["split_root"] = str(split_root / "random_seed0")
    config["data"]["processed_artifact_path"] = str(processed_artifact)
    config["data"]["split_path"] = str(split_path)
    config["train"]["device"] = str(args.device)
    config["train"]["require_cuda"] = bool(args.require_cuda)

    outputs = run_formal_full_database_training(config=config, config_path=CONFIG_BY_MODE[args.mode])
    print(f"mode: {args.mode}")
    print(f"seed: {args.seed}")
    print(f"processed_artifact: {processed['artifact']}")
    print(f"processed_manifest: {processed['manifest']}")
    print(f"processed_reused_existing: {processed['reused_existing']}")
    print(f"processed_molecule_count: {processed['molecule_count']}")
    print(f"source_pickle_count: {processed['source_pickle_count']}")
    print(f"split: {split['split']}")
    print(f"split_manifest: {split['manifest']}")
    print(f"split_reused_existing: {split['reused_existing']}")
    print(f"split_counts: {split['counts']}")
    print(f"manifest: {outputs['manifest']}")
    print(f"results: {outputs['results']}")
    print(f"log: {outputs['log']}")
    print(f"best_checkpoint: {outputs['best_checkpoint']}")
    print(f"latest_checkpoint: {outputs['latest_checkpoint']}")


def _run_label(mode: str, seed: int) -> str:
    label = "stiffness" if mode == "stiffness_structured" else "isotropic"
    return f"{label}_seed{int(seed)}"


if __name__ == "__main__":
    main()
