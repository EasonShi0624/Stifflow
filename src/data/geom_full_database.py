"""Full raw-tree GEOM-Drugs processed/split preparation for formal training."""

from __future__ import annotations

import json
import pickle
import random
import time
from pathlib import Path
from typing import Any

from src.data.geom_preflight import _build_tiny_processed_record
from src.data.geom_split_dry_run import EXPECTED_DATASET_KIND, RESULT_KIND


FULL_DATABASE_SCOPE = "geom_drugs_full_database_one_conformer"
FULL_DATABASE_PROCESSED_ROOT = Path("data/processed/geom_drugs_full_database")
FULL_DATABASE_ARTIFACT_PATH = FULL_DATABASE_PROCESSED_ROOT / "molecules.json"
FULL_DATABASE_PROCESSED_MANIFEST_PATH = FULL_DATABASE_PROCESSED_ROOT / "manifest.json"
FULL_DATABASE_SPLIT_ROOT = Path("data/splits/geom_drugs_full_database")
FULL_DATABASE_SPLIT_PATH = FULL_DATABASE_SPLIT_ROOT / "random_seed0" / "split.json"
FULL_DATABASE_SPLIT_MANIFEST_PATH = FULL_DATABASE_SPLIT_ROOT / "random_seed0" / "manifest.json"
FULL_DATABASE_PROCESSED_RESULT_KIND = "processed_output_dry_run_not_benchmark"
FULL_DATABASE_SPLIT_RESULT_KIND = RESULT_KIND
FULL_DATABASE_SPLIT_FRACTIONS = {"train": 0.90, "val": 0.05, "test": 0.05}


def prepare_geom_full_database_artifact(
    raw_root: str | Path = "data/raw/geom_drugs",
    processed_root: str | Path = FULL_DATABASE_PROCESSED_ROOT,
    max_conformers_per_mol: int = 1,
    max_pickle_bytes: int | None = None,
    max_atoms: int = 512,
    force: bool = False,
) -> dict[str, Any]:
    """Scan the GEOM-Drugs raw tree and write one processed record per extractable molecule.

    This writes processed metadata/coordinates only; it never modifies raw data.
    The top-level dataset/result markers intentionally preserve the existing
    processed-output schema so downstream loaders remain unchanged.
    """

    raw_root = Path(raw_root)
    processed_root = Path(processed_root)
    artifact_path = processed_root / "molecules.json"
    manifest_path = processed_root / "manifest.json"
    if artifact_path.exists() and manifest_path.exists() and not force:
        artifact = _load_processed_artifact(artifact_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "artifact": artifact_path,
            "manifest": manifest_path,
            "reused_existing": True,
            "molecule_count": int(artifact["molecule_count"]),
            "source_pickle_count": int(manifest.get("source_pickle_count", 0)),
            "skip_reasons": manifest.get("skip_reasons", {}),
        }

    start = time.time()
    drugs_dir = raw_root / "rdkit_folder" / "drugs"
    if not drugs_dir.is_dir():
        raise FileNotFoundError(f"GEOM-Drugs rdkit_folder/drugs directory not found: {drugs_dir}")
    if max_conformers_per_mol != 1:
        raise ValueError("full-database formal path preserves one conformer per molecule")
    if max_atoms <= 0:
        raise ValueError("max_atoms must be positive")

    pickle_files = sorted(drugs_dir.glob("*.pickle"), key=lambda path: path.name)
    if not pickle_files:
        raise FileNotFoundError(f"No GEOM-Drugs pickle files found under {drugs_dir}")

    molecules: list[dict[str, Any]] = []
    source_examples: list[dict[str, Any]] = []
    skip_reasons: dict[str, int] = {}
    bytes_read = 0
    pickle_files_considered = 0
    for path in pickle_files:
        pickle_files_considered += 1
        size_bytes = path.stat().st_size
        if max_pickle_bytes is not None and size_bytes > max_pickle_bytes:
            _record_skip(skip_reasons, "pickle_file_exceeds_max_bytes")
            continue
        try:
            with path.open("rb") as handle:
                raw_record = pickle.load(handle)
            bytes_read += size_bytes
        except Exception as exc:  # pragma: no cover - local raw-data failures depend on RDKit/pickle state.
            _record_skip(skip_reasons, f"pickle_load_error:{type(exc).__name__}")
            continue
        processed = _build_tiny_processed_record(
            raw_record=raw_record,
            source_file=path.relative_to(raw_root),
            max_conformers_per_mol=max_conformers_per_mol,
            max_atoms=max_atoms,
        )
        if processed is None:
            _record_skip(skip_reasons, "no_extractable_rdkit_conformer_within_bounds")
            continue
        molecules.append(processed)
        if len(source_examples) < 25:
            source_examples.append(
                {
                    "source_file": str(path.relative_to(raw_root)),
                    "size_bytes": size_bytes,
                    "molecule_id": processed["molecule_id"],
                    "conformers_written": len(processed["conformers"]),
                    "atom_count": len(processed["atom_symbols"]),
                }
            )

    if not molecules:
        raise ValueError("No extractable GEOM-Drugs molecules were found during full-database preparation")

    conformer_count = sum(len(record["conformers"]) for record in molecules)
    artifact = {
        "dataset_kind": EXPECTED_DATASET_KIND,
        "result_kind": FULL_DATABASE_PROCESSED_RESULT_KIND,
        "benchmark_result": False,
        "description": "Full raw-tree GEOM-Drugs one-conformer processed artifact for first formal training. Not benchmark data.",
        "subset_scope": FULL_DATABASE_SCOPE,
        "source_dataset": "geom_drugs",
        "raw_root": str(raw_root),
        "bounds": {
            "max_molecules": None,
            "hard_max_molecules": None,
            "source_pickle_count": len(pickle_files),
            "pickle_files_considered": pickle_files_considered,
            "max_conformers_per_mol": max_conformers_per_mol,
            "max_pickle_bytes": max_pickle_bytes,
            "max_atoms": max_atoms,
            "one_conformer_per_molecule": True,
        },
        "molecule_count": len(molecules),
        "conformer_count": conformer_count,
        "molecules": molecules,
        "notes": _processed_notes(),
    }
    manifest = {
        "run_name": "geom_full_database_one_conformer_processed",
        "dataset_name": "geom_drugs",
        "dataset_kind": EXPECTED_DATASET_KIND,
        "subset_scope": FULL_DATABASE_SCOPE,
        "result_kind": FULL_DATABASE_PROCESSED_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "scheduled_cpu_full_database_processed_preparation",
        "hardware": "cpu_only",
        "scheduler": None,
        "slurm_account": None,
        "training_executed": False,
        "raw_data_modified": False,
        "processed_data_written": True,
        "full_raw_tree_scanned": True,
        "split_files_written": False,
        "artifact_path": str(artifact_path),
        "raw_root": str(raw_root),
        "source_layout": "rdkit_folder/drugs_pickle",
        "source_pickle_count": len(pickle_files),
        "pickle_files_considered": pickle_files_considered,
        "pickle_files_skipped": sum(skip_reasons.values()),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "raw_pickle_bytes_read": bytes_read,
        "molecule_count": len(molecules),
        "conformer_count": conformer_count,
        "source_examples": source_examples,
        "field_contract": {
            "molecule_fields": sorted(molecules[0].keys()),
            "conformer_fields": sorted(molecules[0]["conformers"][0].keys()),
        },
        "bounds": artifact["bounds"],
        "runtime_seconds": time.time() - start,
        "notes": _processed_notes(),
    }

    processed_root.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "artifact": artifact_path,
        "manifest": manifest_path,
        "reused_existing": False,
        "molecule_count": len(molecules),
        "source_pickle_count": len(pickle_files),
        "skip_reasons": manifest["skip_reasons"],
    }


def prepare_geom_full_database_split(
    processed_artifact_path: str | Path = FULL_DATABASE_ARTIFACT_PATH,
    split_root: str | Path = FULL_DATABASE_SPLIT_ROOT,
    seed: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    """Write the full-database random split with the existing 90/5/5 logic."""

    processed_artifact_path = Path(processed_artifact_path)
    split_root = Path(split_root)
    output_dir = split_root / f"random_seed{seed}"
    split_path = output_dir / "split.json"
    manifest_path = output_dir / "manifest.json"
    if split_path.exists() and manifest_path.exists() and not force:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        _validate_full_database_split(split)
        return {
            "split": split_path,
            "manifest": manifest_path,
            "reused_existing": True,
            "counts": {partition: len(split[partition]) for partition in ["train", "val", "test"]},
        }

    start = time.time()
    artifact = _load_processed_artifact(processed_artifact_path)
    records = artifact["molecules"]
    molecule_ids = [record["molecule_id"] for record in records]
    if len(molecule_ids) != len(set(molecule_ids)):
        raise ValueError("Full-database processed artifact has duplicate molecule IDs")
    if len(molecule_ids) < 20:
        raise ValueError("Full-database split preparation requires at least 20 molecules")

    shuffled = list(molecule_ids)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * FULL_DATABASE_SPLIT_FRACTIONS["val"]))
    test_count = max(1, round(len(shuffled) * FULL_DATABASE_SPLIT_FRACTIONS["test"]))
    train_count = len(shuffled) - val_count - test_count
    if train_count <= 0:
        raise ValueError("Full-database split fractions leave no training molecules")
    train = shuffled[:train_count]
    val = shuffled[train_count : train_count + val_count]
    test = shuffled[train_count + val_count :]
    split = {
        "dataset_name": "geom_drugs",
        "dataset_kind": EXPECTED_DATASET_KIND,
        "result_kind": FULL_DATABASE_SPLIT_RESULT_KIND,
        "benchmark_result": False,
        "split_type": "random",
        "seed": int(seed),
        "split_scope": FULL_DATABASE_SCOPE,
        "degenerate_tiny_split": False,
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "notes": [
            "Full-database formal-training random split using the same 90/5/5 construction as the 10000-molecule formal path.",
            "This split is prepared for a first formal training run and is not a benchmark claim.",
        ],
    }
    _validate_full_database_split(split)
    manifest = {
        "run_name": f"geom_full_database_random_split_seed{seed}",
        "dataset_name": "geom_drugs",
        "dataset_kind": EXPECTED_DATASET_KIND,
        "result_kind": FULL_DATABASE_SPLIT_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "scheduled_cpu_full_database_split_preparation",
        "hardware": "cpu_only",
        "scheduler": None,
        "slurm_account": None,
        "training_executed": False,
        "raw_data_modified": False,
        "processed_data_modified": False,
        "full_split_written": True,
        "processed_artifact_path": str(processed_artifact_path),
        "split_path": str(split_path),
        "split_type": "random",
        "split_scope": FULL_DATABASE_SCOPE,
        "seed": int(seed),
        "split_fractions": FULL_DATABASE_SPLIT_FRACTIONS,
        "counts": {partition: len(split[partition]) for partition in ["train", "val", "test"]},
        "input_molecule_count": len(molecule_ids),
        "runtime_seconds": time.time() - start,
        "notes": split["notes"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "split": split_path,
        "manifest": manifest_path,
        "reused_existing": False,
        "counts": manifest["counts"],
    }


def _load_processed_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Full-database processed artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_kind") != EXPECTED_DATASET_KIND:
        raise ValueError(f"Expected dataset_kind={EXPECTED_DATASET_KIND!r}")
    if payload.get("result_kind") != FULL_DATABASE_PROCESSED_RESULT_KIND:
        raise ValueError(f"Expected result_kind={FULL_DATABASE_PROCESSED_RESULT_KIND!r}")
    if payload.get("benchmark_result") is not False:
        raise ValueError("Full-database processed artifact must be labeled benchmark_result=false")
    if payload.get("subset_scope") != FULL_DATABASE_SCOPE:
        raise ValueError(f"Expected subset_scope={FULL_DATABASE_SCOPE!r}")
    records = payload.get("molecules")
    if not isinstance(records, list) or not records:
        raise ValueError("Full-database processed artifact must contain a non-empty molecules list")
    return payload


def _validate_full_database_split(split: dict[str, Any]) -> None:
    if split.get("dataset_kind") != EXPECTED_DATASET_KIND:
        raise ValueError(f"Expected split dataset_kind={EXPECTED_DATASET_KIND!r}")
    if split.get("result_kind") != FULL_DATABASE_SPLIT_RESULT_KIND:
        raise ValueError(f"Expected split result_kind={FULL_DATABASE_SPLIT_RESULT_KIND!r}")
    if split.get("benchmark_result") is not False:
        raise ValueError("Full-database split must be labeled benchmark_result=false")
    if split.get("split_scope") != FULL_DATABASE_SCOPE:
        raise ValueError(f"Expected split_scope={FULL_DATABASE_SCOPE!r}")
    if split.get("split_type") != "random":
        raise ValueError("Full-database split currently supports only split_type=random")
    if split.get("degenerate_tiny_split") is not False:
        raise ValueError("Full-database split must be nondegenerate")
    split_ids = split["train"] + split["val"] + split["test"]
    if not split["train"] or not split["val"] or not split["test"]:
        raise ValueError("Full-database split requires non-empty train/val/test partitions")
    if len(split_ids) != len(set(split_ids)):
        raise ValueError("Full-database split assigns a molecule more than once")


def _record_skip(skip_reasons: dict[str, int], reason: str) -> None:
    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1


def _processed_notes() -> list[str]:
    return [
        "This file scans the full available GEOM-Drugs rdkit_folder/drugs raw tree and extracts one conformer per molecule when possible.",
        "It preserves the existing processed-output schema for downstream compatibility.",
        "It is a prepared formal-training input and is not a benchmark result.",
        "Raw data is read only and never modified.",
    ]
