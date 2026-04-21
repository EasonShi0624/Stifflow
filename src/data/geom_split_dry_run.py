"""Tiny non-benchmark split dry-run for the real GEOM processed artifact."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.data.splits import SplitError, load_split


EXPECTED_DATASET_KIND = "geom_drugs_tiny_processed_dry_run"
RESULT_KIND = "split_dry_run_not_benchmark"
MICRO_PILOT_HARD_MAX_MOLECULES = 6
DATA_SCALE_CHECK_HARD_MAX_MOLECULES = 30
DATA_SCALE_CHECK_500_HARD_MAX_MOLECULES = 500
DATA_SCALE_CHECK_500_VAL_MOLECULES = 25
DATA_SCALE_CHECK_500_TEST_MOLECULES = 25
DATA_SCALE_CHECK_10000_HARD_MAX_MOLECULES = 10_000
DATA_SCALE_CHECK_10000_VAL_MOLECULES = 500
DATA_SCALE_CHECK_10000_TEST_MOLECULES = 500


def run_geom_tiny_split_dry_run(
    processed_artifact_path: str | Path = "data/processed/geom_drugs_subset/tiny_real_dry_run/molecules.json",
    split_root: str | Path = "data/splits/geom_drugs_tiny_real_dry_run",
    split_type: str = "random",
    seed: int = 0,
    max_molecules: int = 3,
    split_scope: str | None = None,
    hard_max_molecules: int = MICRO_PILOT_HARD_MAX_MOLECULES,
) -> dict[str, Path]:
    if max_molecules > hard_max_molecules:
        raise SplitError(f"Tiny GEOM split dry-run scope is capped at {hard_max_molecules} molecules")
    processed_artifact_path = Path(processed_artifact_path)
    split_root = Path(split_root)
    output_dir = split_root / f"{split_type}_seed{seed}"
    split_path = output_dir / "split.json"
    manifest_path = output_dir / "manifest.json"

    payload = _load_tiny_processed_artifact(processed_artifact_path)
    records = payload["molecules"][:max_molecules]
    if split_scope is None:
        split_scope = payload.get("subset_scope") or (
            "micro_pilot_dry_run" if len(records) > 3 else "tiny_real_data_dry_run"
        )
    split = make_tiny_split(records, split_type=split_type, seed=seed, split_scope=split_scope)
    _validate_tiny_split(split, records)

    manifest = {
        "run_name": f"geom_tiny_{split_type}_split_seed{seed}",
        "dataset_name": "geom_drugs",
        "dataset_kind": EXPECTED_DATASET_KIND,
        "result_kind": RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_cpu_tiny_split_dry_run",
        "hardware": "cpu_only",
        "scheduler": None,
        "slurm_account": None,
        "training_executed": False,
        "raw_data_modified": False,
        "processed_data_modified": False,
        "full_split_written": False,
        "split_dry_run_written": True,
        "processed_artifact_path": str(processed_artifact_path),
        "split_path": str(split_path),
        "split_type": split_type,
        "split_scope": split_scope,
        "seed": seed,
        "bounds": {
            "max_molecules": max_molecules,
            "hard_max_molecules": hard_max_molecules,
            "input_molecule_count": len(payload["molecules"]),
            "written_molecule_count": len(records),
        },
        "counts": {partition: len(split[partition]) for partition in ["train", "val", "test"]},
        "molecule_ids": [record["molecule_id"] for record in records],
        "notes": [
            "This split validates the split-file interface on a tiny real-data dry-run artifact.",
            "It is not a real train/validation/test split and must not be used for benchmarking.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_split(split_path)
    return {"split": split_path, "manifest": manifest_path}


def make_tiny_split(
    records: list[dict[str, Any]],
    split_type: str,
    seed: int,
    split_scope: str = "tiny_real_data_dry_run",
) -> dict[str, Any]:
    if split_type != "random":
        raise SplitError("Tiny GEOM split dry-run currently supports only split_type='random'")
    if not records:
        raise SplitError("Tiny GEOM split dry-run requires at least one molecule record")

    molecule_ids = [record["molecule_id"] for record in records]
    rng = random.Random(seed)
    rng.shuffle(molecule_ids)
    if len(molecule_ids) == 1:
        train, val, test = molecule_ids, [], []
    elif len(molecule_ids) == 2:
        train, val, test = molecule_ids[:1], molecule_ids[1:], []
    elif split_scope == "data_scale_check_500_processed_dry_run" and len(molecule_ids) == 500:
        val_count = DATA_SCALE_CHECK_500_VAL_MOLECULES
        test_count = DATA_SCALE_CHECK_500_TEST_MOLECULES
        train = molecule_ids[: -(val_count + test_count)]
        val = molecule_ids[-(val_count + test_count) : -test_count]
        test = molecule_ids[-test_count:]
    elif split_scope == "data_scale_check_10000_processed_dry_run" and len(molecule_ids) == 10_000:
        val_count = DATA_SCALE_CHECK_10000_VAL_MOLECULES
        test_count = DATA_SCALE_CHECK_10000_TEST_MOLECULES
        train = molecule_ids[: -(val_count + test_count)]
        val = molecule_ids[-(val_count + test_count) : -test_count]
        test = molecule_ids[-test_count:]
    else:
        train, val, test = molecule_ids[:-2], molecule_ids[-2:-1], molecule_ids[-1:]

    return {
        "dataset_name": "geom_drugs",
        "dataset_kind": EXPECTED_DATASET_KIND,
        "result_kind": RESULT_KIND,
        "benchmark_result": False,
        "split_type": split_type,
        "seed": seed,
        "split_scope": split_scope,
        "degenerate_tiny_split": len(molecule_ids) < 3,
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "notes": [
            "All conformers for each molecule remain grouped by molecule_id.",
            "Empty validation/test partitions are allowed only because this is a tiny split dry-run.",
        ],
    }


def _load_tiny_processed_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SplitError(f"Tiny processed artifact not found at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_kind") != EXPECTED_DATASET_KIND:
        raise SplitError(f"Expected dataset_kind={EXPECTED_DATASET_KIND!r}")
    if payload.get("benchmark_result") is not False:
        raise SplitError("Tiny processed artifact must be labeled benchmark_result=false")
    records = payload.get("molecules")
    if not isinstance(records, list) or not records:
        raise SplitError("Tiny processed artifact must contain a non-empty molecules list")
    for record in records:
        if not isinstance(record, dict) or not record.get("molecule_id"):
            raise SplitError("Every tiny processed molecule must contain molecule_id")
        conformers = record.get("conformers")
        if not isinstance(conformers, list) or not conformers:
            raise SplitError(f"{record.get('molecule_id')} has no conformers")
    return payload


def _validate_tiny_split(split: dict[str, Any], records: list[dict[str, Any]]) -> None:
    if split.get("result_kind") != RESULT_KIND:
        raise SplitError(f"Tiny split must be labeled result_kind={RESULT_KIND!r}")
    if split.get("benchmark_result") is not False:
        raise SplitError("Tiny split must be labeled benchmark_result=false")
    record_ids = [record["molecule_id"] for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise SplitError("Tiny processed artifact has duplicate molecule IDs")
    split_ids = split["train"] + split["val"] + split["test"]
    if sorted(split_ids) != sorted(record_ids):
        raise SplitError("Tiny split does not cover exactly the bounded molecule IDs")
    if len(split_ids) != len(set(split_ids)):
        raise SplitError("Tiny split assigns a molecule ID to more than one partition")
