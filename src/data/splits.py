"""Molecule-level split helpers for smoke tests."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


class SplitError(ValueError):
    """Raised when split construction or loading fails."""


def split_file_path(split_root: str | Path, split_type: str, dataset_name: str, seed: int) -> Path:
    return Path(split_root) / split_type / f"{dataset_name}_seed{seed}.json"


def load_or_create_split(
    records: list[dict[str, Any]],
    split_root: str | Path,
    split_type: str,
    dataset_name: str,
    seed: int,
) -> dict[str, Any]:
    path = split_file_path(split_root, split_type, dataset_name, seed)
    if path.exists():
        return load_split(path)

    if split_type == "random":
        split = make_random_split(records, dataset_name=dataset_name, seed=seed)
    elif split_type == "scaffold":
        split = make_scaffold_split(records, dataset_name=dataset_name, seed=seed)
    else:
        raise SplitError(f"Unsupported split type: {split_type}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return split


def load_split(path: str | Path) -> dict[str, Any]:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ["train", "val", "test"]:
        if key not in split or not isinstance(split[key], list):
            raise SplitError(f"Split file is missing list field: {key}")
    molecule_ids = split["train"] + split["val"] + split["test"]
    if len(molecule_ids) != len(set(molecule_ids)):
        raise SplitError("Molecule IDs appear in more than one split partition")
    return split


def make_random_split(
    records: list[dict[str, Any]],
    dataset_name: str,
    seed: int,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, Any]:
    molecule_ids = [record["molecule_id"] for record in records]
    rng = random.Random(seed)
    rng.shuffle(molecule_ids)
    train_ids, val_ids, test_ids = _partition_ids(molecule_ids, ratios)
    return _split_payload("random", dataset_name, seed, train_ids, val_ids, test_ids)


def make_scaffold_split(
    records: list[dict[str, Any]],
    dataset_name: str,
    seed: int,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, Any]:
    groups: dict[str, list[str]] = {}
    for record in records:
        scaffold = record.get("scaffold")
        if not scaffold:
            scaffold = _rdkit_scaffold(record.get("smiles", ""))
        groups.setdefault(scaffold, []).append(record["molecule_id"])

    grouped_ids = list(groups.values())
    rng = random.Random(seed)
    rng.shuffle(grouped_ids)

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    target_train = max(1, round(len(records) * ratios[0]))
    target_val = max(1, round(len(records) * ratios[1]))

    for group in grouped_ids:
        if len(train) < target_train:
            train.extend(group)
        elif len(val) < target_val:
            val.extend(group)
        else:
            test.extend(group)
    return _split_payload("scaffold", dataset_name, seed, sorted(train), sorted(val), sorted(test))


def _partition_ids(
    molecule_ids: list[str], ratios: tuple[float, float, float]
) -> tuple[list[str], list[str], list[str]]:
    if len(molecule_ids) < 3:
        raise SplitError("At least three molecules are required for train/val/test split")
    n_total = len(molecule_ids)
    n_train = max(1, round(n_total * ratios[0]))
    n_val = max(1, round(n_total * ratios[1]))
    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = 1
    return (
        sorted(molecule_ids[:n_train]),
        sorted(molecule_ids[n_train : n_train + n_val]),
        sorted(molecule_ids[n_train + n_val :]),
    )


def _split_payload(
    split_type: str,
    dataset_name: str,
    seed: int,
    train: list[str],
    val: list[str],
    test: list[str],
) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "split_type": split_type,
        "seed": seed,
        "train": train,
        "val": val,
        "test": test,
    }


def _rdkit_scaffold(smiles: str) -> str:
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore
    except ImportError as exc:
        raise SplitError(
            "Scaffold split requires either a 'scaffold' field in every smoke record "
            "or an environment with RDKit available."
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SplitError(f"RDKit could not parse SMILES for scaffold split: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or smiles
