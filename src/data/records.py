"""Smoke-test molecule record loading.

The JSON schema here is intentionally small and explicit. It is not a GEOM
replacement; it is a local contract for wiring config, splitting, corruption,
and metrics before the real dataset path exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DataError(ValueError):
    """Raised when smoke-test molecule records are malformed."""


def load_records(processed_root: str | Path) -> list[dict[str, Any]]:
    root = Path(processed_root)
    path = root / "molecules.json"
    if not path.exists():
        raise DataError(
            f"Processed smoke data not found at {path}. "
            "This smoke scaffold expects a small molecules.json file."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_kind") != "synthetic_smoke":
        raise DataError("Smoke scaffold only accepts dataset_kind='synthetic_smoke'")

    records = payload.get("molecules")
    if not isinstance(records, list) or not records:
        raise DataError("molecules.json must contain a non-empty molecules list")

    for record in records:
        validate_record(record)
    return records


def validate_record(record: dict[str, Any]) -> None:
    for key in ["molecule_id", "smiles", "atom_symbols", "bonds", "conformers"]:
        if key not in record:
            raise DataError(f"Missing molecule record field: {key}")

    atom_count = len(record["atom_symbols"])
    if atom_count == 0:
        raise DataError(f"{record['molecule_id']} has no atoms")

    for bond in record["bonds"]:
        if len(bond) < 2:
            raise DataError(f"{record['molecule_id']} has malformed bond: {bond}")
        i, j = int(bond[0]), int(bond[1])
        if not (0 <= i < atom_count and 0 <= j < atom_count):
            raise DataError(f"{record['molecule_id']} has out-of-range bond: {bond}")

    if not record["conformers"]:
        raise DataError(f"{record['molecule_id']} has no conformers")
    for conf in record["conformers"]:
        coords = conf.get("coords")
        if not isinstance(coords, list) or len(coords) != atom_count:
            raise DataError(f"{record['molecule_id']} conformer has wrong atom count")
        for xyz in coords:
            if not isinstance(xyz, list) or len(xyz) != 3:
                raise DataError(f"{record['molecule_id']} conformer coordinate is not 3D")


def select_records(records: list[dict[str, Any]], max_molecules: int | None) -> list[dict[str, Any]]:
    if max_molecules is None:
        return records
    if max_molecules <= 0:
        raise DataError("max_molecules must be positive")
    return records[:max_molecules]
