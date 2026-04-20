"""Read-only GEOM-Drugs raw-data preflight and preprocessing dry-run."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any


EXPECTED_LAYOUT_HINTS = [
    "summary_drugs.json",
    "rdkit_folder",
    "rdkit_folder/summary_drugs.json",
    "rdkit_folder/summary_qm9.json",
    "rdkit_folder/drugs",
    "rdkit_folder/qm9",
    "drugs_crude.msgpack",
    "drugs.msgpack",
    "molecules",
    "mols",
]
TINY_PROCESSED_HARD_MAX_MOLECULES = 6
DATA_SCALE_CHECK_HARD_MAX_MOLECULES = 30


def run_geom_preflight(
    raw_root: str | Path = "data/raw/geom_drugs",
    processed_root: str | Path = "data/processed/geom_drugs_subset",
    manifest_path: str | Path = "experiments/manifests/geom_drugs_preflight.json",
    report_path: str | Path = "reports/tables/geom_drugs_preflight.md",
    max_scan_entries: int = 200,
    max_dry_run_molecules: int = 2,
    max_dry_run_conformers_per_mol: int = 3,
    max_pickle_bytes: int = 1_000_000,
    write_tiny_processed_output: bool = False,
    write_micro_pilot_output: bool = False,
    write_data_scale_check_output: bool = False,
    tiny_processed_tag: str = "tiny_real_dry_run",
    max_processed_molecules: int = 3,
    max_processed_conformers_per_mol: int = 1,
    max_processed_atoms: int = 128,
) -> dict[str, Path]:
    start = time.time()
    raw_root = Path(raw_root)
    processed_root = Path(processed_root)
    manifest_path = Path(manifest_path)
    report_path = Path(report_path)
    if write_micro_pilot_output:
        write_tiny_processed_output = True
        if tiny_processed_tag == "tiny_real_dry_run":
            tiny_processed_tag = "micro_pilot"
        if max_processed_molecules == 3:
            max_processed_molecules = TINY_PROCESSED_HARD_MAX_MOLECULES
    if write_data_scale_check_output:
        write_tiny_processed_output = True
        if tiny_processed_tag == "tiny_real_dry_run":
            tiny_processed_tag = "data_scale_check"
        if max_processed_molecules == 3:
            max_processed_molecules = DATA_SCALE_CHECK_HARD_MAX_MOLECULES
    if write_data_scale_check_output:
        output_scope = "data_scale_check_processed_dry_run"
        hard_max_molecules = DATA_SCALE_CHECK_HARD_MAX_MOLECULES
    elif write_micro_pilot_output:
        output_scope = "micro_pilot_processed_dry_run"
        hard_max_molecules = TINY_PROCESSED_HARD_MAX_MOLECULES
    else:
        output_scope = "tiny_real_processed_dry_run"
        hard_max_molecules = TINY_PROCESSED_HARD_MAX_MOLECULES

    raw_summary = inspect_raw_root(raw_root, max_scan_entries=max_scan_entries)
    preprocessing_dry_run = inspect_preprocessing_dry_run(
        raw_root=raw_root,
        max_molecules=max_dry_run_molecules,
        max_conformers_per_mol=max_dry_run_conformers_per_mol,
        max_pickle_bytes=max_pickle_bytes,
    )
    tiny_processed_output = write_tiny_processed_dry_run(
        raw_root=raw_root,
        processed_root=processed_root,
        output_tag=tiny_processed_tag,
        enabled=write_tiny_processed_output,
        max_molecules=max_processed_molecules,
        max_conformers_per_mol=max_processed_conformers_per_mol,
        max_pickle_bytes=max_pickle_bytes,
        max_atoms=max_processed_atoms,
        output_scope=output_scope,
        hard_max_molecules=hard_max_molecules,
    )
    manifest = {
        "run_name": "geom_drugs_preflight",
        "dataset_name": "geom_drugs",
        "dataset_kind": "raw_data_preflight",
        "execution_mode": "local_read_only_preflight",
        "result_kind": "preflight_not_benchmark",
        "benchmark_result": False,
        "hardware": "cpu_only",
        "scheduler": None,
        "slurm_account": None,
        "training_executed": False,
        "raw_data_modified": False,
        "processed_data_written": tiny_processed_output["written"],
        "processed_output_dry_run_written": tiny_processed_output["written"],
        "preprocessing_executed": False,
        "preprocessing_dry_run_executed": preprocessing_dry_run["executed"],
        "dry_run": True,
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "raw_summary": raw_summary,
        "preprocessing_dry_run": preprocessing_dry_run,
        "tiny_processed_output": tiny_processed_output,
        "dry_run_plan": {
            "would_read_from": str(raw_root),
            "would_write_processed_records_to": str(processed_root),
            "would_create_split_files": False,
            "would_modify_raw_data": False,
            "would_write_full_processed_data": False,
            "would_write_tiny_processed_dry_run": write_tiny_processed_output,
            "would_write_micro_pilot_processed_dry_run": write_micro_pilot_output,
            "would_write_data_scale_check_processed_dry_run": write_data_scale_check_output,
            "next_real_step": "Define and review the processed GEOM record schema before writing full processed data.",
        },
        "runtime_seconds": time.time() - start,
        "notes": [
            "This is a read-only preflight and preprocessing dry-run.",
            "Only a tiny bounded subset of raw GEOM-Drugs pickle metadata was read.",
            "No raw data, full processed data, split files, checkpoints, or training outputs were modified.",
        ],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(manifest), encoding="utf-8")
    paths = {"manifest": manifest_path, "report": report_path}
    if tiny_processed_output["written"]:
        paths["processed_artifact"] = Path(tiny_processed_output["artifact_path"])
        paths["processed_manifest"] = Path(tiny_processed_output["manifest_path"])
    return paths


def inspect_raw_root(raw_root: Path, max_scan_entries: int = 200) -> dict[str, Any]:
    exists = raw_root.exists()
    is_dir = raw_root.is_dir()
    summary: dict[str, Any] = {
        "exists": exists,
        "is_dir": is_dir,
        "status": "missing_root",
        "expected_layout_hints": EXPECTED_LAYOUT_HINTS,
        "present_layout_hints": [],
        "missing_layout_hints": EXPECTED_LAYOUT_HINTS,
        "top_level_entries": [],
        "sample_files": [],
        "sample_dirs": [],
        "extension_counts": {},
        "scanned_entry_count": 0,
        "scan_truncated": False,
    }
    if not exists:
        summary["blocking_missing_items"] = [str(raw_root)]
        return summary
    if not is_dir:
        summary["status"] = "raw_root_not_directory"
        summary["blocking_missing_items"] = [f"{raw_root} is not a directory"]
        return summary

    top_level = sorted(raw_root.iterdir(), key=lambda path: path.name)
    summary["top_level_entries"] = [_entry_payload(path, raw_root) for path in top_level[:max_scan_entries]]
    summary["scan_truncated"] = len(top_level) > max_scan_entries

    present_hints = [hint for hint in EXPECTED_LAYOUT_HINTS if (raw_root / hint).exists()]
    summary["present_layout_hints"] = present_hints
    summary["missing_layout_hints"] = [hint for hint in EXPECTED_LAYOUT_HINTS if hint not in present_hints]

    sample_files: list[str] = []
    sample_dirs: list[str] = []
    extension_counts: dict[str, int] = {}
    scanned = 0
    for path in _bounded_walk(raw_root, max_entries=max_scan_entries):
        scanned += 1
        rel = str(path.relative_to(raw_root))
        if path.is_dir():
            if len(sample_dirs) < 25:
                sample_dirs.append(rel)
        elif path.is_file():
            if len(sample_files) < 25:
                sample_files.append(rel)
            suffix = path.suffix.lower() or "<no_ext>"
            extension_counts[suffix] = extension_counts.get(suffix, 0) + 1

    summary["sample_files"] = sample_files
    summary["sample_dirs"] = sample_dirs
    summary["extension_counts"] = dict(sorted(extension_counts.items()))
    summary["scanned_entry_count"] = scanned
    summary["scan_truncated"] = summary["scan_truncated"] or scanned >= max_scan_entries

    if present_hints:
        summary["status"] = "present_candidate_layout"
        summary["blocking_missing_items"] = []
    elif sample_files or sample_dirs:
        summary["status"] = "present_unrecognized_layout"
        summary["blocking_missing_items"] = ["No known GEOM-Drugs layout hint was detected."]
    else:
        summary["status"] = "present_empty"
        summary["blocking_missing_items"] = ["Raw root exists but contains no visible entries."]
    return summary


def inspect_preprocessing_dry_run(
    raw_root: Path,
    max_molecules: int = 2,
    max_conformers_per_mol: int = 3,
    max_pickle_bytes: int = 1_000_000,
) -> dict[str, Any]:
    bounds = {
        "max_molecules": max_molecules,
        "max_conformers_per_mol": max_conformers_per_mol,
        "max_pickle_bytes": max_pickle_bytes,
        "max_pickle_files_considered": max(1, max_molecules * 20),
    }
    summary: dict[str, Any] = {
        "executed": False,
        "status": "not_started",
        "source_layout": "rdkit_folder/drugs_pickle",
        "source_dir": "rdkit_folder/drugs",
        "bounds": bounds,
        "pickle_files_considered": 0,
        "pickle_files_skipped": 0,
        "skip_reasons": {},
        "molecule_metadata_records_read": 0,
        "conformer_metadata_records_read": 0,
        "raw_pickle_bytes_read": 0,
        "sample_records": [],
        "raw_data_modified": False,
        "processed_data_written": False,
        "split_files_written": False,
    }
    drugs_dir = raw_root / "rdkit_folder" / "drugs"
    if max_molecules <= 0:
        summary["status"] = "disabled_by_zero_molecule_bound"
        return summary
    if not drugs_dir.is_dir():
        summary["status"] = "missing_rdkit_drugs_dir"
        summary["blocking_missing_items"] = [str(drugs_dir)]
        return summary

    summary["executed"] = True
    for path in _iter_limited_pickle_files(drugs_dir, bounds["max_pickle_files_considered"]):
        if len(summary["sample_records"]) >= max_molecules:
            break
        summary["pickle_files_considered"] += 1
        size_bytes = path.stat().st_size
        if size_bytes > max_pickle_bytes:
            _record_skip(summary, "pickle_file_exceeds_max_bytes")
            continue
        try:
            with path.open("rb") as handle:
                raw_record = pickle.load(handle)
        except Exception as exc:  # pragma: no cover - exact pickle failures depend on local RDKit.
            _record_skip(summary, f"pickle_load_error:{type(exc).__name__}")
            continue

        summary["raw_pickle_bytes_read"] += size_bytes
        record = _summarize_geom_pickle(
            raw_record=raw_record,
            path=path,
            raw_root=raw_root,
            size_bytes=size_bytes,
            max_conformers_per_mol=max_conformers_per_mol,
        )
        summary["sample_records"].append(record)
        summary["molecule_metadata_records_read"] += 1
        summary["conformer_metadata_records_read"] += record["sampled_conformer_count"]

    if summary["sample_records"]:
        summary["status"] = "completed_metadata_only"
    elif summary["pickle_files_considered"]:
        summary["status"] = "no_records_read_within_bounds"
    else:
        summary["status"] = "no_pickle_files_found"
    return summary


def write_tiny_processed_dry_run(
    raw_root: Path,
    processed_root: Path,
    output_tag: str = "tiny_real_dry_run",
    enabled: bool = False,
    max_molecules: int = 3,
    max_conformers_per_mol: int = 1,
    max_pickle_bytes: int = 1_000_000,
    max_atoms: int = 128,
    output_scope: str = "tiny_real_processed_dry_run",
    hard_max_molecules: int = TINY_PROCESSED_HARD_MAX_MOLECULES,
) -> dict[str, Any]:
    if max_molecules > hard_max_molecules:
        raise ValueError(f"Tiny processed dry-run scope {output_scope!r} is capped at {hard_max_molecules} molecules")
    output_dir = processed_root / output_tag
    artifact_path = output_dir / "molecules.json"
    manifest_path = output_dir / "manifest.json"
    bounds = {
        "max_molecules": max_molecules,
        "max_conformers_per_mol": max_conformers_per_mol,
        "max_pickle_bytes": max_pickle_bytes,
        "max_atoms": max_atoms,
        "max_pickle_files_considered": max(1, max_molecules * 20),
        "hard_max_molecules": hard_max_molecules,
    }
    summary: dict[str, Any] = {
        "enabled": enabled,
        "written": False,
        "status": "disabled",
        "output_dir": str(output_dir),
        "artifact_path": str(artifact_path),
        "manifest_path": str(manifest_path),
        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
        "subset_scope": output_scope,
        "result_kind": "processed_output_dry_run_not_benchmark",
        "benchmark_result": False,
        "bounds": bounds,
        "raw_data_modified": False,
        "full_processed_data_written": False,
        "split_files_written": False,
        "molecule_count": 0,
        "conformer_count": 0,
        "source_files": [],
        "pickle_files_considered": 0,
        "pickle_files_skipped": 0,
        "skip_reasons": {},
    }
    if not enabled:
        return summary
    if max_molecules <= 0:
        summary["status"] = "disabled_by_zero_molecule_bound"
        return summary

    drugs_dir = raw_root / "rdkit_folder" / "drugs"
    if not drugs_dir.is_dir():
        summary["status"] = "missing_rdkit_drugs_dir"
        summary["blocking_missing_items"] = [str(drugs_dir)]
        return summary

    molecules: list[dict[str, Any]] = []
    for path in _iter_limited_pickle_files(drugs_dir, bounds["max_pickle_files_considered"]):
        if len(molecules) >= max_molecules:
            break
        summary["pickle_files_considered"] += 1
        size_bytes = path.stat().st_size
        if size_bytes > max_pickle_bytes:
            _record_skip(summary, "pickle_file_exceeds_max_bytes")
            continue
        try:
            with path.open("rb") as handle:
                raw_record = pickle.load(handle)
        except Exception as exc:  # pragma: no cover - exact pickle failures depend on local RDKit.
            _record_skip(summary, f"pickle_load_error:{type(exc).__name__}")
            continue

        processed = _build_tiny_processed_record(
            raw_record=raw_record,
            source_file=path.relative_to(raw_root),
            max_conformers_per_mol=max_conformers_per_mol,
            max_atoms=max_atoms,
        )
        if processed is None:
            _record_skip(summary, "no_extractable_rdkit_conformer_within_bounds")
            continue
        molecules.append(processed)
        summary["source_files"].append(
            {
                "source_file": str(path.relative_to(raw_root)),
                "size_bytes": size_bytes,
                "molecule_id": processed["molecule_id"],
                "conformers_written": len(processed["conformers"]),
                "atom_count": len(processed["atom_symbols"]),
            }
        )

    if not molecules:
        summary["status"] = "no_records_written_within_bounds"
        return summary

    conformer_count = sum(len(record["conformers"]) for record in molecules)
    artifact = {
        "dataset_kind": summary["dataset_kind"],
        "result_kind": summary["result_kind"],
        "benchmark_result": False,
        "description": "Tiny bounded real GEOM-Drugs processed-output dry-run. Not benchmark data.",
        "subset_scope": output_scope,
        "source_dataset": "geom_drugs",
        "raw_root": str(raw_root),
        "bounds": bounds,
        "molecule_count": len(molecules),
        "conformer_count": conformer_count,
        "molecules": molecules,
        "notes": [
            "This file is intentionally tiny and exists only to validate real GEOM extraction wiring.",
            f"Subset scope: {output_scope}.",
            "It must not be used as a benchmark dataset or training corpus.",
        ],
    }
    manifest = {
        "run_name": output_tag,
        "dataset_name": "geom_drugs",
        "dataset_kind": summary["dataset_kind"],
        "subset_scope": output_scope,
        "result_kind": summary["result_kind"],
        "benchmark_result": False,
        "execution_mode": "local_cpu_tiny_processed_output_dry_run",
        "hardware": "cpu_only",
        "scheduler": None,
        "slurm_account": None,
        "training_executed": False,
        "raw_data_modified": False,
        "processed_data_written": True,
        "full_processed_data_written": False,
        "split_files_written": False,
        "artifact_path": str(artifact_path),
        "bounds": bounds,
        "molecule_count": len(molecules),
        "conformer_count": conformer_count,
        "source_files": summary["source_files"],
        "field_contract": {
            "molecule_fields": sorted(molecules[0].keys()),
            "conformer_fields": sorted(molecules[0]["conformers"][0].keys()),
        },
        "notes": artifact["notes"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["written"] = True
    summary["status"] = "completed_tiny_processed_output"
    summary["molecule_count"] = len(molecules)
    summary["conformer_count"] = conformer_count
    return summary


def _build_tiny_processed_record(
    raw_record: Any,
    source_file: Path,
    max_conformers_per_mol: int,
    max_atoms: int,
) -> dict[str, Any] | None:
    if not isinstance(raw_record, dict):
        return None
    conformers = raw_record.get("conformers")
    if not isinstance(conformers, list):
        return None

    processed_conformers: list[dict[str, Any]] = []
    atom_symbols: list[str] | None = None
    atomic_numbers: list[int] | None = None
    formal_charges: list[int] | None = None
    ring_atom_indices: list[int] | None = None
    bonds: list[list[int | float]] | None = None
    for conformer in conformers:
        if len(processed_conformers) >= max_conformers_per_mol:
            break
        if not isinstance(conformer, dict):
            continue
        rd_mol = conformer.get("rd_mol")
        if not _looks_like_rdkit_mol(rd_mol):
            continue
        atom_count = rd_mol.GetNumAtoms()
        if atom_count <= 0 or atom_count > max_atoms or rd_mol.GetNumConformers() <= 0:
            continue

        if atom_symbols is None:
            atom_symbols = [atom.GetSymbol() for atom in rd_mol.GetAtoms()]
            atomic_numbers = [int(atom.GetAtomicNum()) for atom in rd_mol.GetAtoms()]
            formal_charges = [int(atom.GetFormalCharge()) for atom in rd_mol.GetAtoms()]
            ring_atom_indices = [atom.GetIdx() for atom in rd_mol.GetAtoms() if atom.IsInRing()]
            bonds = [
                [
                    int(bond.GetBeginAtomIdx()),
                    int(bond.GetEndAtomIdx()),
                    _json_bond_order(bond.GetBondTypeAsDouble()),
                ]
                for bond in rd_mol.GetBonds()
            ]

        coords = _coords_from_rdkit_mol(rd_mol)
        if len(coords) != len(atom_symbols):
            continue
        processed_conformers.append(
            {
                "conformer_id": str(conformer.get("geom_id", len(processed_conformers))),
                "coords": coords,
                "source_metadata": _json_safe_metadata(
                    conformer,
                    exclude_keys={"rd_mol", "mol", "coords", "coordinates", "positions"},
                ),
            }
        )

    if not processed_conformers or atom_symbols is None:
        return None
    return {
        "molecule_id": Path(str(source_file)).stem,
        "smiles": raw_record.get("smiles"),
        "source_file": str(source_file),
        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
        "atom_symbols": atom_symbols,
        "atomic_numbers": atomic_numbers,
        "formal_charges": formal_charges,
        "bonds": bonds,
        "ring_atom_indices": ring_atom_indices,
        "rotatable_bond_count": None,
        "conformers": processed_conformers,
        "source_metadata": _json_safe_metadata(raw_record, exclude_keys={"conformers"}),
    }


def _looks_like_rdkit_mol(value: Any) -> bool:
    return all(
        hasattr(value, method)
        for method in [
            "GetNumAtoms",
            "GetNumConformers",
            "GetAtoms",
            "GetBonds",
            "GetConformer",
        ]
    )


def _coords_from_rdkit_mol(rd_mol: Any) -> list[list[float]]:
    conformer = rd_mol.GetConformer()
    coords: list[list[float]] = []
    for atom_index in range(rd_mol.GetNumAtoms()):
        point = conformer.GetAtomPosition(atom_index)
        coords.append([round(float(point.x), 6), round(float(point.y), 6), round(float(point.z), 6)])
    return coords


def _json_bond_order(order: float) -> int | float:
    rounded = round(float(order), 3)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def _iter_limited_pickle_files(root: Path, max_files_considered: int) -> list[Path]:
    files: list[Path] = []
    try:
        iterator = root.iterdir()
    except OSError:
        return files
    for path in iterator:
        if len(files) >= max_files_considered:
            break
        if path.is_file() and path.suffix == ".pickle":
            files.append(path)
    return files


def _record_skip(summary: dict[str, Any], reason: str) -> None:
    summary["pickle_files_skipped"] += 1
    skip_reasons = summary["skip_reasons"]
    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1


def _summarize_geom_pickle(
    raw_record: Any,
    path: Path,
    raw_root: Path,
    size_bytes: int,
    max_conformers_per_mol: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_file": str(path.relative_to(raw_root)),
        "size_bytes": size_bytes,
        "object_type": type(raw_record).__name__,
        "molecule_id": path.stem,
        "molecule_metadata": {},
        "molecule_keys": [],
        "conformer_count_in_file": 0,
        "sampled_conformer_count": 0,
        "conformer_keys": [],
        "sampled_conformer_metadata": [],
    }
    if not isinstance(raw_record, dict):
        return record

    record["molecule_keys"] = list(raw_record.keys())
    record["molecule_metadata"] = _json_safe_metadata(raw_record, exclude_keys={"conformers"})
    conformers = raw_record.get("conformers")
    if not isinstance(conformers, list):
        return record

    record["conformer_count_in_file"] = len(conformers)
    sampled = conformers[:max(0, max_conformers_per_mol)]
    record["sampled_conformer_count"] = len(sampled)
    if sampled and isinstance(sampled[0], dict):
        record["conformer_keys"] = list(sampled[0].keys())
    record["sampled_conformer_metadata"] = [
        _json_safe_metadata(
            conformer,
            exclude_keys={"rd_mol", "mol", "coords", "coordinates", "positions"},
        )
        for conformer in sampled
        if isinstance(conformer, dict)
    ]
    return record


def _json_safe_metadata(payload: dict[str, Any], exclude_keys: set[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in payload.items():
        if key in exclude_keys:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
        elif isinstance(value, (list, tuple)) and len(value) <= 10:
            if all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
                metadata[key] = list(value)
        elif isinstance(value, dict) and len(value) <= 10:
            nested = _json_safe_metadata(value, exclude_keys=set())
            if len(nested) == len(value):
                metadata[key] = nested
    return metadata


def _bounded_walk(root: Path, max_entries: int) -> list[Path]:
    entries: list[Path] = []
    pending = [root]
    while pending and len(entries) < max_entries:
        current = pending.pop(0)
        try:
            children = current.iterdir()
        except OSError:
            continue
        for child in children:
            entries.append(child)
            if len(entries) >= max_entries:
                break
            if child.is_dir():
                pending.append(child)
    return entries


def _entry_payload(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "kind": "dir" if path.is_dir() else "file" if path.is_file() else "other",
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _markdown_report(manifest: dict[str, Any]) -> str:
    raw = manifest["raw_summary"]
    dry_run = manifest["preprocessing_dry_run"]
    tiny_processed = manifest["tiny_processed_output"]
    lines = [
        "# GEOM-Drugs Preflight",
        "",
        "**Scope:** read-only raw-data preflight and preprocessing dry-run. This is not a benchmark result.",
        "",
        f"- `result_kind`: `{manifest['result_kind']}`",
        f"- `benchmark_result`: `{str(manifest['benchmark_result']).lower()}`",
        f"- `raw_root`: `{manifest['raw_root']}`",
        f"- `processed_root`: `{manifest['processed_root']}`",
        f"- `status`: `{raw['status']}`",
        f"- `preprocessing_dry_run_status`: `{dry_run['status']}`",
        f"- `raw_data_modified`: `{str(manifest['raw_data_modified']).lower()}`",
        f"- `processed_data_written`: `{str(manifest['processed_data_written']).lower()}`",
        f"- `processed_output_dry_run_written`: `{str(manifest['processed_output_dry_run_written']).lower()}`",
        "",
        "## Layout Hints",
        "",
        f"- Present: {', '.join(raw['present_layout_hints']) if raw['present_layout_hints'] else 'none'}",
        f"- Missing: {', '.join(raw['missing_layout_hints']) if raw['missing_layout_hints'] else 'none'}",
        "",
        "## Top-Level Entries",
        "",
    ]
    if raw["top_level_entries"]:
        lines.extend(["| Path | Kind | Size Bytes |", "|---|---|---:|"])
        for entry in raw["top_level_entries"]:
            lines.append(f"| `{entry['path']}` | {entry['kind']} | {entry['size_bytes'] or 'NA'} |")
    else:
        lines.append("No top-level entries found.")

    lines.extend(
        [
            "",
            "## Preprocessing Metadata Dry-Run",
            "",
            f"- Source layout: `{dry_run['source_layout']}`",
            f"- Source directory: `{dry_run['source_dir']}`",
            f"- Status: `{dry_run['status']}`",
            f"- Molecule metadata records read: `{dry_run['molecule_metadata_records_read']}`",
            f"- Conformer metadata records read: `{dry_run['conformer_metadata_records_read']}`",
            f"- Raw pickle bytes read: `{dry_run['raw_pickle_bytes_read']}`",
            f"- Bounds: max molecules `{dry_run['bounds']['max_molecules']}`, "
            f"max conformers per molecule `{dry_run['bounds']['max_conformers_per_mol']}`, "
            f"max pickle bytes `{dry_run['bounds']['max_pickle_bytes']}`",
            "",
        ]
    )
    if dry_run["sample_records"]:
        lines.extend(
            [
                "| Source File | Size Bytes | Conformers In File | Sampled Conformers |",
                "|---|---:|---:|---:|",
            ]
        )
        for record in dry_run["sample_records"]:
            lines.append(
                f"| `{record['source_file']}` | {record['size_bytes']} | "
                f"{record['conformer_count_in_file']} | {record['sampled_conformer_count']} |"
            )
    else:
        lines.append("No molecule metadata records were read within the configured bounds.")

    lines.extend(
        [
            "",
            "## Tiny Processed-Output Dry-Run",
            "",
            f"- Status: `{tiny_processed['status']}`",
            f"- Written: `{str(tiny_processed['written']).lower()}`",
            f"- Result kind: `{tiny_processed['result_kind']}`",
            f"- Subset scope: `{tiny_processed['subset_scope']}`",
            f"- Benchmark result: `{str(tiny_processed['benchmark_result']).lower()}`",
            f"- Molecules written: `{tiny_processed['molecule_count']}`",
            f"- Conformers written: `{tiny_processed['conformer_count']}`",
            f"- Artifact path: `{tiny_processed['artifact_path']}`",
            f"- Manifest path: `{tiny_processed['manifest_path']}`",
            f"- Bounds: max molecules `{tiny_processed['bounds']['max_molecules']}`, "
            f"max conformers per molecule `{tiny_processed['bounds']['max_conformers_per_mol']}`, "
            f"max atoms `{tiny_processed['bounds']['max_atoms']}`",
            "",
        ]
    )

    lines.extend(
        [
            "",
            "## Dry-Run Plan",
            "",
            f"- Would read from: `{manifest['dry_run_plan']['would_read_from']}`",
            f"- Would write processed records to: `{manifest['dry_run_plan']['would_write_processed_records_to']}`",
            "- Would modify raw data: `false`",
            "- Would write full processed data: `false`",
            f"- Would write tiny processed dry-run: `{str(manifest['dry_run_plan']['would_write_tiny_processed_dry_run']).lower()}`",
            f"- Would write micro-pilot processed dry-run: `{str(manifest['dry_run_plan']['would_write_micro_pilot_processed_dry_run']).lower()}`",
            "",
            "## Notes",
            "",
            "- This command reads only bounded pickle metadata and excludes RDKit molecule/coordinate payloads from outputs.",
            "- With the explicit processed-output flag, this command writes only a tiny non-benchmark processed artifact.",
            "- This command does not write a full processed dataset, split files, checkpoints, training outputs, or benchmark results.",
            "",
        ]
    )
    return "\n".join(lines)
