"""Evaluation-only diagnostic backfill for full-database formal A/E outputs."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Any

from src.eval.formal_attribution import write_formal_attribution_phase_outputs
from src.eval.paper_closure import write_paper_closure_outputs
from src.models.checkpoint_sampling_smoke import sample_existing_checkpoint
from src.train import tiny_real as tiny_real_data
from src.train.formal_10000_training import FORMAL_10000_MAX_STEPS
from src.train.formal_full_database_training import (
    FORMAL_FULL_DATABASE_CHECKPOINT_SCOPE,
    FORMAL_FULL_DATABASE_RESULT_KIND,
    FORMAL_FULL_DATABASE_RUNTIME_SCOPE,
)
from src.utils.config import load_config


DEFAULT_BACKFILL_SPLIT_TYPES = ("random", "scaffold")
DEFAULT_BACKFILL_MODES = ("isotropic", "stiffness_structured")
DEFAULT_BACKFILL_SEEDS = (0, 1, 2)
REQUIRED_SAMPLED_KEYS = (
    "sampled_finite_coordinate",
    "sampled_mean_abs_bond_length_delta",
    "sampled_bond_count_checked",
    "rotatable_bond_count",
)
BACKFILL_NOTE = (
    "Existing reusable checkpoint was reloaded in a later evaluation-only pass to backfill sampled physical-validity "
    "and rotatable-bond diagnostics. No training was rerun and this remains non-benchmark."
)

CONFIG_BY_SPLIT_AND_MODE = {
    ("random", "isotropic"): Path("configs/train/geom_full_database_formal_isotropic.yaml"),
    ("random", "stiffness_structured"): Path("configs/train/geom_full_database_formal_stiffness.yaml"),
    ("scaffold", "isotropic"): Path("configs/train/geom_full_database_scaffold_formal_isotropic.yaml"),
    ("scaffold", "stiffness_structured"): Path("configs/train/geom_full_database_scaffold_formal_stiffness.yaml"),
}


def default_backfill_result_paths(
    split_types: tuple[str, ...] = DEFAULT_BACKFILL_SPLIT_TYPES,
    seeds: tuple[int, ...] = DEFAULT_BACKFILL_SEEDS,
    modes: tuple[str, ...] = DEFAULT_BACKFILL_MODES,
) -> list[Path]:
    paths = []
    for split_type in split_types:
        split_label = "" if split_type == "random" else "_scaffold"
        for mode in modes:
            label = "isotropic" if mode == "isotropic" else "stiffness"
            for seed in seeds:
                paths.append(
                    Path("experiments/results")
                    / f"geom_formal_full_database{split_label}_train_{label}_steps{FORMAL_10000_MAX_STEPS}_seed{seed}.json"
                )
    return paths


def backfill_full_database_formal_diagnostics(
    result_paths: list[str | Path] | None = None,
    device: str = "cpu",
    refresh_reports: bool = True,
) -> dict[str, Any]:
    paths = [Path(path) for path in (result_paths or default_backfill_result_paths())]
    record_cache: dict[Path, list[dict[str, Any]]] = {}
    split_cache: dict[tuple[Path, Path], tuple[dict[str, Any], dict[str, str]]] = {}
    refreshed_results = []
    for path in paths:
        refreshed_results.append(
            backfill_single_formal_result(
                result_path=path,
                device=device,
                record_cache=record_cache,
                split_cache=split_cache,
            )
        )

    outputs: dict[str, Any] = {
        "refreshed_results": refreshed_results,
        "report_refresh_executed": bool(refresh_reports),
    }
    if refresh_reports:
        outputs["attribution_outputs"] = {
            key: str(value) for key, value in write_formal_attribution_phase_outputs().items()
        }
        outputs["closure_outputs"] = {
            key: str(value) for key, value in write_paper_closure_outputs().items()
        }
    return outputs


def backfill_single_formal_result(
    result_path: str | Path,
    device: str = "cpu",
    record_cache: dict[Path, list[dict[str, Any]]] | None = None,
    split_cache: dict[tuple[Path, Path], tuple[dict[str, Any], dict[str, str]]] | None = None,
) -> dict[str, Any]:
    result_path = Path(result_path)
    payload = _load_formal_result(result_path)
    sampled = payload["metrics"]["sampled_molecules"]
    status_before = _sampled_diagnostic_status(sampled)
    if status_before["complete"]:
        return {
            "result_path": str(result_path),
            "run_name": payload["run_name"],
            "updated": False,
            "status_before": status_before,
            "status_after": status_before,
            "note": "Result already carries the required sampled diagnostic fields.",
        }

    processed_artifact_path = Path(payload["processed_artifact_path"])
    split_path = Path(payload["split_path"])
    checkpoint_path = Path(payload["best_checkpoint_path"])
    config = _backfill_config(payload)
    records = _cached_records(processed_artifact_path, config, record_cache)
    config["eval"]["eval_subset_size"] = len(records)
    split, split_membership = _cached_split(split_path, processed_artifact_path, records, split_cache)
    if split.get("split_type") != payload["dataset"]["split_type"]:
        raise ValueError(f"Split type mismatch for {result_path}")
    if {key: len(split[key]) for key in ["train", "val", "test"]} != {
        key: int(payload["dataset"]["split_counts"][key]) for key in ["train", "val", "test"]
    }:
        raise ValueError(f"Split counts do not match result payload for {result_path}")

    start = time.time()
    regenerated = sample_existing_checkpoint(
        samples=tiny_real_data._iter_flow_matching_samples(
            config,
            records,
            random.Random(int(payload["random_seed"])),
            split_membership,
        ),
        checkpoint_path=checkpoint_path,
        sample_step_size=float(payload["sample_step_size"]),
        device=device,
        require_cuda=False,
    )
    elapsed = time.time() - start
    regenerated_sampled = regenerated["sampled_molecules"]
    _validate_regenerated_sampled(sampled, regenerated_sampled, result_path)

    payload["metrics"]["sampled_molecules"] = regenerated_sampled
    payload["metrics"]["sampled_physical_validity_diagnostics_backfilled"] = True
    payload["metrics"]["sampled_rotatable_bond_metadata_backfilled"] = True
    payload["diagnostic_backfill"] = {
        "backfilled": True,
        "execution_mode": "local_evaluation_only_checkpoint_reload_sampling",
        "device": regenerated["device"],
        "hardware": "gpu_enabled" if str(regenerated["device"]).startswith("cuda") else "cpu_only",
        "scheduler": None,
        "slurm_account": None,
        "training_reexecuted": False,
        "sampling_executed": True,
        "runtime_seconds": elapsed,
        "sample_count": len(regenerated_sampled),
        "checkpoint_path": str(checkpoint_path),
        "processed_artifact_path": str(processed_artifact_path),
        "split_path": str(split_path),
        "required_sampled_keys": list(REQUIRED_SAMPLED_KEYS),
    }
    payload["notes"] = _append_note_once(payload.get("notes"), BACKFILL_NOTE)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status_after = _sampled_diagnostic_status(regenerated_sampled)
    return {
        "result_path": str(result_path),
        "run_name": payload["run_name"],
        "updated": True,
        "status_before": status_before,
        "status_after": status_after,
        "diagnostic_backfill_runtime_seconds": elapsed,
    }


def _load_formal_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Full-database formal result does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("result_kind") != FORMAL_FULL_DATABASE_RESULT_KIND:
        raise ValueError(f"Unexpected result_kind in {path}")
    if payload.get("benchmark_result") is not False:
        raise ValueError(f"Expected benchmark_result=false in {path}")
    if payload.get("corruption_mode") not in DEFAULT_BACKFILL_MODES:
        raise ValueError(f"Backfill only supports isotropic/stiffness outputs: {path}")
    if payload.get("checkpoint_scope") != FORMAL_FULL_DATABASE_CHECKPOINT_SCOPE:
        raise ValueError(f"Reusable checkpoint scope mismatch in {path}")
    if payload.get("checkpoint_reusable") is not True or payload.get("checkpoint_selected_by_validation") is not True:
        raise ValueError(f"Expected reusable validation-selected checkpoint in {path}")
    if payload.get("runtime_scope") != FORMAL_FULL_DATABASE_RUNTIME_SCOPE:
        raise ValueError(f"Unexpected runtime_scope in {path}")
    if payload.get("sampling_executed") is not True:
        raise ValueError(f"Expected sampling_executed=true in {path}")
    metrics = payload.get("metrics") or {}
    if not isinstance(metrics.get("sampled_molecules"), list):
        raise ValueError(f"Missing sampled_molecules in {path}")
    return payload


def _sampled_diagnostic_status(sampled: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(sampled),
        "physical_key_count": sum(
            1
            for item in sampled
            if all(key in item for key in REQUIRED_SAMPLED_KEYS[:-1])
        ),
        "rotatable_numeric_count": sum(1 for item in sampled if _is_int_like(item.get("rotatable_bond_count"))),
        "complete": bool(sampled)
        and all(all(key in item for key in REQUIRED_SAMPLED_KEYS[:-1]) for item in sampled)
        and all(_is_int_like(item.get("rotatable_bond_count")) for item in sampled),
    }


def _backfill_config(payload: dict[str, Any]) -> dict[str, Any]:
    split_type = str(payload["dataset"]["split_type"])
    corruption_mode = str(payload["corruption_mode"])
    config = copy.deepcopy(load_config(CONFIG_BY_SPLIT_AND_MODE[(split_type, corruption_mode)]))
    config["experiment"]["seed"] = int(payload["random_seed"])
    config["data"]["processed_artifact_path"] = str(payload["processed_artifact_path"])
    config["data"]["split_path"] = str(payload["split_path"])
    config["data"]["split_type"] = split_type
    config["eval"]["compute_physical_validity_diagnostics"] = True
    config["eval"]["compute_rotatable_bond_stratification"] = True
    config["eval"]["compute_rotatable_bond_count_from_smiles"] = True
    config["train"]["device"] = "cpu"
    config["train"]["require_cuda"] = False
    return config


def _cached_records(
    processed_artifact_path: Path,
    config: dict[str, Any],
    record_cache: dict[Path, list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    if record_cache is not None and processed_artifact_path in record_cache:
        return record_cache[processed_artifact_path]
    artifact = tiny_real_data._load_tiny_real_artifact(processed_artifact_path)
    records = artifact["molecules"]
    for record in records:
        if record.get("rotatable_bond_count") is None:
            record["rotatable_bond_count"] = tiny_real_data._rotatable_bond_count(record, config)
    if record_cache is not None:
        record_cache[processed_artifact_path] = records
    return records


def _cached_split(
    split_path: Path,
    processed_artifact_path: Path,
    records: list[dict[str, Any]],
    split_cache: dict[tuple[Path, Path], tuple[dict[str, Any], dict[str, str]]] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    cache_key = (processed_artifact_path, split_path)
    if split_cache is not None and cache_key in split_cache:
        return split_cache[cache_key]
    split = tiny_real_data._load_tiny_real_split(split_path)
    split_membership = tiny_real_data._validate_split_coverage(split, records)
    if split_cache is not None:
        split_cache[cache_key] = (split, split_membership)
    return split, split_membership


def _validate_regenerated_sampled(
    original: list[dict[str, Any]],
    regenerated: list[dict[str, Any]],
    result_path: Path,
) -> None:
    if len(original) != len(regenerated):
        raise ValueError(f"Regenerated sample count mismatch for {result_path}")
    for index, (before, after) in enumerate(zip(original, regenerated, strict=True)):
        for key in ("molecule_id", "split", "atom_count", "sample_step_count"):
            if before.get(key) != after.get(key):
                raise ValueError(f"Sample mismatch on {key} at index {index} for {result_path}")
        if abs(float(before["sample_step_size"]) - float(after["sample_step_size"])) > 1.0e-12:
            raise ValueError(f"Sample mismatch on sample_step_size at index {index} for {result_path}")
        _require_close(
            float(before["sampled_rmsd_to_target"]),
            float(after["sampled_rmsd_to_target"]),
            tolerance=1.0e-6,
            label=f"sampled_rmsd_to_target[{index}]",
            result_path=result_path,
        )
        _require_close(
            float(before["predicted_velocity_norm"]),
            float(after["predicted_velocity_norm"]),
            tolerance=1.0e-6,
            label=f"predicted_velocity_norm[{index}]",
            result_path=result_path,
        )


def _require_close(
    expected: float,
    observed: float,
    tolerance: float,
    label: str,
    result_path: Path,
) -> None:
    if abs(expected - observed) > tolerance:
        raise ValueError(
            f"Regenerated {label} deviates by more than {tolerance} for {result_path}: "
            f"expected {expected}, observed {observed}"
        )


def _append_note_once(notes: Any, note: str) -> list[str]:
    if notes is None:
        items = []
    elif isinstance(notes, list):
        items = [str(item) for item in notes]
    else:
        items = [str(notes)]
    if note not in items:
        items.append(note)
    return items


def _is_int_like(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and value.is_integer()
