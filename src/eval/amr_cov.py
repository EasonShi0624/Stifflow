"""AMR/COV conformer-ensemble evaluation from existing checkpoints.

This module is intentionally reporting-only. It reloads existing full-database
formal checkpoints, streams raw GEOM-Drugs pickle files for reference conformer
ensembles, and computes heavy-atom aligned AMR/COV diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from src.corruption.base import build_corruptor
from src.data.geom_full_database import (
    FULL_DATABASE_SCAFFOLD_SPLIT_PATH,
    FULL_DATABASE_SCOPE,
    FULL_DATABASE_SPLIT_PATH,
)
from src.train.formal_10000_training import FORMAL_10000_MAX_STEPS, FORMAL_10000_SAMPLE_STEP_SIZE
from src.utils.config import load_config


METHODS = ("isotropic", "harmonic_prior", "torsion_only", "stiffness_structured")
METHOD_RUN_LABELS = {
    "isotropic": "isotropic",
    "harmonic_prior": "harmonic",
    "torsion_only": "torsion_only",
    "stiffness_structured": "stiffness",
}
METHOD_DISPLAY = {
    "isotropic": "isotropic",
    "harmonic_prior": "harmonic",
    "torsion_only": "torsion-only",
    "stiffness_structured": "stiffness",
}
SPLIT_PATHS = {
    "random": FULL_DATABASE_SPLIT_PATH,
    "scaffold": FULL_DATABASE_SCAFFOLD_SPLIT_PATH,
}
BOUNDED_RESULT_KIND = "geom_drugs_full_database_bounded_amr_cov_evaluation_not_benchmark"
FULL_TEST_RESULT_KIND = "geom_drugs_full_database_full_test_amr_cov_evaluation_current_protocol"
EXECUTION_MODE = "local_cpu_amr_cov_from_existing_checkpoints"
DEFAULT_RUN_NAME = "geom_full_database_amr_cov_bounded_abde_seed0"
DEFAULT_FULL_TEST_RUN_NAME = "geom_full_database_amr_cov_full_test_abde_seed0"
METRIC_KEYS = ("cov_r", "amr_r", "cov_p", "amr_p")


@dataclass
class ReferenceMolecule:
    molecule_id: str
    smiles: str | None
    record: dict[str, Any]
    anchor_coords: list[list[float]]
    reference_coords: list[list[list[float]]]
    heavy_atom_indices: list[int]
    source_path: str
    available_reference_conformer_count: int
    used_reference_conformer_count: int
    skipped_reference_conformer_count: int
    raw_pickle_bytes: int


@dataclass
class MethodRuntimeState:
    method: str
    config: dict[str, Any]
    checkpoint_path: Path
    checkpoint_payload: dict[str, Any]
    model: Any
    device: str
    start_time: float
    evaluated_count: int = 0
    metric_values: dict[str, list[float]] = field(
        default_factory=lambda: {metric: [] for metric in METRIC_KEYS}
    )
    reference_counts: list[int] = field(default_factory=list)
    generated_counts: list[int] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    per_molecule: list[dict[str, Any]] = field(default_factory=list)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = run_bounded_amr_cov_evaluation(args)
    for label, path in paths.items():
        if path is not None:
            print(f"{label}: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-types", nargs="+", choices=("scaffold", "random"), default=["scaffold", "random"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--max-molecules-per-split", type=int, default=8)
    parser.add_argument("--allow-full-test-split", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--generated-conformers-per-mol", type=int, default=16)
    parser.add_argument("--max-reference-conformers-per-mol", type=int, default=40)
    parser.add_argument("--max-pickle-bytes", type=int, default=25_000_000)
    parser.add_argument("--inline-per-molecule-limit", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--delta", type=float, default=0.75)
    parser.add_argument("--raw-root", default="data/raw/geom_drugs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--figure-path", default=None)
    parser.add_argument("--per-molecule-jsonl-path", default=None)
    return parser.parse_args(argv)


def run_bounded_amr_cov_evaluation(args: argparse.Namespace) -> dict[str, Path | None]:
    start = time.time()
    _validate_args(args)
    _setup_matplotlib_cache()

    split_types = tuple(dict.fromkeys(args.split_types))
    methods = tuple(dict.fromkeys(args.methods))
    raw_root = Path(args.raw_root)
    max_molecules = None if int(args.max_molecules_per_split) == 0 else int(args.max_molecules_per_split)
    max_reference_conformers = (
        None if int(args.max_reference_conformers_per_mol) == 0 else int(args.max_reference_conformers_per_mol)
    )
    full_test_requested = max_molecules is None
    run_name = _resolve_run_name(args, full_test_requested)
    output_paths = _resolve_output_paths(args, run_name, split_types, full_test_requested)
    result_kind = FULL_TEST_RESULT_KIND if full_test_requested else BOUNDED_RESULT_KIND
    coverage_label = "full_test_set_current_protocol" if full_test_requested else "bounded_test_subset"
    if output_paths["per_molecule_jsonl"] is not None:
        _write_text(output_paths["per_molecule_jsonl"], "")

    summary: dict[str, Any] = {
        "run_name": run_name,
        "result_kind": result_kind,
        "benchmark_result": False,
        "coverage_label": coverage_label,
        "full_test_set_evaluation_under_current_protocol": full_test_requested,
        "broader_benchmark_claim": False,
        "execution_mode": EXECUTION_MODE,
        "hardware": "cpu_only" if str(args.device) == "cpu" else "device_requested",
        "gpu_used_by_summary": str(args.device).startswith("cuda"),
        "scheduler": None,
        "slurm_account": None,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": True,
        "raw_data_modified": False,
        "environment_modified": False,
        "dataset_scope": FULL_DATABASE_SCOPE,
        "dataset": {
            "name": "geom_drugs",
            "raw_root": str(raw_root),
            "split_types": list(split_types),
            "partition": "test",
        },
        "evaluation": {
            "delta_angstrom": float(args.delta),
            "rmsd": "heavy_atom_aligned_kabsch",
            "metrics": ["COV-R", "AMR-R", "COV-P", "AMR-P"],
            "aggregation": "mean_and_median_across_molecules",
            "generated_conformers_per_molecule": int(args.generated_conformers_per_mol),
            "max_reference_conformers_per_molecule": max_reference_conformers,
            "max_molecules_per_split": max_molecules,
            "chunk_size": int(args.chunk_size),
            "inline_per_molecule_limit": int(args.inline_per_molecule_limit),
            "per_molecule_jsonl_path": (
                None if output_paths["per_molecule_jsonl"] is None else str(output_paths["per_molecule_jsonl"])
            ),
            "subset_seed": int(args.subset_seed),
            "seed": int(args.seed),
            "sample_step_size": FORMAL_10000_SAMPLE_STEP_SIZE,
            "generation_note": (
                "Generated ensembles use the current one-step checkpoint sampling diagnostic. "
                "Each molecule starts from the first raw conformer as the source anchor; this run "
                "is labeled only for the current repository AMR/COV protocol."
            ),
        },
        "methods": list(methods),
        "input_splits": {split: str(SPLIT_PATHS[split]) for split in split_types},
        "input_result_paths": {},
        "input_checkpoint_paths": {},
        "split_summaries": {},
        "runtime_seconds": None,
        "output_paths": _stringify_paths(output_paths),
        "notes": [
            "Existing validation-selected checkpoints are reloaded; training is not rerun.",
            "Raw GEOM pickle files are read only for reference conformer ensembles.",
            "The 3GB processed molecules.json artifact is intentionally not loaded.",
            "Full-test-set labels apply only to the current one-step sampler and reference-conformer cap protocol.",
        ],
    }

    for split_type in split_types:
        split_summary = _evaluate_split(
            split_type=split_type,
            methods=methods,
            seed=int(args.seed),
            subset_seed=int(args.subset_seed),
            raw_root=raw_root,
            max_molecules=max_molecules,
            max_reference_conformers=max_reference_conformers,
            max_pickle_bytes=int(args.max_pickle_bytes),
            generated_conformers_per_mol=int(args.generated_conformers_per_mol),
            delta=float(args.delta),
            device=str(args.device),
            chunk_size=int(args.chunk_size),
            inline_per_molecule_limit=int(args.inline_per_molecule_limit),
            progress_every=int(args.progress_every),
            per_molecule_jsonl_path=output_paths["per_molecule_jsonl"],
            summary=summary,
        )
        summary["split_summaries"][split_type] = split_summary

    summary["runtime_seconds"] = time.time() - start
    summary["coverage"] = _coverage_summary(summary, max_molecules)
    summary["safety_labels"] = {
        "benchmark_result": False,
        "full_test_set_evaluation_under_current_protocol": full_test_requested,
        "broader_benchmark_claim": False,
        "bounded_test_subset": not summary["coverage"]["all_requested_splits_full_test_covered"],
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": True,
        "raw_data_modified": False,
        "environment_modified": False,
    }

    summary_path = output_paths["summary"]
    manifest_path = output_paths["manifest"]
    log_path = output_paths["log"]
    combined_report_path = output_paths["combined_report"]
    combined_figure_path = output_paths["combined_figure"]
    assert summary_path is not None
    assert manifest_path is not None
    assert log_path is not None
    assert combined_report_path is not None
    assert combined_figure_path is not None

    _write_json(summary_path, summary)
    _write_json(
        manifest_path,
        _manifest_from_summary(summary, summary_path, combined_report_path, combined_figure_path),
    )
    _write_markdown_outputs(summary, output_paths)
    _write_figure_outputs(summary, output_paths)
    _write_text(log_path, _log_text(summary, output_paths))

    return output_paths


def _resolve_run_name(args: argparse.Namespace, full_test_requested: bool) -> str:
    if args.run_name:
        return str(args.run_name)
    if full_test_requested:
        return (
            DEFAULT_FULL_TEST_RUN_NAME
            if int(args.seed) == 0
            else f"geom_full_database_amr_cov_full_test_abde_seed{args.seed}"
        )
    return DEFAULT_RUN_NAME if int(args.seed) == 0 else f"geom_full_database_amr_cov_bounded_abde_seed{args.seed}"


def _resolve_output_paths(
    args: argparse.Namespace,
    run_name: str,
    split_types: tuple[str, ...],
    full_test_requested: bool,
) -> dict[str, Path | None]:
    combined_suffix = "_combined" if full_test_requested else ""
    summary = Path(args.summary_path) if args.summary_path else Path("experiments/results") / f"{run_name}.json"
    manifest = (
        Path(args.manifest_path) if args.manifest_path else Path("experiments/manifests") / f"{run_name}.json"
    )
    log = Path(args.log_path) if args.log_path else Path("experiments/logs") / f"{run_name}.log"
    combined_report = (
        Path(args.report_path)
        if args.report_path
        else Path("reports/tables") / f"{run_name}{combined_suffix}.md"
    )
    combined_figure = (
        Path(args.figure_path)
        if args.figure_path
        else Path("reports/figures") / f"{run_name}{combined_suffix}.png"
    )
    per_molecule_jsonl = (
        Path(args.per_molecule_jsonl_path)
        if args.per_molecule_jsonl_path
        else Path("experiments/results") / f"{run_name}_per_molecule.jsonl"
    )
    split_reports = {
        split: Path("reports/tables") / f"{run_name}_{split}.md"
        for split in split_types
    }
    split_figures = {
        split: Path("reports/figures") / f"{run_name}_{split}.png"
        for split in split_types
    }
    return {
        "summary": summary,
        "manifest": manifest,
        "log": log,
        "combined_report": combined_report,
        "combined_figure": combined_figure,
        "per_molecule_jsonl": per_molecule_jsonl,
        **{f"{split}_report": path for split, path in split_reports.items()},
        **{f"{split}_figure": path for split, path in split_figures.items()},
    }


def _stringify_paths(paths: dict[str, Path | None]) -> dict[str, str | None]:
    return {key: None if value is None else str(value) for key, value in paths.items()}


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.seed) < 0:
        raise ValueError("seed must be non-negative")
    if int(args.subset_seed) < 0:
        raise ValueError("subset-seed must be non-negative")
    if int(args.max_molecules_per_split) < 0:
        raise ValueError("max-molecules-per-split must be >= 0; use 0 for full test split")
    if int(args.max_molecules_per_split) == 0 and not bool(args.allow_full_test_split):
        raise ValueError("Full test split evaluation requires --allow-full-test-split")
    if int(args.chunk_size) <= 0:
        raise ValueError("chunk-size must be positive")
    if int(args.generated_conformers_per_mol) <= 0:
        raise ValueError("generated-conformers-per-mol must be positive")
    if int(args.max_reference_conformers_per_mol) < 0:
        raise ValueError("max-reference-conformers-per-mol must be >= 0; use 0 for all references")
    if int(args.max_pickle_bytes) <= 0:
        raise ValueError("max-pickle-bytes must be positive")
    if int(args.inline_per_molecule_limit) < 0:
        raise ValueError("inline-per-molecule-limit must be non-negative")
    if int(args.progress_every) < 0:
        raise ValueError("progress-every must be non-negative")
    if float(args.delta) <= 0.0:
        raise ValueError("delta must be positive")


def _evaluate_split(
    split_type: str,
    methods: tuple[str, ...],
    seed: int,
    subset_seed: int,
    raw_root: Path,
    max_molecules: int | None,
    max_reference_conformers: int | None,
    max_pickle_bytes: int,
    generated_conformers_per_mol: int,
    delta: float,
    device: str,
    chunk_size: int,
    inline_per_molecule_limit: int,
    progress_every: int,
    per_molecule_jsonl_path: Path | None,
    summary: dict[str, Any],
) -> dict[str, Any]:
    split_start = time.time()
    split = _load_split(SPLIT_PATHS[split_type], split_type)
    selected_ids = _select_test_molecules(split["test"], max_molecules=max_molecules, subset_seed=subset_seed)
    torch = _load_torch()
    device = _resolve_device(torch, device)
    if device == "cpu":
        torch.set_num_threads(1)

    states: dict[str, MethodRuntimeState] = {}
    for method in methods:
        config = _load_method_config(split_type=split_type, method=method, seed=seed)
        result_path, checkpoint_path = _resolve_result_and_checkpoint(split_type=split_type, method=method, seed=seed)
        summary["input_result_paths"].setdefault(split_type, {})[method] = str(result_path)
        summary["input_checkpoint_paths"].setdefault(split_type, {})[method] = str(checkpoint_path)
        model, checkpoint_payload = _load_model(torch, checkpoint_path, device)
        states[method] = MethodRuntimeState(
            method=method,
            config=config,
            checkpoint_path=checkpoint_path,
            checkpoint_payload=checkpoint_payload,
            model=model,
            device=device,
            start_time=time.time(),
        )

    reference_skips = []
    reference_loaded_count = 0
    processed_count = 0
    for chunk_start in range(0, len(selected_ids), chunk_size):
        chunk_ids = selected_ids[chunk_start : chunk_start + chunk_size]
        loaded: list[tuple[int, ReferenceMolecule]] = []
        for local_offset, molecule_id in enumerate(chunk_ids):
            selected_index = chunk_start + local_offset
            try:
                loaded.append(
                    (
                        selected_index,
                        _load_reference_molecule(
                            molecule_id=molecule_id,
                            raw_root=raw_root,
                            max_reference_conformers=max_reference_conformers,
                            max_pickle_bytes=max_pickle_bytes,
                        ),
                    )
                )
            except MoleculeSkip as exc:
                reference_skips.append(
                    {
                        "molecule_id": molecule_id,
                        "selected_index": selected_index,
                        "reason": exc.reason,
                        "detail": exc.detail,
                    }
                )
        reference_loaded_count += len(loaded)
        with torch.no_grad():
            for method, state in states.items():
                for selected_index, molecule in loaded:
                    _evaluate_method_molecule(
                        torch=torch,
                        state=state,
                        split_type=split_type,
                        method=method,
                        seed=seed,
                        molecule=molecule,
                        selected_index=selected_index,
                        generated_conformers_per_mol=generated_conformers_per_mol,
                        delta=delta,
                        inline_per_molecule_limit=inline_per_molecule_limit,
                        per_molecule_jsonl_path=per_molecule_jsonl_path,
                    )
        processed_count += len(chunk_ids)
        if progress_every and (processed_count == len(selected_ids) or processed_count % progress_every < chunk_size):
            _progress(
                f"amr_cov progress split={split_type} processed={processed_count}/{len(selected_ids)} "
                f"loaded={reference_loaded_count} reference_skips={len(reference_skips)}"
            )

    split_summary: dict[str, Any] = {
        "split_type": split_type,
        "split_path": str(SPLIT_PATHS[split_type]),
        "test_molecule_count_in_split": len(split["test"]),
        "selected_molecule_count": len(selected_ids),
        "selected_molecule_ids": selected_ids,
        "reference_loaded_molecule_count": reference_loaded_count,
        "reference_skipped_molecules": reference_skips,
        "chunk_size": chunk_size,
        "methods": {},
        "runtime_seconds": None,
    }

    for method, state in states.items():
        split_summary["methods"][method] = _finalize_method_state(state)
    split_summary["runtime_seconds"] = time.time() - split_start
    return split_summary


def _evaluate_method_molecule(
    torch: Any,
    state: MethodRuntimeState,
    split_type: str,
    method: str,
    seed: int,
    molecule: ReferenceMolecule,
    selected_index: int,
    generated_conformers_per_mol: int,
    delta: float,
    inline_per_molecule_limit: int,
    per_molecule_jsonl_path: Path | None,
) -> None:
    try:
        generated = _generate_conformers(
            torch=torch,
            model=state.model,
            config=state.config,
            molecule=molecule,
            split_type=split_type,
            method=method,
            seed=seed,
            molecule_index=selected_index,
            generated_conformers_per_mol=generated_conformers_per_mol,
            device=state.device,
        )
        pairwise = pairwise_heavy_atom_rmsd(
            generated_coords=generated,
            reference_coords=molecule.reference_coords,
            heavy_atom_indices=molecule.heavy_atom_indices,
        )
        metrics = compute_amr_cov_from_pairwise(pairwise, delta=delta)
        row = {
            "split_type": split_type,
            "method": method,
            "molecule_id": molecule.molecule_id,
            "selected_index": selected_index,
            "smiles": molecule.smiles,
            "source_path": molecule.source_path,
            "atom_count": len(molecule.anchor_coords),
            "heavy_atom_count": len(molecule.heavy_atom_indices),
            "reference_conformer_count_available": molecule.available_reference_conformer_count,
            "reference_conformer_count_used": molecule.used_reference_conformer_count,
            "reference_conformer_count_skipped": molecule.skipped_reference_conformer_count,
            "generated_conformer_count": len(generated),
            "raw_pickle_bytes": molecule.raw_pickle_bytes,
            "metrics": metrics,
        }
        state.evaluated_count += 1
        for metric in METRIC_KEYS:
            state.metric_values[metric].append(float(metrics[metric]))
        state.reference_counts.append(molecule.used_reference_conformer_count)
        state.generated_counts.append(len(generated))
        if len(state.per_molecule) < inline_per_molecule_limit:
            state.per_molecule.append(row)
        if per_molecule_jsonl_path is not None:
            _append_jsonl(per_molecule_jsonl_path, row)
    except Exception as exc:
        state.skipped.append(
            {
                "split_type": split_type,
                "method": method,
                "molecule_id": molecule.molecule_id,
                "selected_index": selected_index,
                "reason": f"method_evaluation_error:{type(exc).__name__}",
                "detail": _exception_detail(exc),
            }
        )


def _finalize_method_state(state: MethodRuntimeState) -> dict[str, Any]:
    return {
        "method": state.method,
        "method_display": METHOD_DISPLAY[state.method],
        "checkpoint_path": str(state.checkpoint_path),
        "checkpoint_result_kind": state.checkpoint_payload.get("result_kind"),
        "checkpoint_scope": state.checkpoint_payload.get("checkpoint_scope"),
        "checkpoint_benchmark_result": state.checkpoint_payload.get("benchmark_result"),
        "evaluated_molecule_count": state.evaluated_count,
        "skipped_molecule_count": len(state.skipped),
        "skipped_molecules": state.skipped,
        "reference_conformers_per_molecule": _count_summary(state.reference_counts),
        "generated_conformers_per_molecule": _count_summary(state.generated_counts),
        "metric_summary": {
            metric: _numeric_summary(state.metric_values[metric])
            for metric in METRIC_KEYS
        },
        "per_molecule_inline_count": len(state.per_molecule),
        "per_molecule": state.per_molecule,
        "runtime_seconds": time.time() - state.start_time,
        "device": state.device,
    }


def _evaluate_method(
    split_type: str,
    method: str,
    seed: int,
    config: dict[str, Any],
    checkpoint_path: Path,
    molecules: list[ReferenceMolecule],
    generated_conformers_per_mol: int,
    delta: float,
    device: str,
) -> dict[str, Any]:
    start = time.time()
    torch = _load_torch()
    device = _resolve_device(torch, device)
    if device == "cpu":
        torch.set_num_threads(1)
    model, checkpoint_payload = _load_model(torch, checkpoint_path, device)
    per_molecule = []
    skipped = []
    with torch.no_grad():
        for molecule_index, molecule in enumerate(molecules):
            try:
                generated = _generate_conformers(
                    torch=torch,
                    model=model,
                    config=config,
                    molecule=molecule,
                    split_type=split_type,
                    method=method,
                    seed=seed,
                    molecule_index=molecule_index,
                    generated_conformers_per_mol=generated_conformers_per_mol,
                    device=device,
                )
                pairwise = pairwise_heavy_atom_rmsd(
                    generated_coords=generated,
                    reference_coords=molecule.reference_coords,
                    heavy_atom_indices=molecule.heavy_atom_indices,
                )
                metrics = compute_amr_cov_from_pairwise(pairwise, delta=delta)
                per_molecule.append(
                    {
                        "molecule_id": molecule.molecule_id,
                        "smiles": molecule.smiles,
                        "source_path": molecule.source_path,
                        "atom_count": len(molecule.anchor_coords),
                        "heavy_atom_count": len(molecule.heavy_atom_indices),
                        "reference_conformer_count_available": molecule.available_reference_conformer_count,
                        "reference_conformer_count_used": molecule.used_reference_conformer_count,
                        "reference_conformer_count_skipped": molecule.skipped_reference_conformer_count,
                        "generated_conformer_count": len(generated),
                        "raw_pickle_bytes": molecule.raw_pickle_bytes,
                        "metrics": metrics,
                    }
                )
            except Exception as exc:
                skipped.append(
                    {
                        "molecule_id": molecule.molecule_id,
                        "reason": f"method_evaluation_error:{type(exc).__name__}",
                        "detail": _exception_detail(exc),
                    }
                )
    return {
        "method": method,
        "method_display": METHOD_DISPLAY[method],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_result_kind": checkpoint_payload.get("result_kind"),
        "checkpoint_scope": checkpoint_payload.get("checkpoint_scope"),
        "checkpoint_benchmark_result": checkpoint_payload.get("benchmark_result"),
        "evaluated_molecule_count": len(per_molecule),
        "skipped_molecule_count": len(skipped),
        "skipped_molecules": skipped,
        "reference_conformers_per_molecule": _count_summary(
            row["reference_conformer_count_used"] for row in per_molecule
        ),
        "generated_conformers_per_molecule": _count_summary(row["generated_conformer_count"] for row in per_molecule),
        "metric_summary": _metric_summary(per_molecule),
        "per_molecule": per_molecule,
        "runtime_seconds": time.time() - start,
        "device": device,
    }


def _load_split(path: Path, split_type: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Split file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split_type") != split_type:
        raise ValueError(f"Split file {path} is not split_type={split_type!r}")
    if payload.get("split_scope") != FULL_DATABASE_SCOPE:
        raise ValueError(f"Split file {path} is not scope={FULL_DATABASE_SCOPE!r}")
    for key in ["train", "val", "test"]:
        if not isinstance(payload.get(key), list) or not payload[key]:
            raise ValueError(f"Split file {path} has an empty or missing {key} partition")
    return payload


def _select_test_molecules(test_ids: list[str], max_molecules: int | None, subset_seed: int) -> list[str]:
    selected = list(test_ids)
    random.Random(subset_seed).shuffle(selected)
    if max_molecules is not None:
        selected = selected[:max_molecules]
    return selected


def _load_method_config(split_type: str, method: str, seed: int) -> dict[str, Any]:
    label = METHOD_RUN_LABELS[method]
    split_label = "_scaffold" if split_type == "scaffold" else ""
    path = Path("configs/train") / f"geom_full_database{split_label}_formal_{label}.yaml"
    config = load_config(path)
    if config["corruption"]["mode"] != method:
        raise ValueError(f"Config {path} corruption mode does not match {method}")
    if config["data"]["split_type"] != split_type:
        raise ValueError(f"Config {path} split type does not match {split_type}")
    config["experiment"]["seed"] = int(seed)
    return config


def _resolve_result_and_checkpoint(split_type: str, method: str, seed: int) -> tuple[Path, Path]:
    label = METHOD_RUN_LABELS[method]
    split_label = "_scaffold" if split_type == "scaffold" else ""
    result_path = (
        Path("experiments/results")
        / f"geom_formal_full_database{split_label}_train_{label}_steps{FORMAL_10000_MAX_STEPS}_seed{seed}.json"
    )
    if not result_path.exists():
        raise FileNotFoundError(f"Full-database result JSON missing: {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("corruption_mode") != method:
        raise ValueError(f"Result {result_path} corruption mode does not match {method}")
    if payload.get("dataset", {}).get("split_type") != split_type:
        raise ValueError(f"Result {result_path} split type does not match {split_type}")
    checkpoint_path = Path(payload["best_checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint from {result_path} does not exist: {checkpoint_path}")
    return result_path, checkpoint_path


class MoleculeSkip(Exception):
    def __init__(self, reason: str, detail: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _load_reference_molecule(
    molecule_id: str,
    raw_root: Path,
    max_reference_conformers: int | None,
    max_pickle_bytes: int,
) -> ReferenceMolecule:
    path = raw_root / "rdkit_folder" / "drugs" / f"{molecule_id}.pickle"
    if not path.exists():
        raise MoleculeSkip("raw_pickle_missing", str(path))
    size_bytes = path.stat().st_size
    if size_bytes > max_pickle_bytes:
        raise MoleculeSkip("raw_pickle_exceeds_max_bytes", f"{size_bytes} > {max_pickle_bytes}")
    try:
        with path.open("rb") as handle:
            raw_record = pickle.load(handle)
    except Exception as exc:  # pragma: no cover - raw-data failures are environment dependent.
        raise MoleculeSkip(f"pickle_load_error:{type(exc).__name__}", _exception_detail(exc)) from exc
    if not isinstance(raw_record, dict):
        raise MoleculeSkip("raw_record_not_dict", type(raw_record).__name__)
    conformers = raw_record.get("conformers")
    if not isinstance(conformers, list) or not conformers:
        raise MoleculeSkip("no_reference_conformers", None)

    anchor_mol = None
    anchor_coords = None
    anchor_atomic_numbers = None
    anchor_index = None
    for index, conformer in enumerate(conformers):
        rd_mol = conformer.get("rd_mol") if isinstance(conformer, dict) else None
        if _looks_like_rdkit_mol(rd_mol) and rd_mol.GetNumConformers() > 0:
            anchor_mol = rd_mol
            anchor_coords = _coords_from_rdkit_mol(rd_mol)
            anchor_atomic_numbers = [int(atom.GetAtomicNum()) for atom in rd_mol.GetAtoms()]
            anchor_index = index
            break
    if anchor_mol is None or anchor_coords is None or anchor_atomic_numbers is None or anchor_index is None:
        raise MoleculeSkip("no_extractable_rdkit_conformer", None)

    reference_coords = []
    skipped_reference_count = 0
    for conformer in conformers:
        if max_reference_conformers is not None and len(reference_coords) >= max_reference_conformers:
            break
        if not isinstance(conformer, dict):
            skipped_reference_count += 1
            continue
        rd_mol = conformer.get("rd_mol")
        if not _looks_like_rdkit_mol(rd_mol) or rd_mol.GetNumConformers() <= 0:
            skipped_reference_count += 1
            continue
        atomic_numbers = [int(atom.GetAtomicNum()) for atom in rd_mol.GetAtoms()]
        if atomic_numbers != anchor_atomic_numbers:
            skipped_reference_count += 1
            continue
        coords = _coords_from_rdkit_mol(rd_mol)
        if len(coords) != len(anchor_coords):
            skipped_reference_count += 1
            continue
        reference_coords.append(coords)
    if not reference_coords:
        raise MoleculeSkip("no_valid_reference_conformers_after_filtering", None)

    heavy_atom_indices = [index for index, atomic_number in enumerate(anchor_atomic_numbers) if atomic_number > 1]
    if not heavy_atom_indices:
        raise MoleculeSkip("no_heavy_atoms", None)
    return ReferenceMolecule(
        molecule_id=molecule_id,
        smiles=raw_record.get("smiles"),
        record=_record_from_rdkit_mol(molecule_id, raw_record, anchor_mol, path.relative_to(raw_root)),
        anchor_coords=anchor_coords,
        reference_coords=reference_coords,
        heavy_atom_indices=heavy_atom_indices,
        source_path=str(path),
        available_reference_conformer_count=len(conformers),
        used_reference_conformer_count=len(reference_coords),
        skipped_reference_conformer_count=skipped_reference_count,
        raw_pickle_bytes=size_bytes,
    )


def _record_from_rdkit_mol(molecule_id: str, raw_record: dict[str, Any], rd_mol: Any, source_file: Path) -> dict[str, Any]:
    return {
        "molecule_id": molecule_id,
        "smiles": raw_record.get("smiles"),
        "source_file": str(source_file),
        "dataset_kind": "geom_drugs_full_database_raw_rdkit_anchor",
        "atom_symbols": [atom.GetSymbol() for atom in rd_mol.GetAtoms()],
        "atomic_numbers": [int(atom.GetAtomicNum()) for atom in rd_mol.GetAtoms()],
        "formal_charges": [int(atom.GetFormalCharge()) for atom in rd_mol.GetAtoms()],
        "bonds": [
            [
                int(bond.GetBeginAtomIdx()),
                int(bond.GetEndAtomIdx()),
                _json_bond_order(bond.GetBondTypeAsDouble()),
            ]
            for bond in rd_mol.GetBonds()
        ],
        "ring_atom_indices": [atom.GetIdx() for atom in rd_mol.GetAtoms() if atom.IsInRing()],
        "rotatable_bond_count": _rotatable_bond_count(raw_record.get("smiles")),
        "conformers": [],
        "source_metadata": {},
    }


def _looks_like_rdkit_mol(value: Any) -> bool:
    return all(
        hasattr(value, method)
        for method in ["GetNumAtoms", "GetNumConformers", "GetAtoms", "GetBonds", "GetConformer"]
    )


def _coords_from_rdkit_mol(rd_mol: Any) -> list[list[float]]:
    conformer = rd_mol.GetConformer()
    coords = []
    for atom_index in range(rd_mol.GetNumAtoms()):
        point = conformer.GetAtomPosition(atom_index)
        coords.append([float(point.x), float(point.y), float(point.z)])
    return coords


def _json_bond_order(order: float) -> int | float:
    rounded = round(float(order), 3)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def _rotatable_bond_count(smiles: Any) -> int | None:
    if not smiles:
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import Lipinski
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return int(Lipinski.NumRotatableBonds(mol))


def _load_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on runtime.
        raise RuntimeError("PyTorch is required to generate conformers from checkpoints") from exc
    return torch


def _resolve_device(torch: Any, requested: str) -> str:
    device = str(requested)
    if device == "auto":
        device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
    if device.startswith("cuda") and not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested for AMR/COV evaluation but is unavailable")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError(f"Unsupported torch device: {requested}")
    return device


def _load_model(torch: Any, checkpoint_path: Path, device: str) -> tuple[Any, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = torch.nn.Linear(4, 3)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def _generate_conformers(
    torch: Any,
    model: Any,
    config: dict[str, Any],
    molecule: ReferenceMolecule,
    split_type: str,
    method: str,
    seed: int,
    molecule_index: int,
    generated_conformers_per_mol: int,
    device: str,
) -> list[list[list[float]]]:
    source_seed = _generation_seed(split_type=split_type, method=method, seed=seed, molecule_index=molecule_index)
    corruptor = build_corruptor(config, source_seed)
    t_rng = random.Random(source_seed + 17_071)
    generated = []
    for _ in range(generated_conformers_per_mol):
        x0 = corruptor.corrupt(molecule.record, molecule.anchor_coords)
        t = t_rng.random()
        xt = _interpolate(x0, molecule.anchor_coords, t)
        xt_tensor = torch.tensor(xt, dtype=torch.float32, device=device)
        t_column = torch.full((xt_tensor.shape[0], 1), float(t), dtype=torch.float32, device=device)
        prediction = model(torch.cat([xt_tensor, t_column], dim=1))
        sampled = xt_tensor + FORMAL_10000_SAMPLE_STEP_SIZE * prediction
        generated.append(sampled.detach().cpu().tolist())
    return generated


def _generation_seed(split_type: str, method: str, seed: int, molecule_index: int) -> int:
    split_offset = {"scaffold": 1_000_000, "random": 2_000_000}[split_type]
    method_offset = {method_name: idx * 100_000 for idx, method_name in enumerate(METHODS)}[method]
    return int(seed) + split_offset + method_offset + molecule_index * 101


def _interpolate(x0: list[list[float]], x1: list[list[float]], t: float) -> list[list[float]]:
    return [[(1.0 - t) * float(a) + t * float(b) for a, b in zip(row0, row1)] for row0, row1 in zip(x0, x1)]


def pairwise_heavy_atom_rmsd(
    generated_coords: list[list[list[float]]],
    reference_coords: list[list[list[float]]],
    heavy_atom_indices: list[int],
) -> np.ndarray:
    if not generated_coords:
        raise ValueError("At least one generated conformer is required")
    if not reference_coords:
        raise ValueError("At least one reference conformer is required")
    matrix = np.empty((len(generated_coords), len(reference_coords)), dtype=float)
    for generated_index, generated in enumerate(generated_coords):
        for reference_index, reference in enumerate(reference_coords):
            matrix[generated_index, reference_index] = heavy_atom_aligned_rmsd(
                generated,
                reference,
                heavy_atom_indices,
            )
    return matrix


def heavy_atom_aligned_rmsd(
    coords_a: list[list[float]],
    coords_b: list[list[float]],
    heavy_atom_indices: list[int],
) -> float:
    if len(coords_a) != len(coords_b):
        raise ValueError("RMSD coordinate arrays must have the same atom count")
    if not heavy_atom_indices:
        raise ValueError("RMSD requires at least one heavy atom")
    a = np.asarray([coords_a[index] for index in heavy_atom_indices], dtype=float)
    b = np.asarray([coords_b[index] for index in heavy_atom_indices], dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("RMSD coordinate arrays must be shaped [n_atoms, 3]")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("RMSD coordinates must be finite")
    return float(_kabsch_rmsd(a, b))


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a - a.mean(axis=0)
    b_centered = b - b.mean(axis=0)
    covariance = a_centered.T @ b_centered
    v, _s, wt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(v @ wt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = v @ correction @ wt
    aligned = a_centered @ rotation
    diff = aligned - b_centered
    return math.sqrt(float(np.mean(np.sum(diff * diff, axis=1))))


def compute_amr_cov_from_pairwise(pairwise_rmsd: np.ndarray, delta: float = 0.75) -> dict[str, float]:
    matrix = np.asarray(pairwise_rmsd, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ValueError("pairwise_rmsd must be a non-empty 2D array")
    if not np.isfinite(matrix).all():
        raise ValueError("pairwise_rmsd must contain only finite values")
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    ref_min = matrix.min(axis=0)
    gen_min = matrix.min(axis=1)
    return {
        "cov_r": float(np.mean(ref_min <= delta)),
        "amr_r": float(np.mean(ref_min)),
        "cov_p": float(np.mean(gen_min <= delta)),
        "amr_p": float(np.mean(gen_min)),
    }


def _metric_summary(per_molecule: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    output = {}
    for metric in ["cov_r", "amr_r", "cov_p", "amr_p"]:
        values = [float(row["metrics"][metric]) for row in per_molecule]
        output[metric] = _numeric_summary(values)
    return output


def _count_summary(values: Any) -> dict[str, float | int | None]:
    items = [int(value) for value in values]
    if not items:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {"min": min(items), "max": max(items), "mean": mean(items), "median": median(items)}


def _numeric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _coverage_summary(summary: dict[str, Any], max_molecules: int | None) -> dict[str, Any]:
    by_split = {}
    for split_type, split_summary in summary["split_summaries"].items():
        full_test = max_molecules is None and split_summary["reference_loaded_molecule_count"] == split_summary[
            "test_molecule_count_in_split"
        ]
        by_split[split_type] = {
            "full_test_split_covered": full_test,
            "test_molecule_count_in_split": split_summary["test_molecule_count_in_split"],
            "selected_molecule_count": split_summary["selected_molecule_count"],
            "reference_loaded_molecule_count": split_summary["reference_loaded_molecule_count"],
            "reference_skipped_molecule_count": len(split_summary["reference_skipped_molecules"]),
        }
    return {
        "all_requested_splits_full_test_covered": bool(by_split) and all(
            row["full_test_split_covered"] for row in by_split.values()
        ),
        "by_split": by_split,
    }


def _manifest_from_summary(
    summary: dict[str, Any],
    summary_path: Path,
    report_path: Path,
    figure_path: Path,
) -> dict[str, Any]:
    return {
        "run_name": summary["run_name"],
        "result_kind": summary["result_kind"],
        "benchmark_result": False,
        "execution_mode": summary["execution_mode"],
        "hardware": summary["hardware"],
        "scheduler": None,
        "slurm_account": None,
        "git_commit": _git_commit(),
        "config_path": None,
        "dataset_scope": summary["dataset_scope"],
        "dataset": summary["dataset"],
        "evaluation": summary["evaluation"],
        "methods": summary["methods"],
        "input_splits": summary["input_splits"],
        "input_result_paths": summary["input_result_paths"],
        "input_checkpoint_paths": summary["input_checkpoint_paths"],
        "metrics_output_path": str(summary_path),
        "report_path": str(report_path),
        "figure_path": str(figure_path),
        "figure_pdf_path": str(figure_path.with_suffix(".pdf")),
        "output_paths": summary["output_paths"],
        "coverage": summary["coverage"],
        "runtime_seconds": summary["runtime_seconds"],
        "training_executed": False,
        "raw_data_modified": False,
        "environment_modified": False,
        "notes": summary["notes"],
    }


def _write_markdown_outputs(summary: dict[str, Any], output_paths: dict[str, Path | None]) -> None:
    combined_report_path = output_paths["combined_report"]
    combined_figure_path = output_paths["combined_figure"]
    assert combined_report_path is not None
    assert combined_figure_path is not None
    _write_text(combined_report_path, _markdown_report(summary, combined_report_path, combined_figure_path))
    for split_type in summary["split_summaries"]:
        report_path = output_paths[f"{split_type}_report"]
        figure_path = output_paths[f"{split_type}_figure"]
        assert report_path is not None
        assert figure_path is not None
        _write_text(report_path, _markdown_split_report(summary, split_type, report_path, figure_path))


def _markdown_report(summary: dict[str, Any], report_path: Path, figure_path: Path) -> str:
    full_protocol = bool(summary["full_test_set_evaluation_under_current_protocol"])
    title = "Full Test-Set AMR/COV Evaluation" if full_protocol else "Bounded Full-Database AMR/COV Evaluation"
    scope = (
        "full test-set AMR/COV evaluation under the current repository protocol"
        if full_protocol
        else "bounded, non-benchmark AMR/COV evaluation"
    )
    lines = [
        f"# {title}",
        "",
        f"**Scope:** {scope} from existing full-database checkpoints.",
        "",
        f"- Result kind: `{summary['result_kind']}`",
        f"- Benchmark result: `{summary['benchmark_result']}`",
        f"- Broader benchmark claim: `{summary['broader_benchmark_claim']}`",
        f"- Coverage label: `{summary['coverage_label']}`",
        f"- Delta: `{summary['evaluation']['delta_angstrom']} Å`",
        f"- Generated conformers per molecule: `{summary['evaluation']['generated_conformers_per_molecule']}`",
        f"- Max reference conformers per molecule: `{summary['evaluation']['max_reference_conformers_per_molecule']}`",
        f"- Max molecules per split: `{summary['evaluation']['max_molecules_per_split']}`",
        f"- Chunk size: `{summary['evaluation']['chunk_size']}`",
        "- Training rerun: `False`",
        "- Raw data modified: `False`",
        "",
        "## Aggregate Metrics",
        "",
        "| Split | Method | Molecules | Ref confs mean/median | Gen confs | COV-R mean/median | AMR-R mean/median | COV-P mean/median | AMR-P mean/median |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_type in summary["split_summaries"]:
        methods = summary["split_summaries"][split_type]["methods"]
        for method in summary["methods"]:
            row = methods[method]
            metrics = row["metric_summary"]
            ref_counts = row["reference_conformers_per_molecule"]
            gen_counts = row["generated_conformers_per_molecule"]
            lines.append(
                f"| {split_type} | {METHOD_DISPLAY[method]} | {row['evaluated_molecule_count']} | "
                f"{_fmt_count(ref_counts['mean'])}/{_fmt_count(ref_counts['median'])} | "
                f"{_fmt_count(gen_counts['mean'])} | "
                f"{_fmt_pct(metrics['cov_r']['mean'])}/{_fmt_pct(metrics['cov_r']['median'])} | "
                f"{_fmt_float(metrics['amr_r']['mean'])}/{_fmt_float(metrics['amr_r']['median'])} | "
                f"{_fmt_pct(metrics['cov_p']['mean'])}/{_fmt_pct(metrics['cov_p']['median'])} | "
                f"{_fmt_float(metrics['amr_p']['mean'])}/{_fmt_float(metrics['amr_p']['median'])} |"
            )
    lines.extend(
        [
            "",
            f"![AMR/COV comparison]({_relative_to_report(report_path, figure_path)})",
            "",
            f"Figure source: `{figure_path}` and `{figure_path.with_suffix('.pdf')}`.",
            "",
            "Split-specific tables and figures:",
        ]
    )
    for split_type in summary["split_summaries"]:
        split_report = summary["output_paths"].get(f"{split_type}_report")
        split_figure = summary["output_paths"].get(f"{split_type}_figure")
        lines.append(f"- {split_type}: `{split_report}`, `{split_figure}`")
    lines.extend(
        [
            "",
            "## Skips",
            "",
            "| Split | Reference skips | Method skips |",
            "|---|---:|---:|",
        ]
    )
    for split_type, split_summary in summary["split_summaries"].items():
        method_skips = sum(row["skipped_molecule_count"] for row in split_summary["methods"].values())
        lines.append(f"| {split_type} | {len(split_summary['reference_skipped_molecules'])} | {method_skips} |")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Full-test labels apply only to the current repository protocol when all requested test molecules are covered.",
            "- Generated conformers are produced by the current one-step checkpoint sampling diagnostic, not a full ODE sampler.",
            "- Each molecule uses the first raw conformer as the source anchor while the raw conformer ensemble is used as the reference set.",
            "- Reference conformers are capped when `max_reference_conformers_per_molecule` is not null.",
            "- The evaluator streams raw pickle files and does not load the full processed `molecules.json` artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_split_report(
    summary: dict[str, Any],
    split_type: str,
    report_path: Path,
    figure_path: Path,
) -> str:
    split_summary = summary["split_summaries"][split_type]
    lines = [
        f"# {split_type.title()} Split AMR/COV Table",
        "",
        f"- Result kind: `{summary['result_kind']}`",
        f"- Coverage label: `{summary['coverage_label']}`",
        f"- Test molecules in split: `{split_summary['test_molecule_count_in_split']}`",
        f"- Selected molecules: `{split_summary['selected_molecule_count']}`",
        f"- Reference-loaded molecules: `{split_summary['reference_loaded_molecule_count']}`",
        f"- Reference skips: `{len(split_summary['reference_skipped_molecules'])}`",
        f"- Delta: `{summary['evaluation']['delta_angstrom']} Å`",
        "",
        "| Method | Molecules | Ref confs mean/median | Gen confs mean | COV-R mean/median | AMR-R mean/median | COV-P mean/median | AMR-P mean/median | Runtime sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in summary["methods"]:
        row = split_summary["methods"][method]
        metrics = row["metric_summary"]
        ref_counts = row["reference_conformers_per_molecule"]
        gen_counts = row["generated_conformers_per_molecule"]
        lines.append(
            f"| {METHOD_DISPLAY[method]} | {row['evaluated_molecule_count']} | "
            f"{_fmt_count(ref_counts['mean'])}/{_fmt_count(ref_counts['median'])} | "
            f"{_fmt_count(gen_counts['mean'])} | "
            f"{_fmt_pct(metrics['cov_r']['mean'])}/{_fmt_pct(metrics['cov_r']['median'])} | "
            f"{_fmt_float(metrics['amr_r']['mean'])}/{_fmt_float(metrics['amr_r']['median'])} | "
            f"{_fmt_pct(metrics['cov_p']['mean'])}/{_fmt_pct(metrics['cov_p']['median'])} | "
            f"{_fmt_float(metrics['amr_p']['mean'])}/{_fmt_float(metrics['amr_p']['median'])} | "
            f"{_fmt_float(row['runtime_seconds'])} |"
        )
    lines.extend(
        [
            "",
            f"![{split_type} AMR/COV comparison]({_relative_to_report(report_path, figure_path)})",
            "",
            f"Figure source: `{figure_path}` and `{figure_path.with_suffix('.pdf')}`.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_figure_outputs(summary: dict[str, Any], output_paths: dict[str, Path | None]) -> None:
    combined_figure_path = output_paths["combined_figure"]
    assert combined_figure_path is not None
    _write_figure(summary, combined_figure_path)
    for split_type in summary["split_summaries"]:
        split_figure_path = output_paths[f"{split_type}_figure"]
        assert split_figure_path is not None
        _write_split_figure(summary, split_type, split_figure_path)


def _write_figure(summary: dict[str, Any], figure_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("cov_r", "COV-R mean (%)", True),
        ("amr_r", "AMR-R mean (Å)", False),
        ("cov_p", "COV-P mean (%)", True),
        ("amr_p", "AMR-P mean (Å)", False),
    ]
    split_types = list(summary["split_summaries"])
    methods = list(summary["methods"])
    colors = {"scaffold": "#3182bd", "random": "#31a354"}
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))
    axes = axes.ravel()
    width = 0.34 if len(split_types) > 1 else 0.55
    x_positions = np.arange(len(methods))
    offsets = {
        split: (index - (len(split_types) - 1) / 2.0) * width
        for index, split in enumerate(split_types)
    }
    for ax, (metric, label, as_percent) in zip(axes, metrics):
        for split in split_types:
            values = []
            for method in methods:
                value = summary["split_summaries"][split]["methods"][method]["metric_summary"][metric]["mean"]
                values.append(np.nan if value is None else float(value) * (100.0 if as_percent else 1.0))
            ax.bar(
                x_positions + offsets[split],
                values,
                width=width * 0.92,
                color=colors.get(split, "#756bb1"),
                label=split,
                alpha=0.9,
            )
        ax.set_title(label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([METHOD_DISPLAY[method] for method in methods], rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.22)
        if metric.startswith("cov"):
            ax.set_ylim(0, 105)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.90, 0.52), frameon=False)
    title_prefix = (
        "Full test-set" if summary["full_test_set_evaluation_under_current_protocol"] else "Bounded"
    )
    fig.suptitle(f"{title_prefix} GEOM-Drugs AMR/COV from existing seed-0 checkpoints", y=0.98)
    fig.text(
        0.5,
        0.01,
        "Heavy-atom aligned RMSD; delta = 0.75 Å. Current one-step sampler protocol; lower AMR and higher COV are better.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 0.88, 0.94))
    fig.savefig(figure_path, bbox_inches="tight", dpi=220)
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_split_figure(summary: dict[str, Any], split_type: str, figure_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("cov_r", "COV-R mean (%)", True),
        ("amr_r", "AMR-R mean (Å)", False),
        ("cov_p", "COV-P mean (%)", True),
        ("amr_p", "AMR-P mean (Å)", False),
    ]
    method_colors = {
        "isotropic": "#756bb1",
        "harmonic_prior": "#2c7fb8",
        "torsion_only": "#de2d26",
        "stiffness_structured": "#31a354",
    }
    methods = list(summary["methods"])
    x_positions = np.arange(len(methods))
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.8))
    axes = axes.ravel()
    for ax, (metric, label, as_percent) in zip(axes, metrics):
        values = []
        for method in methods:
            value = summary["split_summaries"][split_type]["methods"][method]["metric_summary"][metric]["mean"]
            values.append(np.nan if value is None else float(value) * (100.0 if as_percent else 1.0))
        ax.bar(
            x_positions,
            values,
            color=[method_colors[method] for method in methods],
            alpha=0.9,
            width=0.72,
        )
        ax.set_title(label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([METHOD_DISPLAY[method] for method in methods], rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.22)
        if metric.startswith("cov"):
            ax.set_ylim(0, 105)
    title_prefix = (
        "Full test-set" if summary["full_test_set_evaluation_under_current_protocol"] else "Bounded"
    )
    fig.suptitle(f"{title_prefix} GEOM-Drugs AMR/COV: {split_type} split", y=0.98)
    fig.text(
        0.5,
        0.01,
        "Heavy-atom aligned RMSD; delta = 0.75 Å. Current one-step sampler protocol; lower AMR and higher COV are better.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(figure_path, bbox_inches="tight", dpi=220)
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _progress(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _log_text(summary: dict[str, Any], output_paths: dict[str, Path | None]) -> str:
    return (
        f"run_name={summary['run_name']}\n"
        f"result_kind={summary['result_kind']}\n"
        "benchmark_result=false\n"
        f"coverage_label={summary['coverage_label']}\n"
        "training_executed=false\n"
        "raw_data_modified=false\n"
        f"split_types={','.join(summary['split_summaries'])}\n"
        f"methods={','.join(summary['methods'])}\n"
        f"runtime_seconds={summary['runtime_seconds']}\n"
        f"summary={output_paths['summary']}\n"
        f"manifest={output_paths['manifest']}\n"
        f"combined_report={output_paths['combined_report']}\n"
        f"combined_figure={output_paths['combined_figure']}\n"
        f"per_molecule_jsonl={output_paths['per_molecule_jsonl']}\n"
    )


def _setup_matplotlib_cache() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path("experiments/cache/matplotlib")))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path("experiments/cache/xdg")))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def _relative_to_report(report_path: Path, path: Path) -> str:
    return os.path.relpath(path, start=report_path.parent)


def _fmt_float(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100.0:.1f}%"


def _fmt_count(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.1f}"


def _exception_detail(exc: BaseException) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:500]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


if __name__ == "__main__":
    main()
