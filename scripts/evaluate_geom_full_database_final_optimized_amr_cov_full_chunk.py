#!/usr/bin/env python
"""Evaluate one chunk of the full final-optimized val/test AMR/COV plan."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corruption.base import build_corruptor  # noqa: E402
from src.data.geom_full_database import FULL_DATABASE_SCOPE  # noqa: E402
from src.eval.amr_cov import (  # noqa: E402
    MoleculeSkip,
    _load_reference_molecule,
    compute_amr_cov_from_pairwise,
    pairwise_heavy_atom_rmsd,
)
from src.eval.final_optimized_all_methods import METHOD_LABELS, SEEDS, _candidate_paths  # noqa: E402
from src.train.formal_10000_training import FORMAL_10000_SAMPLE_STEP_SIZE  # noqa: E402
from src.utils.config import load_config  # noqa: E402


DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_final_optimized_amr_cov_full_plan.json")
METHODS = (
    "hybrid_harmonic_stiffness_v1_lambda500",
    "harmonic_prior",
    "stiffness_structured",
    "isotropic",
    "torsion_only",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-path", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=128)
    args = parser.parse_args()

    outputs = evaluate_chunk(args)
    for key, value in outputs.items():
        print(f"{key}: {value}")
    print("benchmark_result: false")
    print(
        "note: Evaluated one resumable full val/test AMR/COV chunk from existing checkpoints; "
        "no training, scheduler submission by this script, raw-data mutation, or environment mutation."
    )


def evaluate_chunk(args: argparse.Namespace) -> dict[str, str]:
    start = time.time()
    plan_path = Path(args.plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _validate_plan(plan)
    if not plan["training_result_readiness"]["evaluation_ready"]:
        raise FileNotFoundError("Full AMR/COV evaluation plan is not ready; missing optimized results/checkpoints.")
    task = _resolve_task(plan, args.task_id)
    jsonl_path = Path(task["jsonl_path"])
    manifest_path = Path(task["manifest_path"])
    status_path = Path(task["status_path"])
    for path in (jsonl_path, manifest_path, status_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    if _is_completed(status_path=status_path, jsonl_path=jsonl_path) and not bool(args.overwrite):
        return {
            "status": "skipped_completed",
            "task_id": str(task["task_id"]),
            "jsonl": str(jsonl_path),
            "status_path": str(status_path),
            "manifest": str(manifest_path),
        }

    if bool(args.overwrite):
        jsonl_path.write_text("", encoding="utf-8")
    else:
        jsonl_path.unlink(missing_ok=True)

    _write_json(
        status_path,
        {
            "status": "running",
            "task_id": int(task["task_id"]),
            "started_at_unix": start,
            "plan_path": str(plan_path),
            "jsonl_path": str(jsonl_path),
            "manifest_path": str(manifest_path),
        },
    )

    torch = _load_torch()
    device = _resolve_device(torch, str(args.device))
    if device == "cpu":
        torch.set_num_threads(1)

    result_path, checkpoint_path = _resolve_result_and_checkpoint(
        split_type=task["split_type"],
        method=task["method"],
        seed=int(task["seed"]),
    )
    config = _load_method_config(split_type=task["split_type"], method=task["method"], seed=int(task["seed"]))
    model, checkpoint_payload = _load_model(torch=torch, checkpoint_path=checkpoint_path, device=device)
    split_payload = _load_split(Path(plan["split_paths"][task["split_type"]]), task["split_type"])
    molecule_ids = split_payload[task["partition"]][int(task["chunk_start"]) : int(task["chunk_end"])]

    counts = {
        "planned_molecule_count": len(molecule_ids),
        "attempted_molecule_count": 0,
        "evaluated_molecule_count": 0,
        "skipped_molecule_count": 0,
        "failed_molecule_count": 0,
    }
    raw_root = Path(plan["raw_root"])
    max_reference_conformers = int(plan["evaluation"]["max_reference_conformers_per_molecule"])
    max_pickle_bytes = int(plan["evaluation"]["max_pickle_bytes"])
    generated_conformers_per_mol = int(plan["evaluation"]["generated_conformers_per_molecule"])
    delta = float(plan["evaluation"]["delta_angstrom"])

    with torch.no_grad():
        for local_index, molecule_id in enumerate(molecule_ids):
            global_index = int(task["chunk_start"]) + local_index
            counts["attempted_molecule_count"] += 1
            try:
                molecule = _load_reference_molecule(
                    molecule_id=molecule_id,
                    raw_root=raw_root,
                    max_reference_conformers=max_reference_conformers,
                    max_pickle_bytes=max_pickle_bytes,
                )
            except MoleculeSkip as exc:
                counts["skipped_molecule_count"] += 1
                _append_jsonl(
                    jsonl_path,
                    _skip_row(
                        task=task,
                        molecule_id=molecule_id,
                        global_index=global_index,
                        status="skipped",
                        reason=exc.reason,
                        detail=exc.detail,
                    ),
                )
                continue
            except Exception as exc:
                counts["skipped_molecule_count"] += 1
                _append_jsonl(
                    jsonl_path,
                    _skip_row(
                        task=task,
                        molecule_id=molecule_id,
                        global_index=global_index,
                        status="skipped",
                        reason=f"reference_reload_error:{type(exc).__name__}",
                        detail=_exception_detail(exc),
                    ),
                )
                continue

            try:
                generated = _generate_conformers(
                    torch=torch,
                    model=model,
                    config=config,
                    molecule=molecule,
                    split_type=task["split_type"],
                    partition=task["partition"],
                    method=task["method"],
                    seed=int(task["seed"]),
                    molecule_global_index=global_index,
                    generated_conformers_per_mol=generated_conformers_per_mol,
                    device=device,
                )
                pairwise = pairwise_heavy_atom_rmsd(
                    generated_coords=generated,
                    reference_coords=molecule.reference_coords,
                    heavy_atom_indices=molecule.heavy_atom_indices,
                )
                metrics = compute_amr_cov_from_pairwise(pairwise, delta=delta)
                counts["evaluated_molecule_count"] += 1
                _append_jsonl(
                    jsonl_path,
                    {
                        "status": "evaluated",
                        "task_id": int(task["task_id"]),
                        "split_type": task["split_type"],
                        "partition": task["partition"],
                        "method": task["method"],
                        "seed": int(task["seed"]),
                        "chunk_index": int(task["chunk_index"]),
                        "chunk_start": int(task["chunk_start"]),
                        "chunk_end": int(task["chunk_end"]),
                        "molecule_global_index": global_index,
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
                    },
                )
            except Exception as exc:
                counts["failed_molecule_count"] += 1
                _append_jsonl(
                    jsonl_path,
                    _skip_row(
                        task=task,
                        molecule_id=molecule_id,
                        global_index=global_index,
                        status="failed",
                        reason=f"method_evaluation_error:{type(exc).__name__}",
                        detail=_exception_detail(exc),
                    ),
                )

            if args.progress_every and (
                (local_index + 1) % int(args.progress_every) == 0 or local_index + 1 == len(molecule_ids)
            ):
                print(
                    "amr_cov_full_chunk "
                    f"task={task['task_id']} progress {local_index + 1}/{len(molecule_ids)} "
                    f"evaluated={counts['evaluated_molecule_count']} skipped={counts['skipped_molecule_count']} "
                    f"failed={counts['failed_molecule_count']}"
                )

    runtime_seconds = time.time() - start
    manifest = {
        "result_kind": "geom_drugs_full_database_final_optimized_amr_cov_full_chunk_manifest_not_benchmark",
        "benchmark_result": False,
        "execution_mode": "scheduled_or_local_full_amr_cov_chunk",
        "hardware": "cpu_only" if device == "cpu" else "device_requested",
        "gpu_used_by_chunk": device.startswith("cuda"),
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
        "standalone_task_id": int(task["task_id"]),
        "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_chunk": False,
        "sampling_executed_by_chunk": True,
        "dataset_scope": FULL_DATABASE_SCOPE,
        "plan_path": str(plan_path),
        "task": task,
        "result_path": str(result_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_result_kind": checkpoint_payload.get("result_kind"),
        "checkpoint_benchmark_result": checkpoint_payload.get("benchmark_result"),
        "config_path": str(_method_config_path(task["split_type"], task["method"])),
        "jsonl_path": str(jsonl_path),
        "status_path": str(status_path),
        "counts": counts,
        "runtime_seconds": runtime_seconds,
        "evaluation": plan["evaluation"],
        "notes": [
            "This chunk reloads an existing validation-selected checkpoint; training is not rerun.",
            "Rows are streamed to JSONL; no all-molecule or all-conformer accumulation is used.",
            "Raw GEOM pickles are read only.",
        ],
    }
    _write_json(manifest_path, manifest)
    _write_json(
        status_path,
        {
            "status": "completed",
            "task_id": int(task["task_id"]),
            "completed_at_unix": time.time(),
            "runtime_seconds": runtime_seconds,
            "plan_path": str(plan_path),
            "jsonl_path": str(jsonl_path),
            "manifest_path": str(manifest_path),
            "counts": counts,
        },
    )
    return {
        "status": "completed",
        "task_id": str(task["task_id"]),
        "jsonl": str(jsonl_path),
        "status_path": str(status_path),
        "manifest": str(manifest_path),
    }


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("result_kind") != "geom_drugs_full_database_final_optimized_amr_cov_full_plan_not_benchmark":
        raise ValueError("Unexpected full AMR/COV plan result_kind")
    if plan.get("benchmark_result") is not False:
        raise ValueError("Full AMR/COV plan must be benchmark_result=false")
    if plan.get("dataset_scope") != FULL_DATABASE_SCOPE:
        raise ValueError("Full AMR/COV plan has wrong dataset scope")
    if tuple(plan.get("methods", [])) != METHODS:
        raise ValueError("Full AMR/COV plan method order changed unexpectedly")
    if tuple(plan.get("seeds", [])) != SEEDS:
        raise ValueError("Full AMR/COV plan seed set does not match final optimized matrix")


def _resolve_task(plan: dict[str, Any], task_id: int | None) -> dict[str, Any]:
    if task_id is None:
        raise ValueError("Provide --task-id. Full AMR/COV execution uses standalone single-job wrappers, not SLURM arrays.")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Full AMR/COV plan is missing task list")
    if task_id < 0 or task_id >= len(tasks):
        raise IndexError(f"task_id {task_id} is outside plan task range 0-{len(tasks) - 1}")
    task = tasks[task_id]
    if int(task["task_id"]) != int(task_id):
        raise ValueError(f"Task list is not indexed deterministically at task_id={task_id}")
    return task


def _is_completed(status_path: Path, jsonl_path: Path) -> bool:
    if not status_path.exists() or not jsonl_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return status.get("status") == "completed"


def _load_split(path: Path, split_type: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split_type") != split_type:
        raise ValueError(f"Split file {path} has unexpected split_type={payload.get('split_type')!r}")
    if payload.get("split_scope") != FULL_DATABASE_SCOPE:
        raise ValueError(f"Split file {path} has unexpected split_scope={payload.get('split_scope')!r}")
    return payload


def _load_method_config(split_type: str, method: str, seed: int) -> dict[str, Any]:
    path = _method_config_path(split_type=split_type, method=method)
    config = load_config(path)
    expected_mode = "hybrid_harmonic_stiffness_v1" if method.endswith("lambda500") else method
    if config["corruption"]["mode"] != expected_mode:
        raise ValueError(f"Config {path} does not match {method}")
    if config["data"]["split_type"] != split_type:
        raise ValueError(f"Config {path} does not match split {split_type}")
    config["experiment"]["seed"] = int(seed)
    return config


def _method_config_path(split_type: str, method: str) -> Path:
    split_label = "random" if split_type == "random" else "scaffold"
    config_label = METHOD_LABELS[method]
    return Path("configs/train/final_optimized") / f"geom_full_database_{split_label}_{config_label}_adamw_lr1e3.yaml"


def _resolve_result_and_checkpoint(split_type: str, method: str, seed: int) -> tuple[Path, Path]:
    candidates = _candidate_paths(Path("experiments/results"), split_type=split_type, seed=seed, method=method)
    result_path = next((item["path"] for item in candidates if item["path"].exists()), None)
    if result_path is None:
        raise FileNotFoundError(f"Missing optimized result JSON for {split_type} {method} seed{seed}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(payload["best_checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing optimized checkpoint from {result_path}: {checkpoint_path}")
    return result_path, checkpoint_path


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
    molecule: Any,
    split_type: str,
    partition: str,
    method: str,
    seed: int,
    molecule_global_index: int,
    generated_conformers_per_mol: int,
    device: str,
) -> list[list[list[float]]]:
    source_seed = _generation_seed(
        split_type=split_type,
        partition=partition,
        method=method,
        seed=seed,
        molecule_global_index=molecule_global_index,
    )
    corruptor = build_corruptor(config, source_seed)
    t_rng = random.Random(source_seed + 17_071)
    generated = []
    for _ in range(generated_conformers_per_mol):
        x0 = corruptor.corrupt(molecule.record, molecule.anchor_coords)
        t = t_rng.random()
        xt = [[(1.0 - t) * float(a) + t * float(b) for a, b in zip(row0, row1)] for row0, row1 in zip(x0, molecule.anchor_coords)]
        xt_tensor = torch.tensor(xt, dtype=torch.float32, device=device)
        t_column = torch.full((xt_tensor.shape[0], 1), float(t), dtype=torch.float32, device=device)
        prediction = model(torch.cat([xt_tensor, t_column], dim=1))
        sampled = xt_tensor + FORMAL_10000_SAMPLE_STEP_SIZE * prediction
        generated.append(sampled.detach().cpu().tolist())
    return generated


def _generation_seed(split_type: str, partition: str, method: str, seed: int, molecule_global_index: int) -> int:
    split_offset = {"random": 1_000_000, "scaffold": 2_000_000}[split_type]
    partition_offset = {"train": 0, "val": 10_000, "test": 20_000}[partition]
    method_offset = {method_name: idx * 100_000 for idx, method_name in enumerate(METHODS)}[method]
    return int(seed) + split_offset + partition_offset + method_offset + int(molecule_global_index) * 101


def _skip_row(
    task: dict[str, Any],
    molecule_id: str,
    global_index: int,
    status: str,
    reason: str,
    detail: str | None,
) -> dict[str, Any]:
    return {
        "status": status,
        "task_id": int(task["task_id"]),
        "split_type": task["split_type"],
        "partition": task["partition"],
        "method": task["method"],
        "seed": int(task["seed"]),
        "chunk_index": int(task["chunk_index"]),
        "chunk_start": int(task["chunk_start"]),
        "chunk_end": int(task["chunk_end"]),
        "molecule_global_index": global_index,
        "molecule_id": molecule_id,
        "reason": reason,
        "detail": detail,
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for full AMR/COV checkpoint evaluation") from exc
    return torch


def _resolve_device(torch: Any, requested: str) -> str:
    device = str(requested)
    if device == "auto":
        device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
    if device.startswith("cuda") and not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but is unavailable")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError(f"Unsupported torch device: {requested}")
    return device


def _exception_detail(exc: BaseException) -> str:
    detail = str(exc)
    if detail:
        return detail
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


if __name__ == "__main__":
    main()
