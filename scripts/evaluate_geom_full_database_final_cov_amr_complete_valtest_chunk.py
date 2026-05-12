#!/usr/bin/env python
"""Evaluate one complete val/test COV/AMR chunk from an existing checkpoint."""

from __future__ import annotations

import argparse
import hashlib
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
from src.train.formal_10000_training import FORMAL_10000_SAMPLE_STEP_SIZE  # noqa: E402
from src.utils.config import load_config  # noqa: E402


DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_final_cov_amr_complete_valtest_plan.json")
PLAN_RESULT_KIND = "geom_drugs_full_database_final_cov_amr_complete_valtest_plan_not_benchmark"
CHUNK_RESULT_KIND = "geom_drugs_full_database_final_cov_amr_complete_valtest_chunk_not_benchmark"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-path", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=128)
    args = parser.parse_args()

    outputs = evaluate_chunk(args)
    for key, value in outputs.items():
        print(f"{key}: {value}")
    print("benchmark_result: false")
    print(
        "note: Evaluated one resumable complete val/test COV/AMR chunk from existing checkpoints; "
        "no training, scheduler submission, raw-data mutation, or environment mutation."
    )


def evaluate_chunk(args: argparse.Namespace) -> dict[str, str]:
    start = time.time()
    plan_path = Path(args.plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _validate_plan(plan)
    task = _resolve_task(plan, int(args.task_id))
    input_row = _resolve_input_row(plan, task)

    jsonl_path = Path(task["jsonl_path"])
    manifest_path = Path(task["manifest_path"])
    status_path = Path(task["status_path"])
    for path in (jsonl_path, manifest_path, status_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    if _is_completed(status_path, jsonl_path) and not bool(args.overwrite):
        return {
            "status": "skipped_completed",
            "task_id": str(task["task_id"]),
            "jsonl": str(jsonl_path),
            "manifest": str(manifest_path),
            "status_path": str(status_path),
        }
    jsonl_path.unlink(missing_ok=True)

    _write_json(
        status_path,
        {
            "status": "running",
            "task_id": int(task["task_id"]),
            "started_at_unix": start,
            "plan_path": str(plan_path),
            "jsonl_path": str(jsonl_path),
        },
    )

    torch = _load_torch()
    device = _resolve_device(torch, str(args.device))
    if device == "cpu":
        torch.set_num_threads(1)

    result_path = Path(input_row["result_path"])
    checkpoint_path = Path(input_row["best_checkpoint_path"])
    _validate_result(result_path=result_path, checkpoint_path=checkpoint_path, task=task)
    config = _load_config(input_row=input_row, task=task)
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
                    _status_row(task, molecule_id, global_index, "skipped", exc.reason, exc.detail),
                )
                continue
            except Exception as exc:
                counts["skipped_molecule_count"] += 1
                _append_jsonl(
                    jsonl_path,
                    _status_row(
                        task,
                        molecule_id,
                        global_index,
                        "skipped",
                        f"reference_reload_error:{type(exc).__name__}",
                        _exception_detail(exc),
                    ),
                )
                continue

            try:
                generated = _generate_conformers(
                    torch=torch,
                    model=model,
                    config=config,
                    molecule=molecule,
                    task=task,
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
                        **_base_row(task, molecule.molecule_id, global_index),
                        "status": "evaluated",
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
                    _status_row(
                        task,
                        molecule_id,
                        global_index,
                        "failed",
                        f"method_evaluation_error:{type(exc).__name__}",
                        _exception_detail(exc),
                    ),
                )

            if args.progress_every and (
                (local_index + 1) % int(args.progress_every) == 0 or local_index + 1 == len(molecule_ids)
            ):
                print(
                    "complete_valtest_cov_amr_chunk "
                    f"task={task['task_id']} progress {local_index + 1}/{len(molecule_ids)} "
                    f"evaluated={counts['evaluated_molecule_count']} skipped={counts['skipped_molecule_count']} "
                    f"failed={counts['failed_molecule_count']}"
                )

    runtime_seconds = time.time() - start
    manifest = {
        "result_kind": CHUNK_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "scheduled_or_local_complete_valtest_cov_amr_chunk",
        "hardware": "cpu_only" if device == "cpu" else "device_requested",
        "gpu_used_by_chunk": device.startswith("cuda"),
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
        "standalone_task_id": int(task["task_id"]),
        "slurm_account": os.environ.get("STIFFLOW_SLURM_ACCOUNT") or os.environ.get("SLURM_JOB_ACCOUNT"),
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_chunk": False,
        "sampling_executed_by_chunk": True,
        "dataset_scope": FULL_DATABASE_SCOPE,
        "plan_path": str(plan_path),
        "task": task,
        "input": input_row,
        "checkpoint_result_kind": checkpoint_payload.get("result_kind"),
        "checkpoint_benchmark_result": checkpoint_payload.get("benchmark_result"),
        "jsonl_path": str(jsonl_path),
        "status_path": str(status_path),
        "counts": counts,
        "runtime_seconds": runtime_seconds,
        "evaluation": plan["evaluation"],
        "notes": [
            "The checkpoint is reloaded read-only; training is not rerun.",
            "Rows are streamed to JSONL so no full-matrix molecule/conformer accumulation is used.",
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
        "manifest": str(manifest_path),
        "status_path": str(status_path),
    }


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("result_kind") != PLAN_RESULT_KIND:
        raise ValueError("Unexpected complete val/test COV/AMR plan result_kind")
    if plan.get("benchmark_result") is not False:
        raise ValueError("Complete val/test COV/AMR plan must be benchmark_result=false")
    if plan.get("dataset_scope") != FULL_DATABASE_SCOPE:
        raise ValueError("Complete val/test COV/AMR plan has wrong dataset scope")
    if plan.get("train_partition_included") is not False:
        raise ValueError("Default complete val/test run must not include train")
    if plan.get("chunking", {}).get("slurm_array_used") is not False:
        raise ValueError("Array-style COV/AMR plans are disallowed")


def _resolve_task(plan: dict[str, Any], task_id: int) -> dict[str, Any]:
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Plan is missing tasks")
    if task_id < 0 or task_id >= len(tasks):
        raise IndexError(f"task_id {task_id} is outside plan range 0-{len(tasks) - 1}")
    task = tasks[task_id]
    if int(task["task_id"]) != int(task_id):
        raise ValueError(f"Task list is not indexed deterministically at task_id={task_id}")
    return task


def _resolve_input_row(plan: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    key = f"{task['entry_label']}/{task['split_type']}/seed{task['seed']}"
    try:
        row = plan["entry_seed_inputs"][key]
    except KeyError as exc:
        raise KeyError(f"Plan is missing entry_seed_inputs row {key}") from exc
    if not row.get("ready"):
        raise FileNotFoundError(f"Entry is not checkpoint-ready for task {task['task_id']}: {key}")
    return row


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


def _validate_result(result_path: Path, checkpoint_path: Path, task: dict[str, Any]) -> None:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("optimizer_name") != "AdamW":
        raise ValueError(f"Refusing non-AdamW result for {task['entry_label']}: {result_path}")
    if abs(float(payload.get("learning_rate")) - 1.0e-3) > 1.0e-12:
        raise ValueError(f"Refusing non-lr1e-3 result for {task['entry_label']}: {result_path}")
    expected_mode = "hybrid_harmonic_stiffness_v1" if task["role"] == "hybrid_candidate" else task["method"]
    if payload.get("corruption_mode") != expected_mode:
        raise ValueError(f"Result method mismatch for {task['entry_label']}: {result_path}")
    if task["role"] == "hybrid_candidate":
        strength = payload.get("hybrid_source_strength")
        if not isinstance(strength, (int, float)) or abs(float(strength) - float(task["lambda_value"])) > 1.0e-12:
            raise ValueError(f"Hybrid lambda mismatch for {task['entry_label']}: {result_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for task {task['task_id']}: {checkpoint_path}")


def _load_config(input_row: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    path = Path(input_row["config_path"])
    config = load_config(path)
    expected_mode = "hybrid_harmonic_stiffness_v1" if task["role"] == "hybrid_candidate" else task["method"]
    if config["corruption"]["mode"] != expected_mode:
        raise ValueError(f"Config {path} does not match expected corruption mode {expected_mode}")
    if config["data"]["split_type"] != task["split_type"]:
        raise ValueError(f"Config {path} does not match split {task['split_type']}")
    config["experiment"]["seed"] = int(task["seed"])
    return config


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
    task: dict[str, Any],
    molecule_global_index: int,
    generated_conformers_per_mol: int,
    device: str,
) -> list[list[list[float]]]:
    source_seed = _generation_seed(task=task, molecule_global_index=molecule_global_index)
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


def _generation_seed(task: dict[str, Any], molecule_global_index: int) -> int:
    token = (
        f"{task['entry_label']}|{task['split_type']}|{task['partition']}|"
        f"seed{task['seed']}|mol{molecule_global_index}"
    )
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 2_147_483_647


def _base_row(task: dict[str, Any], molecule_id: str, global_index: int) -> dict[str, Any]:
    return {
        "task_id": int(task["task_id"]),
        "split_type": task["split_type"],
        "partition": task["partition"],
        "entry_label": task["entry_label"],
        "method": task["method"],
        "role": task["role"],
        "lambda_value": task.get("lambda_value"),
        "seed": int(task["seed"]),
        "chunk_index": int(task["chunk_index"]),
        "chunk_start": int(task["chunk_start"]),
        "chunk_end": int(task["chunk_end"]),
        "molecule_global_index": global_index,
        "molecule_id": molecule_id,
    }


def _status_row(
    task: dict[str, Any],
    molecule_id: str,
    global_index: int,
    status: str,
    reason: str,
    detail: str | None,
) -> dict[str, Any]:
    return {**_base_row(task, molecule_id, global_index), "status": status, "reason": reason, "detail": detail}


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for complete val/test COV/AMR checkpoint evaluation") from exc
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
