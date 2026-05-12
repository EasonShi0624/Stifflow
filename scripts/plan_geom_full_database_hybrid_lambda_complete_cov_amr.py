#!/usr/bin/env python
"""Plan complete val/test COV/AMR chunks for hybrid lambda comparison."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.geom_full_database import (  # noqa: E402
    FULL_DATABASE_ARTIFACT_PATH,
    FULL_DATABASE_SCAFFOLD_SPLIT_PATH,
    FULL_DATABASE_SCOPE,
    FULL_DATABASE_SPLIT_PATH,
)
from src.eval.final_optimized_all_methods import SEEDS  # noqa: E402


DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_hybrid_lambda_complete_cov_amr_plan.json")
DEFAULT_REPORT_PATH = Path("reports/tables/geom_full_database_hybrid_lambda_complete_cov_amr_plan.md")
DEFAULT_CHUNKS_DIR = Path("experiments/results/geom_full_database_hybrid_lambda_complete_cov_amr_chunks")
DEFAULT_SUMMARY_PATH = Path("experiments/results/geom_full_database_hybrid_lambda_complete_cov_amr_summary.json")
DEFAULT_SUMMARY_REPORT_PATH = Path("reports/tables/geom_full_database_hybrid_lambda_complete_cov_amr_summary.md")
DEFAULT_WRAPPER_DIR = Path("scripts")
DEFAULT_COMMANDS_PATH = Path("scripts/geom_full_database_hybrid_lambda_complete_cov_amr_single_job_commands.md")
WRAPPER_PREFIX = "slurm_geom_full_database_hybrid_lambda_complete_cov_amr_chunk"
SPLIT_PATHS = {"random": FULL_DATABASE_SPLIT_PATH, "scaffold": FULL_DATABASE_SCAFFOLD_SPLIT_PATH}
SPLITS = ("random", "scaffold")
PARTITIONS = ("val", "test")
REQUESTED_ENTRIES = (
    {
        "entry_label": "harmonic_prior",
        "method": "harmonic_prior",
        "role": "reference",
        "run_label": "harmonic",
        "lambda_value": 0.0,
    },
    {
        "entry_label": "hybrid_lambda0025",
        "method": "hybrid_harmonic_stiffness_v1",
        "role": "hybrid_candidate",
        "lambda_label": "lambda0025",
        "lambda_value": 0.025,
    },
    {
        "entry_label": "hybrid_lambda010",
        "method": "hybrid_harmonic_stiffness_v1",
        "role": "hybrid_candidate",
        "lambda_label": "lambda010",
        "lambda_value": 0.1,
    },
    {
        "entry_label": "hybrid_lambda100",
        "method": "hybrid_harmonic_stiffness_v1",
        "role": "hybrid_candidate",
        "lambda_label": "lambda100",
        "lambda_value": 1.0,
    },
    {
        "entry_label": "hybrid_lambda300",
        "method": "hybrid_harmonic_stiffness_v1",
        "role": "hybrid_candidate",
        "lambda_label": "lambda300",
        "lambda_value": 3.0,
    },
    {
        "entry_label": "hybrid_lambda500",
        "method": "hybrid_harmonic_stiffness_v1",
        "role": "hybrid_candidate",
        "lambda_label": "lambda500",
        "lambda_value": 5.0,
    },
)
DEFAULT_CHUNK_SIZE = 512
DEFAULT_GENERATED_CONFORMERS = 20
DEFAULT_DELTA = 0.75
DEFAULT_MAX_REFERENCE_CONFORMERS = 40
DEFAULT_MAX_PICKLE_BYTES = 25_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--generated-conformers-per-mol", type=int, default=DEFAULT_GENERATED_CONFORMERS)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--max-reference-conformers-per-mol", type=int, default=DEFAULT_MAX_REFERENCE_CONFORMERS)
    parser.add_argument("--max-pickle-bytes", type=int, default=DEFAULT_MAX_PICKLE_BYTES)
    parser.add_argument("--raw-root", default="data/raw/geom_drugs")
    parser.add_argument("--chunks-dir", default=str(DEFAULT_CHUNKS_DIR))
    parser.add_argument("--wrapper-dir", default=str(DEFAULT_WRAPPER_DIR))
    parser.add_argument("--commands-path", default=str(DEFAULT_COMMANDS_PATH))
    parser.add_argument("--plan-path", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    plan = build_plan(args)
    plan_path = Path(args.plan_path)
    report_path = Path(args.report_path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(plan), encoding="utf-8")
    print(f"plan: {plan_path}")
    print(f"report: {report_path}")
    print("benchmark_result: false")
    print("note: Planning-only complete val/test COV/AMR lambda-comparison manifest; no model sampling, training, scheduler submission, raw-data mutation, or environment mutation.")


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    _validate_args(args)
    split_payloads = {split: _load_split(SPLIT_PATHS[split], split) for split in SPLITS}
    readiness = _entry_readiness()
    readiness_counts = _readiness_counts(readiness)
    complete_entries = [row for row in readiness["entries"] if row["complete"]]
    evaluation_ready = len(complete_entries) == len(REQUESTED_ENTRIES)
    entries_for_tasks = complete_entries if (evaluation_ready or bool(args.allow_partial)) else []
    chunks_dir = Path(args.chunks_dir)
    wrapper_dir = Path(args.wrapper_dir)
    commands_path = Path(args.commands_path)
    base_chunks, molecule_slots = _base_chunks(split_payloads, int(args.chunk_size))
    tasks = _tasks(base_chunks, entries_for_tasks, chunks_dir, wrapper_dir)
    total_molecule_slots = sum(sum(row.values()) for row in molecule_slots.values())
    full_task_count = len(base_chunks) * len(REQUESTED_ENTRIES) * len(SEEDS)
    full_method_seed_records = total_molecule_slots * len(REQUESTED_ENTRIES) * len(SEEDS)
    planned_method_seed_records = total_molecule_slots * len(entries_for_tasks) * len(SEEDS)
    return {
        "summary_name": "geom_full_database_hybrid_lambda_complete_cov_amr_plan",
        "result_kind": "geom_drugs_full_database_hybrid_lambda_complete_cov_amr_plan_not_benchmark",
        "benchmark_result": False,
        "execution_mode": "local_reporting_only_hybrid_lambda_complete_cov_amr_planning",
        "hardware": "reporting_only",
        "gpu_used_by_plan": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_plan": False,
        "sampling_executed_by_plan": False,
        "dataset_scope": FULL_DATABASE_SCOPE,
        "processed_artifact_path": str(FULL_DATABASE_ARTIFACT_PATH),
        "raw_root": str(args.raw_root),
        "split_paths": {split: str(SPLIT_PATHS[split]) for split in SPLITS},
        "splits": list(SPLITS),
        "partitions": list(PARTITIONS),
        "train_partition_included": False,
        "seeds": list(SEEDS),
        "requested_entries": list(REQUESTED_ENTRIES),
        "complete_entries": complete_entries,
        "entries_used_for_tasks": entries_for_tasks,
        "entry_readiness": readiness,
        "readiness_counts": readiness_counts,
        "entry_seed_inputs": _entry_seed_inputs(complete_entries),
        "evaluation_ready": evaluation_ready,
        "allow_partial": bool(args.allow_partial),
        "chunking": {
            "chunk_size": int(args.chunk_size),
            "base_molecule_chunk_count": len(base_chunks),
            "full_standalone_job_count_when_ready": full_task_count,
            "standalone_job_count": len(tasks),
            "task_unit": "one_entry_split_seed_partition_chunk",
            "deterministic_assignment": "split-file val/test molecule order sliced by chunk_start/chunk_end, then requested entry order, then seed order",
            "chunks_dir": str(chunks_dir),
            "wrapper_dir": str(wrapper_dir),
            "commands_path": str(commands_path),
            "resume_behavior": "chunk evaluator skips completed status/jsonl pairs unless --overwrite is passed",
            "slurm_array_used": False,
            "submission_style": "standalone_single_job_wrappers_only",
            "molecule_slots_per_chunk": {
                "target_chunk_size": int(args.chunk_size),
                "last_chunk_size_by_split_partition": _last_chunk_sizes(molecule_slots, int(args.chunk_size)),
            },
            "estimated_generated_conformers_per_full_chunk_per_task": int(args.chunk_size)
            * int(args.generated_conformers_per_mol),
        },
        "evaluation": {
            "metrics": ["COV-R", "COV-P", "AMR-R", "AMR-P"],
            "rmsd": "heavy_atom_aligned_kabsch",
            "delta_angstrom": float(args.delta),
            "generated_conformers_per_molecule": int(args.generated_conformers_per_mol),
            "max_reference_conformers_per_molecule": int(args.max_reference_conformers_per_mol),
            "max_pickle_bytes": int(args.max_pickle_bytes),
        },
        "coverage": {
            "molecule_slots_by_split_partition": molecule_slots,
            "total_molecule_slots": total_molecule_slots,
            "requested_entry_count": len(REQUESTED_ENTRIES),
            "entry_count": len(entries_for_tasks),
            "seed_count": len(SEEDS),
            "estimated_full_method_seed_molecule_records_when_ready": full_method_seed_records,
            "estimated_full_generated_conformer_count_when_ready": full_method_seed_records * int(args.generated_conformers_per_mol),
            "estimated_method_seed_molecule_records": planned_method_seed_records,
            "estimated_generated_conformer_count": planned_method_seed_records * int(args.generated_conformers_per_mol),
            "train_slots_excluded_by_split": {split: len(split_payloads[split]["train"]) for split in SPLITS},
        },
        "resource_estimate": {
            "cpu_memory_risk": "moderate_to_high_total_work_but_low_per_task_memory",
            "per_task_memory_request": "16GB",
            "per_task_cpus": 4,
            "per_task_walltime": "04:00:00",
        },
        "commands": {
            "plan_command": "python scripts/plan_geom_full_database_hybrid_lambda_complete_cov_amr.py",
            "wrapper_generation_command": (
                "python scripts/make_geom_full_database_hybrid_lambda_complete_cov_amr_single_job_wrappers.py "
                "--plan-path experiments/results/geom_full_database_hybrid_lambda_complete_cov_amr_plan.json"
            ),
            "single_job_command_list": str(commands_path),
            "first_sbatch_command": tasks[0]["sbatch_command"] if tasks else None,
            "last_sbatch_command": tasks[-1]["sbatch_command"] if tasks else None,
            "merge_command": (
                "python scripts/merge_geom_full_database_hybrid_lambda_complete_cov_amr.py "
                "--plan-path experiments/results/geom_full_database_hybrid_lambda_complete_cov_amr_plan.json "
                "--summary-path experiments/results/geom_full_database_hybrid_lambda_complete_cov_amr_summary.json "
                "--report-path reports/tables/geom_full_database_hybrid_lambda_complete_cov_amr_summary.md --require-complete"
            ),
        },
        "scheduler_recommendation": {
            "submission_recommended_after_user_approval": evaluation_ready,
            "requires_user_approval": True,
            "recommended_account": "torch_pr_872_general",
            "standalone_job_count": len(tasks),
            "full_standalone_job_count_when_ready": full_task_count,
            "slurm_array_used": False,
            "recommended_submission_list": str(commands_path),
            "recommended_resource_request": {
                "cpus_per_task": 4,
                "memory": "16GB",
                "time": "04:00:00",
                "gpu": None,
            },
        },
        "output_targets": {
            "chunk_jsonls": str(chunks_dir / "chunk_<task_id>.jsonl"),
            "chunk_manifests": str(chunks_dir / "chunk_<task_id>_manifest.json"),
            "chunk_status": str(chunks_dir / "chunk_<task_id>_status.json"),
            "single_job_wrappers": str(wrapper_dir / f"{WRAPPER_PREFIX}_<task_id>.sbatch"),
            "single_job_command_list": str(commands_path),
            "summary": str(DEFAULT_SUMMARY_PATH),
            "report": str(DEFAULT_SUMMARY_REPORT_PATH),
        },
        "tasks": tasks,
        "runtime_seconds": time.time() - start,
        "notes": [
            "This is a full val/test COV/AMR lambda-comparison plan, not executed evaluation.",
            "By default tasks are generated only when every requested lambda has random/scaffold seeds 0,1,2 ready.",
            "Train partition is excluded by design.",
            "No SLURM array is used or prepared.",
        ],
    }


def _base_chunks(split_payloads: dict[str, dict[str, Any]], chunk_size: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    base_chunks = []
    molecule_slots: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        molecule_slots[split] = {}
        for partition in PARTITIONS:
            ids = split_payloads[split][partition]
            molecule_slots[split][partition] = len(ids)
            for chunk_index in range(math.ceil(len(ids) / chunk_size)):
                start_index = chunk_index * chunk_size
                end_index = min(start_index + chunk_size, len(ids))
                base_chunks.append(
                    {
                        "split_type": split,
                        "partition": partition,
                        "chunk_index": chunk_index,
                        "chunk_start": start_index,
                        "chunk_end": end_index,
                        "molecule_count": end_index - start_index,
                    }
                )
    return base_chunks, molecule_slots


def _tasks(
    base_chunks: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    chunks_dir: Path,
    wrapper_dir: Path,
) -> list[dict[str, Any]]:
    tasks = []
    task_id = 0
    for base_chunk in base_chunks:
        for entry in entries:
            for seed in SEEDS:
                output_prefix = f"chunk_{task_id:05d}"
                wrapper_path = wrapper_dir / f"{WRAPPER_PREFIX}_{task_id:05d}.sbatch"
                tasks.append(
                    {
                        "task_id": task_id,
                        **base_chunk,
                        "entry_label": entry["entry_label"],
                        "method": entry["method"],
                        "role": entry["role"],
                        "lambda_value": entry.get("lambda_value"),
                        "seed": int(seed),
                        "jsonl_path": str(chunks_dir / f"{output_prefix}.jsonl"),
                        "manifest_path": str(chunks_dir / f"{output_prefix}_manifest.json"),
                        "status_path": str(chunks_dir / f"{output_prefix}_status.json"),
                        "wrapper_path": str(wrapper_path),
                        "sbatch_command": f"sbatch --account=torch_pr_872_general {wrapper_path}",
                    }
                )
                task_id += 1
    return tasks


def _entry_readiness() -> dict[str, Any]:
    entries = []
    for spec in REQUESTED_ENTRIES:
        per_seed = []
        missing = []
        for split in SPLITS:
            for seed in SEEDS:
                row = _resolve_entry_seed(spec, split, int(seed))
                per_seed.append(row)
                if not row["ready"]:
                    missing.append(row)
        complete = len(missing) == 0
        entries.append({**spec, "complete": complete, "seed_inputs": per_seed, "missing": missing})
    return {
        "entries": entries,
        "complete_entry_labels": [row["entry_label"] for row in entries if row["complete"]],
        "incomplete_entry_labels": [row["entry_label"] for row in entries if not row["complete"]],
        "missing_training_jobs": _missing_training_jobs(entries),
        "eligible_hybrid_lambdas": [
            {"entry_label": row["entry_label"], "lambda_value": row["lambda_value"]}
            for row in entries
            if row["role"] == "hybrid_candidate" and row["complete"]
        ],
    }


def _readiness_counts(readiness: dict[str, Any]) -> dict[str, Any]:
    seed_inputs = [item for entry in readiness["entries"] for item in entry["seed_inputs"]]
    missing_results = []
    missing_checkpoints = []
    available_results = 0
    available_checkpoints = 0
    for item in seed_inputs:
        descriptor = {
            "entry_label": item["entry_label"],
            "split_type": item["split_type"],
            "seed": item["seed"],
        }
        result_path = item.get("result_path")
        if result_path and Path(result_path).exists():
            available_results += 1
        else:
            missing_results.append(
                {
                    **descriptor,
                    "expected_result_paths": item.get("expected_result_paths", []),
                }
            )
        for checkpoint_kind, field in (
            ("best", "best_checkpoint_path"),
            ("latest", "latest_checkpoint_path"),
        ):
            checkpoint_path = item.get(field)
            if checkpoint_path and Path(checkpoint_path).exists():
                available_checkpoints += 1
            else:
                missing_checkpoints.append(
                    {
                        **descriptor,
                        "checkpoint_kind": checkpoint_kind,
                        "checkpoint_path": checkpoint_path,
                    }
                )
    return {
        "expected_result_json_count": len(seed_inputs),
        "available_result_json_count": available_results,
        "expected_checkpoint_count": len(seed_inputs) * 2,
        "available_checkpoint_count": available_checkpoints,
        "missing_result_jsons": missing_results,
        "missing_checkpoints": missing_checkpoints,
    }


def _missing_training_jobs(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for entry in entries:
        if entry["role"] != "hybrid_candidate":
            continue
        for row in entry["missing"]:
            if "result_json" in row["missing"]:
                jobs.append(
                    {
                        "entry_label": entry["entry_label"],
                        "lambda_value": entry["lambda_value"],
                        "split_type": row["split_type"],
                        "seed": row["seed"],
                    }
                )
    return jobs


def _entry_seed_inputs(complete_entries: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for entry in complete_entries:
        for item in entry["seed_inputs"]:
            key = f"{entry['entry_label']}/{item['split_type']}/seed{item['seed']}"
            output[key] = item
    return output


def _resolve_entry_seed(spec: dict[str, Any], split_type: str, seed: int) -> dict[str, Any]:
    result_candidates = _result_candidates(spec, split_type, seed)
    config_candidates = _config_candidates(spec, split_type, seed)
    result_path = next((path for path in result_candidates if path.exists()), None)
    config_path = next((path for path in config_candidates if path.exists()), None)
    row = {
        "entry_label": spec["entry_label"],
        "split_type": split_type,
        "seed": seed,
        "expected_result_paths": [str(path) for path in result_candidates],
        "expected_config_paths": [str(path) for path in config_candidates],
        "config_path": None,
        "config_exists": False,
        "result_path": None,
        "best_checkpoint_path": None,
        "latest_checkpoint_path": None,
        "ready": False,
        "missing": [],
    }
    if result_path is None:
        row["missing"].append("result_json")
        if config_path is None:
            row["missing"].append("config")
        return row
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    best = Path(payload.get("best_checkpoint_path", ""))
    latest = Path(payload.get("latest_checkpoint_path", ""))
    row.update(
        {
            "result_path": str(result_path),
            "config_path": str(config_path) if config_path else None,
            "config_exists": config_path is not None,
            "best_checkpoint_path": str(best),
            "latest_checkpoint_path": str(latest),
            "optimizer_name": payload.get("optimizer_name"),
            "learning_rate": payload.get("learning_rate"),
            "hybrid_source_strength": payload.get("hybrid_source_strength"),
        }
    )
    if payload.get("optimizer_name") != "AdamW":
        row["missing"].append("adamw_result")
    if abs(float(payload.get("learning_rate", 999.0)) - 1.0e-3) > 1.0e-12:
        row["missing"].append("lr1e3_result")
    if spec["role"] == "hybrid_candidate":
        strength = payload.get("hybrid_source_strength")
        if not isinstance(strength, (int, float)) or abs(float(strength) - float(spec["lambda_value"])) > 1.0e-12:
            row["missing"].append("lambda_mismatch")
    if not best.exists():
        row["missing"].append("best_checkpoint")
    if not latest.exists():
        row["missing"].append("latest_checkpoint")
    if config_path is None:
        row["missing"].append("config")
    row["ready"] = len(row["missing"]) == 0
    return row


def _result_candidates(spec: dict[str, Any], split_type: str, seed: int) -> list[Path]:
    split_prefix = "geom_formal_full_database" + ("" if split_type == "random" else "_scaffold") + "_train_"
    if spec["entry_label"] == "harmonic_prior":
        paths = []
        if split_type == "random" and seed == 0:
            paths.append(Path(f"experiments/results/{split_prefix}harmonic_steps2000_seed{seed}_opt_adamw_lr1e3.json"))
        paths.append(Path(f"experiments/results/{split_prefix}harmonic_steps2000_seed{seed}_final_adamw_lr1e3.json"))
        return paths
    label = spec["lambda_label"]
    run_label = f"hybrid_harmonic_stiffness_v1_{label}"
    paths = []
    if label == "lambda500" and split_type == "random" and seed == 0:
        paths.append(Path(f"experiments/results/{split_prefix}{run_label}_steps2000_seed{seed}_opt_adamw_lr1e3.json"))
    suffix = "final_adamw_lr1e3" if label == "lambda500" else "low_lambda_adamw_lr1e3"
    paths.append(Path(f"experiments/results/{split_prefix}{run_label}_steps2000_seed{seed}_{suffix}.json"))
    return paths


def _config_candidates(spec: dict[str, Any], split_type: str, seed: int) -> list[Path]:
    if spec["entry_label"] == "harmonic_prior":
        paths = []
        if split_type == "random" and seed == 0:
            paths.append(Path("configs/train/optimization/geom_full_database_random_seed0_harmonic_adamw_lr1e3.yaml"))
        paths.append(Path("configs/train/final_optimized") / f"geom_full_database_{split_type}_harmonic_adamw_lr1e3.yaml")
        return paths
    label = spec["lambda_label"]
    if label == "lambda500":
        return [
            Path("configs/train/final_optimized")
            / f"geom_full_database_{split_type}_hybrid_harmonic_stiffness_v1_lambda500_adamw_lr1e3.yaml"
        ]
    return [
        Path("configs/train/hybrid_lambda_adamw_multiseed")
        / f"geom_full_database_{split_type}_hybrid_harmonic_stiffness_v1_{label}_seed{seed}_adamw_lr1e3.yaml",
        Path("configs/train/final_optimized_low_lambda")
        / f"geom_full_database_{split_type}_hybrid_harmonic_stiffness_v1_{label}_adamw_lr1e3.yaml",
    ]


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.chunk_size) <= 0:
        raise ValueError("chunk-size must be positive")
    if int(args.generated_conformers_per_mol) <= 0:
        raise ValueError("generated-conformers-per-mol must be positive")
    if float(args.delta) <= 0.0:
        raise ValueError("delta must be positive")


def _load_split(path: Path, split_type: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split_type") != split_type:
        raise ValueError(f"Split file {path} has unexpected split_type")
    if payload.get("split_scope") != FULL_DATABASE_SCOPE:
        raise ValueError(f"Split file {path} has unexpected split_scope")
    return payload


def _last_chunk_sizes(molecule_slots: dict[str, dict[str, int]], chunk_size: int) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for split, partitions in molecule_slots.items():
        output[split] = {}
        for partition, count in partitions.items():
            remainder = count % chunk_size
            output[split][partition] = remainder if remainder else chunk_size
    return output


def _markdown_report(plan: dict[str, Any]) -> str:
    counts = plan["readiness_counts"]
    lines = [
        "# Hybrid Lambda Complete Val/Test COV/AMR Plan",
        "",
        "Planning-only full val/test COV/AMR lambda comparison from AdamW checkpoints. This is not a benchmark.",
        "",
        "## Readiness",
        "",
        f"- Evaluation ready: `{plan['evaluation_ready']}`",
        f"- Complete entries: `{plan['entry_readiness']['complete_entry_labels']}`",
        f"- Incomplete entries: `{plan['entry_readiness']['incomplete_entry_labels']}`",
        f"- Available result JSONs: `{counts['available_result_json_count']} / {counts['expected_result_json_count']}`",
        f"- Available checkpoints: `{counts['available_checkpoint_count']} / {counts['expected_checkpoint_count']}`",
        f"- Missing result JSONs: `{counts['missing_result_jsons']}`",
        f"- Missing checkpoints: `{counts['missing_checkpoints']}`",
        f"- Missing training jobs: `{plan['entry_readiness']['missing_training_jobs']}`",
        f"- Standalone jobs in this plan: `{plan['chunking']['standalone_job_count']}`",
        f"- Full standalone jobs when ready: `{plan['chunking']['full_standalone_job_count_when_ready']}`",
        "",
        "## Scope",
        "",
        f"- Splits: `{plan['splits']}`",
        f"- Partitions: `{plan['partitions']}`",
        f"- Seeds: `{plan['seeds']}`",
        f"- Requested entries: `{[row['entry_label'] for row in plan['requested_entries']]}`",
        f"- K generated conformers per molecule: `{plan['evaluation']['generated_conformers_per_molecule']}`",
        f"- RMSD threshold delta: `{plan['evaluation']['delta_angstrom']}`",
        f"- Total molecule slots: `{plan['coverage']['total_molecule_slots']}`",
        f"- Estimated full generated conformers when ready: `{plan['coverage']['estimated_full_generated_conformer_count_when_ready']}`",
        f"- Train included: `{plan['train_partition_included']}`",
        "",
        "## Commands",
        "",
        "Refresh this plan after the missing training jobs complete:",
        "",
        "```bash",
        plan["commands"]["plan_command"],
        "```",
        "",
        "Generate standalone wrappers only after `evaluation_ready` is true:",
        "",
        "```bash",
        plan["commands"]["wrapper_generation_command"],
        "```",
        "",
        "Merge after all approved standalone COV/AMR jobs finish:",
        "",
        "```bash",
        plan["commands"]["merge_command"],
        "```",
        "",
        "## Safety",
        "",
        f"- `benchmark_result`: `{plan['benchmark_result']}`",
        f"- `training_executed_by_plan`: `{plan['training_executed_by_plan']}`",
        f"- `sampling_executed_by_plan`: `{plan['sampling_executed_by_plan']}`",
        f"- `raw_data_modified`: `{plan['raw_data_modified']}`",
        f"- `environment_modified`: `{plan['environment_modified']}`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
