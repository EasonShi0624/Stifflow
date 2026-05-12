#!/usr/bin/env python
"""Plan complete val/test COV/AMR chunks for final AdamW models."""

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


DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_final_cov_amr_complete_valtest_plan.json")
DEFAULT_REPORT_PATH = Path("reports/tables/geom_full_database_final_cov_amr_complete_valtest_plan.md")
DEFAULT_CHUNKS_DIR = Path("experiments/results/geom_full_database_final_cov_amr_complete_valtest_chunks")
DEFAULT_SUMMARY_PATH = Path("experiments/results/geom_full_database_final_cov_amr_complete_valtest_summary.json")
DEFAULT_SUMMARY_REPORT_PATH = Path("reports/tables/geom_full_database_final_cov_amr_complete_valtest_summary.md")
DEFAULT_WRAPPER_DIR = Path("scripts")
DEFAULT_COMMANDS_PATH = Path("scripts/geom_full_database_final_cov_amr_complete_valtest_single_job_commands.md")
WRAPPER_PREFIX = "slurm_geom_full_database_final_cov_amr_complete_valtest_chunk"
SPLIT_PATHS = {"random": FULL_DATABASE_SPLIT_PATH, "scaffold": FULL_DATABASE_SCAFFOLD_SPLIT_PATH}
SPLITS = ("random", "scaffold")
PARTITIONS = ("val", "test")
BASELINE_SPECS = (
    {"entry_label": "harmonic_prior", "method": "harmonic_prior", "role": "baseline", "run_label": "harmonic", "lambda_value": 0.0},
    {"entry_label": "stiffness_structured", "method": "stiffness_structured", "role": "baseline", "run_label": "stiffness", "lambda_value": None},
    {"entry_label": "isotropic", "method": "isotropic", "role": "baseline", "run_label": "isotropic", "lambda_value": None},
    {"entry_label": "torsion_only", "method": "torsion_only", "role": "baseline", "run_label": "torsion_only", "lambda_value": None},
)
HYBRID_CANDIDATES = (
    {"entry_label": "hybrid_lambda00025", "method": "hybrid_harmonic_stiffness_v1", "role": "hybrid_candidate", "run_label": "hybrid_harmonic_stiffness_v1_lambda00025", "lambda_value": 0.0025, "required": False, "label_variants": ["lambda00025", "lambda0002", "lambda0003"]},
    {"entry_label": "hybrid_lambda0005", "method": "hybrid_harmonic_stiffness_v1", "role": "hybrid_candidate", "run_label": "hybrid_harmonic_stiffness_v1_lambda0005", "lambda_value": 0.005, "required": False, "label_variants": ["lambda0005", "lambda001"]},
    {"entry_label": "hybrid_lambda001", "method": "hybrid_harmonic_stiffness_v1", "role": "hybrid_candidate", "run_label": "hybrid_harmonic_stiffness_v1_lambda001", "lambda_value": 0.01, "required": False, "label_variants": ["lambda001", "lambda0010"]},
    {"entry_label": "hybrid_lambda0015", "method": "hybrid_harmonic_stiffness_v1", "role": "hybrid_candidate", "run_label": "hybrid_harmonic_stiffness_v1_lambda0015", "lambda_value": 0.015, "required": False, "label_variants": ["lambda0015", "lambda002"]},
    {"entry_label": "hybrid_lambda002", "method": "hybrid_harmonic_stiffness_v1", "role": "hybrid_candidate", "run_label": "hybrid_harmonic_stiffness_v1_lambda002", "lambda_value": 0.02, "required": False, "label_variants": ["lambda002", "lambda0020"]},
    {"entry_label": "hybrid_lambda0025", "method": "hybrid_harmonic_stiffness_v1", "role": "hybrid_candidate", "run_label": "hybrid_harmonic_stiffness_v1_lambda0025", "lambda_value": 0.025, "required": True, "label_variants": ["lambda0025"]},
    {"entry_label": "hybrid_lambda010", "method": "hybrid_harmonic_stiffness_v1", "role": "hybrid_candidate", "run_label": "hybrid_harmonic_stiffness_v1_lambda010", "lambda_value": 0.1, "required": True, "label_variants": ["lambda010"]},
    {"entry_label": "hybrid_lambda300", "method": "hybrid_harmonic_stiffness_v1", "role": "hybrid_candidate", "run_label": "hybrid_harmonic_stiffness_v1_lambda300", "lambda_value": 3.0, "required": True, "label_variants": ["lambda300"]},
    {"entry_label": "hybrid_lambda500", "method": "hybrid_harmonic_stiffness_v1", "role": "hybrid_candidate", "run_label": "hybrid_harmonic_stiffness_v1_lambda500", "lambda_value": 5.0, "required": True, "label_variants": ["lambda500"]},
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
    print("note: Planning-only complete val/test COV/AMR manifest; no model sampling, training, scheduler submission, raw-data mutation, or environment mutation.")


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    _validate_args(args)
    split_payloads = {split: _load_split(SPLIT_PATHS[split], split) for split in SPLITS}
    entry_readiness = _entry_readiness()
    complete_entries = [row for row in entry_readiness["entries"] if row["complete"]]
    chunks_dir = Path(args.chunks_dir)
    wrapper_dir = Path(args.wrapper_dir)
    commands_path = Path(args.commands_path)
    base_chunks = []
    molecule_slots_by_split_partition = {}
    for split in SPLITS:
        molecule_slots_by_split_partition[split] = {}
        for partition in PARTITIONS:
            ids = split_payloads[split][partition]
            molecule_slots_by_split_partition[split][partition] = len(ids)
            for chunk_index in range(math.ceil(len(ids) / int(args.chunk_size))):
                start_index = chunk_index * int(args.chunk_size)
                end_index = min(start_index + int(args.chunk_size), len(ids))
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

    tasks = []
    task_id = 0
    for base_chunk in base_chunks:
        for entry in complete_entries:
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

    total_molecule_slots = sum(sum(row.values()) for row in molecule_slots_by_split_partition.values())
    method_seed_records = total_molecule_slots * len(complete_entries) * len(SEEDS)
    plan = {
        "summary_name": "geom_full_database_final_cov_amr_complete_valtest_plan",
        "result_kind": "geom_drugs_full_database_final_cov_amr_complete_valtest_plan_not_benchmark",
        "benchmark_result": False,
        "execution_mode": "local_reporting_only_complete_valtest_cov_amr_planning",
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
        "methods": [entry["entry_label"] for entry in complete_entries],
        "all_requested_entries": [entry["entry_label"] for entry in entry_readiness["entries"]],
        "entry_readiness": entry_readiness,
        "complete_entries": complete_entries,
        "entry_seed_inputs": _entry_seed_inputs(complete_entries),
        "chunking": {
            "chunk_size": int(args.chunk_size),
            "base_molecule_chunk_count": len(base_chunks),
            "standalone_job_count": len(tasks),
            "task_unit": "one_entry_split_seed_partition_chunk",
            "deterministic_assignment": "split-file val/test molecule order sliced by chunk_start/chunk_end, then complete entry order, then seed order",
            "chunks_dir": str(chunks_dir),
            "wrapper_dir": str(wrapper_dir),
            "commands_path": str(commands_path),
            "resume_behavior": "chunk evaluator skips completed status/jsonl pairs unless --overwrite is passed",
            "slurm_array_used": False,
            "submission_style": "standalone_single_job_wrappers_only",
            "molecule_slots_per_chunk": {
                "target_chunk_size": int(args.chunk_size),
                "last_chunk_size_by_split_partition": _last_chunk_sizes(
                    molecule_slots_by_split_partition,
                    int(args.chunk_size),
                ),
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
            "molecule_slots_by_split_partition": molecule_slots_by_split_partition,
            "total_molecule_slots": total_molecule_slots,
            "entry_count": len(complete_entries),
            "seed_count": len(SEEDS),
            "estimated_method_seed_molecule_records": method_seed_records,
            "estimated_generated_conformer_count": method_seed_records * int(args.generated_conformers_per_mol),
            "train_slots_excluded_by_split": {split: len(split_payloads[split]["train"]) for split in SPLITS},
        },
        "resource_estimate": {
            "cpu_memory_risk": "moderate_to_high_total_work_but_low_per_task_memory",
            "per_task_memory_request": "16GB",
            "per_task_cpus": 4,
            "per_task_walltime": "04:00:00",
        },
        "commands": {
            "plan_command": "python scripts/plan_geom_full_database_final_cov_amr_complete_valtest.py",
            "wrapper_generation_command": (
                "python scripts/make_geom_full_database_final_cov_amr_complete_valtest_single_job_wrappers.py "
                "--plan-path experiments/results/geom_full_database_final_cov_amr_complete_valtest_plan.json"
            ),
            "single_job_command_list": str(commands_path),
            "first_sbatch_command": tasks[0]["sbatch_command"] if tasks else None,
            "last_sbatch_command": tasks[-1]["sbatch_command"] if tasks else None,
            "merge_command": (
                "python scripts/merge_geom_full_database_final_cov_amr_complete_valtest.py "
                "--plan-path experiments/results/geom_full_database_final_cov_amr_complete_valtest_plan.json "
                "--summary-path experiments/results/geom_full_database_final_cov_amr_complete_valtest_summary.json "
                "--report-path reports/tables/geom_full_database_final_cov_amr_complete_valtest_summary.md --require-complete"
            ),
            "local_single_task_example": (
                "python scripts/evaluate_geom_full_database_final_cov_amr_complete_valtest_chunk.py "
                "--plan-path experiments/results/geom_full_database_final_cov_amr_complete_valtest_plan.json --task-id 0"
            ),
        },
        "scheduler_recommendation": {
            "submission_recommended_after_user_approval": bool(complete_entries),
            "requires_user_approval": True,
            "recommended_account": "torch_pr_872_general",
            "standalone_job_count": len(tasks),
            "slurm_array_used": False,
            "recommended_submission_list": str(commands_path),
            "recommended_resource_request": {
                "cpus_per_task": 4,
                "memory": "16GB",
                "time": "04:00:00",
                "gpu": None,
            },
            "reason": "CPU-only checkpoint ensemble evaluation; standalone jobs are easier to allocate and arrays are explicitly disallowed.",
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
            "Complete means full val/test only. Train is excluded by design because it is much larger and not needed for val/test ensemble-quality decisions.",
            "Only checkpoint-ready AdamW entries with random/scaffold seeds 0,1,2 coverage are included.",
            "Old SGD lambda runs are intentionally excluded.",
            "No SLURM array is used or prepared.",
        ],
    }
    return plan


def _entry_readiness() -> dict[str, Any]:
    specs = list(BASELINE_SPECS) + list(HYBRID_CANDIDATES)
    entries = []
    for spec in specs:
        per_seed = []
        missing = []
        for split in SPLITS:
            for seed in SEEDS:
                found = _resolve_entry_seed(spec, split, int(seed))
                per_seed.append(found)
                if not found["ready"]:
                    missing.append(found)
        complete = len(missing) == 0
        entries.append({**spec, "complete": complete, "seed_inputs": per_seed, "missing": missing})
    return {
        "entries": entries,
        "complete_entry_labels": [row["entry_label"] for row in entries if row["complete"]],
        "incomplete_entry_labels": [row["entry_label"] for row in entries if not row["complete"]],
        "baseline_complete": [row["entry_label"] for row in entries if row["role"] == "baseline" and row["complete"]],
        "eligible_hybrid_lambdas": [
            {"entry_label": row["entry_label"], "lambda_value": row["lambda_value"]}
            for row in entries
            if row["role"] == "hybrid_candidate" and row["complete"]
        ],
    }


def _entry_seed_inputs(complete_entries: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for entry in complete_entries:
        for item in entry["seed_inputs"]:
            key = f"{entry['entry_label']}/{item['split_type']}/seed{item['seed']}"
            output[key] = item
    return output


def _resolve_entry_seed(spec: dict[str, Any], split_type: str, seed: int) -> dict[str, Any]:
    result_candidates, config_path = _candidate_paths(spec, split_type, seed)
    result_path = next((path for path in result_candidates if path.exists()), None)
    row = {
        "entry_label": spec["entry_label"],
        "split_type": split_type,
        "seed": seed,
        "expected_result_paths": [str(path) for path in result_candidates],
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "result_path": None,
        "best_checkpoint_path": None,
        "latest_checkpoint_path": None,
        "ready": False,
        "missing": [],
    }
    if result_path is None:
        row["missing"].append("result_json")
        return row
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    best = Path(payload.get("best_checkpoint_path", ""))
    latest = Path(payload.get("latest_checkpoint_path", ""))
    row.update(
        {
            "result_path": str(result_path),
            "best_checkpoint_path": str(best),
            "latest_checkpoint_path": str(latest),
            "optimizer_name": payload.get("optimizer_name"),
            "learning_rate": payload.get("learning_rate"),
            "hybrid_source_strength": payload.get("hybrid_source_strength"),
        }
    )
    if payload.get("optimizer_name") != "AdamW":
        row["missing"].append("adamw_result")
    if abs(float(payload.get("learning_rate", math.nan)) - 1.0e-3) > 1.0e-12:
        row["missing"].append("lr1e3_result")
    if not best.exists():
        row["missing"].append("best_checkpoint")
    if not latest.exists():
        row["missing"].append("latest_checkpoint")
    if not config_path.exists():
        row["missing"].append("config")
    row["ready"] = len(row["missing"]) == 0
    return row


def _candidate_paths(spec: dict[str, Any], split_type: str, seed: int) -> tuple[list[Path], Path]:
    split_prefix = "geom_formal_full_database" + ("" if split_type == "random" else "_scaffold") + "_train_"
    if spec["role"] == "baseline":
        run_label = spec["run_label"]
        paths = []
        if split_type == "random" and seed == 0 and run_label == "harmonic":
            paths.append(Path(f"experiments/results/{split_prefix}{run_label}_steps2000_seed{seed}_opt_adamw_lr1e3.json"))
        paths.append(Path(f"experiments/results/{split_prefix}{run_label}_steps2000_seed{seed}_final_adamw_lr1e3.json"))
        config_path = Path("configs/train/final_optimized") / f"geom_full_database_{split_type}_{run_label}_adamw_lr1e3.yaml"
        return paths, config_path
    paths = []
    for label in spec["label_variants"]:
        run_label = f"hybrid_harmonic_stiffness_v1_{label}"
        if split_type == "random" and seed == 0 and label == "lambda500":
            paths.append(Path(f"experiments/results/{split_prefix}{run_label}_steps2000_seed{seed}_opt_adamw_lr1e3.json"))
        paths.append(Path(f"experiments/results/{split_prefix}{run_label}_steps2000_seed{seed}_final_adamw_lr1e3.json"))
        paths.append(Path(f"experiments/results/{split_prefix}{run_label}_steps2000_seed{seed}_low_lambda_adamw_lr1e3.json"))
    label = spec["label_variants"][0]
    if label == "lambda500":
        config_path = Path("configs/train/final_optimized") / f"geom_full_database_{split_type}_hybrid_harmonic_stiffness_v1_lambda500_adamw_lr1e3.yaml"
    else:
        config_path = Path("configs/train/final_optimized_low_lambda") / f"geom_full_database_{split_type}_hybrid_harmonic_stiffness_v1_{label}_adamw_lr1e3.yaml"
    return paths, config_path


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


def _last_chunk_sizes(molecule_slots_by_split_partition: dict[str, dict[str, int]], chunk_size: int) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for split, partitions in molecule_slots_by_split_partition.items():
        output[split] = {}
        for partition, count in partitions.items():
            remainder = count % chunk_size
            output[split][partition] = remainder if remainder else chunk_size
    return output


def _markdown_report(plan: dict[str, Any]) -> str:
    lines = [
        "# Final Complete Val/Test COV/AMR Plan",
        "",
        "Planning-only full val/test COV/AMR evaluation from existing AdamW checkpoints. This is not a benchmark.",
        "",
        "## Scope",
        "",
        f"- Splits: `{plan['splits']}`",
        f"- Partitions: `{plan['partitions']}`",
        f"- Seeds: `{plan['seeds']}`",
        f"- Complete entries: `{plan['entry_readiness']['complete_entry_labels']}`",
        f"- Incomplete entries: `{plan['entry_readiness']['incomplete_entry_labels']}`",
        f"- Eligible hybrid lambdas: `{plan['entry_readiness']['eligible_hybrid_lambdas']}`",
        f"- Total molecule slots: `{plan['coverage']['total_molecule_slots']}`",
        f"- Chunk size: `{plan['chunking']['chunk_size']}`",
        f"- Chunk assignment: `{plan['chunking']['deterministic_assignment']}`",
        f"- Standalone jobs: `{plan['chunking']['standalone_job_count']}`",
        f"- Generated conformers per full chunk/task: `{plan['chunking']['estimated_generated_conformers_per_full_chunk_per_task']}`",
        f"- Estimated generated conformers: `{plan['coverage']['estimated_generated_conformer_count']}`",
        f"- Train included: `{plan['train_partition_included']}`",
        f"- Train slots excluded by split: `{plan['coverage']['train_slots_excluded_by_split']}`",
        "",
        "## Molecule Slots",
        "",
        "| Split | Partition | Molecules |",
        "|---|---|---:|",
    ]
    for split, partitions in plan["coverage"]["molecule_slots_by_split_partition"].items():
        for partition, count in partitions.items():
            lines.append(f"| {split} | {partition} | {count} |")
    lines.extend(
        [
            "",
            "## Chunking",
            "",
            f"- Base molecule chunks: `{plan['chunking']['base_molecule_chunk_count']}`",
            f"- Task unit: `{plan['chunking']['task_unit']}`",
            f"- Target molecule slots per chunk: `{plan['chunking']['molecule_slots_per_chunk']['target_chunk_size']}`",
            f"- Last chunk sizes: `{plan['chunking']['molecule_slots_per_chunk']['last_chunk_size_by_split_partition']}`",
            f"- Resumable: `{plan['chunking']['resume_behavior']}`",
            f"- SLURM array used: `{plan['chunking']['slurm_array_used']}`",
            "",
            "## Output Paths",
            "",
            f"- Chunk JSONLs: `{plan['output_targets']['chunk_jsonls']}`",
            f"- Chunk manifests: `{plan['output_targets']['chunk_manifests']}`",
            f"- Chunk status: `{plan['output_targets']['chunk_status']}`",
            f"- Standalone wrappers: `{plan['output_targets']['single_job_wrappers']}`",
            f"- Command list: `{plan['output_targets']['single_job_command_list']}`",
            f"- Future summary JSON: `{plan['output_targets']['summary']}`",
            f"- Future summary report: `{plan['output_targets']['report']}`",
            "",
            "## Scheduler Recommendation",
            "",
            f"- Recommended account: `{plan['scheduler_recommendation']['recommended_account']}`",
            f"- Resource request: `{plan['scheduler_recommendation']['recommended_resource_request']}`",
            f"- Submission requires approval: `{plan['scheduler_recommendation']['requires_user_approval']}`",
        ]
    )
    lines.extend(
        [
            "",
            "## Missing Hybrid Coverage",
            "",
            "| Entry | Missing count |",
            "|---|---:|",
        ]
    )
    for entry in plan["entry_readiness"]["entries"]:
        if not entry["complete"]:
            lines.append(f"| {entry['entry_label']} | {len(entry['missing'])} |")
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "Generate standalone wrappers and command list:",
            "",
            "```bash",
            plan["commands"]["wrapper_generation_command"],
            "```",
            "",
            "Command-list path:",
            "",
            f"`{plan['commands']['single_job_command_list']}`",
            "",
            "First and last command:",
            "",
            "```bash",
            str(plan["commands"]["first_sbatch_command"]),
            str(plan["commands"]["last_sbatch_command"]),
            "```",
            "",
            "Merge after all approved standalone jobs complete:",
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
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
