#!/usr/bin/env python
"""Plan full val/test final-optimized AMR/COV evaluation chunks."""

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
from src.eval.final_optimized_all_methods import METHOD_LABELS, SEEDS, _candidate_paths  # noqa: E402


DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_final_optimized_amr_cov_full_plan.json")
DEFAULT_REPORT_PATH = Path("reports/tables/geom_full_database_final_optimized_amr_cov_full_plan.md")
DEFAULT_CHUNKS_DIR = Path("experiments/results/geom_full_database_final_optimized_amr_cov_full_chunks")
DEFAULT_SUMMARY_PATH = Path("experiments/results/geom_full_database_final_optimized_amr_cov_full_summary.json")
DEFAULT_SUMMARY_REPORT_PATH = Path("reports/tables/geom_full_database_final_optimized_amr_cov_full_summary.md")
DEFAULT_WRAPPER_DIR = Path("scripts")
DEFAULT_COMMANDS_PATH = Path("scripts/geom_full_database_final_optimized_amr_cov_full_single_job_commands.md")
WRAPPER_PREFIX = "slurm_geom_full_database_final_optimized_amr_cov_full_chunk"
SPLIT_PATHS = {"random": FULL_DATABASE_SPLIT_PATH, "scaffold": FULL_DATABASE_SCAFFOLD_SPLIT_PATH}
METHODS = (
    "hybrid_harmonic_stiffness_v1_lambda500",
    "harmonic_prior",
    "stiffness_structured",
    "isotropic",
    "torsion_only",
)
DEFAULT_SPLITS = ("random", "scaffold")
DEFAULT_PARTITIONS = ("val", "test")
DEFAULT_CHUNK_SIZE = 512
DEFAULT_GENERATED_CONFORMERS = 20
DEFAULT_DELTA = 0.75
DEFAULT_MAX_REFERENCE_CONFORMERS = 40
DEFAULT_MAX_PICKLE_BYTES = 25_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=("random", "scaffold"), default=list(DEFAULT_SPLITS))
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=("train", "val", "test"),
        default=list(DEFAULT_PARTITIONS),
        help="Default excludes train. Add train only for an explicit separate train-partition run.",
    )
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
    print("note: Planning-only full val/test AMR/COV single-job manifest; no model sampling, training, scheduler submission, raw-data mutation, or environment mutation.")


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    _validate_args(args)
    splits = tuple(dict.fromkeys(args.splits))
    partitions = tuple(dict.fromkeys(args.partitions))
    chunks_dir = Path(args.chunks_dir)
    wrapper_dir = Path(args.wrapper_dir)
    commands_path = Path(args.commands_path)
    split_payloads = {split: _load_split(SPLIT_PATHS[split], split) for split in splits}
    readiness = _training_readiness(splits=splits)

    base_chunks: list[dict[str, Any]] = []
    molecule_slots_by_split_partition: dict[str, dict[str, int]] = {}
    for split in splits:
        molecule_slots_by_split_partition[split] = {}
        for partition in partitions:
            ids = split_payloads[split][partition]
            molecule_slots_by_split_partition[split][partition] = len(ids)
            chunk_count = math.ceil(len(ids) / int(args.chunk_size))
            for chunk_index in range(chunk_count):
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
        for method in METHODS:
            for seed in SEEDS:
                output_prefix = f"chunk_{task_id:05d}"
                wrapper_path = wrapper_dir / f"{WRAPPER_PREFIX}_{task_id:05d}.sbatch"
                task = {
                    "task_id": task_id,
                    **base_chunk,
                    "method": method,
                    "seed": int(seed),
                    "jsonl_path": str(chunks_dir / f"{output_prefix}.jsonl"),
                    "manifest_path": str(chunks_dir / f"{output_prefix}_manifest.json"),
                    "status_path": str(chunks_dir / f"{output_prefix}_status.json"),
                    "wrapper_path": str(wrapper_path),
                    "sbatch_command": f"sbatch --account=torch_pr_872_general {wrapper_path}",
                }
                tasks.append(task)
                task_id += 1

    total_molecule_slots = sum(sum(row.values()) for row in molecule_slots_by_split_partition.values())
    estimated_method_seed_molecule_records = total_molecule_slots * len(METHODS) * len(SEEDS)
    estimated_generated_conformer_count = (
        estimated_method_seed_molecule_records * int(args.generated_conformers_per_mol)
    )
    plan = {
        "summary_name": "geom_full_database_final_optimized_amr_cov_full_plan",
        "result_kind": "geom_drugs_full_database_final_optimized_amr_cov_full_plan_not_benchmark",
        "benchmark_result": False,
        "execution_mode": "local_reporting_only_full_amr_cov_planning",
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
        "split_paths": {split: str(SPLIT_PATHS[split]) for split in splits},
        "splits": list(splits),
        "partitions": list(partitions),
        "train_partition_included": "train" in partitions,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "chunking": {
            "chunk_size": int(args.chunk_size),
            "base_molecule_chunk_count": len(base_chunks),
            "standalone_job_count": len(tasks),
            "task_unit": "one_method_split_seed_partition_chunk",
            "deterministic_assignment": "split-file molecule order sliced by chunk_start/chunk_end, then method order, then seed order",
            "chunks_dir": str(chunks_dir),
            "wrapper_dir": str(wrapper_dir),
            "commands_path": str(commands_path),
            "resume_behavior": "chunk evaluator skips completed status/jsonl pairs unless --overwrite is passed",
            "slurm_array_used": False,
            "submission_style": "standalone_single_job_wrappers_only",
            "molecule_slots_per_chunk": {
                "target_chunk_size": int(args.chunk_size),
                "last_chunk_size_by_partition": _last_chunk_sizes(molecule_slots_by_split_partition, int(args.chunk_size)),
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
            "method_count": len(METHODS),
            "seed_count": len(SEEDS),
            "estimated_method_seed_molecule_records": estimated_method_seed_molecule_records,
            "estimated_generated_conformer_count": estimated_generated_conformer_count,
        },
        "training_result_readiness": readiness,
        "resource_estimate": {
            "cpu_memory_risk": "moderate_to_high_total_work_but_low_per_task_memory",
            "per_task_memory_request": "16GB",
            "per_task_cpus": 4,
            "per_task_walltime": "04:00:00",
            "reason": (
                "Each task loads one checkpoint and one split/partition chunk, streams raw pickles, "
                "and writes scalar per-molecule JSONL rows. It does not accumulate all molecules in memory."
            ),
        },
        "commands": {
            "plan_command": "python scripts/plan_geom_full_database_final_optimized_amr_cov_full.py",
            "wrapper_generation_command": (
                "python scripts/make_geom_full_database_final_optimized_amr_cov_full_single_job_wrappers.py "
                "--plan-path experiments/results/geom_full_database_final_optimized_amr_cov_full_plan.json"
            ),
            "single_job_command_list": str(commands_path),
            "first_sbatch_command": tasks[0]["sbatch_command"] if tasks else None,
            "last_sbatch_command": tasks[-1]["sbatch_command"] if tasks else None,
            "recommended_merge_command": (
                "python scripts/merge_geom_full_database_final_optimized_amr_cov_full.py "
                "--plan-path experiments/results/geom_full_database_final_optimized_amr_cov_full_plan.json "
                "--summary-path experiments/results/geom_full_database_final_optimized_amr_cov_full_summary.json "
                "--report-path reports/tables/geom_full_database_final_optimized_amr_cov_full_summary.md "
                "--require-complete"
            ),
            "local_single_task_example": (
                "python scripts/evaluate_geom_full_database_final_optimized_amr_cov_full_chunk.py "
                "--plan-path experiments/results/geom_full_database_final_optimized_amr_cov_full_plan.json --task-id 0"
            ),
        },
        "scheduler_recommendation": {
            "submission_recommended_after_user_approval": readiness["evaluation_ready"],
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
            "reason": "CPU-only chunked checkpoint evaluation; no A100/GPU-specific need in the current minimal model path.",
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
            "This is a full val/test AMR/COV evaluation plan, not executed evaluation.",
            "The default plan intentionally excludes train because train-partition AMR/COV is much larger.",
            "Chunk tasks are resumable and deterministic. Rerun failed/missing task IDs only, or use --overwrite intentionally.",
            "No SLURM array is used or prepared; submit standalone wrappers only after explicit user approval.",
        ],
    }
    return plan


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.chunk_size) <= 0:
        raise ValueError("chunk-size must be positive")
    if int(args.generated_conformers_per_mol) <= 0:
        raise ValueError("generated-conformers-per-mol must be positive")
    if float(args.delta) <= 0.0:
        raise ValueError("delta must be positive")
    if int(args.max_reference_conformers_per_mol) <= 0:
        raise ValueError("max-reference-conformers-per-mol must be positive")
    if int(args.max_pickle_bytes) <= 0:
        raise ValueError("max-pickle-bytes must be positive")


def _load_split(path: Path, split_type: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split_type") != split_type:
        raise ValueError(f"Split file {path} has unexpected split_type={payload.get('split_type')!r}")
    if payload.get("split_scope") != FULL_DATABASE_SCOPE:
        raise ValueError(f"Split file {path} has unexpected split_scope={payload.get('split_scope')!r}")
    for partition in ("train", "val", "test"):
        if not isinstance(payload.get(partition), list):
            raise ValueError(f"Split file {path} is missing partition {partition}")
    return payload


def _last_chunk_sizes(molecule_slots_by_split_partition: dict[str, dict[str, int]], chunk_size: int) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for split, partitions in molecule_slots_by_split_partition.items():
        output[split] = {}
        for partition, count in partitions.items():
            remainder = count % chunk_size
            output[split][partition] = remainder if remainder else chunk_size
    return output


def _training_readiness(splits: tuple[str, ...]) -> dict[str, Any]:
    missing = []
    available = []
    for split in splits:
        for method in METHODS:
            for seed in SEEDS:
                candidates = _candidate_paths(Path("experiments/results"), split_type=split, seed=int(seed), method=method)
                result_path = next((item["path"] for item in candidates if item["path"].exists()), None)
                if result_path is None:
                    missing.append(
                        {
                            "split_type": split,
                            "method": method,
                            "seed": int(seed),
                            "missing": "result_json",
                            "expected_paths": [str(item["path"]) for item in candidates],
                        }
                    )
                    continue
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                checkpoint_paths = [Path(payload["best_checkpoint_path"]), Path(payload["latest_checkpoint_path"])]
                missing_checkpoints = [str(path) for path in checkpoint_paths if not path.exists()]
                if missing_checkpoints:
                    missing.append(
                        {
                            "split_type": split,
                            "method": method,
                            "seed": int(seed),
                            "result_path": str(result_path),
                            "missing": "checkpoint",
                            "missing_checkpoint_paths": missing_checkpoints,
                        }
                    )
                    continue
                available.append(
                    {
                        "split_type": split,
                        "method": method,
                        "seed": int(seed),
                        "result_path": str(result_path),
                        "best_checkpoint_path": str(checkpoint_paths[0]),
                        "latest_checkpoint_path": str(checkpoint_paths[1]),
                    }
                )
    return {
        "expected_result_count": len(splits) * len(METHODS) * len(SEEDS),
        "available_result_count": len(available),
        "missing_result_or_checkpoint_count": len(missing),
        "missing": missing,
        "available": available,
        "evaluation_ready": len(missing) == 0,
    }


def _markdown_report(plan: dict[str, Any]) -> str:
    lines = [
        "# Final Optimized Full Val/Test AMR/COV Plan",
        "",
        "Planning-only artifact for a scalable full val/test AMR/COV evaluation. This is not executed evaluation.",
        "",
        "## Scope",
        "",
        f"- Splits: `{plan['splits']}`",
        f"- Partitions: `{plan['partitions']}`",
        f"- Methods: `{plan['methods']}`",
        f"- Seeds: `{plan['seeds']}`",
        f"- K generated conformers per molecule: `{plan['evaluation']['generated_conformers_per_molecule']}`",
        f"- RMSD threshold delta: `{plan['evaluation']['delta_angstrom']}`",
        f"- Total molecule slots: `{plan['coverage']['total_molecule_slots']}`",
        f"- Base molecule chunks: `{plan['chunking']['base_molecule_chunk_count']}`",
        f"- Standalone jobs: `{plan['chunking']['standalone_job_count']}`",
        f"- SLURM array used: `{plan['chunking']['slurm_array_used']}`",
        f"- Estimated generated conformers: `{plan['coverage']['estimated_generated_conformer_count']}`",
        f"- Estimated generated conformers per full chunk/task: `{plan['chunking']['estimated_generated_conformers_per_full_chunk_per_task']}`",
        f"- Evaluation ready: `{plan['training_result_readiness']['evaluation_ready']}`",
        f"- Resumable: `True`",
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
            "## Resource Estimate",
            "",
            f"- CPU/memory risk: `{plan['resource_estimate']['cpu_memory_risk']}`",
            f"- Per-task CPUs: `{plan['resource_estimate']['per_task_cpus']}`",
            f"- Per-task memory: `{plan['resource_estimate']['per_task_memory_request']}`",
            f"- Per-task wall time: `{plan['resource_estimate']['per_task_walltime']}`",
            f"- Recommended account: `{plan['scheduler_recommendation']['recommended_account']}`",
            f"- Standalone job count: `{plan['scheduler_recommendation']['standalone_job_count']}`",
            f"- Command list: `{plan['scheduler_recommendation']['recommended_submission_list']}`",
            "",
            "## Commands",
            "",
            "Generate standalone single-job wrappers and command list:",
            "",
            "```bash",
            plan["commands"]["wrapper_generation_command"],
            "```",
            "",
            "Command-list file for standalone job submission:",
            "",
            f"`{plan['commands']['single_job_command_list']}`",
            "",
            "First and last generated commands:",
            "",
            "```bash",
            str(plan["commands"]["first_sbatch_command"]),
            str(plan["commands"]["last_sbatch_command"]),
            "```",
            "",
            "Do not submit any command without explicit approval.",
            "",
            "Merge after all standalone jobs complete:",
            "",
            "```bash",
            plan["commands"]["recommended_merge_command"],
            "```",
            "",
            "Single-task local debugging example:",
            "",
            "```bash",
            plan["commands"]["local_single_task_example"],
            "```",
            "",
            "## Readiness",
            "",
            f"- Available optimized result/checkpoint sets: `{plan['training_result_readiness']['available_result_count']}` / `{plan['training_result_readiness']['expected_result_count']}`",
            f"- Missing result/checkpoint sets: `{plan['training_result_readiness']['missing_result_or_checkpoint_count']}`",
            "",
            "```json",
            json.dumps(plan["training_result_readiness"]["missing"], indent=2, sort_keys=True),
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
