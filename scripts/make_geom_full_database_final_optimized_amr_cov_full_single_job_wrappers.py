#!/usr/bin/env python
"""Generate standalone SLURM wrappers for the full final-optimized AMR/COV plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_final_optimized_amr_cov_full_plan.json")
DEFAULT_COMMANDS_PATH = Path("scripts/geom_full_database_final_optimized_amr_cov_full_single_job_commands.md")
ACCOUNT = "torch_pr_872_general"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-path", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--commands-path", default=None)
    parser.add_argument("--wrapper-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-every", type=int, default=128)
    parser.add_argument("--overwrite-wrappers", action="store_true")
    args = parser.parse_args()

    outputs = generate(args)
    for key, value in outputs.items():
        print(f"{key}: {value}")
    print("benchmark_result: false")
    print("note: Generated standalone single-job SLURM wrappers only; no scheduler submission, training, raw-data mutation, or environment mutation.")


def generate(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = Path(args.plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _validate_plan(plan)
    commands_path = Path(args.commands_path or plan["chunking"]["commands_path"])
    wrapper_dir = Path(args.wrapper_dir or plan["chunking"]["wrapper_dir"])
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    commands_path.parent.mkdir(parents=True, exist_ok=True)

    generated = []
    skipped = []
    for task in plan["tasks"]:
        wrapper_path = wrapper_dir / Path(task["wrapper_path"]).name
        if wrapper_path.exists() and not bool(args.overwrite_wrappers):
            skipped.append(str(wrapper_path))
        else:
            wrapper_path.write_text(
                _wrapper_text(
                    plan_path=plan_path,
                    task=task,
                    wrapper_path=wrapper_path,
                    device=str(args.device),
                    progress_every=int(args.progress_every),
                ),
                encoding="utf-8",
            )
            generated.append(str(wrapper_path))
        task["wrapper_path"] = str(wrapper_path)
        task["sbatch_command"] = f"sbatch --account={ACCOUNT} {wrapper_path}"

    commands_path.write_text(_commands_markdown(plan=plan, plan_path=plan_path), encoding="utf-8")
    return {
        "wrapper_count": len(plan["tasks"]),
        "generated_wrapper_count": len(generated),
        "existing_wrapper_count": len(skipped),
        "wrapper_dir": str(wrapper_dir),
        "commands": str(commands_path),
    }


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("result_kind") != "geom_drugs_full_database_final_optimized_amr_cov_full_plan_not_benchmark":
        raise ValueError("Unexpected full AMR/COV plan result_kind")
    if plan.get("benchmark_result") is not False:
        raise ValueError("Full AMR/COV plan must be benchmark_result=false")
    if plan.get("chunking", {}).get("slurm_array_used") is not False:
        raise ValueError("Single-job wrapper generation requires slurm_array_used=false")
    if plan.get("scheduler_recommendation", {}).get("slurm_array_used") is not False:
        raise ValueError("Single-job wrapper generation refuses array-style plans")


def _wrapper_text(plan_path: Path, task: dict[str, Any], wrapper_path: Path, device: str, progress_every: int) -> str:
    task_id = int(task["task_id"])
    log_stem = f"slurm_geom_full_database_final_optimized_amr_cov_full_chunk_{task_id:05d}"
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name=geom_amr_{task_id:05d}",
            "#SBATCH --nodes=1",
            "#SBATCH --cpus-per-task=4",
            "#SBATCH --mem=16GB",
            "#SBATCH --time=4:00:00",
            f"#SBATCH --output=experiments/logs/{log_stem}_%j.out",
            f"#SBATCH --error=experiments/logs/{log_stem}_%j.err",
            f"#SBATCH --account={ACCOUNT}",
            "",
            "set -euo pipefail",
            "",
            "cd /scratch/ys6132/project/0_stiff_flow_matching",
            "mkdir -p experiments/logs experiments/results/geom_full_database_final_optimized_amr_cov_full_chunks",
            f"export STIFFLOW_SLURM_ACCOUNT={ACCOUNT}",
            "",
            "source scripts/activate_env.sh",
            "",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "",
            "# Standalone single-job chunk. Array directives are intentionally absent.",
            "python scripts/evaluate_geom_full_database_final_optimized_amr_cov_full_chunk.py \\",
            f"  --plan-path {plan_path} \\",
            f"  --task-id {task_id} \\",
            f"  --device {device} \\",
            f"  --progress-every {progress_every}",
            "",
            f"# Wrapper: {wrapper_path}",
            f"# Split/partition/method/seed/chunk: {task['split_type']} {task['partition']} {task['method']} seed{task['seed']} chunk{task['chunk_index']}",
        ]
    ) + "\n"


def _commands_markdown(plan: dict[str, Any], plan_path: Path) -> str:
    lines = [
        "# Full Val/Test AMR/COV Standalone Job Commands",
        "",
        "Submit only after explicit approval. These are standalone single-job submissions; no SLURM array is used.",
        "",
        f"- Plan: `{plan_path}`",
        f"- Standalone jobs: `{len(plan['tasks'])}`",
        f"- Account: `{ACCOUNT}`",
        f"- Chunk directory: `{plan['chunking']['chunks_dir']}`",
        f"- K generated conformers per molecule: `{plan['evaluation']['generated_conformers_per_molecule']}`",
        f"- RMSD threshold delta: `{plan['evaluation']['delta_angstrom']}`",
        "",
        "Merge after all approved jobs complete:",
        "",
        "```bash",
        plan["commands"]["recommended_merge_command"],
        "```",
        "",
        "## Commands",
        "",
        "```bash",
    ]
    for task in plan["tasks"]:
        lines.append(task["sbatch_command"])
    lines.extend(["```", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
