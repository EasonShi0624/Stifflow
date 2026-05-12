#!/usr/bin/env python
"""Generate standalone SLURM wrappers for complete val/test COV/AMR chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_final_cov_amr_complete_valtest_plan.json")
DEFAULT_COMMANDS_PATH = Path("scripts/geom_full_database_final_cov_amr_complete_valtest_single_job_commands.md")
PLAN_RESULT_KIND = "geom_drugs_full_database_final_cov_amr_complete_valtest_plan_not_benchmark"
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
    existing = []
    for task in plan["tasks"]:
        wrapper_path = wrapper_dir / Path(task["wrapper_path"]).name
        if wrapper_path.exists() and not bool(args.overwrite_wrappers):
            existing.append(str(wrapper_path))
        else:
            wrapper_path.write_text(
                _wrapper_text(
                    plan_path=plan_path,
                    task=task,
                    wrapper_path=wrapper_path,
                    device=str(args.device),
                    progress_every=int(args.progress_every),
                    chunks_dir=Path(plan["chunking"]["chunks_dir"]),
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
        "existing_wrapper_count": len(existing),
        "wrapper_dir": str(wrapper_dir),
        "commands": str(commands_path),
    }


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("result_kind") != PLAN_RESULT_KIND:
        raise ValueError("Unexpected complete val/test COV/AMR plan result_kind")
    if plan.get("benchmark_result") is not False:
        raise ValueError("Complete val/test COV/AMR plan must be benchmark_result=false")
    if plan.get("train_partition_included") is not False:
        raise ValueError("Complete val/test wrapper generation refuses train-partition plans")
    if plan.get("chunking", {}).get("slurm_array_used") is not False:
        raise ValueError("Standalone wrapper generation refuses array-style plans")
    if plan.get("scheduler_recommendation", {}).get("slurm_array_used") is not False:
        raise ValueError("Standalone wrapper generation refuses array-style scheduler recommendations")


def _wrapper_text(
    plan_path: Path,
    task: dict[str, Any],
    wrapper_path: Path,
    device: str,
    progress_every: int,
    chunks_dir: Path,
) -> str:
    task_id = int(task["task_id"])
    log_stem = f"slurm_geom_full_database_final_cov_amr_complete_valtest_chunk_{task_id:05d}"
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name=covamr_{task_id:05d}",
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
            f"mkdir -p experiments/logs {chunks_dir}",
            f"export STIFFLOW_SLURM_ACCOUNT={ACCOUNT}",
            "",
            "source scripts/activate_env.sh",
            "",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "",
            "python scripts/evaluate_geom_full_database_final_cov_amr_complete_valtest_chunk.py \\",
            f"  --plan-path {plan_path} \\",
            f"  --task-id {task_id} \\",
            f"  --device {device} \\",
            f"  --progress-every {progress_every}",
            "",
            f"# Wrapper: {wrapper_path}",
            (
                "# Scope: "
                f"{task['split_type']} {task['partition']} {task['entry_label']} "
                f"seed{task['seed']} chunk{task['chunk_index']}"
            ),
        ]
    ) + "\n"


def _commands_markdown(plan: dict[str, Any], plan_path: Path) -> str:
    lines = [
        "# Complete Val/Test COV/AMR Standalone Job Commands",
        "",
        "Submit only after explicit approval. These are independent single-job submissions.",
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
        plan["commands"]["merge_command"],
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
