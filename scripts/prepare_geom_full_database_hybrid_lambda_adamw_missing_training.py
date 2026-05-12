#!/usr/bin/env python
"""Prepare missing AdamW hybrid-lambda multiseed training jobs."""

from __future__ import annotations

import argparse
import json
import sys
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


LAMBDA_SPECS = (
    ("lambda0025", 0.025),
    ("lambda010", 0.1),
    ("lambda100", 1.0),
    ("lambda300", 3.0),
)
SPLITS = ("random", "scaffold")
SEEDS = (1, 2)
ACCOUNT = "torch_pr_872_general"
DEFAULT_CONFIG_DIR = Path("configs/train/hybrid_lambda_adamw_multiseed")
DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_hybrid_lambda_adamw_missing_training_plan.json")
DEFAULT_REPORT_PATH = Path("reports/tables/geom_full_database_hybrid_lambda_adamw_missing_training_plan.md")
DEFAULT_COMMANDS_PATH = Path("scripts/geom_full_database_hybrid_lambda_adamw_missing_training_commands.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--plan-path", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--commands-path", default=str(DEFAULT_COMMANDS_PATH))
    args = parser.parse_args()

    plan = build_plan(args)
    plan_path = Path(args.plan_path)
    report_path = Path(args.report_path)
    commands_path = Path(args.commands_path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(plan), encoding="utf-8")
    commands_path.write_text(_commands_markdown(plan), encoding="utf-8")
    print(f"plan: {plan_path}")
    print(f"report: {report_path}")
    print(f"commands: {commands_path}")
    print("benchmark_result: false")
    print("note: Prepared standalone missing-training wrappers only; no scheduler submission, training, raw-data mutation, or environment mutation.")


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split_type in SPLITS:
        for lambda_label, lambda_value in LAMBDA_SPECS:
            for seed in SEEDS:
                result_path = _expected_result_path(split_type, lambda_label, seed)
                best_path = _checkpoint_path(result_path, "best")
                latest_path = _checkpoint_path(result_path, "latest")
                complete = result_path.exists() and best_path.exists() and latest_path.exists()
                config_path = config_dir / (
                    f"geom_full_database_{split_type}_hybrid_harmonic_stiffness_v1_"
                    f"{lambda_label}_seed{seed}_adamw_lr1e3.yaml"
                )
                wrapper_path = Path("scripts") / (
                    f"slurm_geom_full_database_hybrid_lambda_adamw_{split_type}_"
                    f"{lambda_label}_seed{seed}_cpu.sbatch"
                )
                config_path.write_text(_config_text(split_type, lambda_label, lambda_value, seed), encoding="utf-8")
                wrapper_path.write_text(_wrapper_text(config_path, split_type, lambda_label, seed), encoding="utf-8")
                jobs.append(
                    {
                        "split_type": split_type,
                        "seed": seed,
                        "lambda_label": lambda_label,
                        "lambda_value": lambda_value,
                        "config_path": str(config_path),
                        "wrapper_path": str(wrapper_path),
                        "sbatch_command": f"sbatch --account={ACCOUNT} {wrapper_path}",
                        "expected_result_path": str(result_path),
                        "expected_best_checkpoint_path": str(best_path),
                        "expected_latest_checkpoint_path": str(latest_path),
                        "already_complete": complete,
                    }
                )
    missing_jobs = [job for job in jobs if not job["already_complete"]]
    return {
        "summary_name": "geom_full_database_hybrid_lambda_adamw_missing_training_plan",
        "result_kind": "geom_drugs_full_database_hybrid_lambda_adamw_missing_training_plan_not_benchmark",
        "benchmark_result": False,
        "execution_mode": "local_reporting_only_hybrid_lambda_missing_training_planning",
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
        "split_paths": {
            "random": str(FULL_DATABASE_SPLIT_PATH),
            "scaffold": str(FULL_DATABASE_SCAFFOLD_SPLIT_PATH),
        },
        "method": "hybrid_harmonic_stiffness_v1",
        "optimizer": "AdamW",
        "learning_rate": 1.0e-3,
        "scheduler_used": False,
        "splits": list(SPLITS),
        "seeds": list(SEEDS),
        "lambdas": [{"lambda_label": label, "lambda_value": value} for label, value in LAMBDA_SPECS],
        "training_jobs": jobs,
        "job_count": len(jobs),
        "missing_training_jobs": missing_jobs,
        "missing_training_job_count": len(missing_jobs),
        "completed_training_job_count": len(jobs) - len(missing_jobs),
        "commands": {
            "command_list": str(DEFAULT_COMMANDS_PATH),
            "first_missing_sbatch_command": missing_jobs[0]["sbatch_command"] if missing_jobs else None,
            "last_missing_sbatch_command": missing_jobs[-1]["sbatch_command"] if missing_jobs else None,
        },
        "resource_request": {
            "account": ACCOUNT,
            "cpus_per_task": 32,
            "memory": "64GB",
            "time": "2:00:00",
            "gpu": None,
        },
        "notes": [
            "Only missing seeds 1/2 for lambdas 0.025, 0.1, 1.0, and 3.0 are prepared.",
            "Lambda 5.0 is already complete and is not rerun.",
            "Each wrapper is standalone; no SLURM array is prepared.",
        ],
    }


def _config_text(split_type: str, lambda_label: str, lambda_value: float, seed: int) -> str:
    split_root = "data/splits/geom_drugs_full_database/random_seed0"
    split_path = "data/splits/geom_drugs_full_database/random_seed0/split.json"
    sample_scope = "all_full_database_molecules"
    split_fragment = ""
    if split_type == "scaffold":
        split_root = "data/splits/geom_drugs_full_database/scaffold_seed0"
        split_path = "data/splits/geom_drugs_full_database/scaffold_seed0/split.json"
        sample_scope = "all_full_database_scaffold_molecules"
        split_fragment = "_scaffold"
    name = (
        f"geom_formal_full_database{split_fragment}_train_hybrid_harmonic_stiffness_v1_"
        f"{lambda_label}_steps2000_seed{seed}_low_lambda_adamw_lr1e3"
    )
    return f"""experiment:
  name: {name}
  run_name_suffix: low_lambda_adamw_lr1e3
  seed: {seed}
  output_root: experiments

data:
  dataset_name: geom_drugs
  dataset_kind: geom_drugs_tiny_processed_dry_run
  raw_root: data/raw/geom_drugs
  processed_root: data/processed/geom_drugs_full_database
  split_root: {split_root}
  processed_artifact_path: data/processed/geom_drugs_full_database/molecules.json
  split_path: {split_path}
  split_type: {split_type}
  subset:
    max_molecules: full_database
    max_conformers_per_mol: 1
    tag: geom_drugs_full_database_one_conformer

model:
  backbone: minimal_linear_velocity_pilot
  pilot_model: single_linear_velocity_pilot
  hidden_dim: 0
  num_layers: 1
  time_embedding_dim: 1

flow_matching:
  objective: cartesian_linear
  interpolation: linear

corruption:
  mode: hybrid_harmonic_stiffness_v1
  stiffness_table: heuristic_v1
  protect_rings: true
  protect_amides: true
  protect_aromatic_conjugated: true
  isotropic_sigma: 0.25
  harmonic_sigma: 0.25
  harmonic_smoothing_steps: 2
  harmonic_neighbor_weight: 0.65
  harmonic_anchor_weight: 0.15
  modulation_strength: {lambda_value:.6g}
  hybrid_min_noise_scale: 0.50
  hybrid_max_noise_scale: 1.25
  bond_sigma: 0.02
  angle_sigma: 0.08
  torsion_sigma: 0.30

train:
  enabled: true
  formal_training: true
  pilot_only: false
  optimizer_pilot: false
  low_lambda_adamw_refinement: true
  batch_scope: full_database_train_split
  max_steps: 2000
  hard_max_steps: 2000
  optimizer: AdamW
  optimizer_weight_decay: 0.0
  learning_rate: 1.0e-3
  num_workers: 0
  checkpointing_enabled: true
  checkpoint_scope: reusable_formal_full_database_validation_selected_checkpoint
  early_stop:
    enabled: true
    monitor: validation_loss_after_step
    patience: 200
    min_delta: 1.0e-6

sample:
  enabled: true
  pilot_only: false
  sample_step_count: 1
  sample_step_size: 0.1
  sample_scope: {sample_scope}

eval:
  enabled: true
  eval_subset_size: 0
  compute_train_val_test_losses: true
  compute_cov_mat: false
  compute_validity: false
  compute_energy_checks: false
  compute_physical_validity_diagnostics: true
  compute_rotatable_bond_stratification: true
  compute_rotatable_bond_count_from_smiles: true

logging:
  save_manifest: true
  save_results_json: true
  save_log: true
"""


def _wrapper_text(config_path: Path, split_type: str, lambda_label: str, seed: int) -> str:
    stem = f"slurm_geom_full_database_hybrid_lambda_adamw_{split_type}_{lambda_label}_seed{seed}_cpu"
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name=hlam_{split_type[:4]}_{lambda_label.replace('lambda', 'l')}_s{seed}",
            "#SBATCH --nodes=1",
            "#SBATCH --cpus-per-task=32",
            "#SBATCH --mem=64GB",
            "#SBATCH --time=2:00:00",
            f"#SBATCH --output=experiments/logs/{stem}_%j.out",
            f"#SBATCH --error=experiments/logs/{stem}_%j.err",
            f"#SBATCH --account={ACCOUNT}",
            "",
            "set -euo pipefail",
            "",
            "cd /scratch/ys6132/project/0_stiff_flow_matching",
            "mkdir -p experiments/logs",
            f"export STIFFLOW_SLURM_ACCOUNT={ACCOUNT}",
            "source scripts/activate_env.sh",
            "",
            "python scripts/run_geom_full_database_formal_training_config_single.py \\",
            f"  --config {config_path} \\",
            f"  --split-type {split_type} \\",
            f"  --seed {seed} \\",
            "  --device cpu \\",
            "  --no-require-cuda",
            "",
        ]
    )


def _expected_result_path(split_type: str, lambda_label: str, seed: int) -> Path:
    split_fragment = "" if split_type == "random" else "_scaffold"
    return Path(
        f"experiments/results/geom_formal_full_database{split_fragment}_train_"
        f"hybrid_harmonic_stiffness_v1_{lambda_label}_steps2000_seed{seed}_low_lambda_adamw_lr1e3.json"
    )


def _checkpoint_path(result_path: Path, kind: str) -> Path:
    return Path(str(result_path).replace("experiments/results/", "experiments/checkpoints/").replace(".json", f"_{kind}.pt"))


def _commands_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Lambda AdamW Missing Training Commands",
        "",
        "Submit only after explicit approval. These are standalone single-job submissions; no SLURM arrays are used.",
        "",
        "```bash",
    ]
    for job in plan["missing_training_jobs"]:
        lines.append(job["sbatch_command"])
    lines.extend(["```", ""])
    return "\n".join(lines)


def _markdown_report(plan: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Lambda AdamW Missing Training Plan",
        "",
        "Planning-only artifact for missing multiseed low-lambda AdamW hybrid runs. This is not a benchmark.",
        "",
        f"- Missing training jobs: `{plan['missing_training_job_count']}` / `{plan['job_count']}`",
        f"- Account: `{plan['resource_request']['account']}`",
        f"- Resource request: `{plan['resource_request']}`",
        "",
        "| Split | Lambda | Seed | Config | Wrapper | Complete |",
        "|---|---:|---:|---|---|---|",
    ]
    for job in plan["training_jobs"]:
        lines.append(
            f"| {job['split_type']} | {job['lambda_value']} | {job['seed']} | "
            f"`{job['config_path']}` | `{job['wrapper_path']}` | `{job['already_complete']}` |"
        )
    lines.extend(
        [
            "",
            "## Commands",
            "",
            f"- Command list: `{plan['commands']['command_list']}`",
            f"- First missing command: `{plan['commands']['first_missing_sbatch_command']}`",
            f"- Last missing command: `{plan['commands']['last_missing_sbatch_command']}`",
            "",
            "## Safety",
            "",
            f"- `benchmark_result`: `{plan['benchmark_result']}`",
            f"- `training_executed_by_plan`: `{plan['training_executed_by_plan']}`",
            f"- `raw_data_modified`: `{plan['raw_data_modified']}`",
            f"- `environment_modified`: `{plan['environment_modified']}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
