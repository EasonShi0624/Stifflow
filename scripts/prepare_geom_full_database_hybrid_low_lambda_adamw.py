#!/usr/bin/env python
"""Prepare Stage 1 low-lambda AdamW hybrid training and COV/AMR selection paths."""

from __future__ import annotations

import argparse
import json
import math
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
    ("lambda005", 0.05),
    ("lambda010", 0.1),
    ("lambda015", 0.15),
    ("lambda020", 0.2),
    ("lambda025", 0.25),
    ("lambda050", 0.5),
    ("lambda100", 1.0),
    ("lambda300", 3.0),
)
SPLITS = ("random", "scaffold")
DEFAULT_CONFIG_DIR = Path("configs/train/final_optimized_low_lambda")
DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_hybrid_low_lambda_adamw_plan.json")
DEFAULT_REPORT_PATH = Path("reports/tables/geom_full_database_hybrid_low_lambda_adamw_plan.md")
DEFAULT_COMMANDS_PATH = Path("scripts/geom_full_database_hybrid_low_lambda_adamw_stage1_commands.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--plan-path", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--commands-path", default=str(DEFAULT_COMMANDS_PATH))
    args = parser.parse_args()

    plan = build_outputs(args)
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
    print("note: Prepared low-lambda AdamW configs and standalone wrappers only; no training, evaluation, scheduler submission, raw-data mutation, or environment mutation.")


def build_outputs(args: argparse.Namespace) -> dict[str, Any]:
    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split_type in SPLITS:
        for lambda_label, lambda_value in LAMBDA_SPECS:
            config_path = config_dir / f"geom_full_database_{split_type}_hybrid_harmonic_stiffness_v1_{lambda_label}_adamw_lr1e3.yaml"
            wrapper_path = Path("scripts") / f"slurm_geom_full_database_hybrid_low_lambda_adamw_{split_type}_{lambda_label}_seed0_cpu.sbatch"
            config_path.write_text(_config_text(split_type, lambda_label, lambda_value), encoding="utf-8")
            wrapper_path.write_text(_wrapper_text(config_path, split_type, lambda_label), encoding="utf-8")
            expected_result_path = _expected_result_path(split_type, lambda_value)
            jobs.append(
                {
                    "split_type": split_type,
                    "seed": 0,
                    "lambda_label": lambda_label,
                    "lambda_value": lambda_value,
                    "config_path": str(config_path),
                    "wrapper_path": str(wrapper_path),
                    "sbatch_command": f"sbatch --account=torch_pr_872_general {wrapper_path}",
                    "expected_result_path": str(expected_result_path),
                    "expected_best_checkpoint_path": str(expected_result_path).replace("experiments/results/", "experiments/checkpoints/").replace(".json", "_best.pt"),
                    "expected_latest_checkpoint_path": str(expected_result_path).replace("experiments/results/", "experiments/checkpoints/").replace(".json", "_latest.pt"),
                    "already_complete": expected_result_path.exists(),
                }
            )

    harmonic_refs = _reference_rows("harmonic")
    hybrid_lambda5_refs = _reference_rows("hybrid_lambda500")
    subset_plan = _subset_plan_summary()
    plan = {
        "summary_name": "geom_full_database_hybrid_low_lambda_adamw_plan",
        "result_kind": "geom_drugs_full_database_hybrid_low_lambda_adamw_plan_not_benchmark",
        "benchmark_result": False,
        "execution_mode": "local_reporting_only_low_lambda_adamw_planning",
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
        "stage": "stage1_random_and_scaffold_seed0",
        "method": "hybrid_harmonic_stiffness_v1",
        "optimizer": "AdamW",
        "learning_rate": 1.0e-3,
        "scheduler_used": False,
        "lambdas": [{"lambda_label": label, "lambda_value": value} for label, value in LAMBDA_SPECS],
        "splits": list(SPLITS),
        "seeds": [0],
        "training_jobs": jobs,
        "job_count": len(jobs),
        "completed_training_job_count": sum(1 for job in jobs if job["already_complete"]),
        "missing_training_job_count": sum(1 for job in jobs if not job["already_complete"]),
        "references": {
            "lambda0_harmonic_prior": harmonic_refs,
            "lambda5_current_hybrid_reference": hybrid_lambda5_refs,
            "reference_policy": (
                "lambda=0 is represented by optimized harmonic_prior AdamW lr=1e-3; "
                "lambda=5 is included as the existing RMSD-optimal high-lambda reference and is not rerun."
            ),
        },
        "cov_amr_subset": subset_plan,
        "estimated_cov_amr_size": {
            "molecule_slots": subset_plan["selected_molecule_slots"],
            "candidate_lambda_count": len(LAMBDA_SPECS),
            "reference_count": 2,
            "generated_conformers_per_molecule": 20,
            "candidate_generated_conformers": subset_plan["selected_molecule_slots"] * len(LAMBDA_SPECS) * 20,
            "with_references_generated_conformers": subset_plan["selected_molecule_slots"] * (len(LAMBDA_SPECS) + 2) * 20,
        },
        "commands": {
            "stage1_training_command_list": str(DEFAULT_COMMANDS_PATH),
            "cov_amr_sbatch_command": "sbatch --account=torch_pr_872_general scripts/slurm_geom_full_database_hybrid_low_lambda_cov_amr_cpu.sbatch",
            "training_summary_command": (
                "python scripts/summarize_geom_full_database_hybrid_low_lambda_adamw.py "
                "--plan-path experiments/results/geom_full_database_hybrid_low_lambda_adamw_plan.json --require-results"
            ),
            "cov_amr_evaluation_command": (
                "python scripts/evaluate_geom_full_database_hybrid_low_lambda_cov_amr.py "
                "--plan-path experiments/results/geom_full_database_hybrid_low_lambda_adamw_plan.json --device cpu"
            ),
        },
        "output_targets": {
            "training_summary": "experiments/results/geom_full_database_hybrid_low_lambda_adamw_summary.json",
            "training_report": "reports/tables/geom_full_database_hybrid_low_lambda_adamw_summary.md",
            "cov_amr_summary": "experiments/results/geom_full_database_hybrid_low_lambda_cov_amr_summary.json",
            "cov_amr_report": "reports/tables/geom_full_database_hybrid_low_lambda_cov_amr_summary.md",
        },
        "notes": [
            "Old SGD-trained lambda results are intentionally not used as final lambda evidence.",
            "Configs use the narrow low_lambda_adamw_refinement validation gate for seed0 random/scaffold hybrid candidates under AdamW lr=1e-3.",
            "Stage 2 should choose at most 2-3 lambdas for seeds 1/2 after Stage 1 COV/AMR is summarized.",
            "No scheduler jobs are submitted by this preparation script.",
        ],
    }
    return plan


def _config_text(split_type: str, lambda_label: str, lambda_value: float) -> str:
    split_root = "data/splits/geom_drugs_full_database/random_seed0"
    split_path = "data/splits/geom_drugs_full_database/random_seed0/split.json"
    sample_scope = "all_full_database_molecules"
    split_fragment = ""
    if split_type == "scaffold":
        split_root = "data/splits/geom_drugs_full_database/scaffold_seed0"
        split_path = "data/splits/geom_drugs_full_database/scaffold_seed0/split.json"
        sample_scope = "all_full_database_scaffold_molecules"
        split_fragment = "_scaffold"
    name = f"geom_formal_full_database{split_fragment}_train_hybrid_harmonic_stiffness_v1_{lambda_label}_steps2000_seed0_low_lambda_adamw_lr1e3"
    return f"""experiment:
  name: {name}
  run_name_suffix: low_lambda_adamw_lr1e3
  seed: 0
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


def _wrapper_text(config_path: Path, split_type: str, lambda_label: str) -> str:
    stem = f"slurm_geom_full_database_hybrid_low_lambda_adamw_{split_type}_{lambda_label}_seed0_cpu"
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name=geom_ll_{split_type[:4]}_{lambda_label.replace('lambda', 'l')}",
            "#SBATCH --nodes=1",
            "#SBATCH --cpus-per-task=32",
            "#SBATCH --mem=64GB",
            "#SBATCH --time=2:00:00",
            f"#SBATCH --output=experiments/logs/{stem}_%j.out",
            f"#SBATCH --error=experiments/logs/{stem}_%j.err",
            "#SBATCH --account=torch_pr_872_general",
            "",
            "set -euo pipefail",
            "",
            "cd /scratch/ys6132/project/0_stiff_flow_matching",
            "mkdir -p experiments/logs",
            "export STIFFLOW_SLURM_ACCOUNT=torch_pr_872_general",
            "source scripts/activate_env.sh",
            "",
            "python scripts/run_geom_full_database_formal_training_config_single.py \\",
            f"  --config {config_path} \\",
            f"  --split-type {split_type} \\",
            "  --seed 0 \\",
            "  --device cpu \\",
            "  --no-require-cuda",
            "",
        ]
    )


def _expected_result_path(split_type: str, lambda_value: float) -> Path:
    label = _lambda_label(lambda_value)
    split_label = "" if split_type == "random" else "_scaffold"
    return Path(
        f"experiments/results/geom_formal_full_database{split_label}_train_"
        f"hybrid_harmonic_stiffness_v1_{label}_steps2000_seed0_low_lambda_adamw_lr1e3.json"
    )


def _lambda_label(value: float) -> str:
    centi_value = float(value) * 100.0
    if abs(centi_value - round(centi_value)) <= 1.0e-9:
        return f"lambda{int(round(centi_value)):03d}"
    return f"lambda{int(round(float(value) * 1000.0)):04d}"


def _reference_rows(kind: str) -> list[dict[str, Any]]:
    rows = []
    for split_type in SPLITS:
        if kind == "harmonic":
            if split_type == "random":
                result_path = Path("experiments/results/geom_formal_full_database_train_harmonic_steps2000_seed0_opt_adamw_lr1e3.json")
                config_path = Path("configs/train/optimization/geom_full_database_random_seed0_harmonic_adamw_lr1e3.yaml")
            else:
                result_path = Path("experiments/results/geom_formal_full_database_scaffold_train_harmonic_steps2000_seed0_final_adamw_lr1e3.json")
                config_path = Path("configs/train/final_optimized/geom_full_database_scaffold_harmonic_adamw_lr1e3.yaml")
        else:
            if split_type == "random":
                result_path = Path(
                    "experiments/results/geom_formal_full_database_train_hybrid_harmonic_stiffness_v1_lambda500_steps2000_seed0_opt_adamw_lr1e3.json"
                )
                config_path = Path("configs/train/optimization/geom_full_database_random_seed0_hybrid_harmonic_stiffness_v1_lambda500_adamw_lr1e3.yaml")
            else:
                result_path = Path(
                    "experiments/results/geom_formal_full_database_scaffold_train_hybrid_harmonic_stiffness_v1_lambda500_steps2000_seed0_final_adamw_lr1e3.json"
                )
                config_path = Path("configs/train/final_optimized/geom_full_database_scaffold_hybrid_harmonic_stiffness_v1_lambda500_adamw_lr1e3.yaml")
        rows.append(
            {
                "split_type": split_type,
                "result_path": str(result_path),
                "config_path": str(config_path),
                "result_exists": result_path.exists(),
                "config_exists": config_path.exists(),
            }
        )
    return rows


def _subset_plan_summary() -> dict[str, Any]:
    path = Path("experiments/results/geom_full_database_final_optimized_amr_cov_subset_plan.json")
    if not path.exists():
        return {"available": False, "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_molecules", [])
    return {
        "available": True,
        "path": str(path),
        "selected_molecule_slots": len(selected),
        "random_slots": sum(1 for row in selected if row.get("split_type") == "random"),
        "scaffold_slots": sum(1 for row in selected if row.get("split_type") == "scaffold"),
        "partitions": payload.get("partitions"),
        "rotatable_bins": payload.get("rotatable_bins"),
        "source_reference": "same bounded COV/AMR subset plan and raw GEOM reference conformer source as previous evaluations",
    }


def _commands_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Low-Lambda AdamW Stage 1 Commands",
        "",
        "Do not submit without explicit approval. These are standalone single-job submissions; no SLURM arrays are used.",
        "",
        "```bash",
    ]
    for job in plan["training_jobs"]:
        if not job["already_complete"]:
            lines.append(job["sbatch_command"])
    lines.extend(
        [
            "```",
            "",
            "After approved training jobs complete, summarize training:",
            "",
            "```bash",
            plan["commands"]["training_summary_command"],
            "```",
            "",
            "Then run the bounded COV/AMR evaluation only after approval:",
            "",
            "```bash",
            plan["commands"]["cov_amr_sbatch_command"],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_report(plan: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Low-Lambda AdamW Plan",
        "",
        "Planning-only Stage 1 low-lambda refinement under AdamW lr=1e-3. This is not a benchmark.",
        "",
        "## Scope",
        "",
        f"- Method: `{plan['method']}`",
        f"- Optimizer: `{plan['optimizer']}`",
        f"- Learning rate: `{plan['learning_rate']}`",
        f"- Splits: `{plan['splits']}`",
        f"- Seeds: `{plan['seeds']}`",
        f"- Lambdas: `{[row['lambda_value'] for row in plan['lambdas']]}`",
        f"- Standalone training jobs: `{plan['job_count']}`",
        f"- Missing training jobs: `{plan['missing_training_job_count']}`",
        "",
        "## Training Jobs",
        "",
        "| Split | Lambda | Config | Wrapper | Complete |",
        "|---|---:|---|---|---|",
    ]
    for job in plan["training_jobs"]:
        lines.append(
            f"| {job['split_type']} | {job['lambda_value']} | `{job['config_path']}` | "
            f"`{job['wrapper_path']}` | {job['already_complete']} |"
        )
    lines.extend(
        [
            "",
            "## COV/AMR Evaluation",
            "",
            f"- Molecule slots: `{plan['cov_amr_subset'].get('selected_molecule_slots')}`",
            f"- Random slots: `{plan['cov_amr_subset'].get('random_slots')}`",
            f"- Scaffold slots: `{plan['cov_amr_subset'].get('scaffold_slots')}`",
            f"- Candidate generated conformers: `{plan['estimated_cov_amr_size']['candidate_generated_conformers']}`",
            f"- With references generated conformers: `{plan['estimated_cov_amr_size']['with_references_generated_conformers']}`",
            "",
            "## References",
            "",
            f"- Policy: {plan['references']['reference_policy']}",
            "",
            "## Commands",
            "",
            f"- Training command list: `{plan['commands']['stage1_training_command_list']}`",
            f"- Training summary command: `{plan['commands']['training_summary_command']}`",
            f"- COV/AMR command: `{plan['commands']['cov_amr_sbatch_command']}`",
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
