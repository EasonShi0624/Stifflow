#!/usr/bin/env python
"""Bounded COV/AMR lambda-selection evaluation for hybrid harmonic-stiffness."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
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


DEFAULT_SOURCE_SUBSET_PLAN_PATH = Path("experiments/results/geom_full_database_final_optimized_amr_cov_subset_plan.json")
DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_hybrid_lambda_cov_amr_selection_plan.json")
DEFAULT_PLAN_REPORT_PATH = Path("reports/tables/geom_full_database_hybrid_lambda_cov_amr_selection_plan.md")
DEFAULT_SUMMARY_PATH = Path("experiments/results/geom_full_database_hybrid_lambda_cov_amr_selection_summary.json")
DEFAULT_REPORT_PATH = Path("reports/tables/geom_full_database_hybrid_lambda_cov_amr_selection_summary.md")
DEFAULT_PER_MOLECULE_PATH = Path("experiments/results/geom_full_database_hybrid_lambda_cov_amr_selection_per_molecule.jsonl")
LAMBDA_CODES = (
    ("025", 0.25),
    ("050", 0.5),
    ("075", 0.75),
    ("100", 1.0),
    ("150", 1.5),
    ("200", 2.0),
    ("300", 3.0),
    ("500", 5.0),
)
METRICS = ("cov_r", "cov_p", "amr_r", "amr_p")
HIGHER_IS_BETTER = {"cov_r", "cov_p"}
TIE_TOLERANCE = 1.0e-12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-plan-only", action="store_true")
    parser.add_argument("--source-subset-plan-path", default=str(DEFAULT_SOURCE_SUBSET_PLAN_PATH))
    parser.add_argument("--plan-path", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--plan-report-path", default=str(DEFAULT_PLAN_REPORT_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--per-molecule-jsonl-path", default=str(DEFAULT_PER_MOLECULE_PATH))
    parser.add_argument("--raw-root", default="data/raw/geom_drugs")
    parser.add_argument("--generated-conformers-per-mol", type=int, default=20)
    parser.add_argument("--delta", type=float, default=0.75)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()

    if args.write_plan_only:
        outputs = write_plan(args)
    else:
        outputs = evaluate(args)
    for key, path in outputs.items():
        print(f"{key}: {path}")
    print("benchmark_result: false")
    print("note: Hybrid lambda COV/AMR selection path uses existing checkpoints only; no training, scheduler submission, raw-data mutation, or environment mutation.")


def write_plan(args: argparse.Namespace) -> dict[str, Path]:
    plan = build_plan(args)
    plan_path = Path(args.plan_path)
    report_path = Path(args.plan_report_path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_plan_markdown(plan), encoding="utf-8")
    return {"plan": plan_path, "report": report_path}


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    source_plan_path = Path(args.source_subset_plan_path)
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    _validate_source_plan(source_plan)
    selected = [
        {**row, "lambda_selection_index": index}
        for index, row in enumerate(source_plan["selected_molecules"])
        if row["split_type"] == "scaffold" and row["partition"] in {"val", "test"}
    ]
    entries = _evaluation_entries()
    random_availability = _random_lambda_availability()
    available_entries = [entry for entry in entries if entry["available"]]
    candidate_entries = [entry for entry in available_entries if entry["role"] == "candidate_lambda_sweep"]
    reference_entries = [entry for entry in available_entries if entry["role"] != "candidate_lambda_sweep"]
    plan = {
        "summary_name": "geom_full_database_hybrid_lambda_cov_amr_selection_plan",
        "result_kind": "geom_drugs_full_database_hybrid_lambda_cov_amr_selection_plan_not_benchmark",
        "benchmark_result": False,
        "execution_mode": "local_reporting_only_hybrid_lambda_cov_amr_selection_planning",
        "hardware": "reporting_only",
        "gpu_used_by_plan": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_plan": False,
        "sampling_executed_by_plan": False,
        "dataset_scope": FULL_DATABASE_SCOPE,
        "source_subset_plan_path": str(source_plan_path),
        "raw_root": str(args.raw_root),
        "split_type": "scaffold",
        "seed": 0,
        "partitions": ["val", "test"],
        "rotatable_bins": source_plan["rotatable_bins"],
        "subset_design": {
            "selected_molecule_slots": len(selected),
            "unique_molecule_ids": len({row["molecule_id"] for row in selected}),
            "scaffold_val_slots": sum(1 for row in selected if row["partition"] == "val"),
            "scaffold_test_slots": sum(1 for row in selected if row["partition"] == "test"),
            "generated_conformers_per_molecule": int(args.generated_conformers_per_mol),
            "delta_angstrom": float(args.delta),
            "max_reference_conformers_per_molecule": int(
                source_plan["subset_design"]["max_reference_conformers_per_molecule"]
            ),
            "max_pickle_bytes": int(source_plan["subset_design"]["max_pickle_bytes"]),
            "source": "scaffold-only rows from existing final optimized bounded COV/AMR subset plan",
        },
        "candidate_lambdas": [
            {"lambda_code": code, "lambda_value": value} for code, value in LAMBDA_CODES
        ],
        "evaluation_entries": entries,
        "availability": {
            "expected_candidate_lambda_count": len(LAMBDA_CODES),
            "available_candidate_lambda_count": len(candidate_entries),
            "available_reference_count": len(reference_entries),
            "missing_entries": [entry for entry in entries if not entry["available"]],
            "random_seed0_lambda_availability": random_availability,
            "evaluation_ready": len(candidate_entries) == len(LAMBDA_CODES),
        },
        "estimated_evaluation_size": {
            "candidate_lambda_records": len(selected) * len(candidate_entries),
            "candidate_lambda_generated_conformers": len(selected)
            * len(candidate_entries)
            * int(args.generated_conformers_per_mol),
            "all_entry_records": len(selected) * len(available_entries),
            "all_entry_generated_conformers": len(selected)
            * len(available_entries)
            * int(args.generated_conformers_per_mol),
        },
        "selected_molecules": selected,
        "decision_rules": {
            "precision_optimal": "lowest AMR-P mean, sampled RMSD overall as tie-breaker",
            "recall_optimal": "highest COV-R mean, lowest AMR-R mean as tie-breaker",
            "balanced": "rank aggregation over AMR-P, AMR-R, 1-COV-R, and 1-COV-P",
            "pareto": "not dominated across COV-R/COV-P higher-is-better and AMR-R/AMR-P lower-is-better",
        },
        "commands": {
            "plan_command": (
                "python scripts/evaluate_geom_full_database_hybrid_lambda_cov_amr_selection.py "
                "--write-plan-only"
            ),
            "evaluation_command": (
                "python scripts/evaluate_geom_full_database_hybrid_lambda_cov_amr_selection.py "
                "--plan-path experiments/results/geom_full_database_hybrid_lambda_cov_amr_selection_plan.json "
                "--summary-path experiments/results/geom_full_database_hybrid_lambda_cov_amr_selection_summary.json "
                "--report-path reports/tables/geom_full_database_hybrid_lambda_cov_amr_selection_summary.md "
                "--per-molecule-jsonl-path experiments/results/geom_full_database_hybrid_lambda_cov_amr_selection_per_molecule.jsonl "
                "--device cpu"
            ),
            "standalone_sbatch_command": (
                "sbatch --account=torch_pr_872_general "
                "scripts/slurm_geom_full_database_hybrid_lambda_cov_amr_selection_cpu.sbatch"
            ),
        },
        "output_targets": {
            "summary": str(DEFAULT_SUMMARY_PATH),
            "report": str(DEFAULT_REPORT_PATH),
            "per_molecule_jsonl": str(DEFAULT_PER_MOLECULE_PATH),
        },
        "notes": [
            "The primary lambda decision uses the original scaffold seed0 SGD lambda-sweep checkpoints for comparable training protocol.",
            "The final optimized AdamW lambda=5.0 checkpoint is included as a reference because it is the current selected production setting.",
            "Harmonic and stiffness references use final optimized AdamW seed0 checkpoints unless noted in the entry metadata.",
            "This is a bounded lambda-selection evaluation, not a full benchmark or physical validity run.",
        ],
    }
    return plan


def evaluate(args: argparse.Namespace) -> dict[str, Path]:
    start = time.time()
    plan_path = Path(args.plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _validate_plan(plan)
    if not plan["availability"]["evaluation_ready"]:
        raise FileNotFoundError("Lambda COV/AMR selection plan is missing one or more candidate lambda checkpoints.")

    torch = _load_torch()
    device = _resolve_device(torch, str(args.device))
    if device == "cpu":
        torch.set_num_threads(1)

    summary_path = Path(args.summary_path)
    report_path = Path(args.report_path)
    per_molecule_path = Path(args.per_molecule_jsonl_path)
    for path in (summary_path, report_path, per_molecule_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    per_molecule_path.write_text("", encoding="utf-8")

    entries = [entry for entry in plan["evaluation_entries"] if entry["available"]]
    states = _load_states(torch=torch, device=device, entries=entries)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    selected = plan["selected_molecules"]
    raw_root = Path(plan["raw_root"])
    with torch.no_grad():
        for index, planned in enumerate(selected):
            try:
                molecule = _load_reference_molecule(
                    molecule_id=planned["molecule_id"],
                    raw_root=raw_root,
                    max_reference_conformers=int(plan["subset_design"]["max_reference_conformers_per_molecule"]),
                    max_pickle_bytes=int(plan["subset_design"]["max_pickle_bytes"]),
                )
            except MoleculeSkip as exc:
                skipped.append(
                    {
                        "status": "skipped",
                        "selection_index": planned["selection_index"],
                        "molecule_id": planned["molecule_id"],
                        "reason": exc.reason,
                        "detail": exc.detail,
                    }
                )
                continue
            except Exception as exc:
                skipped.append(
                    {
                        "status": "skipped",
                        "selection_index": planned["selection_index"],
                        "molecule_id": planned["molecule_id"],
                        "reason": f"reference_reload_error:{type(exc).__name__}",
                        "detail": _exception_detail(exc),
                    }
                )
                continue
            for entry_index, entry in enumerate(entries):
                try:
                    state = states[entry["entry_label"]]
                    generated = _generate_conformers(
                        torch=torch,
                        model=state["model"],
                        config=state["config"],
                        molecule=molecule,
                        entry_label=entry["entry_label"],
                        entry_index=entry_index,
                        partition=planned["partition"],
                        selection_index=int(planned["selection_index"]),
                        generated_conformers_per_mol=int(plan["subset_design"]["generated_conformers_per_molecule"]),
                        device=device,
                    )
                    pairwise = pairwise_heavy_atom_rmsd(
                        generated_coords=generated,
                        reference_coords=molecule.reference_coords,
                        heavy_atom_indices=molecule.heavy_atom_indices,
                    )
                    metrics = compute_amr_cov_from_pairwise(
                        pairwise,
                        delta=float(plan["subset_design"]["delta_angstrom"]),
                    )
                    row = {
                        "status": "evaluated",
                        "split_type": planned["split_type"],
                        "partition": planned["partition"],
                        "rotatable_bin": planned["rotatable_bin"],
                        "selection_index": planned["selection_index"],
                        "molecule_id": planned["molecule_id"],
                        "entry_label": entry["entry_label"],
                        "role": entry["role"],
                        "lambda_code": entry.get("lambda_code"),
                        "lambda_value": entry.get("lambda_value"),
                        "optimizer_name": entry["optimizer_name"],
                        "learning_rate": entry["learning_rate"],
                        "sampled_rmsd_overall": entry.get("sampled_rmsd_overall"),
                        "reference_conformer_count_used": molecule.used_reference_conformer_count,
                        "generated_conformer_count": len(generated),
                        "metrics": metrics,
                    }
                    rows.append(row)
                    _append_jsonl(per_molecule_path, row)
                except Exception as exc:
                    skipped.append(
                        {
                            "status": "failed",
                            "selection_index": planned["selection_index"],
                            "molecule_id": planned["molecule_id"],
                            "entry_label": entry["entry_label"],
                            "role": entry["role"],
                            "reason": f"entry_evaluation_error:{type(exc).__name__}",
                            "detail": _exception_detail(exc),
                        }
                    )
            if args.progress_every and ((index + 1) % int(args.progress_every) == 0 or index + 1 == len(selected)):
                print(f"lambda_cov_amr_selection progress {index + 1}/{len(selected)} scaffold molecule slots")

    summary = _build_summary(
        plan=plan,
        rows=rows,
        skipped=skipped,
        states=states,
        runtime_seconds=time.time() - start,
        plan_path=plan_path,
        per_molecule_path=per_molecule_path,
        device=device,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_summary_markdown(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path, "per_molecule_jsonl": per_molecule_path}


def _evaluation_entries() -> list[dict[str, Any]]:
    entries = []
    for code, value in LAMBDA_CODES:
        result_path = Path(
            f"experiments/results/geom_formal_full_database_scaffold_train_"
            f"hybrid_harmonic_stiffness_v1_lambda{code}_steps2000_seed0.json"
        )
        config_path = Path(f"configs/train/geom_full_database_scaffold_formal_hybrid_harmonic_stiffness_v1_lambda{code}.yaml")
        entries.append(
            _entry_from_result(
                entry_label=f"hybrid_lambda{code}_sweep_sgd",
                role="candidate_lambda_sweep",
                result_path=result_path,
                config_path=config_path,
                lambda_code=code,
                lambda_value=value,
            )
        )
    references = [
        (
            "harmonic_prior_final_adamw_lr1e3",
            "reference_harmonic_prior",
            Path("experiments/results/geom_formal_full_database_scaffold_train_harmonic_steps2000_seed0_final_adamw_lr1e3.json"),
            Path("configs/train/final_optimized/geom_full_database_scaffold_harmonic_adamw_lr1e3.yaml"),
            None,
            None,
        ),
        (
            "stiffness_structured_final_adamw_lr1e3",
            "reference_stiffness_structured",
            Path("experiments/results/geom_formal_full_database_scaffold_train_stiffness_steps2000_seed0_final_adamw_lr1e3.json"),
            Path("configs/train/final_optimized/geom_full_database_scaffold_stiffness_adamw_lr1e3.yaml"),
            None,
            None,
        ),
        (
            "hybrid_lambda500_final_adamw_lr1e3",
            "reference_current_hybrid_lambda500",
            Path(
                "experiments/results/geom_formal_full_database_scaffold_train_"
                "hybrid_harmonic_stiffness_v1_lambda500_steps2000_seed0_final_adamw_lr1e3.json"
            ),
            Path("configs/train/final_optimized/geom_full_database_scaffold_hybrid_harmonic_stiffness_v1_lambda500_adamw_lr1e3.yaml"),
            "500",
            5.0,
        ),
    ]
    for label, role, result_path, config_path, code, value in references:
        entries.append(
            _entry_from_result(
                entry_label=label,
                role=role,
                result_path=result_path,
                config_path=config_path,
                lambda_code=code,
                lambda_value=value,
            )
        )
    return entries


def _entry_from_result(
    entry_label: str,
    role: str,
    result_path: Path,
    config_path: Path,
    lambda_code: str | None,
    lambda_value: float | None,
) -> dict[str, Any]:
    entry = {
        "entry_label": entry_label,
        "role": role,
        "lambda_code": lambda_code,
        "lambda_value": lambda_value,
        "result_path": str(result_path),
        "config_path": str(config_path),
        "available": False,
        "missing": [],
    }
    if not result_path.exists():
        entry["missing"].append("result_json")
        return entry
    if not config_path.exists():
        entry["missing"].append("config")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(payload.get("best_checkpoint_path", ""))
    if not checkpoint_path.exists():
        entry["missing"].append("best_checkpoint")
    entry.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "run_name": payload.get("run_name"),
            "corruption_mode": payload.get("corruption_mode"),
            "hybrid_source_strength": payload.get("hybrid_source_strength"),
            "optimizer_name": payload.get("optimizer_name"),
            "learning_rate": payload.get("learning_rate"),
            "sampled_rmsd_overall": _sampled_rmsd_overall(payload),
        }
    )
    entry["available"] = len(entry["missing"]) == 0
    return entry


def _random_lambda_availability() -> list[dict[str, Any]]:
    rows = []
    for code, value in LAMBDA_CODES:
        matches = sorted(
            Path("experiments/results").glob(
                f"geom_formal_full_database_train_hybrid_harmonic_stiffness_v1_lambda{code}_steps2000_seed0*.json"
            )
        )
        checkpoint_ready = []
        missing_checkpoint = []
        for path in matches:
            payload = json.loads(path.read_text(encoding="utf-8"))
            checkpoint_path = Path(payload.get("best_checkpoint_path", ""))
            row = {"result_path": str(path), "best_checkpoint_path": str(checkpoint_path)}
            if checkpoint_path.exists():
                checkpoint_ready.append(row)
            else:
                missing_checkpoint.append(row)
        rows.append(
            {
                "lambda_code": code,
                "lambda_value": value,
                "available_result_count": len(matches),
                "available_checkpoint_count": len(checkpoint_ready),
                "result_paths": [str(path) for path in matches],
                "checkpoint_ready_results": checkpoint_ready,
                "missing_checkpoint_results": missing_checkpoint,
                "available": len(checkpoint_ready) > 0,
            }
        )
    return rows


def _load_states(torch: Any, device: str, entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    states = {}
    for entry in entries:
        config = load_config(entry["config_path"])
        config["experiment"]["seed"] = 0
        checkpoint_payload = torch.load(entry["checkpoint_path"], map_location=device, weights_only=False)
        model = torch.nn.Linear(4, 3)
        model.load_state_dict(checkpoint_payload["model_state_dict"])
        model.to(device)
        model.eval()
        states[entry["entry_label"]] = {
            "entry": entry,
            "config": config,
            "checkpoint_payload": checkpoint_payload,
            "model": model,
        }
    return states


def _generate_conformers(
    torch: Any,
    model: Any,
    config: dict[str, Any],
    molecule: Any,
    entry_label: str,
    entry_index: int,
    partition: str,
    selection_index: int,
    generated_conformers_per_mol: int,
    device: str,
) -> list[list[list[float]]]:
    source_seed = _generation_seed(entry_label, entry_index, partition, selection_index)
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


def _generation_seed(entry_label: str, entry_index: int, partition: str, selection_index: int) -> int:
    digest = hashlib.sha256(entry_label.encode("utf-8")).hexdigest()
    label_offset = int(digest[:8], 16) % 50_000
    partition_offset = {"val": 10_000, "test": 20_000}[partition]
    return 3_000_000 + int(entry_index) * 100_000 + label_offset + partition_offset + int(selection_index) * 101


def _build_summary(
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    runtime_seconds: float,
    plan_path: Path,
    per_molecule_path: Path,
    device: str,
) -> dict[str, Any]:
    aggregate_by_entry = _aggregate(rows, ("entry_label",))
    aggregate_by_entry_partition = _aggregate(rows, ("entry_label", "partition"))
    aggregate_by_entry_bin = _aggregate(rows, ("entry_label", "rotatable_bin"))
    decisions = _decision_analysis(plan, aggregate_by_entry)
    return {
        "summary_name": "geom_full_database_hybrid_lambda_cov_amr_selection_summary",
        "result_kind": "geom_drugs_full_database_hybrid_lambda_cov_amr_selection_summary_not_benchmark",
        "benchmark_result": False,
        "execution_mode": "scheduled_or_local_bounded_hybrid_lambda_cov_amr_selection",
        "hardware": "cpu_only" if device == "cpu" else "device_requested",
        "gpu_used_by_summary": device.startswith("cuda"),
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": True,
        "dataset_scope": FULL_DATABASE_SCOPE,
        "plan_path": str(plan_path),
        "per_molecule_jsonl_path": str(per_molecule_path),
        "split_type": plan["split_type"],
        "seed": plan["seed"],
        "subset_design": plan["subset_design"],
        "coverage": {
            "planned_molecule_slots": plan["subset_design"]["selected_molecule_slots"],
            "available_entry_count": len(states),
            "evaluated_records": len(rows),
            "skipped_or_failed_records": len(skipped),
            "expected_records": plan["subset_design"]["selected_molecule_slots"] * len(states),
        },
        "entries": plan["evaluation_entries"],
        "aggregate_by_entry": aggregate_by_entry,
        "aggregate_by_entry_partition": aggregate_by_entry_partition,
        "aggregate_by_entry_rotatable_bin": aggregate_by_entry_bin,
        "decision_analysis": decisions,
        "skipped_records": skipped[:500],
        "skipped_record_count": len(skipped),
        "runtime_seconds": runtime_seconds,
        "notes": [
            "Primary candidate comparison uses the scaffold seed0 SGD lambda sweep to avoid retraining.",
            "Current optimized lambda=5.0 AdamW is included as a reference, not as a same-protocol lambda candidate.",
            "This is bounded COV/AMR lambda selection, not benchmark-style physical validation.",
        ],
    }


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = {}
    for key, group in grouped.items():
        name = "/".join(str(part) for part in key)
        first = group[0]
        output[name] = {
            **{keys[index]: key[index] for index in range(len(keys))},
            "role": first["role"],
            "lambda_code": first.get("lambda_code"),
            "lambda_value": first.get("lambda_value"),
            "optimizer_name": first.get("optimizer_name"),
            "learning_rate": first.get("learning_rate"),
            "sampled_rmsd_overall": first.get("sampled_rmsd_overall"),
            "record_count": len(group),
            "metric_summary": {
                metric: _numeric_summary([float(row["metrics"][metric]) for row in group]) for metric in METRICS
            },
            "reference_conformers_per_molecule": _count_summary(
                [int(row["reference_conformer_count_used"]) for row in group]
            ),
            "generated_conformers_per_molecule": _count_summary(
                [int(row["generated_conformer_count"]) for row in group]
            ),
        }
    return output


def _decision_analysis(plan: dict[str, Any], aggregate_by_entry: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        row for row in aggregate_by_entry.values() if row["role"] == "candidate_lambda_sweep"
    ]
    if not candidates:
        return {"available": False, "reason": "no_candidate_rows"}
    precision = sorted(
        candidates,
        key=lambda row: (
            row["metric_summary"]["amr_p"]["mean"],
            _none_last(row.get("sampled_rmsd_overall")),
        ),
    )[0]
    recall = sorted(
        candidates,
        key=lambda row: (
            -float(row["metric_summary"]["cov_r"]["mean"]),
            row["metric_summary"]["amr_r"]["mean"],
        ),
    )[0]
    balanced_rows = _balanced_rank_rows(candidates)
    balanced = sorted(balanced_rows, key=lambda row: (row["balanced_rank_sum"], row["lambda_value"]))[0]
    comparisons = _reference_comparisons(aggregate_by_entry)
    return {
        "available": True,
        "precision_optimal_lambda": _decision_row(precision),
        "recall_optimal_lambda": _decision_row(recall),
        "balanced_lambda": balanced,
        "balanced_rank_table": balanced_rows,
        "pareto_frontier": [_decision_row(row) for row in _pareto_frontier(candidates)],
        "comparisons": comparisons,
        "candidate_protocol_note": plan["notes"][0],
    }


def _balanced_rank_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank_maps = {
        "amr_p": _rank_map(candidates, "amr_p", higher_is_better=False),
        "amr_r": _rank_map(candidates, "amr_r", higher_is_better=False),
        "one_minus_cov_r": _rank_map(candidates, "cov_r", higher_is_better=True),
        "one_minus_cov_p": _rank_map(candidates, "cov_p", higher_is_better=True),
    }
    rows = []
    for row in candidates:
        label = row["entry_label"]
        rank_sum = sum(rank_map[label] for rank_map in rank_maps.values())
        rows.append(
            {
                **_decision_row(row),
                "ranks": {name: rank_map[label] for name, rank_map in rank_maps.items()},
                "balanced_rank_sum": rank_sum,
            }
        )
    return sorted(rows, key=lambda row: (row["balanced_rank_sum"], row["lambda_value"]))


def _rank_map(candidates: list[dict[str, Any]], metric: str, higher_is_better: bool) -> dict[str, int]:
    values = [
        (row["entry_label"], float(row["metric_summary"][metric]["mean"]))
        for row in candidates
    ]
    ordered = sorted(values, key=lambda item: item[1], reverse=higher_is_better)
    return {label: index + 1 for index, (label, _value) in enumerate(ordered)}


def _pareto_frontier(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other["entry_label"] == candidate["entry_label"]:
                continue
            if _dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda row: row["lambda_value"])


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    better_or_equal = []
    strictly_better = []
    for metric in METRICS:
        left_value = float(left["metric_summary"][metric]["mean"])
        right_value = float(right["metric_summary"][metric]["mean"])
        if metric in HIGHER_IS_BETTER:
            better_or_equal.append(left_value >= right_value - TIE_TOLERANCE)
            strictly_better.append(left_value > right_value + TIE_TOLERANCE)
        else:
            better_or_equal.append(left_value <= right_value + TIE_TOLERANCE)
            strictly_better.append(left_value < right_value - TIE_TOLERANCE)
    return all(better_or_equal) and any(strictly_better)


def _reference_comparisons(aggregate_by_entry: dict[str, Any]) -> dict[str, Any]:
    references = {
        "harmonic_prior_final_adamw_lr1e3": aggregate_by_entry.get("harmonic_prior_final_adamw_lr1e3"),
        "stiffness_structured_final_adamw_lr1e3": aggregate_by_entry.get("stiffness_structured_final_adamw_lr1e3"),
        "hybrid_lambda500_final_adamw_lr1e3": aggregate_by_entry.get("hybrid_lambda500_final_adamw_lr1e3"),
        "hybrid_lambda500_sweep_sgd": aggregate_by_entry.get("hybrid_lambda500_sweep_sgd"),
    }
    candidates = [row for row in aggregate_by_entry.values() if row["role"] == "candidate_lambda_sweep"]
    output = {}
    for reference_label, reference in references.items():
        if reference is None:
            output[reference_label] = {"available": False}
            continue
        rows = []
        for candidate in candidates:
            rows.append(
                {
                    "entry_label": candidate["entry_label"],
                    "lambda_value": candidate["lambda_value"],
                    "metric_outcomes": {
                        metric: _metric_outcome(
                            metric,
                            candidate["metric_summary"][metric]["mean"],
                            reference["metric_summary"][metric]["mean"],
                        )
                        for metric in METRICS
                    },
                }
            )
        output[reference_label] = {"available": True, "candidate_outcomes": rows}
    return output


def _decision_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["metric_summary"]
    return {
        "entry_label": row["entry_label"],
        "lambda_code": row.get("lambda_code"),
        "lambda_value": row.get("lambda_value"),
        "optimizer_name": row.get("optimizer_name"),
        "learning_rate": row.get("learning_rate"),
        "sampled_rmsd_overall": row.get("sampled_rmsd_overall"),
        "cov_r_mean": metrics["cov_r"]["mean"],
        "cov_p_mean": metrics["cov_p"]["mean"],
        "amr_r_mean": metrics["amr_r"]["mean"],
        "amr_p_mean": metrics["amr_p"]["mean"],
    }


def _validate_source_plan(plan: dict[str, Any]) -> None:
    if plan.get("result_kind") != "geom_drugs_full_database_final_optimized_amr_cov_subset_plan_not_benchmark":
        raise ValueError("Unexpected source subset plan result_kind")
    if plan.get("benchmark_result") is not False:
        raise ValueError("Source subset plan must be benchmark_result=false")
    if plan.get("dataset_scope") != FULL_DATABASE_SCOPE:
        raise ValueError("Source subset plan has wrong dataset scope")


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("result_kind") != "geom_drugs_full_database_hybrid_lambda_cov_amr_selection_plan_not_benchmark":
        raise ValueError("Unexpected lambda COV/AMR selection plan result_kind")
    if plan.get("benchmark_result") is not False:
        raise ValueError("Lambda COV/AMR selection plan must be benchmark_result=false")
    if plan.get("dataset_scope") != FULL_DATABASE_SCOPE:
        raise ValueError("Lambda COV/AMR selection plan has wrong dataset scope")


def _sampled_rmsd_overall(payload: dict[str, Any]) -> float | None:
    sampled = (payload.get("metrics") or {}).get("sampled_molecules")
    if not isinstance(sampled, list) or not sampled:
        return None
    values = [float(row["sampled_rmsd_to_target"]) for row in sampled if "sampled_rmsd_to_target" in row]
    if not values:
        return None
    return float(mean(values))


def _metric_outcome(metric: str, candidate: float | None, reference: float | None) -> str:
    if candidate is None or reference is None:
        return "missing"
    if metric in HIGHER_IS_BETTER:
        if candidate > reference + TIE_TOLERANCE:
            return "better"
        if candidate < reference - TIE_TOLERANCE:
            return "worse"
        return "tied"
    if candidate < reference - TIE_TOLERANCE:
        return "better"
    if candidate > reference + TIE_TOLERANCE:
        return "worse"
    return "tied"


def _numeric_summary(values: list[float]) -> dict[str, float | None | int]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": min(values),
        "max": max(values),
    }


def _count_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": min(values),
        "max": max(values),
    }


def _none_last(value: float | None) -> tuple[int, float]:
    if value is None:
        return (1, float("inf"))
    return (0, float(value))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _load_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for checkpoint COV/AMR evaluation") from exc
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


def _plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Lambda COV/AMR Selection Plan",
        "",
        "Planning-only bounded scaffold seed0 lambda-selection evaluation. This is not a benchmark.",
        "",
        "## Scope",
        "",
        f"- Split: `{plan['split_type']}`",
        f"- Seed: `{plan['seed']}`",
        f"- Partitions: `{plan['partitions']}`",
        f"- Molecule slots: `{plan['subset_design']['selected_molecule_slots']}`",
        f"- K generated conformers per molecule: `{plan['subset_design']['generated_conformers_per_molecule']}`",
        f"- Delta: `{plan['subset_design']['delta_angstrom']}`",
        f"- Candidate lambdas available: `{plan['availability']['available_candidate_lambda_count']}` / `{plan['availability']['expected_candidate_lambda_count']}`",
        f"- Evaluation ready: `{plan['availability']['evaluation_ready']}`",
        f"- Candidate generated conformers: `{plan['estimated_evaluation_size']['candidate_lambda_generated_conformers']}`",
        f"- All-entry generated conformers: `{plan['estimated_evaluation_size']['all_entry_generated_conformers']}`",
        "",
        "## Entries",
        "",
        "| Entry | Role | Lambda | Optimizer | LR | Available | Result |",
        "|---|---|---:|---|---:|---|---|",
    ]
    for entry in plan["evaluation_entries"]:
        lines.append(
            f"| {entry['entry_label']} | {entry['role']} | {entry.get('lambda_value')} | "
            f"{entry.get('optimizer_name')} | {entry.get('learning_rate')} | {entry['available']} | `{entry['result_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Random Availability",
            "",
            "| Lambda | Available result count | Checkpoint-ready count |",
            "|---:|---:|---:|",
        ]
    )
    for row in plan["availability"]["random_seed0_lambda_availability"]:
        lines.append(
            f"| {row['lambda_value']} | {row['available_result_count']} | {row['available_checkpoint_count']} |"
        )
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "Evaluation command, for an approved scheduler wrapper or a controlled run:",
            "",
            "```bash",
            plan["commands"]["evaluation_command"],
            "```",
            "",
            "Standalone scheduler command, do not submit without approval:",
            "",
            "```bash",
            plan["commands"]["standalone_sbatch_command"],
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


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Lambda COV/AMR Selection Summary",
        "",
        "Bounded scaffold seed0 COV/AMR lambda-selection summary. This is not a benchmark.",
        "",
        "## Coverage",
        "",
        f"- Evaluated records: `{summary['coverage']['evaluated_records']}` / `{summary['coverage']['expected_records']}`",
        f"- Skipped/failed records: `{summary['coverage']['skipped_or_failed_records']}`",
        f"- Runtime seconds: `{summary['runtime_seconds']:.3f}`",
        "",
        "## Candidate Metrics",
        "",
        "| Lambda | Entry | Records | COV-R mean | COV-R median | COV-P mean | COV-P median | AMR-R mean | AMR-R median | AMR-P mean | AMR-P median | RMSD overall |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        [row for row in summary["aggregate_by_entry"].values() if row["role"] == "candidate_lambda_sweep"],
        key=lambda row: float(row["lambda_value"]),
    ):
        metrics = row["metric_summary"]
        lines.append(
            f"| {row['lambda_value']} | {row['entry_label']} | {row['record_count']} | "
            f"{_fmt(metrics['cov_r']['mean'])} | {_fmt(metrics['cov_r']['median'])} | "
            f"{_fmt(metrics['cov_p']['mean'])} | {_fmt(metrics['cov_p']['median'])} | "
            f"{_fmt(metrics['amr_r']['mean'])} | {_fmt(metrics['amr_r']['median'])} | "
            f"{_fmt(metrics['amr_p']['mean'])} | {_fmt(metrics['amr_p']['median'])} | "
            f"{_fmt(row.get('sampled_rmsd_overall'))} |"
        )
    decision = summary["decision_analysis"]
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            f"- Precision-optimal lambda: `{decision.get('precision_optimal_lambda', {}).get('lambda_value')}`",
            f"- Recall-optimal lambda: `{decision.get('recall_optimal_lambda', {}).get('lambda_value')}`",
            f"- Balanced lambda: `{decision.get('balanced_lambda', {}).get('lambda_value')}`",
            f"- Pareto frontier: `{[row['lambda_value'] for row in decision.get('pareto_frontier', [])]}`",
            "",
            "## Safety",
            "",
            f"- `benchmark_result`: `{summary['benchmark_result']}`",
            f"- `training_executed_by_summary`: `{summary['training_executed_by_summary']}`",
            f"- `raw_data_modified`: `{summary['raw_data_modified']}`",
            f"- `environment_modified`: `{summary['environment_modified']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.8g}"


if __name__ == "__main__":
    main()
