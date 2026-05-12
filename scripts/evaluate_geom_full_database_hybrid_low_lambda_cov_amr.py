#!/usr/bin/env python
"""Evaluate bounded COV/AMR for Stage 1 low-lambda AdamW hybrid checkpoints."""

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


DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_hybrid_low_lambda_adamw_plan.json")
DEFAULT_SUMMARY_PATH = Path("experiments/results/geom_full_database_hybrid_low_lambda_cov_amr_summary.json")
DEFAULT_REPORT_PATH = Path("reports/tables/geom_full_database_hybrid_low_lambda_cov_amr_summary.md")
DEFAULT_PER_MOLECULE_PATH = Path("experiments/results/geom_full_database_hybrid_low_lambda_cov_amr_per_molecule.jsonl")
METRICS = ("cov_r", "cov_p", "amr_r", "amr_p")
HIGHER_IS_BETTER = {"cov_r", "cov_p"}
TIE_TOLERANCE = 1.0e-12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-path", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--per-molecule-jsonl-path", default=str(DEFAULT_PER_MOLECULE_PATH))
    parser.add_argument("--source-subset-plan-path", default="experiments/results/geom_full_database_final_optimized_amr_cov_subset_plan.json")
    parser.add_argument("--raw-root", default="data/raw/geom_drugs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--delta", type=float, default=0.75)
    parser.add_argument("--generated-conformers-per-mol", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--require-results", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    outputs = evaluate(args)
    for key, path in outputs.items():
        print(f"{key}: {path}")
    print("benchmark_result: false")
    print("note: Bounded low-lambda COV/AMR evaluation reloads existing checkpoints only; no training, scheduler submission, raw-data mutation, or environment mutation.")


def evaluate(args: argparse.Namespace) -> dict[str, Path]:
    start = time.time()
    plan_path = Path(args.plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _validate_plan(plan)
    subset_plan = json.loads(Path(args.source_subset_plan_path).read_text(encoding="utf-8"))
    selected = subset_plan["selected_molecules"]
    entries, missing = _entries(plan)
    if args.require_results and missing:
        raise FileNotFoundError("Missing low-lambda COV/AMR inputs:\n" + json.dumps(missing, indent=2, sort_keys=True))

    torch = _load_torch()
    device = _resolve_device(torch, str(args.device))
    if device == "cpu":
        torch.set_num_threads(1)
    states = _load_states(torch, device, entries)

    summary_path = Path(args.summary_path)
    report_path = Path(args.report_path)
    per_molecule_path = Path(args.per_molecule_jsonl_path)
    for path in (summary_path, report_path, per_molecule_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    per_molecule_path.write_text("", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    raw_root = Path(args.raw_root)
    with torch.no_grad():
        for index, planned in enumerate(selected):
            try:
                molecule = _load_reference_molecule(
                    molecule_id=planned["molecule_id"],
                    raw_root=raw_root,
                    max_reference_conformers=int(subset_plan["subset_design"]["max_reference_conformers_per_molecule"]),
                    max_pickle_bytes=int(subset_plan["subset_design"]["max_pickle_bytes"]),
                )
            except MoleculeSkip as exc:
                skipped.append({"molecule_id": planned["molecule_id"], "reason": exc.reason, "detail": exc.detail})
                continue
            except Exception as exc:
                skipped.append(
                    {
                        "molecule_id": planned["molecule_id"],
                        "reason": f"reference_reload_error:{type(exc).__name__}",
                        "detail": _exception_detail(exc),
                    }
                )
                continue
            split_entries = [entry for entry in entries if entry["split_type"] == planned["split_type"]]
            for entry_index, entry in enumerate(split_entries):
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
                        generated_conformers_per_mol=int(args.generated_conformers_per_mol),
                        device=device,
                    )
                    pairwise = pairwise_heavy_atom_rmsd(
                        generated_coords=generated,
                        reference_coords=molecule.reference_coords,
                        heavy_atom_indices=molecule.heavy_atom_indices,
                    )
                    metrics = compute_amr_cov_from_pairwise(pairwise, delta=float(args.delta))
                    row = {
                        "status": "evaluated",
                        "split_type": planned["split_type"],
                        "partition": planned["partition"],
                        "rotatable_bin": planned["rotatable_bin"],
                        "selection_index": planned["selection_index"],
                        "molecule_id": planned["molecule_id"],
                        "entry_label": entry["entry_label"],
                        "role": entry["role"],
                        "lambda_value": entry.get("lambda_value"),
                        "lambda_label": entry.get("lambda_label"),
                        "sampled_rmsd_overall": entry.get("sampled_rmsd_overall"),
                        "best_validation_loss": entry.get("best_validation_loss"),
                        "reloaded_validation_loss": entry.get("reloaded_validation_loss"),
                        "reference_conformer_count_used": molecule.used_reference_conformer_count,
                        "generated_conformer_count": len(generated),
                        "metrics": metrics,
                    }
                    rows.append(row)
                    _append_jsonl(per_molecule_path, row)
                except Exception as exc:
                    skipped.append(
                        {
                            "molecule_id": planned["molecule_id"],
                            "entry_label": entry["entry_label"],
                            "reason": f"entry_evaluation_error:{type(exc).__name__}",
                            "detail": _exception_detail(exc),
                        }
                    )
            if args.progress_every and ((index + 1) % int(args.progress_every) == 0 or index + 1 == len(selected)):
                print(f"low_lambda_cov_amr progress {index + 1}/{len(selected)} molecule slots")

    aggregate_by_split_entry = _aggregate(rows, ("split_type", "entry_label"))
    aggregate_by_split_entry_partition = _aggregate(rows, ("split_type", "entry_label", "partition"))
    aggregate_by_split_entry_bin = _aggregate(rows, ("split_type", "entry_label", "rotatable_bin"))
    summary = {
        "summary_name": "geom_full_database_hybrid_low_lambda_cov_amr_summary",
        "result_kind": "geom_drugs_full_database_hybrid_low_lambda_cov_amr_summary_not_benchmark",
        "benchmark_result": False,
        "execution_mode": "scheduled_or_local_bounded_low_lambda_cov_amr",
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
        "source_subset_plan_path": str(args.source_subset_plan_path),
        "per_molecule_jsonl_path": str(per_molecule_path),
        "delta_angstrom": float(args.delta),
        "generated_conformers_per_molecule": int(args.generated_conformers_per_mol),
        "coverage": {
            "selected_molecule_slots": len(selected),
            "entry_count": len(entries),
            "evaluated_records": len(rows),
            "expected_records": sum(1 for item in selected for entry in entries if entry["split_type"] == item["split_type"]),
            "skipped_or_failed_records": len(skipped),
        },
        "entries": entries,
        "missing_inputs": missing,
        "aggregate_by_split_entry": aggregate_by_split_entry,
        "aggregate_by_split_entry_partition": aggregate_by_split_entry_partition,
        "aggregate_by_split_entry_rotatable_bin": aggregate_by_split_entry_bin,
        "decision_analysis": _decision_analysis(aggregate_by_split_entry),
        "skipped_records": skipped[:500],
        "skipped_record_count": len(skipped),
        "runtime_seconds": time.time() - start,
        "notes": [
            "Candidate lambdas are Stage 1 AdamW lr=1e-3 low-lambda runs only.",
            "lambda=0 is optimized harmonic_prior; lambda=5 is the existing optimized high-lambda reference.",
            "Old SGD lambda-sweep results are not used.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path, "per_molecule_jsonl": per_molecule_path}


def _entries(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries = []
    missing = []
    for job in plan["training_jobs"]:
        entry = _entry_from_paths(
            split_type=job["split_type"],
            entry_label=f"{job['split_type']}_{job['lambda_label']}_adamw",
            role="candidate_low_lambda_adamw",
            lambda_label=job["lambda_label"],
            lambda_value=float(job["lambda_value"]),
            result_path=Path(job["expected_result_path"]),
            config_path=Path(job["config_path"]),
        )
        if entry["available"]:
            entries.append(entry)
        else:
            missing.append(entry)
    for split_type, ref in _reference_inputs(plan).items():
        for role, item in ref.items():
            entry = _entry_from_paths(
                split_type=split_type,
                entry_label=f"{split_type}_{role}",
                role=role,
                lambda_label=item.get("lambda_label"),
                lambda_value=item.get("lambda_value"),
                result_path=Path(item["result_path"]),
                config_path=Path(item["config_path"]),
            )
            if entry["available"]:
                entries.append(entry)
            else:
                missing.append(entry)
    return entries, missing


def _reference_inputs(plan: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for row in plan["references"]["lambda0_harmonic_prior"]:
        output.setdefault(row["split_type"], {})["reference_lambda0_harmonic"] = {
            **row,
            "lambda_label": "lambda000",
            "lambda_value": 0.0,
        }
    for row in plan["references"]["lambda5_current_hybrid_reference"]:
        output.setdefault(row["split_type"], {})["reference_lambda5_hybrid"] = {
            **row,
            "lambda_label": "lambda500",
            "lambda_value": 5.0,
        }
    return output


def _entry_from_paths(
    split_type: str,
    entry_label: str,
    role: str,
    lambda_label: str | None,
    lambda_value: float | None,
    result_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "split_type": split_type,
        "entry_label": entry_label,
        "role": role,
        "lambda_label": lambda_label,
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
    metrics = payload.get("metrics") or {}
    entry.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "optimizer_name": payload.get("optimizer_name"),
            "learning_rate": payload.get("learning_rate"),
            "best_validation_loss": metrics.get("best_validation_loss"),
            "reloaded_validation_loss": metrics.get("reloaded_validation_loss"),
            "sampled_rmsd_overall": _sampled_rmsd_overall(payload),
            "available": len(entry["missing"]) == 0,
        }
    )
    return entry


def _load_states(torch: Any, device: str, entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    states = {}
    for entry in entries:
        config = load_config(entry["config_path"])
        config["experiment"]["seed"] = 0
        payload = torch.load(entry["checkpoint_path"], map_location=device, weights_only=False)
        model = torch.nn.Linear(4, 3)
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)
        model.eval()
        states[entry["entry_label"]] = {"config": config, "model": model, "checkpoint_payload": payload}
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
    return 4_000_000 + int(entry_index) * 100_000 + label_offset + partition_offset + int(selection_index) * 101


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = {}
    for key, group in grouped.items():
        first = group[0]
        name = "/".join(str(part) for part in key)
        output[name] = {
            **{keys[index]: key[index] for index in range(len(keys))},
            "role": first["role"],
            "lambda_value": first.get("lambda_value"),
            "lambda_label": first.get("lambda_label"),
            "best_validation_loss": first.get("best_validation_loss"),
            "reloaded_validation_loss": first.get("reloaded_validation_loss"),
            "sampled_rmsd_overall": first.get("sampled_rmsd_overall"),
            "record_count": len(group),
            "metric_summary": {
                metric: _numeric_summary([float(row["metrics"][metric]) for row in group]) for metric in METRICS
            },
        }
    return output


def _decision_analysis(aggregate_by_split_entry: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for split_type in sorted({row["split_type"] for row in aggregate_by_split_entry.values()}):
        candidates = [
            row for row in aggregate_by_split_entry.values()
            if row["split_type"] == split_type and row["role"] == "candidate_low_lambda_adamw"
        ]
        if not candidates:
            output[split_type] = {"available": False}
            continue
        precision = sorted(candidates, key=lambda row: (row["metric_summary"]["amr_p"]["mean"], _none_last(row["sampled_rmsd_overall"])))[0]
        recall = sorted(candidates, key=lambda row: (-float(row["metric_summary"]["cov_r"]["mean"]), row["metric_summary"]["amr_r"]["mean"]))[0]
        balanced_rows = _balanced_rank_rows(candidates)
        output[split_type] = {
            "available": True,
            "precision_optimal_lambda": _decision_row(precision),
            "recall_optimal_lambda": _decision_row(recall),
            "balanced_lambda": balanced_rows[0],
            "balanced_rank_table": balanced_rows,
            "pareto_frontier": [_decision_row(row) for row in _pareto_frontier(candidates)],
            "trend": _trend(candidates),
            "lambda025_balanced_optimal": balanced_rows[0]["lambda_value"] == 0.25,
            "smaller_than_lambda025_improves_balanced": balanced_rows[0]["lambda_value"] < 0.25,
            "lambda5_reference_role": "RMSD/high-lambda reference only, not a Stage 1 low-lambda candidate",
        }
    return output


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
        rows.append(
            {
                **_decision_row(row),
                "ranks": {name: rank_map[label] for name, rank_map in rank_maps.items()},
                "balanced_rank_sum": sum(rank_map[label] for rank_map in rank_maps.values()),
            }
        )
    return sorted(rows, key=lambda row: (row["balanced_rank_sum"], row["lambda_value"]))


def _rank_map(candidates: list[dict[str, Any]], metric: str, higher_is_better: bool) -> dict[str, int]:
    ordered = sorted(
        [(row["entry_label"], float(row["metric_summary"][metric]["mean"])) for row in candidates],
        key=lambda item: item[1],
        reverse=higher_is_better,
    )
    return {label: index + 1 for index, (label, _value) in enumerate(ordered)}


def _pareto_frontier(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [candidate for candidate in candidates if not any(_dominates(other, candidate) for other in candidates if other["entry_label"] != candidate["entry_label"])],
        key=lambda row: row["lambda_value"],
    )


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


def _trend(candidates: list[dict[str, Any]]) -> str:
    ordered = sorted(candidates, key=lambda row: row["lambda_value"])
    balanced = _balanced_rank_rows(ordered)
    best_lambda = balanced[0]["lambda_value"]
    if best_lambda in {ordered[0]["lambda_value"], ordered[-1]["lambda_value"]}:
        return "edge_optimum_or_monotonic_candidate"
    near = [row for row in balanced if row["balanced_rank_sum"] <= balanced[0]["balanced_rank_sum"] + 2]
    if len(near) >= 3:
        return "saturated_or_noisy"
    return "peaked"


def _decision_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["metric_summary"]
    return {
        "entry_label": row["entry_label"],
        "lambda_value": row["lambda_value"],
        "sampled_rmsd_overall": row.get("sampled_rmsd_overall"),
        "best_validation_loss": row.get("best_validation_loss"),
        "reloaded_validation_loss": row.get("reloaded_validation_loss"),
        "cov_r_mean": metrics["cov_r"]["mean"],
        "cov_p_mean": metrics["cov_p"]["mean"],
        "amr_r_mean": metrics["amr_r"]["mean"],
        "amr_p_mean": metrics["amr_p"]["mean"],
    }


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("result_kind") != "geom_drugs_full_database_hybrid_low_lambda_adamw_plan_not_benchmark":
        raise ValueError("Unexpected low-lambda AdamW plan result_kind")
    if plan.get("benchmark_result") is not False:
        raise ValueError("Low-lambda AdamW plan must be benchmark_result=false")
    if plan.get("dataset_scope") != FULL_DATABASE_SCOPE:
        raise ValueError("Low-lambda AdamW plan has wrong dataset scope")


def _sampled_rmsd_overall(payload: dict[str, Any]) -> float | None:
    sampled = (payload.get("metrics") or {}).get("sampled_molecules")
    if not isinstance(sampled, list) or not sampled:
        return None
    values = [float(row["sampled_rmsd_to_target"]) for row in sampled if "sampled_rmsd_to_target" in row]
    return float(mean(values)) if values else None


def _numeric_summary(values: list[float]) -> dict[str, float | None | int]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"count": len(values), "mean": float(mean(values)), "median": float(median(values)), "min": min(values), "max": max(values)}


def _none_last(value: float | None) -> tuple[int, float]:
    return (1, float("inf")) if value is None else (0, float(value))


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


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Low-Lambda COV/AMR Summary",
        "",
        "Bounded random/scaffold seed0 low-lambda AdamW COV/AMR evaluation. This is not a benchmark.",
        "",
        f"- Evaluated records: `{summary['coverage']['evaluated_records']}` / `{summary['coverage']['expected_records']}`",
        f"- Skipped/failed records: `{summary['coverage']['skipped_or_failed_records']}`",
        "",
        "| Split | Lambda | Records | COV-R mean | COV-P mean | AMR-R mean | AMR-P mean | RMSD overall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [
        row for row in summary["aggregate_by_split_entry"].values()
        if row["role"] == "candidate_low_lambda_adamw"
    ]
    for row in sorted(rows, key=lambda item: (item["split_type"], item["lambda_value"])):
        metrics = row["metric_summary"]
        lines.append(
            f"| {row['split_type']} | {row['lambda_value']} | {row['record_count']} | "
            f"{_fmt(metrics['cov_r']['mean'])} | {_fmt(metrics['cov_p']['mean'])} | "
            f"{_fmt(metrics['amr_r']['mean'])} | {_fmt(metrics['amr_p']['mean'])} | "
            f"{_fmt(row.get('sampled_rmsd_overall'))} |"
        )
    lines.extend(["", "## Decisions", ""])
    for split_type, decision in summary["decision_analysis"].items():
        if not decision.get("available"):
            continue
        lines.append(
            f"- {split_type}: precision `{decision['precision_optimal_lambda']['lambda_value']}`, "
            f"recall `{decision['recall_optimal_lambda']['lambda_value']}`, "
            f"balanced `{decision['balanced_lambda']['lambda_value']}`, trend `{decision['trend']}`"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.8g}"


if __name__ == "__main__":
    main()
