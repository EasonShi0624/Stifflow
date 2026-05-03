#!/usr/bin/env python
"""Evaluate bounded final-optimized GEOM AMR/COV subset from existing checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
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
    _load_reference_molecule,
    compute_amr_cov_from_pairwise,
    pairwise_heavy_atom_rmsd,
)
from src.eval.final_optimized_all_methods import (  # noqa: E402
    METHODS,
    METHOD_LABELS,
    SEEDS,
    _candidate_paths,
)
from src.train.formal_10000_training import FORMAL_10000_SAMPLE_STEP_SIZE  # noqa: E402
from src.utils.config import load_config  # noqa: E402


DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_final_optimized_amr_cov_subset_plan.json")
DEFAULT_SUMMARY_PATH = Path("experiments/results/geom_full_database_final_optimized_amr_cov_subset_summary.json")
DEFAULT_REPORT_PATH = Path("reports/tables/geom_full_database_final_optimized_amr_cov_subset_summary.md")
DEFAULT_PER_MOLECULE_PATH = Path(
    "experiments/results/geom_full_database_final_optimized_amr_cov_subset_per_molecule.jsonl"
)
METRICS = ("cov_r", "cov_p", "amr_r", "amr_p")
METRIC_DISPLAY = {"cov_r": "COV-R", "cov_p": "COV-P", "amr_r": "AMR-R", "amr_p": "AMR-P"}
HIGHER_IS_BETTER = {"cov_r", "cov_p"}
TIE_TOLERANCE = 1.0e-12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-path", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--per-molecule-jsonl-path", default=str(DEFAULT_PER_MOLECULE_PATH))
    parser.add_argument("--training-summary-path", default="experiments/results/geom_full_database_final_optimized_all_methods_summary.json")
    parser.add_argument("--raw-root", default="data/raw/geom_drugs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--delta", type=float, default=0.75)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    outputs = evaluate(args)
    for key, path in outputs.items():
        print(f"{key}: {path}")
    print("benchmark_result: false")
    print("note: Bounded final-optimized AMR/COV evaluation from existing checkpoints; no training, scheduler submission, raw-data mutation, or environment mutation.")


def evaluate(args: argparse.Namespace) -> dict[str, Path]:
    start = time.time()
    plan = json.loads(Path(args.plan_path).read_text(encoding="utf-8"))
    _validate_plan(plan)
    if not plan["training_result_readiness"]["evaluation_ready"]:
        raise FileNotFoundError(
            "Final optimized AMR/COV subset evaluation requires all optimized training results. Missing: "
            + json.dumps(plan["training_result_readiness"]["missing_results"], sort_keys=True)
        )
    torch = _load_torch()
    device = _resolve_device(torch, str(args.device))
    if device == "cpu":
        torch.set_num_threads(1)

    summary_path = Path(args.summary_path)
    report_path = Path(args.report_path)
    per_molecule_path = Path(args.per_molecule_jsonl_path)
    for path in [summary_path, report_path, per_molecule_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    per_molecule_path.write_text("", encoding="utf-8")

    states = _load_method_seed_states(torch=torch, device=device)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    selected = plan["selected_molecules"]
    with torch.no_grad():
        for index, planned in enumerate(selected):
            try:
                molecule = _load_reference_molecule(
                    molecule_id=planned["molecule_id"],
                    raw_root=Path(args.raw_root),
                    max_reference_conformers=int(plan["subset_design"]["max_reference_conformers_per_molecule"]),
                    max_pickle_bytes=int(plan["subset_design"]["max_pickle_bytes"]),
                )
            except Exception as exc:
                skipped.append(
                    {
                        "selection_index": planned["selection_index"],
                        "molecule_id": planned["molecule_id"],
                        "reason": f"reference_reload_error:{type(exc).__name__}",
                        "detail": str(exc),
                    }
                )
                continue
            for method in METHODS:
                for seed in SEEDS:
                    try:
                        state = states[(planned["split_type"], method, seed)]
                        generated = _generate_conformers(
                            torch=torch,
                            model=state["model"],
                            config=state["config"],
                            molecule=molecule,
                            split_type=planned["split_type"],
                            partition=planned["partition"],
                            method=method,
                            seed=seed,
                            selection_index=int(planned["selection_index"]),
                            generated_conformers_per_mol=int(
                                plan["subset_design"]["generated_conformers_per_molecule"]
                            ),
                            device=device,
                        )
                        pairwise = pairwise_heavy_atom_rmsd(
                            generated_coords=generated,
                            reference_coords=molecule.reference_coords,
                            heavy_atom_indices=molecule.heavy_atom_indices,
                        )
                        metrics = compute_amr_cov_from_pairwise(pairwise, delta=float(args.delta))
                        row = {
                            "split_type": planned["split_type"],
                            "partition": planned["partition"],
                            "rotatable_bin": planned["rotatable_bin"],
                            "method": method,
                            "seed": seed,
                            "selection_index": planned["selection_index"],
                            "molecule_id": planned["molecule_id"],
                            "reference_conformer_count_used": molecule.used_reference_conformer_count,
                            "generated_conformer_count": len(generated),
                            "metrics": metrics,
                        }
                        rows.append(row)
                        _append_jsonl(per_molecule_path, row)
                    except Exception as exc:
                        skipped.append(
                            {
                                "selection_index": planned["selection_index"],
                                "molecule_id": planned["molecule_id"],
                                "split_type": planned["split_type"],
                                "partition": planned["partition"],
                                "rotatable_bin": planned["rotatable_bin"],
                                "method": method,
                                "seed": seed,
                                "reason": f"method_evaluation_error:{type(exc).__name__}",
                                "detail": str(exc),
                            }
                        )
            if args.progress_every and ((index + 1) % int(args.progress_every) == 0 or index + 1 == len(selected)):
                print(f"amr_cov_subset progress {index + 1}/{len(selected)} selected molecule slots")

    summary = _build_summary(
        plan=plan,
        rows=rows,
        skipped=skipped,
        states=states,
        delta=float(args.delta),
        device=device,
        runtime_seconds=time.time() - start,
        per_molecule_path=per_molecule_path,
        plan_path=Path(args.plan_path),
        training_summary_path=Path(args.training_summary_path),
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path, "per_molecule_jsonl": per_molecule_path}


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("result_kind") != "geom_drugs_full_database_final_optimized_amr_cov_subset_plan_not_benchmark":
        raise ValueError("Unexpected AMR/COV subset plan result_kind")
    if plan.get("benchmark_result") is not False:
        raise ValueError("AMR/COV subset plan must be benchmark_result=false")
    if plan.get("dataset_scope") != FULL_DATABASE_SCOPE:
        raise ValueError("AMR/COV subset plan has wrong dataset scope")
    if tuple(plan.get("methods", [])) != METHODS:
        raise ValueError("AMR/COV subset plan method set does not match final optimized all-method matrix")
    if tuple(plan.get("seeds", [])) != SEEDS:
        raise ValueError("AMR/COV subset plan seed set does not match final optimized all-method matrix")


def _load_method_seed_states(torch: Any, device: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    states = {}
    for split_type in ("random", "scaffold"):
        for method in METHODS:
            config = _load_method_config(split_type=split_type, method=method)
            for seed in SEEDS:
                result_path, checkpoint_path = _resolve_result_and_checkpoint(split_type, method, seed)
                payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model = torch.nn.Linear(4, 3)
                model.load_state_dict(payload["model_state_dict"])
                model.to(device)
                model.eval()
                states[(split_type, method, seed)] = {
                    "config": config,
                    "result_path": result_path,
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_payload": payload,
                    "model": model,
                }
    return states


def _load_method_config(split_type: str, method: str) -> dict[str, Any]:
    split_label = "random" if split_type == "random" else "scaffold"
    config_label = METHOD_LABELS[method]
    path = Path("configs/train/final_optimized") / f"geom_full_database_{split_label}_{config_label}_adamw_lr1e3.yaml"
    config = load_config(path)
    expected_mode = "hybrid_harmonic_stiffness_v1" if method.endswith("lambda500") else method
    if config["corruption"]["mode"] != expected_mode:
        raise ValueError(f"Config {path} does not match {method}")
    if config["data"]["split_type"] != split_type:
        raise ValueError(f"Config {path} does not match split {split_type}")
    return config


def _resolve_result_and_checkpoint(split_type: str, method: str, seed: int) -> tuple[Path, Path]:
    candidates = _candidate_paths(Path("experiments/results"), split_type=split_type, seed=seed, method=method)
    result_path = next((item["path"] for item in candidates if item["path"].exists()), None)
    if result_path is None:
        raise FileNotFoundError(f"Missing optimized result JSON for {split_type} {method} seed{seed}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(payload["best_checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing optimized checkpoint from {result_path}: {checkpoint_path}")
    return result_path, checkpoint_path


def _generate_conformers(
    torch: Any,
    model: Any,
    config: dict[str, Any],
    molecule: Any,
    split_type: str,
    partition: str,
    method: str,
    seed: int,
    selection_index: int,
    generated_conformers_per_mol: int,
    device: str,
) -> list[list[list[float]]]:
    source_seed = _generation_seed(
        split_type=split_type,
        partition=partition,
        method=method,
        seed=seed,
        selection_index=selection_index,
    )
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


def _generation_seed(split_type: str, partition: str, method: str, seed: int, selection_index: int) -> int:
    split_offset = {"random": 1_000_000, "scaffold": 2_000_000}[split_type]
    partition_offset = {"val": 10_000, "test": 20_000}[partition]
    method_offset = {method_name: idx * 100_000 for idx, method_name in enumerate(METHODS)}[method]
    return int(seed) + split_offset + partition_offset + method_offset + int(selection_index) * 101


def _build_summary(
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    states: dict[tuple[str, str, int], dict[str, Any]],
    delta: float,
    device: str,
    runtime_seconds: float,
    per_molecule_path: Path,
    plan_path: Path,
    training_summary_path: Path,
) -> dict[str, Any]:
    aggregate_by_split_method = _aggregate(rows, ("split_type", "method"))
    aggregate_by_split_method_seed = _aggregate(rows, ("split_type", "method", "seed"))
    aggregate_by_split_partition_bin_method = _aggregate(rows, ("split_type", "partition", "rotatable_bin", "method"))
    comparisons = _hybrid_vs_harmonic(aggregate_by_split_method)
    rankings = _rankings(aggregate_by_split_method)
    return {
        "summary_name": "geom_full_database_final_optimized_amr_cov_subset_summary",
        "result_kind": "geom_drugs_full_database_final_optimized_amr_cov_subset_summary_not_benchmark",
        "benchmark_result": False,
        "execution_mode": "scheduled_or_local_bounded_final_optimized_amr_cov_subset",
        "hardware": "cpu_only" if device == "cpu" else "device_requested",
        "gpu_used_by_summary": device.startswith("cuda"),
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": True,
        "dataset_scope": FULL_DATABASE_SCOPE,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "delta_angstrom": delta,
        "plan_path": str(plan_path),
        "subset_design": plan["subset_design"],
        "coverage": {
            "planned_molecule_slots": plan["coverage"]["selected_molecule_slots"],
            "evaluated_method_seed_molecule_records": len(rows),
            "skipped_record_count": len(skipped),
            "expected_method_seed_molecule_records": plan["coverage"]["selected_molecule_slots"] * len(METHODS) * len(SEEDS),
        },
        "input_result_paths": {
            f"{split_type}/{method}/seed{seed}": str(state["result_path"])
            for (split_type, method, seed), state in states.items()
        },
        "input_checkpoint_paths": {
            f"{split_type}/{method}/seed{seed}": str(state["checkpoint_path"])
            for (split_type, method, seed), state in states.items()
        },
        "aggregate_by_split_method": aggregate_by_split_method,
        "aggregate_by_split_method_seed": aggregate_by_split_method_seed,
        "aggregate_by_split_partition_bin_method": aggregate_by_split_partition_bin_method,
        "hybrid_lambda500_vs_harmonic": comparisons,
        "method_rankings": rankings,
        "training_ranking_agreement": _training_ranking_agreement(training_summary_path, rankings),
        "skipped_records": skipped[:500],
        "skipped_record_count": len(skipped),
        "per_molecule_jsonl_path": str(per_molecule_path),
        "runtime_seconds": runtime_seconds,
        "notes": [
            "Bounded final optimized AMR/COV subset evaluation. This is not full benchmark evaluation.",
            "Existing validation-selected checkpoints are reloaded; training is not rerun.",
            "Generated conformer ensembles are produced on the fly and per-molecule scalar metrics are written to JSONL.",
            "Raw GEOM reference pickles are read only; raw data are not modified.",
        ],
    }


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = {}
    for key, group in grouped.items():
        name = "/".join(str(part) for part in key)
        output[name] = {
            **{keys[index]: key[index] for index in range(len(keys))},
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


def _hybrid_vs_harmonic(aggregate_by_split_method: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for split_type in ("random", "scaffold"):
        hybrid = aggregate_by_split_method.get(f"{split_type}/hybrid_harmonic_stiffness_v1_lambda500")
        harmonic = aggregate_by_split_method.get(f"{split_type}/harmonic_prior")
        if hybrid is None or harmonic is None:
            output[split_type] = {"available": False}
            continue
        metric_outcomes = {}
        counts = {"better": 0, "tied": 0, "worse": 0}
        for metric in METRICS:
            candidate = hybrid["metric_summary"][metric]["mean"]
            reference = harmonic["metric_summary"][metric]["mean"]
            outcome = _metric_outcome(metric, candidate, reference)
            counts[outcome] += 1
            metric_outcomes[metric] = {
                "hybrid": candidate,
                "harmonic": reference,
                "outcome": outcome,
            }
        output[split_type] = {
            "available": True,
            "metric_outcomes": metric_outcomes,
            "counts": counts,
            "hybrid_beats_harmonic_on_cov_r": metric_outcomes["cov_r"]["outcome"] == "better",
            "hybrid_beats_harmonic_on_cov_p": metric_outcomes["cov_p"]["outcome"] == "better",
            "hybrid_beats_harmonic_on_amr_r": metric_outcomes["amr_r"]["outcome"] == "better",
            "hybrid_beats_harmonic_on_amr_p": metric_outcomes["amr_p"]["outcome"] == "better",
        }
    return output


def _rankings(aggregate_by_split_method: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for split_type in ("random", "scaffold"):
        output[split_type] = {}
        for metric in METRICS:
            values = []
            for method in METHODS:
                row = aggregate_by_split_method.get(f"{split_type}/{method}")
                if row is not None:
                    values.append((method, row["metric_summary"][metric]["mean"]))
            reverse = metric in HIGHER_IS_BETTER
            output[split_type][metric] = [
                {"method": method, "mean": value, "rank": index + 1}
                for index, (method, value) in enumerate(sorted(values, key=lambda item: item[1], reverse=reverse))
            ]
    return output


def _training_ranking_agreement(training_summary_path: Path, rankings: dict[str, Any]) -> dict[str, Any]:
    if not training_summary_path.exists():
        return {"available": False, "reason": "training_summary_missing", "path": str(training_summary_path)}
    summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    if summary.get("coverage", {}).get("missing_result_count") != 0:
        return {"available": False, "reason": "training_summary_incomplete", "path": str(training_summary_path)}
    output = {"available": True, "path": str(training_summary_path), "by_split": {}}
    for split_type in ("random", "scaffold"):
        train_order = summary["split_summaries"][split_type]["rankings"]["overall_order_by_mean_rank"]
        metric_orders = {
            metric: [row["method"] for row in rankings[split_type][metric]]
            for metric in METRICS
        }
        output["by_split"][split_type] = {
            "training_order": train_order,
            "amr_cov_metric_orders": metric_orders,
            "all_metric_orders_match_training": all(order == train_order for order in metric_orders.values()),
        }
    return output


def _metric_outcome(metric: str, candidate: float, reference: float) -> str:
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


def _numeric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {"mean": float(mean(values)), "median": float(median(values)), "min": min(values), "max": max(values)}


def _count_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {"mean": float(mean(values)), "median": float(median(values)), "min": min(values), "max": max(values)}


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _load_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for checkpoint AMR/COV evaluation") from exc
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


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Final Optimized AMR/COV Subset Summary",
        "",
        "Bounded final optimized AMR/COV evaluation from existing checkpoints. This is not full benchmark evaluation.",
        "",
        "## Scope",
        "",
        f"- Evaluated records: `{summary['coverage']['evaluated_method_seed_molecule_records']}` / `{summary['coverage']['expected_method_seed_molecule_records']}`",
        f"- Skipped records: `{summary['coverage']['skipped_record_count']}`",
        f"- Delta: `{summary['delta_angstrom']}`",
        f"- Per-molecule JSONL: `{summary['per_molecule_jsonl_path']}`",
        "",
        "## Hybrid Lambda500 vs Harmonic",
        "",
        "| Split | COV-R | COV-P | AMR-R | AMR-P | Better | Tied | Worse |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for split_type in ("random", "scaffold"):
        row = summary["hybrid_lambda500_vs_harmonic"].get(split_type, {"available": False})
        if not row.get("available"):
            lines.append(f"| {split_type} | NA | NA | NA | NA | 0 | 0 | 0 |")
            continue
        outcomes = row["metric_outcomes"]
        counts = row["counts"]
        lines.append(
            f"| {split_type} | {outcomes['cov_r']['outcome']} | {outcomes['cov_p']['outcome']} | "
            f"{outcomes['amr_r']['outcome']} | {outcomes['amr_p']['outcome']} | "
            f"{counts['better']} | {counts['tied']} | {counts['worse']} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            "| Split | Method | Records | COV-R mean | COV-P mean | AMR-R mean | AMR-P mean |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split_type in ("random", "scaffold"):
        for method in METHODS:
            row = summary["aggregate_by_split_method"].get(f"{split_type}/{method}")
            if row is None:
                lines.append(f"| {split_type} | {method} | 0 | NA | NA | NA | NA |")
                continue
            metrics = row["metric_summary"]
            lines.append(
                f"| {split_type} | {method} | {row['record_count']} | "
                f"{_fmt(metrics['cov_r']['mean'])} | {_fmt(metrics['cov_p']['mean'])} | "
                f"{_fmt(metrics['amr_r']['mean'])} | {_fmt(metrics['amr_p']['mean'])} |"
            )
    lines.extend(
        [
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
