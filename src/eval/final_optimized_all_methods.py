"""Reporting for the final optimized full-method matrix."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.data.geom_full_database import FULL_DATABASE_SCOPE
from src.train.formal_10000_training import FORMAL_10000_MAX_STEPS
from src.train.formal_full_database_training import FORMAL_FULL_DATABASE_RESULT_KIND


DEFAULT_FINAL_OPTIMIZED_ALL_METHODS_SUMMARY_PATH = Path(
    "experiments/results/geom_full_database_final_optimized_all_methods_summary.json"
)
DEFAULT_FINAL_OPTIMIZED_ALL_METHODS_REPORT_PATH = Path(
    "reports/tables/geom_full_database_final_optimized_all_methods_summary.md"
)

RESULT_KIND = "geom_drugs_full_database_final_optimized_all_methods_summary_not_benchmark"

METHODS = (
    "isotropic",
    "stiffness_structured",
    "torsion_only",
    "harmonic_prior",
    "hybrid_harmonic_stiffness_v1_lambda500",
)
METHOD_LABELS = {
    "isotropic": "isotropic",
    "stiffness_structured": "stiffness",
    "torsion_only": "torsion_only",
    "harmonic_prior": "harmonic",
    "hybrid_harmonic_stiffness_v1_lambda500": "hybrid_harmonic_stiffness_v1_lambda500",
}
EXPECTED_ORDER = (
    "hybrid_harmonic_stiffness_v1_lambda500",
    "harmonic_prior",
    "stiffness_structured",
    "isotropic",
    "torsion_only",
)
SPLITS = ("random", "scaffold")
SEEDS = (0, 1, 2)
FINAL_SUFFIX = "final_adamw_lr1e3"
REUSED_RANDOM_SEED0_SUFFIX = "opt_adamw_lr1e3"
CORE_LOWER_IS_BETTER_METRICS = (
    "best_validation_loss",
    "reloaded_validation_loss",
    "sampled_rmsd_to_target_overall_mean",
    "sampled_rmsd_to_target_train_mean",
    "sampled_rmsd_to_target_val_mean",
    "sampled_rmsd_to_target_test_mean",
)
TIE_TOLERANCE = 1.0e-12


def write_final_optimized_all_methods_summary(
    results_root: str | Path = "experiments/results",
    summary_path: str | Path = DEFAULT_FINAL_OPTIMIZED_ALL_METHODS_SUMMARY_PATH,
    report_path: str | Path = DEFAULT_FINAL_OPTIMIZED_ALL_METHODS_REPORT_PATH,
    require_results: bool = False,
) -> dict[str, Path]:
    rows, missing, reused = _load_rows(Path(results_root))
    if require_results and missing:
        missing_text = "\n".join(
            f"- {item['split_type']} seed{item['seed']} {item['method']}: {item['expected_paths']}"
            for item in missing
        )
        raise FileNotFoundError(f"Missing required final optimized all-method result JSONs:\n{missing_text}")
    split_summaries = {
        split_type: _split_summary(split_type=split_type, rows=rows, missing=missing)
        for split_type in SPLITS
    }
    summary = _master_summary(rows=rows, missing=missing, reused=reused, split_summaries=split_summaries)
    summary_path = Path(summary_path)
    report_path = Path(report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def _load_rows(results_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []
    for split_type in SPLITS:
        for seed in SEEDS:
            for method in METHODS:
                candidates = _candidate_paths(results_root, split_type=split_type, seed=seed, method=method)
                selected = next((item for item in candidates if item["path"].exists()), None)
                if selected is None:
                    missing.append(
                        {
                            "split_type": split_type,
                            "seed": seed,
                            "method": method,
                            "expected_paths": [str(item["path"]) for item in candidates],
                        }
                    )
                    continue
                payload = _load_result(selected["path"], split_type=split_type, seed=seed, method=method)
                rows.append(
                    _row_from_result(
                        path=selected["path"],
                        payload=payload,
                        split_type=split_type,
                        seed=seed,
                        method=method,
                        source_role=selected["source_role"],
                    )
                )
                if selected["source_role"] == "reused_optimizer_pilot":
                    reused.append(
                        {
                            "split_type": split_type,
                            "seed": seed,
                            "method": method,
                            "source_path": str(selected["path"]),
                        }
                    )
    return rows, missing, reused


def _candidate_paths(results_root: Path, split_type: str, seed: int, method: str) -> list[dict[str, Any]]:
    split_label = "" if split_type == "random" else "_scaffold"
    label = METHOD_LABELS[method]
    base = f"geom_formal_full_database{split_label}_train_{label}_steps{FORMAL_10000_MAX_STEPS}_seed{seed}"
    paths = [{"source_role": "final_optimized", "path": results_root / f"{base}_{FINAL_SUFFIX}.json"}]
    if split_type == "random" and seed == 0 and method in {
        "harmonic_prior",
        "hybrid_harmonic_stiffness_v1_lambda500",
    }:
        paths.append(
            {
                "source_role": "reused_optimizer_pilot",
                "path": results_root / f"{base}_{REUSED_RANDOM_SEED0_SUFFIX}.json",
            }
        )
    return paths


def _load_result(path: Path, split_type: str, seed: int, method: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("result_kind") != FORMAL_FULL_DATABASE_RESULT_KIND:
        raise ValueError(f"Not a full-database formal result JSON: {path}")
    if payload.get("benchmark_result") is not False:
        raise ValueError(f"Result must be benchmark_result=false: {path}")
    if payload.get("dataset_subset_scope") != FULL_DATABASE_SCOPE:
        raise ValueError(f"Unexpected dataset scope in {path}")
    payload_split_type = payload.get("split_type") or (payload.get("dataset") or {}).get("split_type")
    if payload_split_type != split_type:
        raise ValueError(f"Result split mismatch for {path}")
    if int(payload.get("random_seed")) != int(seed):
        raise ValueError(f"Result seed mismatch for {path}")
    expected_mode = "hybrid_harmonic_stiffness_v1" if method.endswith("lambda500") else method
    if payload.get("corruption_mode") != expected_mode:
        raise ValueError(f"Result method mismatch for {path}")
    if str(payload.get("optimizer_name")) != "AdamW":
        raise ValueError(f"Final optimized result must use AdamW: {path}")
    if abs(float(payload.get("learning_rate")) - 1.0e-3) > 1.0e-12:
        raise ValueError(f"Final optimized result must use lr=1e-3: {path}")
    scheduler = payload.get("lr_scheduler") or payload.get("lr_scheduler_name")
    if scheduler not in {None, False, "None"}:
        raise ValueError(f"Final optimized result must not use an LR scheduler: {path}")
    if method == "hybrid_harmonic_stiffness_v1_lambda500":
        strength = payload.get("hybrid_source_strength")
        if not isinstance(strength, (int, float)) or abs(float(strength) - 5.0) > 1.0e-12:
            raise ValueError(f"Final optimized hybrid result must use lambda=5.0: {path}")
    metrics = payload.get("metrics") or {}
    for key in ["best_validation_loss", "reloaded_validation_loss", "sampled_molecules"]:
        if key not in metrics:
            raise ValueError(f"Result metrics missing {key}: {path}")
    return payload


def _row_from_result(
    path: Path,
    payload: dict[str, Any],
    split_type: str,
    seed: int,
    method: str,
    source_role: str,
) -> dict[str, Any]:
    metrics = payload["metrics"]
    sampled = _sampled_metrics(metrics["sampled_molecules"])
    return {
        "split_type": split_type,
        "seed": int(seed),
        "method": method,
        "source_role": source_role,
        "run_name": payload["run_name"],
        "source_path": str(path),
        "optimizer_name": payload.get("optimizer_name"),
        "learning_rate": float(payload.get("learning_rate")),
        "final_learning_rate": payload.get("final_learning_rate"),
        "lr_scheduler_name": payload.get("lr_scheduler_name"),
        "executed_train_steps": int(metrics["executed_train_steps"]),
        "stop_reason": str(metrics.get("stop_reason") or payload.get("early_stop_reason")),
        "best_validation_loss": float(metrics["best_validation_loss"]),
        "best_validation_loss_step": int(metrics["best_validation_loss_step"]),
        "reloaded_validation_loss": float(metrics["reloaded_validation_loss"]),
        **sampled,
    }


def _sampled_metrics(sampled: list[dict[str, Any]]) -> dict[str, float | None]:
    by_split = {
        split: _mean([float(item["sampled_rmsd_to_target"]) for item in sampled if item.get("split") == split])
        for split in ("train", "val", "test")
    }
    return {
        "sampled_rmsd_to_target_overall_mean": _mean(
            [float(item["sampled_rmsd_to_target"]) for item in sampled]
        ),
        "sampled_rmsd_to_target_train_mean": by_split["train"],
        "sampled_rmsd_to_target_val_mean": by_split["val"],
        "sampled_rmsd_to_target_test_mean": by_split["test"],
    }


def _split_summary(split_type: str, rows: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    split_rows = [row for row in rows if row["split_type"] == split_type]
    split_missing = [row for row in missing if row["split_type"] == split_type]
    aggregate = {method: _aggregate_method([row for row in split_rows if row["method"] == method]) for method in METHODS}
    pairwise = _pairwise_counts(split_rows)
    rankings = _rankings(aggregate)
    return {
        "split_type": split_type,
        "available_result_count": len(split_rows),
        "expected_result_count": len(METHODS) * len(SEEDS),
        "missing_result_count": len(split_missing),
        "missing_results": split_missing,
        "complete_seed_set": len(split_missing) == 0 and len(split_rows) == len(METHODS) * len(SEEDS),
        "seed_level_rows": sorted(split_rows, key=lambda row: (row["seed"], row["method"])),
        "aggregate_metrics": aggregate,
        "pairwise_win_tie_loss_counts": pairwise,
        "rankings": rankings,
        "expected_order": list(EXPECTED_ORDER),
        "matches_expected_order": (
            rankings["complete_for_all_methods"]
            and rankings["overall_order_by_mean_rank"] == list(EXPECTED_ORDER)
        ),
    }


def _master_summary(
    rows: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    reused: list[dict[str, Any]],
    split_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    complete = len(missing) == 0
    cross_split_summary = _cross_split_summary(rows=rows, missing=missing)
    ordering_by_split = {
        split: summary["rankings"]["overall_order_by_mean_rank"] for split, summary in split_summaries.items()
    }
    same_order = complete and len({tuple(order) for order in ordering_by_split.values()}) == 1
    expected_order = complete and all(summary["matches_expected_order"] for summary in split_summaries.values())
    if not complete:
        verdict = "pending_final_optimized_all_method_runs"
        decision = "submit_only_the_18_prepared_missing_cpu_jobs_after_user_approval"
    elif expected_order:
        verdict = "optimized_order_matches_expected_hybrid_harmonic_stiffness_isotropic_torsion"
        decision = "use_this_completed_matrix_as_training_input_for_approved_bounded_cov_amr_evaluation"
    else:
        verdict = "optimized_order_changed_after_all_methods_were_optimized"
        decision = "inspect_rank_changes_before_claiming_method_order"
    return {
        "summary_name": "geom_full_database_final_optimized_all_methods_summary",
        "result_kind": RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_reporting_only_final_optimized_all_methods_summary",
        "hardware": "reporting_only",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "dataset_scope": FULL_DATABASE_SCOPE,
        "methods": list(METHODS),
        "splits": list(SPLITS),
        "seeds": list(SEEDS),
        "optimizer": {"name": "AdamW", "learning_rate": 1.0e-3, "lr_scheduler": None},
        "expected_order": list(EXPECTED_ORDER),
        "coverage": {
            "expected_result_count": len(SPLITS) * len(SEEDS) * len(METHODS),
            "available_result_count": len(rows),
            "missing_result_count": len(missing),
            "missing_results": missing,
            "reused_optimizer_pilot_results": reused,
        },
        "split_summaries": split_summaries,
        "cross_split_summary": cross_split_summary,
        "directional_verdict": {
            "verdict": verdict,
            "decision_rule": decision,
            "random_and_scaffold_same_qualitative_ordering": same_order,
            "optimized_order_matches_expected": expected_order,
            "requires_user_approval_before_any_scheduler_submission": True,
        },
        "safety_labels": {
            "benchmark_result": False,
            "non_benchmark": True,
            "training_executed_by_summary": False,
            "sampling_executed_by_summary": False,
            "scheduler_submission_executed": False,
            "raw_data_modified": False,
            "environment_modified": False,
        },
        "notes": [
            "Reporting-only final optimized all-method summary. This is not benchmark evaluation.",
            "Random seed0 optimized pilot outputs are reused for harmonic and hybrid lambda500 when final duplicate outputs are absent.",
            "No training, sampling, scheduler submission, benchmark evaluation, raw-data edit, or environment change is performed by this summary.",
        ],
    }


def _cross_split_summary(rows: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {method: _aggregate_method([row for row in rows if row["method"] == method]) for method in METHODS}
    rankings = _rankings(aggregate)
    return {
        "available_result_count": len(rows),
        "expected_result_count": len(SPLITS) * len(SEEDS) * len(METHODS),
        "missing_result_count": len(missing),
        "complete_matrix": len(missing) == 0,
        "aggregate_metrics": aggregate,
        "rankings": rankings,
        "expected_order": list(EXPECTED_ORDER),
        "matches_expected_order": (
            rankings["complete_for_all_methods"]
            and rankings["overall_order_by_mean_rank"] == list(EXPECTED_ORDER)
        ),
    }


def _aggregate_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"available_seed_count": 0, "seeds": [], "mean": None, "per_seed": []}
    return {
        "available_seed_count": len(rows),
        "seeds": sorted(row["seed"] for row in rows),
        "mean": {metric: _mean([float(row[metric]) for row in rows]) for metric in CORE_LOWER_IS_BETTER_METRICS},
        "per_seed": sorted(rows, key=lambda row: row["seed"]),
    }


def _pairwise_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for candidate in METHODS:
        output[candidate] = {}
        for reference in METHODS:
            if candidate == reference:
                continue
            comparisons = []
            for seed in SEEDS:
                candidate_row = _find_row(rows, seed=seed, method=candidate)
                reference_row = _find_row(rows, seed=seed, method=reference)
                if candidate_row is None or reference_row is None:
                    continue
                comparisons.append(_compare_rows(candidate_row, reference_row))
            output[candidate][reference] = _win_tie_loss_counts(comparisons)
    return output


def _rankings(aggregate: dict[str, dict[str, Any]]) -> dict[str, Any]:
    available_counts = [aggregate[method]["available_seed_count"] for method in METHODS]
    complete = len(set(available_counts)) == 1 and available_counts[0] > 0
    metric_orders = {}
    rank_sums = {method: 0.0 for method in METHODS}
    available_rank_counts = {method: 0 for method in METHODS}
    for metric in CORE_LOWER_IS_BETTER_METRICS:
        values = []
        for method in METHODS:
            mean = aggregate[method].get("mean")
            value = None if mean is None else mean.get(metric)
            if value is not None:
                values.append((method, float(value)))
        ordered = sorted(values, key=lambda item: item[1])
        metric_orders[metric] = [{"method": method, "mean": value, "rank": index + 1} for index, (method, value) in enumerate(ordered)]
        for index, (method, _value) in enumerate(ordered):
            rank_sums[method] += index + 1
            available_rank_counts[method] += 1
    mean_ranks = {
        method: (rank_sums[method] / available_rank_counts[method] if available_rank_counts[method] else None)
        for method in METHODS
    }
    overall_order = [
        method
        for method, value in sorted(
            ((method, rank) for method, rank in mean_ranks.items() if rank is not None),
            key=lambda item: item[1],
        )
    ]
    return {
        "complete_for_all_methods": complete,
        "metric_orders": metric_orders,
        "mean_rank_by_method": mean_ranks,
        "overall_order_by_mean_rank": overall_order,
    }


def _find_row(rows: list[dict[str, Any]], seed: int, method: str) -> dict[str, Any] | None:
    return next((row for row in rows if row["seed"] == seed and row["method"] == method), None)


def _compare_rows(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    counts = {"better": 0, "tied": 0, "worse": 0}
    metric_comparisons = {}
    for metric in CORE_LOWER_IS_BETTER_METRICS:
        candidate_value = float(candidate[metric])
        reference_value = float(reference[metric])
        outcome = _outcome(candidate_value, reference_value)
        counts[outcome] += 1
        metric_comparisons[metric] = {
            "candidate": candidate_value,
            "reference": reference_value,
            "delta_candidate_minus_reference": candidate_value - reference_value,
            "outcome": outcome,
        }
    return {
        "seed": candidate["seed"],
        "candidate_method": candidate["method"],
        "reference_method": reference["method"],
        "counts": counts,
        "metric_comparisons": metric_comparisons,
        "candidate_better_on_all_core_metrics": counts["better"] == len(CORE_LOWER_IS_BETTER_METRICS),
    }


def _win_tie_loss_counts(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"better": 0, "tied": 0, "worse": 0}
    seed_all_core_better = 0
    for comparison in comparisons:
        for metric in CORE_LOWER_IS_BETTER_METRICS:
            counts[comparison["metric_comparisons"][metric]["outcome"]] += 1
        if comparison["candidate_better_on_all_core_metrics"]:
            seed_all_core_better += 1
    return {
        "available_seed_count": len(comparisons),
        "metric_counts_over_available_seeds": counts,
        "seed_all_core_better_count": seed_all_core_better,
    }


def _outcome(candidate_value: float, reference_value: float) -> str:
    if candidate_value < reference_value - TIE_TOLERANCE:
        return "better"
    if candidate_value > reference_value + TIE_TOLERANCE:
        return "worse"
    return "tied"


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Full-Database Final Optimized All-Methods Summary",
        "",
        "Reporting-only summary for final optimized `AdamW lr=1e-3` all-method runs. This is not benchmark evaluation.",
        "",
        "## Scope",
        "",
        f"- Available results: `{summary['coverage']['available_result_count']}` / `{summary['coverage']['expected_result_count']}`",
        f"- Missing results: `{summary['coverage']['missing_result_count']}`",
        f"- Reused optimizer-pilot results: `{summary['coverage']['reused_optimizer_pilot_results']}`",
        f"- Verdict: `{summary['directional_verdict']['verdict']}`",
        f"- Expected order: `{' > '.join(summary['expected_order'])}`",
        "",
        "## Aggregate Means",
        "",
        "| Split | Method | Seeds | Best val | Reload val | RMSD overall | RMSD train | RMSD val | RMSD test |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_type in SPLITS:
        split_summary = summary["split_summaries"][split_type]
        for method in METHODS:
            row = split_summary["aggregate_metrics"][method]
            mean = row.get("mean") or {}
            lines.append(
                f"| {split_type} | {method} | {row['available_seed_count']} | "
                f"{_fmt(mean.get('best_validation_loss'))} | "
                f"{_fmt(mean.get('reloaded_validation_loss'))} | "
                f"{_fmt(mean.get('sampled_rmsd_to_target_overall_mean'))} | "
                f"{_fmt(mean.get('sampled_rmsd_to_target_train_mean'))} | "
                f"{_fmt(mean.get('sampled_rmsd_to_target_val_mean'))} | "
                f"{_fmt(mean.get('sampled_rmsd_to_target_test_mean'))} |"
            )
    lines.extend(["", "## Final Ranking", ""])
    for split_type in SPLITS:
        ranking = summary["split_summaries"][split_type]["rankings"]
        order = " > ".join(ranking["overall_order_by_mean_rank"]) if ranking["overall_order_by_mean_rank"] else "NA"
        lines.append(f"- `{split_type}`: `{order}`")
    cross_order = " > ".join(summary["cross_split_summary"]["rankings"]["overall_order_by_mean_rank"])
    lines.append(f"- `cross_split`: `{cross_order if cross_order else 'NA'}`")
    lines.extend(
        [
            "",
            "## Adjacent Expected-Order Win/Tie/Loss Counts",
            "",
            "| Scope | Candidate | Reference | Seeds/rows | Better | Tied | Worse |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    adjacent_pairs = tuple(zip(EXPECTED_ORDER, EXPECTED_ORDER[1:]))
    for split_type in SPLITS:
        pairwise = summary["split_summaries"][split_type]["pairwise_win_tie_loss_counts"]
        for candidate, reference in adjacent_pairs:
            counts = pairwise[candidate][reference]
            metric_counts = counts["metric_counts_over_available_seeds"]
            lines.append(
                f"| {split_type} | {candidate} | {reference} | {counts['available_seed_count']} | "
                f"{metric_counts['better']} | {metric_counts['tied']} | {metric_counts['worse']} |"
            )
    lines.extend(
        [
            "",
            "## Missing Results",
            "",
            "```json",
            json.dumps(summary["coverage"]["missing_results"], indent=2, sort_keys=True),
            "```",
            "",
            "No jobs were submitted and no training, sampling, benchmark evaluation, raw-data edits, or environment changes were run by this summary.",
            "",
        ]
    )
    return "\n".join(lines)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NA"
    return f"{float(value):.8g}"
