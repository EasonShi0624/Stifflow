"""Reporting for final optimized harmonic-prior vs hybrid follow-up."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.data.geom_full_database import FULL_DATABASE_SCOPE
from src.train.formal_10000_training import FORMAL_10000_MAX_STEPS
from src.train.formal_full_database_training import FORMAL_FULL_DATABASE_RESULT_KIND


DEFAULT_FINAL_OPTIMIZED_SUMMARY_PATH = Path(
    "experiments/results/geom_full_database_final_optimized_harmonic_vs_hybrid_summary.json"
)
DEFAULT_FINAL_OPTIMIZED_REPORT_PATH = Path(
    "reports/tables/geom_full_database_final_optimized_harmonic_vs_hybrid_summary.md"
)

FINAL_OPTIMIZED_RESULT_KIND = (
    "geom_drugs_full_database_final_optimized_harmonic_vs_hybrid_summary_not_benchmark"
)

METHODS = ("harmonic_prior", "hybrid_harmonic_stiffness_v1_lambda500")
METHOD_LABELS = {
    "harmonic_prior": "harmonic",
    "hybrid_harmonic_stiffness_v1_lambda500": "hybrid_harmonic_stiffness_v1_lambda500",
}
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


def write_final_optimized_harmonic_vs_hybrid_summary(
    results_root: str | Path = "experiments/results",
    summary_path: str | Path = DEFAULT_FINAL_OPTIMIZED_SUMMARY_PATH,
    report_path: str | Path = DEFAULT_FINAL_OPTIMIZED_REPORT_PATH,
    require_results: bool = False,
) -> dict[str, Path]:
    results_root = Path(results_root)
    rows, missing, reused = _load_rows(results_root)
    if require_results and missing:
        missing_paths = "\n".join(
            f"- {item['split_type']} seed{item['seed']} {item['method']}: {item['expected_paths']}"
            for item in missing
        )
        raise FileNotFoundError(f"Missing required final optimized result JSONs:\n{missing_paths}")
    split_summaries = {
        split_type: _split_summary(split_type=split_type, rows=rows, missing=missing)
        for split_type in SPLITS
    }
    master = _master_summary(rows=rows, missing=missing, reused=reused, split_summaries=split_summaries)

    summary_path = Path(summary_path)
    report_path = Path(report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(master, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(master), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def _load_rows(results_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    missing = []
    reused = []
    for split_type in SPLITS:
        for seed in SEEDS:
            for method in METHODS:
                paths = _candidate_paths(results_root, split_type=split_type, method=method, seed=seed)
                selected = next((item for item in paths if item["path"].exists()), None)
                if selected is None:
                    missing.append(
                        {
                            "split_type": split_type,
                            "seed": seed,
                            "method": method,
                            "expected_paths": [str(item["path"]) for item in paths],
                        }
                    )
                    continue
                payload = _load_result(
                    path=selected["path"],
                    split_type=split_type,
                    seed=seed,
                    method=method,
                )
                row = _row_from_result(
                    path=selected["path"],
                    payload=payload,
                    split_type=split_type,
                    seed=seed,
                    method=method,
                    source_role=selected["source_role"],
                )
                rows.append(row)
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


def _candidate_paths(results_root: Path, split_type: str, method: str, seed: int) -> list[dict[str, Any]]:
    split_label = "" if split_type == "random" else "_scaffold"
    label = METHOD_LABELS[method]
    base = f"geom_formal_full_database{split_label}_train_{label}_steps{FORMAL_10000_MAX_STEPS}_seed{seed}"
    paths = [
        {"source_role": "final_optimized", "path": results_root / f"{base}_{FINAL_SUFFIX}.json"},
    ]
    if split_type == "random" and seed == 0:
        paths.append(
            {"source_role": "reused_optimizer_pilot", "path": results_root / f"{base}_{REUSED_RANDOM_SEED0_SUFFIX}.json"}
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
    expected_mode = "harmonic_prior" if method == "harmonic_prior" else "hybrid_harmonic_stiffness_v1"
    if payload.get("corruption_mode") != expected_mode:
        raise ValueError(f"Result method mismatch for {path}")
    if str(payload.get("optimizer_name")) != "AdamW":
        raise ValueError(f"Final optimized result must use AdamW: {path}")
    if abs(float(payload.get("learning_rate")) - 1.0e-3) > 1.0e-12:
        raise ValueError(f"Final optimized result must use lr=1e-3: {path}")
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
    if not sampled:
        raise ValueError("sampled_molecules must be non-empty")
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
    comparisons = []
    for seed in SEEDS:
        harmonic = _find_row(split_rows, seed=seed, method="harmonic_prior")
        hybrid = _find_row(split_rows, seed=seed, method="hybrid_harmonic_stiffness_v1_lambda500")
        if harmonic is None or hybrid is None:
            comparisons.append(
                {
                    "seed": seed,
                    "available": False,
                    "missing_methods": [
                        method
                        for method, row in [
                            ("harmonic_prior", harmonic),
                            ("hybrid_harmonic_stiffness_v1_lambda500", hybrid),
                        ]
                        if row is None
                    ],
                }
            )
            continue
        comparisons.append(
            {
                "seed": seed,
                "available": True,
                **_compare_rows(candidate=hybrid, reference=harmonic),
            }
        )
    aggregate_metrics = {
        method: _aggregate_method([row for row in split_rows if row["method"] == method])
        for method in METHODS
    }
    available_comparisons = [row for row in comparisons if row.get("available")]
    complete = len(split_missing) == 0 and len(available_comparisons) == len(SEEDS)
    hybrid_wins_all_available = bool(available_comparisons) and all(
        row["candidate_better_on_all_core_metrics"] for row in available_comparisons
    )
    return {
        "split_type": split_type,
        "available_result_count": len(split_rows),
        "expected_result_count": len(METHODS) * len(SEEDS),
        "missing_result_count": len(split_missing),
        "missing_results": split_missing,
        "complete_seed_set": complete,
        "aggregate_metrics": aggregate_metrics,
        "seed_comparisons": comparisons,
        "win_tie_loss_counts": _win_tie_loss_counts(available_comparisons),
        "hybrid_beats_harmonic_on_all_tracked_metrics_all_available_seeds": hybrid_wins_all_available,
        "hybrid_beats_harmonic_on_all_tracked_metrics_complete_seed_set": complete and hybrid_wins_all_available,
    }


def _master_summary(
    rows: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    reused: list[dict[str, Any]],
    split_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    complete = len(missing) == 0
    random_win = split_summaries["random"]["hybrid_beats_harmonic_on_all_tracked_metrics_complete_seed_set"]
    scaffold_win = split_summaries["scaffold"]["hybrid_beats_harmonic_on_all_tracked_metrics_complete_seed_set"]
    if not complete:
        verdict = "pending_final_optimized_runs"
        decision_rule = "submit_only_the_prepared_missing_final_optimized_jobs_after_approval"
        proceed = False
    elif random_win and scaffold_win:
        verdict = "optimized_hybrid_remains_better_than_optimized_harmonic_both_splits"
        decision_rule = "proceed_to_benchmark_style_physical_validation_after_user_approval"
        proceed = True
    else:
        verdict = "optimized_harmonic_catches_up_or_beats_hybrid_on_at_least_one_split"
        decision_rule = "do_not_proceed_to_benchmark_without_discussing_limitation"
        proceed = False
    return {
        "summary_name": "geom_full_database_final_optimized_harmonic_vs_hybrid_summary",
        "result_kind": FINAL_OPTIMIZED_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_reporting_only_final_optimized_harmonic_vs_hybrid_summary",
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
        "hybrid_method": {"corruption_mode": "hybrid_harmonic_stiffness_v1", "lambda": 5.0},
        "coverage": {
            "expected_result_count": len(SPLITS) * len(SEEDS) * len(METHODS),
            "available_result_count": len(rows),
            "missing_result_count": len(missing),
            "missing_results": missing,
            "reused_optimizer_pilot_results": reused,
        },
        "split_summaries": split_summaries,
        "directional_verdict": {
            "verdict": verdict,
            "decision_rule": decision_rule,
            "proceed_to_benchmark_style_physical_validation_supported": proceed,
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
            "Reporting-only final optimized harmonic-vs-hybrid summary. This is not benchmark evaluation.",
            "Random seed0 optimized pilot outputs are reused when final-optimized duplicate outputs are absent, because they match the final method, source, optimizer, learning rate, split, seed, checkpoint/reload behavior, sampling diagnostics, and hard cap.",
            "No training, sampling, scheduler submission, benchmark evaluation, raw-data edit, or environment change is performed by this summary.",
        ],
    }


def _find_row(rows: list[dict[str, Any]], seed: int, method: str) -> dict[str, Any] | None:
    return next((row for row in rows if row["seed"] == seed and row["method"] == method), None)


def _aggregate_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"available_seed_count": 0, "seeds": [], "mean": None, "per_seed": []}
    return {
        "available_seed_count": len(rows),
        "seeds": sorted(row["seed"] for row in rows),
        "mean": {
            metric: _mean([float(row[metric]) for row in rows])
            for metric in CORE_LOWER_IS_BETTER_METRICS
        },
        "per_seed": sorted(rows, key=lambda row: row["seed"]),
    }


def _compare_rows(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    comparisons = {}
    counts = {"better": 0, "tied": 0, "worse": 0}
    for metric in CORE_LOWER_IS_BETTER_METRICS:
        candidate_value = float(candidate[metric])
        reference_value = float(reference[metric])
        outcome = _outcome(candidate_value, reference_value)
        counts[outcome] += 1
        comparisons[metric] = {
            "hybrid": candidate_value,
            "harmonic": reference_value,
            "delta_hybrid_minus_harmonic": candidate_value - reference_value,
            "outcome": outcome,
        }
    return {
        "candidate_method": candidate["method"],
        "reference_method": reference["method"],
        "candidate_run_name": candidate["run_name"],
        "reference_run_name": reference["run_name"],
        "metric_comparisons": comparisons,
        "counts": counts,
        "candidate_better_on_all_core_metrics": counts["better"] == len(CORE_LOWER_IS_BETTER_METRICS),
        "candidate_better_or_tied_on_all_core_metrics": counts["worse"] == 0,
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
        "metric_counts_over_available_seeds": counts,
        "available_seed_count": len(comparisons),
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
        "# Full-Database Final Optimized Harmonic vs Hybrid Summary",
        "",
        "Reporting-only summary for final optimized `AdamW lr=1e-3` harmonic vs hybrid runs. This is not benchmark evaluation.",
        "",
        "## Scope",
        "",
        f"- Available results: `{summary['coverage']['available_result_count']}` / `{summary['coverage']['expected_result_count']}`",
        f"- Missing results: `{summary['coverage']['missing_result_count']}`",
        f"- Reused optimizer-pilot results: `{summary['coverage']['reused_optimizer_pilot_results']}`",
        f"- Verdict: `{summary['directional_verdict']['verdict']}`",
        f"- Decision rule: `{summary['directional_verdict']['decision_rule']}`",
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
    lines.extend(
        [
            "",
            "## Per-Seed Hybrid vs Harmonic",
            "",
            "| Split | Seed | Available | Better | Tied | Worse | Hybrid all-core better | Missing methods |",
            "|---|---:|---|---:|---:|---:|---|---|",
        ]
    )
    for split_type in SPLITS:
        for comparison in summary["split_summaries"][split_type]["seed_comparisons"]:
            if not comparison["available"]:
                lines.append(
                    f"| {split_type} | {comparison['seed']} | false | NA | NA | NA | false | `{comparison['missing_methods']}` |"
                )
                continue
            counts = comparison["counts"]
            lines.append(
                f"| {split_type} | {comparison['seed']} | true | {counts['better']} | {counts['tied']} | "
                f"{counts['worse']} | {comparison['candidate_better_on_all_core_metrics']} | `[]` |"
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
            "## Safety",
            "",
            f"- `benchmark_result`: `{summary['benchmark_result']}`",
            f"- `training_executed_by_summary`: `{summary['training_executed_by_summary']}`",
            f"- `sampling_executed_by_summary`: `{summary['sampling_executed_by_summary']}`",
            f"- `raw_data_modified`: `{summary['raw_data_modified']}`",
            f"- `environment_modified`: `{summary['environment_modified']}`",
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
