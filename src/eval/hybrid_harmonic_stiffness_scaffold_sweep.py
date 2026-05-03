"""Reporting utilities for the scaffold hybrid harmonic + stiffness lambda sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.geom_full_database import FULL_DATABASE_SCOPE
from src.eval.hybrid_harmonic_stiffness_pilot import (
    BASELINE_LABELS,
    BASELINE_MODES,
    CORE_LOWER_IS_BETTER_METRICS,
    RMSD_METRICS,
    TIE_TOLERANCE,
    VALIDATION_LOSS_METRICS,
    _baseline_metric_table,
    _best_hybrid_by_core_rank,
    _dataset_summary,
    _fmt,
    _fmt_int,
    _group_rows,
    _hybrid_comparison,
    _load_rows,
)
from src.train.formal_10000_training import (
    FORMAL_10000_HARD_MAX_STEPS,
    FORMAL_10000_LEARNING_RATE,
    FORMAL_10000_MAX_STEPS,
    FORMAL_10000_MIN_DELTA,
    FORMAL_10000_PATIENCE,
    FORMAL_10000_SAMPLE_STEP_SIZE,
)


DEFAULT_SCAFFOLD_SWEEP_SUMMARY_PATH = Path(
    "experiments/results/geom_full_database_scaffold_hybrid_harmonic_stiffness_lambda_sweep_summary.json"
)
DEFAULT_SCAFFOLD_SWEEP_REPORT_PATH = Path(
    "reports/tables/geom_full_database_scaffold_hybrid_harmonic_stiffness_lambda_sweep_summary.md"
)
SCAFFOLD_SWEEP_SUMMARY_RESULT_KIND = (
    "geom_drugs_full_database_scaffold_hybrid_harmonic_stiffness_lambda_sweep_summary_not_benchmark"
)
SCAFFOLD_HYBRID_LAMBDAS = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00)
SCAFFOLD_HYBRID_MODE = "hybrid_harmonic_stiffness_v1"


def write_scaffold_hybrid_harmonic_stiffness_lambda_sweep_summary(
    result_paths: list[str | Path] | None = None,
    results_root: str | Path = "experiments/results",
    summary_path: str | Path = DEFAULT_SCAFFOLD_SWEEP_SUMMARY_PATH,
    report_path: str | Path = DEFAULT_SCAFFOLD_SWEEP_REPORT_PATH,
    seed: int = 0,
    require_hybrid_results: bool = False,
) -> dict[str, Path]:
    paths = _resolve_result_paths(
        result_paths=result_paths,
        results_root=Path(results_root),
        seed=int(seed),
    )
    rows = _load_rows(paths, expected_split_type="scaffold", expected_seed=int(seed))
    summary = _build_summary(
        rows=rows,
        input_paths=paths,
        seed=int(seed),
        require_hybrid_results=require_hybrid_results,
    )
    summary_path = Path(summary_path)
    report_path = Path(report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def _resolve_result_paths(
    result_paths: list[str | Path] | None,
    results_root: Path,
    seed: int,
) -> list[Path]:
    if result_paths:
        return [Path(path) for path in result_paths]
    paths = []
    for mode in BASELINE_MODES:
        paths.append(
            results_root
            / (
                "geom_formal_full_database_scaffold_train_"
                f"{BASELINE_LABELS[mode]}_steps{FORMAL_10000_MAX_STEPS}_seed{seed}.json"
            )
        )
    for strength in SCAFFOLD_HYBRID_LAMBDAS:
        paths.append(
            results_root
            / (
                "geom_formal_full_database_scaffold_train_"
                f"{SCAFFOLD_HYBRID_MODE}_{_lambda_label(strength)}"
                f"_steps{FORMAL_10000_MAX_STEPS}_seed{seed}.json"
            )
        )
    return paths


def _build_summary(
    rows: list[dict[str, Any]],
    input_paths: list[Path],
    seed: int,
    require_hybrid_results: bool,
) -> dict[str, Any]:
    by_mode = _group_rows(rows)
    baseline_rows = {mode: by_mode.get((mode, None)) for mode in BASELINE_MODES}
    missing_baselines = [mode for mode, row in baseline_rows.items() if row is None]
    hybrid_rows = {
        strength: by_mode.get((SCAFFOLD_HYBRID_MODE, round(strength, 2)))
        for strength in SCAFFOLD_HYBRID_LAMBDAS
    }
    missing_hybrids = [
        {"corruption_mode": SCAFFOLD_HYBRID_MODE, "source_strength": strength}
        for strength, row in hybrid_rows.items()
        if row is None
    ]
    if require_hybrid_results and missing_hybrids:
        raise FileNotFoundError(f"Missing expected scaffold hybrid result rows: {missing_hybrids}")

    hybrid_comparisons = [
        _hybrid_comparison(
            mode=SCAFFOLD_HYBRID_MODE,
            strength=strength,
            hybrid_row=hybrid_rows[strength],
            baseline_rows=baseline_rows,
        )
        for strength in SCAFFOLD_HYBRID_LAMBDAS
    ]
    best_scaffold_lambda = _best_hybrid_by_core_rank(hybrid_comparisons)
    directional_verdict = _directional_verdict(
        comparisons=hybrid_comparisons,
        missing_baselines=missing_baselines,
    )
    harmonic_prior_outcome_summary = _harmonic_prior_outcome_summary(hybrid_comparisons)
    lambda_trend = _lambda_trend(hybrid_comparisons)
    large_lambda_assessment = _large_lambda_assessment(hybrid_comparisons)
    summary = {
        "summary_name": "geom_full_database_scaffold_hybrid_harmonic_stiffness_lambda_sweep_summary",
        "result_kind": SCAFFOLD_SWEEP_SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_full_database_scaffold_hybrid_harmonic_stiffness_lambda_sweep_report",
        "hardware": "reporting_only",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "split_type": "scaffold",
        "seed": int(seed),
        "dataset_scope": FULL_DATABASE_SCOPE,
        "input_result_paths": [str(path) for path in input_paths],
        "input_result_paths_found": [row["source_path"] for row in rows],
        "row_count": len(rows),
        "baseline_modes": list(BASELINE_MODES),
        "hybrid_mode": SCAFFOLD_HYBRID_MODE,
        "hybrid_lambdas": list(SCAFFOLD_HYBRID_LAMBDAS),
        "coverage": {
            "baseline_rows_available": [mode for mode, row in baseline_rows.items() if row is not None],
            "missing_baselines": missing_baselines,
            "hybrid_rows_available": [
                {"corruption_mode": SCAFFOLD_HYBRID_MODE, "source_strength": strength}
                for strength, row in hybrid_rows.items()
                if row is not None
            ],
            "missing_hybrids": missing_hybrids,
            "complete_scaffold_lambda_sweep_matrix": not missing_hybrids,
        },
        "dataset": _dataset_summary(rows),
        "training_horizon": {
            "max_steps": FORMAL_10000_MAX_STEPS,
            "hard_max_steps": FORMAL_10000_HARD_MAX_STEPS,
            "optimizer": "SGD",
            "learning_rate": FORMAL_10000_LEARNING_RATE,
            "early_stop_monitor": "validation_loss_after_step",
            "early_stop_patience": FORMAL_10000_PATIENCE,
            "early_stop_min_delta": FORMAL_10000_MIN_DELTA,
            "sample_step_count": 1,
            "sample_step_size": FORMAL_10000_SAMPLE_STEP_SIZE,
        },
        "rows": rows,
        "baseline_metric_table": _baseline_metric_table(baseline_rows),
        "hybrid_comparisons": hybrid_comparisons,
        "best_scaffold_lambda": best_scaffold_lambda,
        "best_hybrid": best_scaffold_lambda,
        "harmonic_prior_outcome_summary": harmonic_prior_outcome_summary,
        "lambda_trend": lambda_trend,
        "large_lambda_assessment": large_lambda_assessment,
        "directional_verdict": directional_verdict,
        "supports_multiseed_followup": directional_verdict["supports_multiseed_followup"],
        "multiseed_followup": directional_verdict["multiseed_followup"],
        "safety_labels": {
            "benchmark_result": False,
            "non_benchmark": True,
            "scaffold_lambda_sweep_summary": True,
            "training_executed_by_summary": False,
            "sampling_executed_by_summary": False,
            "raw_data_modified": False,
            "environment_modified": False,
            "gpu_used_by_summary": False,
            "result_scope": "full_database_scaffold_seed0_hybrid_harmonic_stiffness_lambda_sweep_reporting_only",
        },
        "notes": [
            "Reporting-only scaffold-split seed0 hybrid harmonic + stiffness lambda-sweep summary. This is not a benchmark result.",
            "The summary compares only hybrid_harmonic_stiffness_v1 lambdas against existing scaffold seed0 harmonic_prior, stiffness_structured, isotropic, and torsion_only references when available.",
            "Success is not claimed unless a lambda beats harmonic_prior on best validation loss, reloaded validation loss, and sampled RMSD overall/train/val/test.",
            "Missing hybrid result JSONs are reported as pending rather than treated as wins, losses, or ties.",
            "No residual variants, old baselines, multiseed runs, scaffold jobs, preprocessing, scheduler action, GPU use, physical benchmark jobs, xTB, raw-data mutation, or environment change is performed by this summary.",
        ],
    }
    return summary


def _harmonic_prior_outcome_summary(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in comparisons if row["result_available"]]
    all_core = [
        row for row in available
        if row["beats_harmonic_prior_on_all_core_metrics"]
    ]
    validation_losses = [
        row for row in available
        if row["baseline_core_summary"]["harmonic_prior"]["validation_losses_better"]
    ]
    rmsd_metrics = [
        row for row in available
        if row["baseline_core_summary"]["harmonic_prior"]["rmsd_metrics_better"]
    ]
    return {
        "any_lambda_beats_harmonic_prior_on_all_tracked_core_metrics": bool(all_core),
        "lambdas_beating_harmonic_prior_on_all_tracked_core_metrics": [
            row["source_strength"] for row in all_core
        ],
        "any_lambda_beats_harmonic_prior_on_best_and_reloaded_validation_loss": bool(validation_losses),
        "lambdas_beating_harmonic_prior_on_best_and_reloaded_validation_loss": [
            row["source_strength"] for row in validation_losses
        ],
        "any_lambda_beats_harmonic_prior_on_sampled_rmsd_overall_train_val_test": bool(rmsd_metrics),
        "lambdas_beating_harmonic_prior_on_sampled_rmsd_overall_train_val_test": [
            row["source_strength"] for row in rmsd_metrics
        ],
        "validation_loss_metrics": list(VALIDATION_LOSS_METRICS),
        "rmsd_metrics": list(RMSD_METRICS),
        "best_harmonic_beating_lambda_by_core_rank": _best_hybrid_by_core_rank(all_core),
        "direction": "lower_is_better",
    }


def _directional_verdict(
    comparisons: list[dict[str, Any]],
    missing_baselines: list[str],
) -> dict[str, Any]:
    available = [row for row in comparisons if row["result_available"]]
    harmonic_winners = [row for row in available if row["beats_harmonic_prior_on_all_core_metrics"]]
    partial_success_rows = [
        row for row in available if row["partial_success_validation_loss_only_vs_harmonic"]
    ]
    stiffness_winners = [row for row in available if row["beats_stiffness_structured_on_all_core_metrics"]]
    best_harmonic_winner = _best_hybrid_by_core_rank(harmonic_winners)
    best_available = _best_hybrid_by_core_rank(available)
    over_modulation_risk = bool(
        best_harmonic_winner and float(best_harmonic_winner["source_strength"]) > 0.50
    )

    if missing_baselines:
        verdict = "missing_required_scaffold_baseline_results"
        decision_rule = "missing_required_baseline_results"
    elif not available:
        verdict = "scaffold_hybrid_results_not_available"
        decision_rule = "await_completed_scaffold_lambda_sweep_results"
    elif best_harmonic_winner:
        strength = float(best_harmonic_winner["source_strength"])
        if abs(strength - 0.50) <= 1.0e-12:
            verdict = "lambda_0_5_remains_best_and_beats_harmonic_prior"
            decision_rule = "recommend_lambda_0_5_for_scaffold_random_multiseed_followup"
        elif strength > 0.50:
            verdict = "larger_lambda_beats_harmonic_prior_flag_over_modulation_risk"
            decision_rule = "recommend_best_lambda_with_over_modulation_risk_flag"
        else:
            verdict = "smaller_lambda_beats_harmonic_prior_on_scaffold"
            decision_rule = "recommend_best_lambda_for_scaffold_random_multiseed_followup"
    elif partial_success_rows:
        verdict = "partial_success_validation_loss_only_do_not_automatically_proceed"
        decision_rule = "partial_success_do_not_proceed_without_review"
    elif stiffness_winners:
        verdict = "partial_transfer_beats_stiffness_but_not_harmonic_prior"
        decision_rule = "do_not_start_multiseed_until_review"
    else:
        verdict = "no_lambda_beats_stiffness_stop_scaffold_hybrid_branch"
        decision_rule = "stop_scaffold_hybrid_branch_and_analyze_failure"

    multiseed_followup = {
        "supported": bool(best_harmonic_winner),
        "recommendation": None,
        "requires_user_approval_before_submission": True,
        "reason": "No scaffold lambda beats harmonic_prior on all tracked core metrics.",
        "over_modulation_risk": over_modulation_risk,
    }
    if best_harmonic_winner:
        multiseed_followup["recommendation"] = {
            "corruption_mode": SCAFFOLD_HYBRID_MODE,
            "source_strength": best_harmonic_winner["source_strength"],
            "selection_basis": best_harmonic_winner["selection_basis"],
            "recommended_scope": "scaffold_and_random_multiseed_followup_only_after_approval",
        }
        multiseed_followup["reason"] = (
            "This scaffold lambda beats harmonic_prior on best validation loss, reloaded validation loss, "
            "and sampled RMSD overall/train/val/test."
        )
        if over_modulation_risk:
            multiseed_followup["reason"] += " The selected lambda is above 0.5, so over-modulation risk should be tracked."

    return {
        "verdict": verdict,
        "decision_rule": decision_rule,
        "success_claim_supported": bool(best_harmonic_winner),
        "partial_success_present": bool(partial_success_rows),
        "supports_multiseed_followup": bool(best_harmonic_winner),
        "best_scaffold_lambda": best_available,
        "best_harmonic_beating_lambda": best_harmonic_winner,
        "lambdas_beating_harmonic_prior": [
            row["source_strength"] for row in harmonic_winners
        ],
        "lambdas_beating_stiffness_structured": [
            row["source_strength"] for row in stiffness_winners
        ],
        "partial_success_validation_loss_only_lambdas": [
            row["source_strength"] for row in partial_success_rows
        ],
        "over_modulation_risk_flag": over_modulation_risk,
        "multiseed_followup": multiseed_followup,
        "available_hybrid_result_count": len(available),
        "missing_baselines": missing_baselines,
        "direction": "lower_is_better",
    }


def _lambda_trend(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    available = sorted(
        [row for row in comparisons if row["result_available"]],
        key=lambda row: float(row["source_strength"]),
    )
    if not available:
        return {
            "status": "not_available",
            "classification": "not_available_until_hybrid_results_exist",
            "requested_bucket": "not_available",
        }
    metric_trends = {}
    monotonic_metric_count = 0
    for metric in CORE_LOWER_IS_BETTER_METRICS:
        series = [
            {"source_strength": row["source_strength"], "value": float(row["metrics"][metric])}
            for row in available
        ]
        is_monotonic = _nonincreasing([item["value"] for item in series])
        if is_monotonic:
            monotonic_metric_count += 1
        best = min(series, key=lambda item: item["value"])
        metric_trends[metric] = {
            "series": series,
            "is_monotonic_improving": is_monotonic,
            "best_source_strength": best["source_strength"],
            "best_value": best["value"],
        }

    rank_summary = _average_core_metric_rank_by_lambda(available)
    best_ranked = min(
        rank_summary,
        key=lambda row: (
            row["average_core_metric_rank"],
            row["best_validation_loss"],
            row["sampled_rmsd_to_target_overall_mean"],
        ),
    )
    best_strength = float(best_ranked["source_strength"])
    max_strength = max(float(row["source_strength"]) for row in available)
    high_end = [
        row for row in available
        if float(row["source_strength"]) in {3.0, 5.0}
    ]
    high_end_close = False
    if len(high_end) >= 2:
        high_end_close = all(
            _relative_delta(
                float(high_end[0]["metrics"][metric]),
                float(high_end[-1]["metrics"][metric]),
            )
            <= 0.10
            for metric in CORE_LOWER_IS_BETTER_METRICS
        )
    all_monotonic = monotonic_metric_count == len(CORE_LOWER_IS_BETTER_METRICS)
    any_mid_sweep_bump = any(not trend["is_monotonic_improving"] for trend in metric_trends.values())
    if all_monotonic:
        classification = "monotonic_improving"
        requested_bucket = "monotonic"
    elif best_strength == max_strength and high_end_close:
        classification = "saturated_high_lambda_with_minor_nonmonotonicity"
        requested_bucket = "saturated"
    elif 0.25 < best_strength < max_strength:
        classification = "peaked"
        requested_bucket = "peaked"
    else:
        classification = "nonmonotonic_mixed"
        requested_bucket = "nonmonotonic"
    return {
        "status": "available",
        "classification": classification,
        "requested_bucket": requested_bucket,
        "best_source_strength_by_average_core_metric_rank": best_strength,
        "best_source_strength_is_max_tested_lambda": best_strength == max_strength,
        "all_core_metrics_monotonic_improving": all_monotonic,
        "monotonic_core_metric_count": monotonic_metric_count,
        "core_metric_count": len(CORE_LOWER_IS_BETTER_METRICS),
        "minor_nonmonotonicity_detected": any_mid_sweep_bump,
        "metric_trends": metric_trends,
        "average_core_metric_rank_by_lambda": rank_summary,
        "note": (
            "Trend classification is based on lower-is-better core metrics across the tested scaffold lambdas. "
            "A saturated high-lambda label means the best tested lambda is at the high end and large-lambda "
            "metrics are close, even if a smaller mid-sweep bump prevents strict monotonicity."
        ),
    }


def _large_lambda_assessment(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in comparisons if row["result_available"]]
    lower = [row for row in available if float(row["source_strength"]) < 3.0]
    large = [row for row in available if float(row["source_strength"]) in {3.0, 5.0}]
    if not large:
        return {
            "status": "not_available",
            "overall_assessment": "not_available_until_large_lambda_results_exist",
        }
    lower_frontier = _best_hybrid_by_core_rank(lower)
    rows = []
    for row in sorted(large, key=lambda item: float(item["source_strength"])):
        improves_lower_frontier = False
        if lower_frontier:
            improves_lower_frontier = all(
                float(row["metrics"][metric]) < float(lower_frontier["metrics"][metric]) - TIE_TOLERANCE
                for metric in CORE_LOWER_IS_BETTER_METRICS
            )
        beats_harmonic = row["beats_harmonic_prior_on_all_core_metrics"]
        beats_stiffness = row["beats_stiffness_structured_on_all_core_metrics"]
        if beats_harmonic and improves_lower_frontier:
            assessment = "beneficial"
        elif beats_harmonic:
            assessment = "neutral_or_saturated"
        else:
            assessment = "harmful"
        rows.append(
            {
                "source_strength": row["source_strength"],
                "assessment": assessment,
                "beats_harmonic_prior_on_all_core_metrics": beats_harmonic,
                "beats_stiffness_structured_on_all_core_metrics": beats_stiffness,
                "improves_best_lower_lambda_on_all_core_metrics": improves_lower_frontier,
                "metrics": dict(row["metrics"]),
            }
        )
    assessments = {row["assessment"] for row in rows}
    if assessments == {"beneficial"}:
        overall = "beneficial"
    elif "harmful" in assessments:
        overall = "mixed_or_harmful"
    else:
        overall = "neutral_or_saturated"
    return {
        "status": "available",
        "large_lambdas": rows,
        "best_lower_lambda_reference": lower_frontier,
        "overall_assessment": overall,
        "note": "Large lambdas are assessed against harmonic_prior and the best lower-lambda scaffold hybrid by tracked core metrics.",
    }


def _average_core_metric_rank_by_lambda(available: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranks = {id(row): 0.0 for row in available}
    for metric in CORE_LOWER_IS_BETTER_METRICS:
        sorted_rows = sorted(available, key=lambda row: float(row["metrics"][metric]))
        for rank, row in enumerate(sorted_rows, start=1):
            ranks[id(row)] += float(rank)
    return [
        {
            "source_strength": row["source_strength"],
            "average_core_metric_rank": ranks[id(row)] / len(CORE_LOWER_IS_BETTER_METRICS),
            "best_validation_loss": float(row["metrics"]["best_validation_loss"]),
            "reloaded_validation_loss": float(row["metrics"]["reloaded_validation_loss"]),
            "sampled_rmsd_to_target_overall_mean": float(row["metrics"]["sampled_rmsd_to_target_overall_mean"]),
        }
        for row in sorted(available, key=lambda item: float(item["source_strength"]))
    ]


def _nonincreasing(values: list[float]) -> bool:
    return all(next_value <= value + TIE_TOLERANCE for value, next_value in zip(values, values[1:]))


def _relative_delta(first: float, second: float) -> float:
    denominator = max(abs(first), 1.0e-12)
    return abs(second - first) / denominator


def _lambda_label(value: float) -> str:
    return f"lambda{int(round(float(value) * 100)):03d}"


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Full-Database Scaffold Hybrid Harmonic + Stiffness Lambda Sweep",
        "",
        summary["notes"][0],
        "",
        "## Scope",
        "",
        f"- Split type: `{summary['split_type']}`",
        f"- Seed: `{summary['seed']}`",
        f"- Dataset scope: `{summary['dataset_scope']}`",
        f"- Molecule count: `{summary['dataset'].get('molecule_count', 'NA')}`",
        f"- Verdict: `{summary['directional_verdict']['verdict']}`",
        f"- Decision rule: `{summary['directional_verdict']['decision_rule']}`",
        f"- Multiseed follow-up supported: `{summary['supports_multiseed_followup']}`",
        f"- Best scaffold lambda: `{_best_lambda_text(summary['best_scaffold_lambda'])}`",
        "",
        "## Coverage",
        "",
        f"- Baselines available: `{summary['coverage']['baseline_rows_available']}`",
        f"- Missing baselines: `{summary['coverage']['missing_baselines']}`",
        f"- Hybrid rows available: `{summary['coverage']['hybrid_rows_available']}`",
        f"- Missing hybrids: `{summary['coverage']['missing_hybrids']}`",
        "",
        "## Harmonic Prior Outcomes",
        "",
        f"- Any lambda beats harmonic on all tracked core metrics: `{summary['harmonic_prior_outcome_summary']['any_lambda_beats_harmonic_prior_on_all_tracked_core_metrics']}`",
        f"- Lambdas beating harmonic on all tracked core metrics: `{summary['harmonic_prior_outcome_summary']['lambdas_beating_harmonic_prior_on_all_tracked_core_metrics']}`",
        f"- Any lambda beats harmonic on best/reloaded validation loss: `{summary['harmonic_prior_outcome_summary']['any_lambda_beats_harmonic_prior_on_best_and_reloaded_validation_loss']}`",
        f"- Lambdas beating harmonic on best/reloaded validation loss: `{summary['harmonic_prior_outcome_summary']['lambdas_beating_harmonic_prior_on_best_and_reloaded_validation_loss']}`",
        f"- Any lambda beats harmonic on sampled RMSD overall/train/val/test: `{summary['harmonic_prior_outcome_summary']['any_lambda_beats_harmonic_prior_on_sampled_rmsd_overall_train_val_test']}`",
        f"- Lambdas beating harmonic on sampled RMSD overall/train/val/test: `{summary['harmonic_prior_outcome_summary']['lambdas_beating_harmonic_prior_on_sampled_rmsd_overall_train_val_test']}`",
        "",
        "## Scaffold Reference Metrics",
        "",
        "| Mode | Steps | Stop reason | Best val loss | Reloaded val loss | RMSD overall | RMSD train | RMSD val | RMSD test |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in BASELINE_MODES:
        row = summary["baseline_metric_table"].get(mode, {})
        lines.append(
            f"| {mode} | {_fmt_int(row.get('executed_train_steps'))} | {row.get('stop_reason', 'NA')} | "
            f"{_fmt(row.get('best_validation_loss'))} | {_fmt(row.get('reloaded_validation_loss'))} | "
            f"{_fmt(row.get('sampled_rmsd_to_target_overall_mean'))} | "
            f"{_fmt(row.get('sampled_rmsd_to_target_train_mean'))} | "
            f"{_fmt(row.get('sampled_rmsd_to_target_val_mean'))} | "
            f"{_fmt(row.get('sampled_rmsd_to_target_test_mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Lambda Comparisons",
            "",
            "| Lambda | Status | Steps | Stop reason | Best val loss | Reloaded val loss | RMSD overall | RMSD train | RMSD val | RMSD test | Beats harmonic | Beats stiffness | Beats isotropic | Beats torsion | Beats iso+torsion | Failure modes |",
            "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|",
        ]
    )
    for row in summary["hybrid_comparisons"]:
        metrics = row.get("metrics", {})
        lines.append(
            f"| {row['source_strength']} | {row['status']} | "
            f"{_fmt_int(row.get('executed_train_steps'))} | {row.get('stop_reason') or 'NA'} | "
            f"{_fmt(metrics.get('best_validation_loss'))} | "
            f"{_fmt(metrics.get('reloaded_validation_loss'))} | "
            f"{_fmt(metrics.get('sampled_rmsd_to_target_overall_mean'))} | "
            f"{_fmt(metrics.get('sampled_rmsd_to_target_train_mean'))} | "
            f"{_fmt(metrics.get('sampled_rmsd_to_target_val_mean'))} | "
            f"{_fmt(metrics.get('sampled_rmsd_to_target_test_mean'))} | "
            f"{row['beats_harmonic_prior_on_all_core_metrics']} | "
            f"{row['beats_stiffness_structured_on_all_core_metrics']} | "
            f"{row['beats_isotropic_on_all_core_metrics']} | "
            f"{row['beats_torsion_only_on_all_core_metrics']} | "
            f"{row['beats_isotropic_and_torsion_only_on_all_core_metrics']} | "
            f"{', '.join(row['failure_modes']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Lambda Trend",
            "",
            f"- Trend classification: `{summary['lambda_trend']['classification']}`",
            f"- Requested trend bucket: `{summary['lambda_trend']['requested_bucket']}`",
            f"- Best lambda by average core-metric rank: `{summary['lambda_trend'].get('best_source_strength_by_average_core_metric_rank')}`",
            f"- Strictly monotonic across all core metrics: `{summary['lambda_trend'].get('all_core_metrics_monotonic_improving')}`",
            f"- Monotonic core metrics: `{summary['lambda_trend'].get('monotonic_core_metric_count')}` / `{summary['lambda_trend'].get('core_metric_count')}`",
            f"- Minor nonmonotonicity detected: `{summary['lambda_trend'].get('minor_nonmonotonicity_detected')}`",
            "",
            "## Large Lambda Assessment",
            "",
            f"- Overall assessment for lambda 3.0 and 5.0: `{summary['large_lambda_assessment']['overall_assessment']}`",
            "",
            "| Lambda | Assessment | Beats harmonic | Beats stiffness | Improves best lower lambda | Best val loss | RMSD overall |",
            "|---:|---|---|---|---|---:|---:|",
        ]
    )
    for row in summary["large_lambda_assessment"].get("large_lambdas", []):
        metrics = row["metrics"]
        lines.append(
            f"| {row['source_strength']} | {row['assessment']} | "
            f"{row['beats_harmonic_prior_on_all_core_metrics']} | "
            f"{row['beats_stiffness_structured_on_all_core_metrics']} | "
            f"{row['improves_best_lower_lambda_on_all_core_metrics']} | "
            f"{_fmt(metrics.get('best_validation_loss'))} | "
            f"{_fmt(metrics.get('sampled_rmsd_to_target_overall_mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Follow-Up Decision",
            "",
            f"- Verdict: `{summary['directional_verdict']['verdict']}`",
            f"- Decision rule: `{summary['directional_verdict']['decision_rule']}`",
            f"- Multiseed follow-up supported: `{summary['supports_multiseed_followup']}`",
            f"- Recommendation: `{_recommendation_text(summary['multiseed_followup']['recommendation'])}`",
            f"- Reason: {summary['multiseed_followup']['reason']}",
            f"- Over-modulation risk flag: `{summary['directional_verdict']['over_modulation_risk_flag']}`",
            "- No scaffold or multiseed jobs were submitted by this summary.",
            "",
            "## Safety Labels",
            "",
            f"- `benchmark_result`: `{summary['benchmark_result']}`",
            f"- `training_executed_by_summary`: `{summary['training_executed_by_summary']}`",
            f"- `sampling_executed_by_summary`: `{summary['sampling_executed_by_summary']}`",
            f"- `raw_data_modified`: `{summary['raw_data_modified']}`",
            f"- `environment_modified`: `{summary['environment_modified']}`",
            f"- `gpu_used_by_summary`: `{summary['gpu_used_by_summary']}`",
            "",
            "No training, sampling, preprocessing, scheduler action, GPU use, physical benchmark job, xTB, or environment change was performed by this summary.",
            "",
        ]
    )
    return "\n".join(lines)


def _best_lambda_text(row: dict[str, Any] | None) -> str:
    if not row:
        return "None"
    return (
        f"lambda {row['source_strength']} "
        f"(average core rank {float(row['average_core_metric_rank']):.3g}, "
        f"best val {_fmt(row['metrics']['best_validation_loss'])}, "
        f"RMSD overall {_fmt(row['metrics']['sampled_rmsd_to_target_overall_mean'])})"
    )


def _recommendation_text(row: dict[str, Any] | None) -> str:
    if not row:
        return "None"
    return (
        f"{row['corruption_mode']} lambda {row['source_strength']} "
        f"for {row['recommended_scope']}"
    )
