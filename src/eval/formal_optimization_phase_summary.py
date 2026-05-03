"""Summaries for completed optimizer/LR and lambda-refinement pilot outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.data.geom_full_database import FULL_DATABASE_SCOPE
from src.train.formal_10000_training import FORMAL_10000_MAX_STEPS
from src.train.formal_full_database_training import FORMAL_FULL_DATABASE_RESULT_KIND


DEFAULT_OPTIMIZER_SUMMARY_PATH = Path(
    "experiments/results/geom_full_database_optimizer_lr_pilot_summary.json"
)
DEFAULT_OPTIMIZER_REPORT_PATH = Path(
    "reports/tables/geom_full_database_optimizer_lr_pilot_summary.md"
)
DEFAULT_LAMBDA_SUMMARY_PATH = Path(
    "experiments/results/geom_full_database_hybrid_lambda_refinement_summary.json"
)
DEFAULT_LAMBDA_REPORT_PATH = Path(
    "reports/tables/geom_full_database_hybrid_lambda_refinement_summary.md"
)
DEFAULT_DECISION_SUMMARY_PATH = Path(
    "experiments/results/geom_full_database_optimization_and_lambda_decision_summary.json"
)
DEFAULT_DECISION_REPORT_PATH = Path(
    "reports/tables/geom_full_database_optimization_and_lambda_decision_summary.md"
)
DEFAULT_DIAGNOSTICS_PATH = Path(
    "experiments/results/geom_full_database_optimization_diagnostics.json"
)

OPTIMIZER_SUMMARY_RESULT_KIND = "geom_drugs_full_database_optimizer_lr_pilot_summary_not_benchmark"
LAMBDA_SUMMARY_RESULT_KIND = "geom_drugs_full_database_hybrid_lambda_refinement_summary_not_benchmark"
DECISION_SUMMARY_RESULT_KIND = (
    "geom_drugs_full_database_optimization_and_lambda_decision_summary_not_benchmark"
)

OPTIMIZER_METHODS = (
    "harmonic_prior",
    "hybrid_harmonic_stiffness_v1_lambda500",
)
METHOD_LABELS = {
    "harmonic_prior": "harmonic",
    "hybrid_harmonic_stiffness_v1_lambda500": "hybrid_harmonic_stiffness_v1_lambda500",
}
OPTIMIZER_SETTINGS = (
    "sgd_lr1e2",
    "sgd_lr3e3",
    "sgd_lr1e3",
    "adamw_lr1e3",
    "adamw_lr3e4",
    "adamw_lr1e4",
    "sgd_lr1e2_plateau",
)
SETTING_DETAILS = {
    "sgd_lr1e2": {"optimizer": "SGD", "learning_rate": 1.0e-2, "scheduler": None},
    "sgd_lr3e3": {"optimizer": "SGD", "learning_rate": 3.0e-3, "scheduler": None},
    "sgd_lr1e3": {"optimizer": "SGD", "learning_rate": 1.0e-3, "scheduler": None},
    "adamw_lr1e3": {"optimizer": "AdamW", "learning_rate": 1.0e-3, "scheduler": None},
    "adamw_lr3e4": {"optimizer": "AdamW", "learning_rate": 3.0e-4, "scheduler": None},
    "adamw_lr1e4": {"optimizer": "AdamW", "learning_rate": 1.0e-4, "scheduler": None},
    "sgd_lr1e2_plateau": {
        "optimizer": "SGD",
        "learning_rate": 1.0e-2,
        "scheduler": "ReduceLROnPlateau",
    },
}

LAMBDA_VALUES = (2.0, 3.0, 5.0, 7.5, 10.0, 20.0, 50.0)
CORE_LOWER_IS_BETTER_METRICS = (
    "best_validation_loss",
    "reloaded_validation_loss",
    "sampled_rmsd_to_target_overall_mean",
    "sampled_rmsd_to_target_train_mean",
    "sampled_rmsd_to_target_val_mean",
    "sampled_rmsd_to_target_test_mean",
)
TIE_TOLERANCE = 1.0e-12


def write_optimization_phase_summaries(
    results_root: str | Path = "experiments/results",
    diagnostics_path: str | Path = DEFAULT_DIAGNOSTICS_PATH,
    optimizer_summary_path: str | Path = DEFAULT_OPTIMIZER_SUMMARY_PATH,
    optimizer_report_path: str | Path = DEFAULT_OPTIMIZER_REPORT_PATH,
    lambda_summary_path: str | Path = DEFAULT_LAMBDA_SUMMARY_PATH,
    lambda_report_path: str | Path = DEFAULT_LAMBDA_REPORT_PATH,
    decision_summary_path: str | Path = DEFAULT_DECISION_SUMMARY_PATH,
    decision_report_path: str | Path = DEFAULT_DECISION_REPORT_PATH,
) -> dict[str, Path]:
    results_root = Path(results_root)
    diagnostics = _load_diagnostics(Path(diagnostics_path))
    optimizer_summary = _build_optimizer_summary(results_root=results_root, diagnostics=diagnostics)
    lambda_summary = _build_lambda_summary(results_root=results_root, diagnostics=diagnostics)
    decision_summary = _build_decision_summary(
        optimizer_summary=optimizer_summary,
        lambda_summary=lambda_summary,
        diagnostics=diagnostics,
    )

    outputs = {
        "optimizer_summary": Path(optimizer_summary_path),
        "optimizer_report": Path(optimizer_report_path),
        "lambda_summary": Path(lambda_summary_path),
        "lambda_report": Path(lambda_report_path),
        "decision_summary": Path(decision_summary_path),
        "decision_report": Path(decision_report_path),
    }
    _write_json(outputs["optimizer_summary"], optimizer_summary)
    _write_text(outputs["optimizer_report"], _optimizer_markdown(optimizer_summary))
    _write_json(outputs["lambda_summary"], lambda_summary)
    _write_text(outputs["lambda_report"], _lambda_markdown(lambda_summary))
    _write_json(outputs["decision_summary"], decision_summary)
    _write_text(outputs["decision_report"], _decision_markdown(decision_summary))
    return outputs


def _build_optimizer_summary(results_root: Path, diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    rows, missing = _load_optimizer_rows(results_root)
    rows_by_method = {
        method: [row for row in rows if row["method"] == method]
        for method in OPTIMIZER_METHODS
    }
    rows_by_method_setting = {
        f"{row['method']}:{row['setting']}": row
        for row in rows
    }
    best_by_method = {
        method: _best_setting_summary(rows_by_method[method])
        for method in OPTIMIZER_METHODS
        if rows_by_method[method]
    }
    shared_setting_comparisons = _shared_setting_comparisons(rows_by_method_setting)
    best_shared_setting = _best_shared_setting(rows_by_method)
    decisions = _optimizer_decisions(
        rows_by_method_setting=rows_by_method_setting,
        best_by_method=best_by_method,
        best_shared_setting=best_shared_setting,
        shared_setting_comparisons=shared_setting_comparisons,
    )
    return {
        "summary_name": "geom_full_database_optimizer_lr_pilot_summary",
        "result_kind": OPTIMIZER_SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_reporting_only_optimizer_lr_pilot_summary",
        "hardware": "reporting_only",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "dataset_scope": FULL_DATABASE_SCOPE,
        "split_type": "random",
        "seed": 0,
        "methods": list(OPTIMIZER_METHODS),
        "settings": list(OPTIMIZER_SETTINGS),
        "coverage": {
            "expected_result_count": len(OPTIMIZER_METHODS) * len(OPTIMIZER_SETTINGS),
            "available_result_count": len(rows),
            "missing_result_count": len(missing),
            "missing_results": missing,
        },
        "diagnostics_context": _diagnostics_context(diagnostics),
        "rows": rows,
        "best_by_method": best_by_method,
        "shared_setting_comparisons": shared_setting_comparisons,
        "best_shared_setting": best_shared_setting,
        "decisions": decisions,
        "safety_labels": _safety_labels(),
        "notes": [
            "Reporting-only optimizer/LR pilot summary. This is not benchmark evaluation.",
            "The summary reads completed random-split seed0 result JSONs for harmonic_prior and hybrid_harmonic_stiffness_v1 lambda 5.0.",
            "No training, sampling, preprocessing, scheduler submission, GPU use, benchmark job, xTB, raw-data change, or environment change is performed by this summary.",
        ],
    }


def _build_lambda_summary(results_root: Path, diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    rows, missing = _load_lambda_rows(results_root)
    best_by_metric = {
        metric: _row_reference(_best_row(rows, metric))
        for metric in CORE_LOWER_IS_BETTER_METRICS
        if _best_row(rows, metric) is not None
    }
    best_overall = _best_overall_by_core_rank(rows)
    trend = _lambda_trend(rows)
    decisions = _lambda_decisions(rows=rows, best_by_metric=best_by_metric, trend=trend, best_overall=best_overall)
    return {
        "summary_name": "geom_full_database_hybrid_lambda_refinement_summary",
        "result_kind": LAMBDA_SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_reporting_only_hybrid_lambda_refinement_summary",
        "hardware": "reporting_only",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "dataset_scope": FULL_DATABASE_SCOPE,
        "split_type": "random",
        "seed": 0,
        "method": "hybrid_harmonic_stiffness_v1",
        "lambdas": list(LAMBDA_VALUES),
        "coverage": {
            "expected_result_count": len(LAMBDA_VALUES),
            "available_result_count": len(rows),
            "missing_result_count": len(missing),
            "missing_results": missing,
        },
        "diagnostics_context": _diagnostics_context(diagnostics),
        "rows": rows,
        "best_by_metric": best_by_metric,
        "best_overall_by_core_rank": best_overall,
        "lambda_trend": trend,
        "decisions": decisions,
        "safety_labels": _safety_labels(),
        "notes": [
            "Reporting-only hybrid lambda-refinement summary. This is not benchmark evaluation.",
            "The summary reads completed random-split seed0 result JSONs for hybrid_harmonic_stiffness_v1 lambdas 2.0 through 50.0.",
            "No residual variants, training, sampling, preprocessing, scheduler submission, GPU use, benchmark job, xTB, raw-data change, or environment change is performed by this summary.",
        ],
    }


def _build_decision_summary(
    optimizer_summary: dict[str, Any],
    lambda_summary: dict[str, Any],
    diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    optimizer_decisions = optimizer_summary["decisions"]
    lambda_decisions = lambda_summary["decisions"]
    if optimizer_decisions["hybrid_gain_robust_best_per_method"] is True:
        optimizer_verdict = "hybrid_gain_robust_to_optimizer_lr_changes"
    elif optimizer_decisions["optimized_harmonic_catches_up_or_beats_hybrid"] is True:
        optimizer_verdict = "previous_hybrid_advantage_optimizer_regime_dependent"
    else:
        optimizer_verdict = "optimizer_lr_effect_mixed_or_incomplete"

    if lambda_decisions["lambda_above_5_improves_without_clear_degradation"] is True:
        lambda_verdict = "lambda_above_5_supports_minimal_scaffold_followup"
    elif lambda_decisions["high_lambdas_20_or_50_degrade"] is True:
        lambda_verdict = "lambda_above_5_degrades_keep_lambda5"
    elif lambda_decisions["lambda5_remains_best_choice"] is True:
        lambda_verdict = "lambda5_remains_current_best_robust_setting"
    else:
        lambda_verdict = "lambda_refinement_inconclusive_or_missing"

    if optimizer_verdict == "hybrid_gain_robust_to_optimizer_lr_changes":
        benchmark_gate = "optimizer_phase_supports_next_physical_validation_after_user_approval"
    else:
        benchmark_gate = "do_not_start_benchmark_validation_until_optimizer_dependency_is_resolved"

    final_recommendation = _final_recommendation(
        optimizer_decisions=optimizer_decisions,
        lambda_decisions=lambda_decisions,
    )
    return {
        "summary_name": "geom_full_database_optimization_and_lambda_decision_summary",
        "result_kind": DECISION_SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_reporting_only_optimization_and_lambda_decision_summary",
        "hardware": "reporting_only",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "diagnostics_context": _diagnostics_context(diagnostics),
        "optimizer_lr_pilot": {
            "coverage": optimizer_summary["coverage"],
            "best_by_method": optimizer_summary["best_by_method"],
            "best_shared_setting": optimizer_summary["best_shared_setting"],
            "decisions": optimizer_decisions,
        },
        "lambda_refinement": {
            "coverage": lambda_summary["coverage"],
            "best_by_metric": lambda_summary["best_by_metric"],
            "best_overall_by_core_rank": lambda_summary["best_overall_by_core_rank"],
            "lambda_trend": lambda_summary["lambda_trend"],
            "decisions": lambda_decisions,
        },
        "decision_verdicts": {
            "optimizer_lr_verdict": optimizer_verdict,
            "lambda_verdict": lambda_verdict,
            "benchmark_validation_gate": benchmark_gate,
            "final_recommendation": final_recommendation,
            "requires_user_approval_before_any_scheduler_submission": True,
        },
        "safety_labels": _safety_labels(),
        "notes": [
            "Decision summary for completed optimization-prep phase only. This is not benchmark evaluation.",
            "The summary does not start benchmark validation, submit jobs, rerun training, modify raw data, or modify the environment.",
        ],
    }


def _load_optimizer_rows(results_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    missing = []
    for method in OPTIMIZER_METHODS:
        label = METHOD_LABELS[method]
        for setting in OPTIMIZER_SETTINGS:
            path = results_root / (
                f"geom_formal_full_database_train_{label}_steps{FORMAL_10000_MAX_STEPS}_seed0_opt_{setting}.json"
            )
            if not path.exists():
                missing.append({"method": method, "setting": setting, "expected_path": str(path)})
                continue
            payload = _load_formal_result(path)
            rows.append(_row_from_result(path=path, payload=payload, method=method, setting=setting))
    return rows, missing


def _load_lambda_rows(results_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    missing = []
    for value in LAMBDA_VALUES:
        label = _lambda_label(value)
        path = results_root / (
            f"geom_formal_full_database_train_hybrid_harmonic_stiffness_v1_{label}"
            f"_steps{FORMAL_10000_MAX_STEPS}_seed0_lambda_refinement.json"
        )
        if not path.exists():
            missing.append({"lambda": value, "lambda_label": label, "expected_path": str(path)})
            continue
        payload = _load_formal_result(path)
        rows.append(
            _row_from_result(
                path=path,
                payload=payload,
                method="hybrid_harmonic_stiffness_v1",
                setting=None,
                lambda_value=value,
            )
        )
    return rows, missing


def _load_formal_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("result_kind") != FORMAL_FULL_DATABASE_RESULT_KIND:
        raise ValueError(f"Not a full-database formal result JSON: {path}")
    if payload.get("benchmark_result") is not False:
        raise ValueError(f"Result must be benchmark_result=false: {path}")
    if payload.get("dataset_subset_scope") != FULL_DATABASE_SCOPE:
        raise ValueError(f"Unexpected dataset scope in {path}")
    split_type = payload.get("split_type") or (payload.get("dataset") or {}).get("split_type")
    if split_type != "random":
        raise ValueError(f"Optimization-prep summaries use only random split seed0 outputs: {path}")
    if int(payload.get("random_seed")) != 0:
        raise ValueError(f"Optimization-prep summaries use only seed0 outputs: {path}")
    metrics = payload.get("metrics") or {}
    for key in ["best_validation_loss", "reloaded_validation_loss", "sampled_molecules"]:
        if key not in metrics:
            raise ValueError(f"Result metrics missing {key}: {path}")
    return payload


def _row_from_result(
    path: Path,
    payload: dict[str, Any],
    method: str,
    setting: str | None,
    lambda_value: float | None = None,
) -> dict[str, Any]:
    metrics = payload["metrics"]
    sampled_metrics = _sampled_metrics(metrics["sampled_molecules"])
    train_steps = metrics.get("train_steps") or []
    row = {
        "method": method,
        "setting": setting,
        "lambda": lambda_value,
        "run_name": payload["run_name"],
        "source_path": str(path),
        "corruption_mode": payload["corruption_mode"],
        "hybrid_source_strength": payload.get("hybrid_source_strength"),
        "optimizer_name": payload.get("optimizer_name"),
        "learning_rate": float(payload.get("learning_rate")),
        "optimizer_weight_decay": float(payload.get("optimizer_weight_decay", 0.0)),
        "lr_scheduler_name": payload.get("lr_scheduler_name"),
        "final_learning_rate": payload.get("final_learning_rate"),
        "executed_train_steps": int(metrics["executed_train_steps"]),
        "stop_reason": str(metrics.get("stop_reason") or payload.get("early_stop_reason")),
        "hit_hard_cap": int(metrics["executed_train_steps"]) >= int(payload["hard_max_steps"]),
        "best_validation_loss": float(metrics["best_validation_loss"]),
        "best_validation_loss_step": int(metrics["best_validation_loss_step"]),
        "final_validation_loss": float(metrics["final_validation_loss"]),
        "reloaded_validation_loss": float(metrics["reloaded_validation_loss"]),
        "sample_count": int(payload["sample_count"]),
        **sampled_metrics,
    }
    if train_steps:
        row["last_100_validation_delta"] = _window_delta(train_steps, "validation_loss_after_step", 100)
        row["last_500_validation_delta"] = _window_delta(train_steps, "validation_loss_after_step", 500)
    return row


def _sampled_metrics(sampled: list[dict[str, Any]]) -> dict[str, Any]:
    if not sampled:
        raise ValueError("sampled_molecules must be non-empty")
    rmsd_values = [float(item["sampled_rmsd_to_target"]) for item in sampled]
    by_split = {
        split: _mean([float(item["sampled_rmsd_to_target"]) for item in sampled if item.get("split") == split])
        for split in ("train", "val", "test")
    }
    bond_values = [
        float(item["sampled_mean_abs_bond_length_delta"])
        for item in sampled
        if isinstance(item.get("sampled_mean_abs_bond_length_delta"), (int, float))
    ]
    return {
        "sampled_rmsd_to_target_overall_mean": _mean(rmsd_values),
        "sampled_rmsd_to_target_train_mean": by_split["train"],
        "sampled_rmsd_to_target_val_mean": by_split["val"],
        "sampled_rmsd_to_target_test_mean": by_split["test"],
        "sampled_mean_abs_bond_length_delta": _mean(bond_values),
    }


def _window_delta(train_steps: list[dict[str, Any]], field: str, window_size: int) -> float | None:
    values = [float(row[field]) for row in train_steps[-window_size:] if field in row]
    if len(values) < 2:
        return None
    return values[-1] - values[0]


def _best_setting_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_by_metric = {
        metric: _row_reference(_best_row(rows, metric))
        for metric in CORE_LOWER_IS_BETTER_METRICS
    }
    return {
        "best_by_metric": best_by_metric,
        "best_overall_by_core_rank": _best_overall_by_core_rank(rows),
        "current_sgd_lr1e2": _row_reference(next((row for row in rows if row.get("setting") == "sgd_lr1e2"), None)),
        "settings_ranked_by_core": _rank_rows(rows),
    }


def _best_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if isinstance(row.get(metric), (int, float))]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (float(row[metric]), row.get("setting") or "", row.get("lambda") or 0.0))


def _best_overall_by_core_rank(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ranked = _rank_rows(rows)
    if not ranked:
        return None
    return ranked[0]


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    scores = {id(row): 0.0 for row in rows}
    counts = {id(row): 0 for row in rows}
    for metric in CORE_LOWER_IS_BETTER_METRICS:
        ranked = sorted(rows, key=lambda row: float(row[metric]))
        for rank, row in enumerate(ranked, start=1):
            scores[id(row)] += float(rank)
            counts[id(row)] += 1
    ranked_refs = []
    for row in rows:
        average_rank = scores[id(row)] / counts[id(row)]
        ranked_refs.append(
            {
                **_row_reference(row),
                "average_core_metric_rank": average_rank,
            }
        )
    return sorted(
        ranked_refs,
        key=lambda item: (
            item["average_core_metric_rank"],
            item["metrics"]["best_validation_loss"],
            item["metrics"]["sampled_rmsd_to_target_overall_mean"],
        ),
    )


def _shared_setting_comparisons(rows_by_method_setting: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    for setting in OPTIMIZER_SETTINGS:
        harmonic = rows_by_method_setting.get(f"harmonic_prior:{setting}")
        hybrid = rows_by_method_setting.get(f"hybrid_harmonic_stiffness_v1_lambda500:{setting}")
        if harmonic is None or hybrid is None:
            comparisons.append({"setting": setting, "available": False})
            continue
        comparison = _compare_rows(
            candidate=hybrid,
            reference=harmonic,
            candidate_label="hybrid_harmonic_stiffness_v1_lambda500",
            reference_label="harmonic_prior",
        )
        comparisons.append({"setting": setting, "available": True, **comparison})
    return comparisons


def _best_shared_setting(rows_by_method: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    if any(not rows_by_method.get(method) for method in OPTIMIZER_METHODS):
        return None
    method_rank_maps: dict[str, dict[str, float]] = {}
    for method, rows in rows_by_method.items():
        method_rank_maps[method] = {
            str(item["setting"]): float(item["average_core_metric_rank"])
            for item in _rank_rows(rows)
        }
    available_settings = [
        setting
        for setting in OPTIMIZER_SETTINGS
        if all(setting in method_rank_maps[method] for method in OPTIMIZER_METHODS)
    ]
    if not available_settings:
        return None
    ranked = []
    for setting in available_settings:
        score = sum(method_rank_maps[method][setting] for method in OPTIMIZER_METHODS) / len(OPTIMIZER_METHODS)
        ranked.append(
            {
                "setting": setting,
                "setting_details": SETTING_DETAILS[setting],
                "mean_method_core_rank": score,
                "method_core_ranks": {
                    method: method_rank_maps[method][setting]
                    for method in OPTIMIZER_METHODS
                },
            }
        )
    ranked.sort(key=lambda item: (item["mean_method_core_rank"], OPTIMIZER_SETTINGS.index(item["setting"])))
    return ranked[0]


def _optimizer_decisions(
    rows_by_method_setting: dict[str, dict[str, Any]],
    best_by_method: dict[str, dict[str, Any]],
    best_shared_setting: dict[str, Any] | None,
    shared_setting_comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    harmonic_current = rows_by_method_setting.get("harmonic_prior:sgd_lr1e2")
    hybrid_current = rows_by_method_setting.get("hybrid_harmonic_stiffness_v1_lambda500:sgd_lr1e2")
    harmonic_best_ref = (best_by_method.get("harmonic_prior") or {}).get("best_overall_by_core_rank")
    hybrid_best_ref = (best_by_method.get("hybrid_harmonic_stiffness_v1_lambda500") or {}).get(
        "best_overall_by_core_rank"
    )
    harmonic_best = _row_from_reference(harmonic_best_ref, rows_by_method_setting)
    hybrid_best = _row_from_reference(hybrid_best_ref, rows_by_method_setting)
    harmonic_improvement = (
        _compare_rows(candidate=harmonic_best, reference=harmonic_current, candidate_label="harmonic_best", reference_label="harmonic_sgd_lr1e2")
        if harmonic_best is not None and harmonic_current is not None
        else None
    )
    hybrid_improvement = (
        _compare_rows(candidate=hybrid_best, reference=hybrid_current, candidate_label="hybrid_best", reference_label="hybrid_sgd_lr1e2")
        if hybrid_best is not None and hybrid_current is not None
        else None
    )
    best_per_method_comparison = (
        _compare_rows(candidate=hybrid_best, reference=harmonic_best, candidate_label="hybrid_best", reference_label="harmonic_best")
        if hybrid_best is not None and harmonic_best is not None
        else None
    )
    shared_comparison = None
    if best_shared_setting is not None:
        setting = best_shared_setting["setting"]
        shared_comparison = next((row for row in shared_setting_comparisons if row["setting"] == setting), None)
    current_shared_comparison = next(
        (row for row in shared_setting_comparisons if row["setting"] == "sgd_lr1e2"),
        None,
    )
    scheduler_comparison = _scheduler_comparison(rows_by_method_setting)
    adamw_best_methods = [
        method
        for method, summary in best_by_method.items()
        if (summary.get("best_overall_by_core_rank") or {}).get("optimizer_name") == "AdamW"
    ]
    return {
        "best_setting_for_harmonic_prior": harmonic_best_ref,
        "best_setting_for_hybrid_harmonic_stiffness_v1_lambda500": hybrid_best_ref,
        "harmonic_best_vs_current_sgd_lr1e2": harmonic_improvement,
        "hybrid_best_vs_current_sgd_lr1e2": hybrid_improvement,
        "best_per_method_hybrid_vs_harmonic": best_per_method_comparison,
        "best_shared_setting": best_shared_setting,
        "best_shared_setting_hybrid_vs_harmonic": shared_comparison,
        "current_sgd_lr1e2_hybrid_vs_harmonic": current_shared_comparison,
        "optimized_harmonic_catches_up_or_beats_hybrid": (
            None
            if best_per_method_comparison is None
            else _reference_better_or_tied_on_all_core_metrics(best_per_method_comparison)
        ),
        "hybrid_gain_robust_best_per_method": (
            None if best_per_method_comparison is None else best_per_method_comparison["candidate_better_on_all_core_metrics"]
        ),
        "hybrid_gain_robust_best_shared": (
            None if shared_comparison is None else shared_comparison.get("candidate_better_on_all_core_metrics")
        ),
        "sgd_lr1e2_should_be_replaced": _should_replace_sgd_lr1e2(
            harmonic_best_ref=harmonic_best_ref,
            hybrid_best_ref=hybrid_best_ref,
            best_shared_setting=best_shared_setting,
        ),
        "adamw_appears_beneficial": bool(adamw_best_methods),
        "adamw_best_methods": adamw_best_methods,
        "adam_baseline_tested": False,
        "scheduler_variant_appears_beneficial": scheduler_comparison["scheduler_beneficial_for_any_method"],
        "scheduler_comparison": scheduler_comparison,
        "interpretation": _optimizer_interpretation(
            harmonic_best_ref=harmonic_best_ref,
            hybrid_best_ref=hybrid_best_ref,
            best_per_method_comparison=best_per_method_comparison,
            scheduler_comparison=scheduler_comparison,
        ),
    }


def _scheduler_comparison(rows_by_method_setting: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_method = {}
    beneficial = False
    for method in OPTIMIZER_METHODS:
        base = rows_by_method_setting.get(f"{method}:sgd_lr1e2")
        plateau = rows_by_method_setting.get(f"{method}:sgd_lr1e2_plateau")
        if base is None or plateau is None:
            per_method[method] = {"available": False}
            continue
        comparison = _compare_rows(
            candidate=plateau,
            reference=base,
            candidate_label=f"{method}_sgd_lr1e2_plateau",
            reference_label=f"{method}_sgd_lr1e2",
        )
        per_method[method] = {"available": True, **comparison}
        beneficial = beneficial or comparison["candidate_better_on_any_core_metric"]
    return {
        "scheduler": "ReduceLROnPlateau",
        "per_method": per_method,
        "scheduler_beneficial_for_any_method": beneficial,
    }


def _should_replace_sgd_lr1e2(
    harmonic_best_ref: dict[str, Any] | None,
    hybrid_best_ref: dict[str, Any] | None,
    best_shared_setting: dict[str, Any] | None,
) -> dict[str, Any]:
    harmonic_setting = None if harmonic_best_ref is None else harmonic_best_ref.get("setting")
    hybrid_setting = None if hybrid_best_ref is None else hybrid_best_ref.get("setting")
    shared_setting = None if best_shared_setting is None else best_shared_setting.get("setting")
    if harmonic_setting == "sgd_lr1e2" and hybrid_setting == "sgd_lr1e2":
        status = "keep_current_sgd_lr1e2"
        replace = False
    elif harmonic_setting == hybrid_setting and harmonic_setting is not None:
        status = "replace_with_shared_best_setting"
        replace = harmonic_setting != "sgd_lr1e2"
    elif shared_setting and shared_setting != "sgd_lr1e2":
        status = "replace_for_shared_followup_with_best_shared_setting"
        replace = True
    else:
        status = "method_dependent_do_not_replace_blindly"
        replace = None
    return {
        "status": status,
        "replace": replace,
        "harmonic_best_setting": harmonic_setting,
        "hybrid_best_setting": hybrid_setting,
        "best_shared_setting": shared_setting,
    }


def _optimizer_interpretation(
    harmonic_best_ref: dict[str, Any] | None,
    hybrid_best_ref: dict[str, Any] | None,
    best_per_method_comparison: dict[str, Any] | None,
    scheduler_comparison: dict[str, Any],
) -> str:
    if harmonic_best_ref is None or hybrid_best_ref is None or best_per_method_comparison is None:
        return "missing_results_prevent_optimizer_decision"
    if best_per_method_comparison["candidate_better_on_all_core_metrics"]:
        if harmonic_best_ref.get("setting") == hybrid_best_ref.get("setting"):
            return "hybrid_remains_better_under_same_best_optimizer_lr_setting"
        return "hybrid_remains_better_under_best_per_method_optimizer_lr_settings"
    if _reference_better_or_tied_on_all_core_metrics(best_per_method_comparison):
        return "optimized_harmonic_catches_up_or_beats_hybrid"
    if scheduler_comparison["scheduler_beneficial_for_any_method"]:
        return "scheduler_helped_some_metric_but_hybrid_harmonic_ordering_is_mixed"
    return "optimizer_lr_changes_do_not_support_reversing_hybrid_advantage"


def _lambda_decisions(
    rows: list[dict[str, Any]],
    best_by_metric: dict[str, dict[str, Any]],
    trend: dict[str, Any],
    best_overall: dict[str, Any] | None,
) -> dict[str, Any]:
    lambda5 = next((row for row in rows if row.get("lambda") == 5.0), None)
    high_rows = [row for row in rows if row.get("lambda") in {20.0, 50.0}]
    high_degradation = []
    for row in high_rows:
        if lambda5 is None:
            continue
        comparison = _compare_rows(
            candidate=row,
            reference=lambda5,
            candidate_label=f"lambda{row['lambda']}",
            reference_label="lambda5.0",
        )
        high_degradation.append(comparison)
    high_lambdas_degrade = any(_reference_better_on_any_core_metric(row) for row in high_degradation)
    high_lambdas_improve = any(row["candidate_better_on_any_core_metric"] for row in high_degradation)
    best_lambdas = {
        metric: item.get("lambda")
        for metric, item in best_by_metric.items()
    }
    best_values = {value for value in best_lambdas.values() if value is not None}
    lambda5_best_or_tied = bool(best_values) and all(float(value) >= 5.0 for value in best_values)
    lambda_above5_strict_best = any(float(value) > 5.0 for value in best_values)
    return {
        "best_lambda_by_best_validation_loss": best_lambdas.get("best_validation_loss"),
        "best_lambda_by_reloaded_validation_loss": best_lambdas.get("reloaded_validation_loss"),
        "best_lambda_by_sampled_rmsd_overall": best_lambdas.get("sampled_rmsd_to_target_overall_mean"),
        "best_overall_by_core_rank": best_overall,
        "high_lambda_20_or_50_comparisons_vs_lambda5": high_degradation,
        "high_lambdas_20_or_50_degrade": high_lambdas_degrade,
        "high_lambdas_20_or_50_improve_any_core_metric": high_lambdas_improve,
        "lambda_above_5_improves_without_clear_degradation": lambda_above5_strict_best and not high_lambdas_degrade,
        "lambda5_remains_best_choice": lambda5_best_or_tied and not lambda_above5_strict_best,
        "recommended_lambda": (
            5.0
            if lambda5_best_or_tied and not lambda_above5_strict_best
            else (None if best_overall is None else best_overall.get("lambda"))
        ),
        "interpretation": _lambda_interpretation(
            trend=trend,
            high_lambdas_degrade=high_lambdas_degrade,
            high_lambdas_improve=high_lambdas_improve,
            lambda_above5_strict_best=lambda_above5_strict_best,
        ),
    }


def _lambda_interpretation(
    trend: dict[str, Any],
    high_lambdas_degrade: bool,
    high_lambdas_improve: bool,
    lambda_above5_strict_best: bool,
) -> str:
    if high_lambdas_degrade:
        return "high_lambdas_show_over_modulation_keep_lambda5"
    if trend["overall_trend"] == "saturated_at_or_above_lambda5" and not lambda_above5_strict_best:
        return "lambda5_is_first_point_on_observed_saturation_plateau"
    if lambda_above5_strict_best and not high_lambdas_degrade:
        return "lambda_above5_improves_without_clear_degradation"
    if high_lambdas_improve:
        return "high_lambdas_improve_some_metrics_but_not_all"
    return "lambda_trend_mixed_or_inconclusive"


def _lambda_trend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_rows = sorted(rows, key=lambda row: float(row["lambda"]))
    per_metric = {
        metric: _metric_trend(sorted_rows, metric)
        for metric in CORE_LOWER_IS_BETTER_METRICS
    }
    if not rows:
        overall = "missing"
    elif all(item["trend"] == "saturated_at_or_above_lambda5" for item in per_metric.values()):
        overall = "saturated_at_or_above_lambda5"
    elif any(item["trend"] == "over_modulated_at_high_lambda" for item in per_metric.values()):
        overall = "over_modulated_at_high_lambda"
    elif all(item["trend"] in {"monotonic_improving", "saturated_at_or_above_lambda5"} for item in per_metric.values()):
        overall = "monotonic_or_saturated_improving"
    elif any(item["trend"] == "peaked" for item in per_metric.values()):
        overall = "peaked"
    else:
        overall = "mixed"
    return {
        "overall_trend": overall,
        "per_metric": per_metric,
    }


def _metric_trend(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    if not rows:
        return {"metric": metric, "trend": "missing", "values": []}
    values = [{"lambda": row["lambda"], "value": row[metric]} for row in rows]
    numeric = [float(item["value"]) for item in values]
    lambdas = [float(item["lambda"]) for item in values]
    lambda5_index = lambdas.index(5.0) if 5.0 in lambdas else None
    best_value = min(numeric)
    best_lambdas = [lambdas[index] for index, value in enumerate(numeric) if _tied(value, best_value)]
    if lambda5_index is not None and all(_tied(value, numeric[lambda5_index]) for value in numeric[lambda5_index:]):
        trend = "saturated_at_or_above_lambda5"
    elif all(numeric[index + 1] <= numeric[index] + TIE_TOLERANCE for index in range(len(numeric) - 1)):
        trend = "monotonic_improving"
    elif all(numeric[index + 1] >= numeric[index] - TIE_TOLERANCE for index in range(len(numeric) - 1)):
        trend = "monotonic_degrading"
    elif min(best_lambdas) not in {lambdas[0], lambdas[-1]}:
        trend = "peaked"
    else:
        trend = "mixed"
    if lambda5_index is not None:
        high_values = [numeric[index] for index, value in enumerate(lambdas) if value in {20.0, 50.0}]
        if any(value > numeric[lambda5_index] + TIE_TOLERANCE for value in high_values):
            trend = "over_modulated_at_high_lambda"
    return {
        "metric": metric,
        "trend": trend,
        "values": values,
        "best_value": best_value,
        "best_lambdas": best_lambdas,
    }


def _compare_rows(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    candidate_label: str,
    reference_label: str,
) -> dict[str, Any]:
    metrics = {}
    counts = {"better": 0, "tied": 0, "worse": 0}
    for metric in CORE_LOWER_IS_BETTER_METRICS:
        candidate_value = float(candidate[metric])
        reference_value = float(reference[metric])
        outcome = _outcome(candidate_value, reference_value)
        counts[outcome] += 1
        metrics[metric] = {
            "candidate": candidate_value,
            "reference": reference_value,
            "delta_candidate_minus_reference": candidate_value - reference_value,
            "outcome": outcome,
        }
    return {
        "candidate_label": candidate_label,
        "reference_label": reference_label,
        "candidate": _row_reference(candidate),
        "reference": _row_reference(reference),
        "metric_comparisons": metrics,
        "counts": counts,
        "candidate_better_on_all_core_metrics": counts["better"] == len(CORE_LOWER_IS_BETTER_METRICS),
        "candidate_better_or_tied_on_all_core_metrics": counts["worse"] == 0,
        "candidate_better_on_any_core_metric": counts["better"] > 0,
    }


def _reference_better_or_tied_on_all_core_metrics(comparison: dict[str, Any]) -> bool:
    return comparison["counts"]["better"] == 0


def _reference_better_on_any_core_metric(comparison: dict[str, Any]) -> bool:
    return comparison["counts"]["better"] == 0 and comparison["counts"]["worse"] > 0


def _outcome(candidate_value: float, reference_value: float) -> str:
    if candidate_value < reference_value - TIE_TOLERANCE:
        return "better"
    if candidate_value > reference_value + TIE_TOLERANCE:
        return "worse"
    return "tied"


def _row_reference(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "method": row.get("method"),
        "setting": row.get("setting"),
        "lambda": row.get("lambda"),
        "run_name": row.get("run_name"),
        "source_path": row.get("source_path"),
        "optimizer_name": row.get("optimizer_name"),
        "learning_rate": row.get("learning_rate"),
        "lr_scheduler_name": row.get("lr_scheduler_name"),
        "final_learning_rate": row.get("final_learning_rate"),
        "executed_train_steps": row.get("executed_train_steps"),
        "stop_reason": row.get("stop_reason"),
        "metrics": {metric: row.get(metric) for metric in CORE_LOWER_IS_BETTER_METRICS},
    }


def _row_from_reference(
    reference: dict[str, Any] | None,
    rows_by_method_setting: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if reference is None:
        return None
    key = f"{reference['method']}:{reference['setting']}"
    return rows_by_method_setting.get(key)


def _load_diagnostics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _diagnostics_context(diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    if diagnostics is None:
        return {"available": False, "source_path": str(DEFAULT_DIAGNOSTICS_PATH)}
    return {
        "available": True,
        "source_path": str(DEFAULT_DIAGNOSTICS_PATH),
        "overall_verdict": (diagnostics.get("overall_assessment") or {}).get("verdict"),
        "coverage": diagnostics.get("coverage"),
    }


def _final_recommendation(
    optimizer_decisions: dict[str, Any],
    lambda_decisions: dict[str, Any],
) -> dict[str, Any]:
    optimizer_setting = None
    replace = optimizer_decisions["sgd_lr1e2_should_be_replaced"]
    if replace["replace"] is True:
        optimizer_setting = replace["best_shared_setting"]
    recommended_lambda = lambda_decisions["recommended_lambda"]
    if optimizer_decisions["hybrid_gain_robust_best_per_method"] is True:
        source_recommendation = "hybrid_harmonic_stiffness_v1"
    elif optimizer_decisions["optimized_harmonic_catches_up_or_beats_hybrid"] is True:
        source_recommendation = "do_not_claim_hybrid_until_optimizer_dependency_resolved"
    else:
        source_recommendation = "inconclusive"
    return {
        "source_recommendation": source_recommendation,
        "recommended_lambda": recommended_lambda,
        "recommended_optimizer_setting_for_minimal_followup": optimizer_setting,
        "recommend_rerun_only_final_hybrid_and_harmonic_under_better_optimizer": optimizer_setting is not None,
        "recommend_minimal_scaffold_followup_for_lambda_above5": (
            lambda_decisions["lambda_above_5_improves_without_clear_degradation"] is True
        ),
        "submit_jobs_automatically": False,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _optimizer_markdown(summary: dict[str, Any]) -> str:
    decisions = summary["decisions"]
    lines = [
        "# Full-Database Optimizer/LR Pilot Summary",
        "",
        "Reporting-only summary of completed random-split seed0 optimizer/LR pilot outputs. This is not benchmark evaluation.",
        "",
        "## Decisions",
        "",
        f"- Best harmonic setting: `{decisions['best_setting_for_harmonic_prior']['setting']}`",
        f"- Best hybrid lambda 5.0 setting: `{decisions['best_setting_for_hybrid_harmonic_stiffness_v1_lambda500']['setting']}`",
        f"- Hybrid robust under best-per-method settings: `{decisions['hybrid_gain_robust_best_per_method']}`",
        f"- Hybrid robust under best shared setting: `{decisions['hybrid_gain_robust_best_shared']}`",
        f"- Optimized harmonic catches up or beats hybrid: `{decisions['optimized_harmonic_catches_up_or_beats_hybrid']}`",
        f"- Replace SGD lr=1e-2: `{decisions['sgd_lr1e2_should_be_replaced']['status']}`",
        f"- AdamW beneficial: `{decisions['adamw_appears_beneficial']}`",
        f"- Adam tested: `{decisions['adam_baseline_tested']}`",
        f"- Scheduler beneficial: `{decisions['scheduler_variant_appears_beneficial']}`",
        f"- Interpretation: `{decisions['interpretation']}`",
        "",
        "## Per-Run Metrics",
        "",
        "| Method | Setting | Optimizer | LR | Scheduler | Final LR | Steps | Stop | Best val | Reload val | RMSD overall | RMSD train | RMSD val | RMSD test |",
        "|---|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary["rows"], key=lambda item: (item["method"], OPTIMIZER_SETTINGS.index(item["setting"]))):
        lines.append(
            f"| {row['method']} | {row['setting']} | {row['optimizer_name']} | {_fmt(row['learning_rate'])} | "
            f"{row.get('lr_scheduler_name') or 'None'} | {_fmt(row.get('final_learning_rate'))} | "
            f"{row['executed_train_steps']} | {row['stop_reason']} | {_fmt(row['best_validation_loss'])} | "
            f"{_fmt(row['reloaded_validation_loss'])} | {_fmt(row['sampled_rmsd_to_target_overall_mean'])} | "
            f"{_fmt(row['sampled_rmsd_to_target_train_mean'])} | {_fmt(row['sampled_rmsd_to_target_val_mean'])} | "
            f"{_fmt(row['sampled_rmsd_to_target_test_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Shared Setting Comparison",
            "",
            "| Setting | Hybrid better | Better | Tied | Worse |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for comparison in summary["shared_setting_comparisons"]:
        if not comparison.get("available"):
            lines.append(f"| {comparison['setting']} | missing | NA | NA | NA |")
            continue
        counts = comparison["counts"]
        lines.append(
            f"| {comparison['setting']} | {comparison['candidate_better_on_all_core_metrics']} | "
            f"{counts['better']} | {counts['tied']} | {counts['worse']} |"
        )
    lines.extend(_safety_markdown(summary))
    return "\n".join(lines)


def _lambda_markdown(summary: dict[str, Any]) -> str:
    decisions = summary["decisions"]
    lines = [
        "# Full-Database Hybrid Lambda-Refinement Summary",
        "",
        "Reporting-only summary of completed random-split seed0 hybrid lambda-refinement outputs. This is not benchmark evaluation.",
        "",
        "## Decisions",
        "",
        f"- Best lambda by best validation loss: `{decisions['best_lambda_by_best_validation_loss']}`",
        f"- Best lambda by reloaded validation loss: `{decisions['best_lambda_by_reloaded_validation_loss']}`",
        f"- Best lambda by RMSD overall: `{decisions['best_lambda_by_sampled_rmsd_overall']}`",
        f"- Overall trend: `{summary['lambda_trend']['overall_trend']}`",
        f"- High lambdas 20 or 50 degrade: `{decisions['high_lambdas_20_or_50_degrade']}`",
        f"- Lambda >5 improves without clear degradation: `{decisions['lambda_above_5_improves_without_clear_degradation']}`",
        f"- Lambda 5 remains best choice: `{decisions['lambda5_remains_best_choice']}`",
        f"- Recommended lambda: `{decisions['recommended_lambda']}`",
        f"- Interpretation: `{decisions['interpretation']}`",
        "",
        "## Per-Lambda Metrics",
        "",
        "| Lambda | Steps | Stop | Best val | Reload val | RMSD overall | RMSD train | RMSD val | RMSD test |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary["rows"], key=lambda item: float(item["lambda"])):
        lines.append(
            f"| {_fmt(row['lambda'])} | {row['executed_train_steps']} | {row['stop_reason']} | "
            f"{_fmt(row['best_validation_loss'])} | {_fmt(row['reloaded_validation_loss'])} | "
            f"{_fmt(row['sampled_rmsd_to_target_overall_mean'])} | "
            f"{_fmt(row['sampled_rmsd_to_target_train_mean'])} | {_fmt(row['sampled_rmsd_to_target_val_mean'])} | "
            f"{_fmt(row['sampled_rmsd_to_target_test_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Metric Trends",
            "",
            "| Metric | Trend | Best lambdas |",
            "|---|---|---|",
        ]
    )
    for metric, trend in summary["lambda_trend"]["per_metric"].items():
        lines.append(f"| {metric} | {trend['trend']} | `{trend['best_lambdas']}` |")
    lines.extend(_safety_markdown(summary))
    return "\n".join(lines)


def _decision_markdown(summary: dict[str, Any]) -> str:
    verdicts = summary["decision_verdicts"]
    final = verdicts["final_recommendation"]
    lines = [
        "# Full-Database Optimization And Lambda Decision Summary",
        "",
        "Reporting-only decision summary for the completed optimization-prep phase. This is not benchmark evaluation.",
        "",
        "## Verdicts",
        "",
        f"- Optimizer/LR verdict: `{verdicts['optimizer_lr_verdict']}`",
        f"- Lambda verdict: `{verdicts['lambda_verdict']}`",
        f"- Benchmark validation gate: `{verdicts['benchmark_validation_gate']}`",
        f"- Source recommendation: `{final['source_recommendation']}`",
        f"- Recommended lambda: `{final['recommended_lambda']}`",
        f"- Recommended optimizer setting for minimal follow-up: `{final['recommended_optimizer_setting_for_minimal_followup']}`",
        f"- Rerun only final hybrid and harmonic under better optimizer: `{final['recommend_rerun_only_final_hybrid_and_harmonic_under_better_optimizer']}`",
        f"- Minimal scaffold follow-up for lambda above 5: `{final['recommend_minimal_scaffold_followup_for_lambda_above5']}`",
        f"- Submit jobs automatically: `{final['submit_jobs_automatically']}`",
        "",
        "## Optimizer/LR Pilot",
        "",
        f"- Best harmonic setting: `{summary['optimizer_lr_pilot']['decisions']['best_setting_for_harmonic_prior']['setting']}`",
        f"- Best hybrid setting: `{summary['optimizer_lr_pilot']['decisions']['best_setting_for_hybrid_harmonic_stiffness_v1_lambda500']['setting']}`",
        f"- Hybrid robust best per method: `{summary['optimizer_lr_pilot']['decisions']['hybrid_gain_robust_best_per_method']}`",
        f"- Replace SGD lr=1e-2: `{summary['optimizer_lr_pilot']['decisions']['sgd_lr1e2_should_be_replaced']['status']}`",
        "",
        "## Lambda Refinement",
        "",
        f"- Best by validation loss: `{summary['lambda_refinement']['decisions']['best_lambda_by_best_validation_loss']}`",
        f"- Best by reloaded validation loss: `{summary['lambda_refinement']['decisions']['best_lambda_by_reloaded_validation_loss']}`",
        f"- Best by RMSD overall: `{summary['lambda_refinement']['decisions']['best_lambda_by_sampled_rmsd_overall']}`",
        f"- Trend: `{summary['lambda_refinement']['lambda_trend']['overall_trend']}`",
        "",
    ]
    lines.extend(_safety_markdown(summary))
    return "\n".join(lines)


def _safety_markdown(summary: dict[str, Any]) -> list[str]:
    return [
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


def _safety_labels() -> dict[str, Any]:
    return {
        "benchmark_result": False,
        "non_benchmark": True,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "scheduler_submission_executed": False,
        "raw_data_modified": False,
        "environment_modified": False,
    }


def _lambda_label(value: float) -> str:
    return f"lambda{int(round(float(value) * 100)):03d}"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _tied(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= TIE_TOLERANCE


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NA"
    return f"{float(value):.8g}"
