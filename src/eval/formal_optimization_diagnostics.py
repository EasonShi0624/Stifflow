"""Optimization diagnostics for completed full-database formal runs."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.data.geom_full_database import FULL_DATABASE_SCOPE
from src.eval.hybrid_harmonic_stiffness_pilot import BASELINE_LABELS
from src.train.formal_10000_training import (
    FORMAL_10000_HARD_MAX_STEPS,
    FORMAL_10000_LEARNING_RATE,
    FORMAL_10000_MAX_STEPS,
)
from src.train.formal_full_database_training import FORMAL_FULL_DATABASE_RESULT_KIND


DEFAULT_OPTIMIZATION_DIAGNOSTICS_PATH = Path(
    "experiments/results/geom_full_database_optimization_diagnostics.json"
)
DEFAULT_OPTIMIZATION_DIAGNOSTICS_REPORT_PATH = Path(
    "reports/tables/geom_full_database_optimization_diagnostics.md"
)

OPTIMIZATION_DIAGNOSTICS_RESULT_KIND = (
    "geom_drugs_full_database_optimization_diagnostics_not_benchmark"
)

METHODS = (
    "harmonic_prior",
    "hybrid_harmonic_stiffness_v1_lambda500",
    "stiffness_structured",
    "isotropic",
    "torsion_only",
)
METHOD_PATH_LABELS = {
    **BASELINE_LABELS,
    "hybrid_harmonic_stiffness_v1_lambda500": "hybrid_harmonic_stiffness_v1_lambda500",
}
SPLITS = ("random", "scaffold")
SEEDS = (0, 1, 2)
LOSS_FIELDS = ("train_loss_before_step", "validation_loss_after_step", "gradient_l2_norm")


def write_optimization_diagnostics(
    results_root: str | Path = "experiments/results",
    summary_path: str | Path = DEFAULT_OPTIMIZATION_DIAGNOSTICS_PATH,
    report_path: str | Path = DEFAULT_OPTIMIZATION_DIAGNOSTICS_REPORT_PATH,
    require_results: bool = False,
) -> dict[str, Path]:
    """Read completed result JSONs and write optimization diagnostics."""

    results_root = Path(results_root)
    expected_paths = _expected_result_paths(results_root)
    runs = []
    missing = []
    for split_type in SPLITS:
        for method in METHODS:
            for seed in SEEDS:
                path = expected_paths[split_type][method][seed]
                if not path.exists():
                    missing.append(
                        {
                            "split_type": split_type,
                            "method": method,
                            "seed": seed,
                            "expected_path": str(path),
                        }
                    )
                    continue
                runs.append(_diagnose_result(path=path, split_type=split_type, method=method, seed=seed))

    if require_results and missing:
        raise FileNotFoundError(f"Missing expected full-database result JSONs: {missing}")

    aggregate_by_method_split = _aggregate_by_method_split(runs)
    aggregate_by_method = _aggregate_by_method(runs)
    overall = _overall_assessment(runs)
    summary = {
        "summary_name": "geom_full_database_optimization_diagnostics",
        "result_kind": OPTIMIZATION_DIAGNOSTICS_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_reporting_only_optimization_diagnostics",
        "hardware": "reporting_only",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "dataset_scope": FULL_DATABASE_SCOPE,
        "splits": list(SPLITS),
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "coverage": {
            "expected_result_count": len(SPLITS) * len(METHODS) * len(SEEDS),
            "available_result_count": len(runs),
            "missing_result_count": len(missing),
            "missing_results": missing,
        },
        "training_horizon": {
            "max_steps": FORMAL_10000_MAX_STEPS,
            "hard_max_steps": FORMAL_10000_HARD_MAX_STEPS,
            "baseline_optimizer": "SGD",
            "baseline_learning_rate": FORMAL_10000_LEARNING_RATE,
        },
        "run_diagnostics": runs,
        "aggregate_by_method_split": aggregate_by_method_split,
        "aggregate_by_method": aggregate_by_method,
        "overall_assessment": overall,
        "prepared_followup_plan": _prepared_followup_plan(),
        "safety_labels": {
            "benchmark_result": False,
            "non_benchmark": True,
            "training_executed_by_summary": False,
            "sampling_executed_by_summary": False,
            "raw_data_modified": False,
            "environment_modified": False,
            "scheduler_submission_executed": False,
        },
        "notes": [
            "Optimization diagnostics are computed only from existing completed full-database formal result JSONs.",
            "Loss-curve trend diagnostics use stored per-step train/validation losses from each JSON.",
            "This is not benchmark evaluation and does not run PoseBusters, MMFF/UFF benchmarks, xTB, training, sampling, preprocessing, scheduler actions, raw-data edits, or environment changes.",
        ],
    }

    summary_path = Path(summary_path)
    report_path = Path(report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def _expected_result_paths(results_root: Path) -> dict[str, dict[str, dict[int, Path]]]:
    expected: dict[str, dict[str, dict[int, Path]]] = {}
    for split_type in SPLITS:
        split_label = "" if split_type == "random" else "_scaffold"
        expected[split_type] = {}
        for method in METHODS:
            label = METHOD_PATH_LABELS[method]
            expected[split_type][method] = {
                seed: results_root
                / f"geom_formal_full_database{split_label}_train_{label}_steps{FORMAL_10000_MAX_STEPS}_seed{seed}.json"
                for seed in SEEDS
            }
    return expected


def _diagnose_result(path: Path, split_type: str, method: str, seed: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_result(payload=payload, path=path, split_type=split_type, seed=seed)
    metrics = payload["metrics"]
    train_steps = metrics.get("train_steps") or []
    if not train_steps:
        raise ValueError(f"Result has no train_steps for optimization diagnostics: {path}")

    executed_steps = int(metrics["executed_train_steps"])
    max_steps = int(payload["max_steps"])
    hard_max_steps = int(payload["hard_max_steps"])
    stop_reason = str(metrics.get("stop_reason") or payload.get("early_stop_reason"))
    best_step = int(metrics.get("best_validation_loss_step") or metrics.get("selected_checkpoint_step") or 0)
    best_gap = None if best_step <= 0 else executed_steps - best_step
    windows = {
        str(window_size): {
            field: _window_diagnostics(train_steps, field=field, window_size=window_size)
            for field in LOSS_FIELDS
        }
        for window_size in (100, 500)
    }
    validation_last100 = windows["100"]["validation_loss_after_step"]
    validation_last500 = windows["500"]["validation_loss_after_step"]
    train_last500 = windows["500"]["train_loss_before_step"]
    hard_cap_hit = executed_steps >= hard_max_steps and stop_reason == "hard_cap_reached"
    diagnostics = {
        "split_type": split_type,
        "method": method,
        "seed": int(seed),
        "run_name": payload["run_name"],
        "source_path": str(path),
        "corruption_mode": payload["corruption_mode"],
        "hybrid_source_strength": payload.get("hybrid_source_strength"),
        "optimizer_name": payload.get("optimizer_name"),
        "learning_rate": float(payload.get("learning_rate")),
        "lr_scheduler_name": payload.get("lr_scheduler_name"),
        "max_steps": max_steps,
        "hard_max_steps": hard_max_steps,
        "executed_train_steps": executed_steps,
        "stop_reason": stop_reason,
        "hit_hard_cap": hard_cap_hit,
        "stopped_early": bool(payload.get("stopped_early")),
        "best_validation_loss": float(metrics["best_validation_loss"]),
        "best_validation_loss_step": best_step,
        "best_step_vs_final_gap_steps": best_gap,
        "best_step_position": _best_step_position(best_gap),
        "final_validation_loss": float(metrics["final_validation_loss"]),
        "reloaded_validation_loss": float(metrics["reloaded_validation_loss"]),
        "final_train_loss": float(metrics["final_train_loss"]),
        "reloaded_train_loss": float(metrics["reloaded_train_loss"]),
        "window_diagnostics": windows,
        "late_stage_still_decreasing": {
            "last_100_validation": _still_decreasing(validation_last100),
            "last_500_validation": _still_decreasing(validation_last500),
            "last_500_train": _still_decreasing(train_last500),
        },
        "late_stage_curve_shape": {
            "last_100_validation": _curve_shape(validation_last100),
            "last_500_validation": _curve_shape(validation_last500),
            "last_500_train": _curve_shape(train_last500),
        },
    }
    diagnostics["lr_interpretation"] = _run_lr_interpretation(diagnostics)
    return diagnostics


def _validate_result(payload: dict[str, Any], path: Path, split_type: str, seed: int) -> None:
    if payload.get("result_kind") != FORMAL_FULL_DATABASE_RESULT_KIND:
        raise ValueError(f"Not a formal full-database result JSON: {path}")
    if payload.get("benchmark_result") is not False:
        raise ValueError(f"Optimization diagnostics require benchmark_result=false: {path}")
    if payload.get("dataset_subset_scope") != FULL_DATABASE_SCOPE:
        raise ValueError(f"Unexpected dataset scope in {path}")
    if payload.get("formal_training_run") is not True or payload.get("full_database_formal_training_run") is not True:
        raise ValueError(f"Not a full-database formal training output: {path}")
    payload_split_type = payload.get("split_type") or (payload.get("dataset") or {}).get("split_type")
    if str(payload_split_type) != split_type:
        raise ValueError(f"Result split mismatch for {path}")
    if int(payload.get("random_seed")) != int(seed):
        raise ValueError(f"Result seed mismatch for {path}")
    metrics = payload.get("metrics") or {}
    for key in [
        "executed_train_steps",
        "best_validation_loss",
        "best_validation_loss_step",
        "final_validation_loss",
        "reloaded_validation_loss",
        "train_steps",
    ]:
        if key not in metrics:
            raise ValueError(f"Result metrics missing {key}: {path}")


def _window_diagnostics(train_steps: list[dict[str, Any]], field: str, window_size: int) -> dict[str, Any]:
    values = [float(row[field]) for row in train_steps[-int(window_size) :] if field in row]
    if not values:
        return {"available": False, "count": 0}
    deltas = [values[index + 1] - values[index] for index in range(len(values) - 1)]
    negative = sum(1 for value in deltas if value < -1.0e-12)
    positive = sum(1 for value in deltas if value > 1.0e-12)
    signs = [_sign(delta) for delta in deltas if _sign(delta) != 0]
    sign_changes = sum(1 for index in range(len(signs) - 1) if signs[index + 1] != signs[index])
    slope = _slope(values)
    first = values[0]
    last = values[-1]
    delta = last - first
    relative_delta = None if first == 0.0 else delta / abs(first)
    denominator = max(1, len(deltas))
    return {
        "available": True,
        "count": len(values),
        "first": first,
        "last": last,
        "delta": delta,
        "relative_delta": relative_delta,
        "slope_per_step": slope,
        "decrease_fraction": negative / denominator,
        "increase_fraction": positive / denominator,
        "flat_fraction": (denominator - negative - positive) / denominator,
        "sign_change_count": sign_changes,
        "sign_change_fraction": sign_changes / max(1, len(signs) - 1),
    }


def _slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean_x = (len(values) - 1) / 2.0
    mean_y = sum(values) / len(values)
    numerator = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
    denominator = sum((index - mean_x) ** 2 for index in range(len(values)))
    if denominator == 0.0:
        return None
    return numerator / denominator


def _sign(value: float) -> int:
    if value > 1.0e-12:
        return 1
    if value < -1.0e-12:
        return -1
    return 0


def _still_decreasing(window: dict[str, Any]) -> bool | None:
    if not window.get("available"):
        return None
    return bool(
        window.get("delta") is not None
        and float(window["delta"]) < -1.0e-10
        and window.get("slope_per_step") is not None
        and float(window["slope_per_step"]) < -1.0e-12
    )


def _curve_shape(window: dict[str, Any]) -> str:
    if not window.get("available"):
        return "missing"
    decrease_fraction = float(window.get("decrease_fraction", 0.0))
    increase_fraction = float(window.get("increase_fraction", 0.0))
    sign_change_fraction = float(window.get("sign_change_fraction", 0.0))
    delta = float(window.get("delta", 0.0))
    relative_delta = abs(float(window.get("relative_delta") or 0.0))
    if sign_change_fraction >= 0.30 and increase_fraction >= 0.20:
        return "oscillatory"
    if abs(delta) <= 1.0e-10 or relative_delta <= 1.0e-4:
        return "flat_or_saturated"
    if delta < 0.0 and decrease_fraction >= 0.90 and sign_change_fraction <= 0.05:
        return "smooth_decreasing"
    if delta < 0.0:
        return "decreasing_with_noise"
    if delta > 0.0:
        return "increasing_or_degrading"
    return "mixed"


def _best_step_position(best_gap: int | None) -> str:
    if best_gap is None:
        return "unknown"
    if best_gap == 0:
        return "final_step"
    if best_gap <= 100:
        return "within_last_100_steps"
    if best_gap <= 500:
        return "within_last_500_steps"
    return "earlier_than_last_500_steps"


def _run_lr_interpretation(run: dict[str, Any]) -> str:
    if run["hit_hard_cap"]:
        val500_shape = run["late_stage_curve_shape"]["last_500_validation"]
        val100_shape = run["late_stage_curve_shape"]["last_100_validation"]
        if val500_shape in {"smooth_decreasing", "decreasing_with_noise"} and val100_shape in {
            "smooth_decreasing",
            "decreasing_with_noise",
        }:
            return "still_decreasing_at_hard_cap_likely_needs_schedule_or_more_steps"
        if val500_shape == "oscillatory" or val100_shape == "oscillatory":
            return "late_loss_oscillatory_lr_may_be_too_large_or_schedule_needed"
        if val500_shape == "flat_or_saturated" and run["best_step_position"] in {
            "final_step",
            "within_last_100_steps",
            "within_last_500_steps",
        }:
            return "near_plateau_at_hard_cap_schedule_may_help"
        return "hard_cap_hit_mixed_late_trend"
    if run["stop_reason"] == "validation_patience_exhausted":
        return "early_stop_reached_under_current_lr"
    return "inconclusive"


def _aggregate_by_method_split(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["split_type"], run["method"])].append(run)
    return {
        f"{split_type}:{method}": _aggregate_runs(group)
        for (split_type, method), group in sorted(grouped.items())
    }


def _aggregate_by_method(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["method"]].append(run)
    return {method: _aggregate_runs(group) for method, group in sorted(grouped.items())}


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    counter = Counter(run["lr_interpretation"] for run in runs)
    best_gaps = [
        int(run["best_step_vs_final_gap_steps"])
        for run in runs
        if run["best_step_vs_final_gap_steps"] is not None
    ]
    return {
        "run_count": len(runs),
        "splits": sorted({run["split_type"] for run in runs}),
        "methods": sorted({run["method"] for run in runs}),
        "seeds": sorted({run["seed"] for run in runs}),
        "hard_cap_hit_count": sum(1 for run in runs if run["hit_hard_cap"]),
        "early_stop_count": sum(1 for run in runs if run["stop_reason"] == "validation_patience_exhausted"),
        "best_at_final_count": sum(1 for run in runs if run["best_step_position"] == "final_step"),
        "best_within_last_100_count": sum(
            1 for run in runs if run["best_step_position"] in {"final_step", "within_last_100_steps"}
        ),
        "best_within_last_500_count": sum(
            1
            for run in runs
            if run["best_step_position"]
            in {"final_step", "within_last_100_steps", "within_last_500_steps"}
        ),
        "mean_best_step_vs_final_gap_steps": _mean(best_gaps),
        "last_100_validation_decreasing_count": sum(
            1 for run in runs if run["late_stage_still_decreasing"]["last_100_validation"] is True
        ),
        "last_500_validation_decreasing_count": sum(
            1 for run in runs if run["late_stage_still_decreasing"]["last_500_validation"] is True
        ),
        "last_500_train_decreasing_count": sum(
            1 for run in runs if run["late_stage_still_decreasing"]["last_500_train"] is True
        ),
        "last_500_validation_shape_counts": dict(
            Counter(run["late_stage_curve_shape"]["last_500_validation"] for run in runs)
        ),
        "lr_interpretation_counts": dict(counter),
        "dominant_lr_interpretation": counter.most_common(1)[0][0],
    }


def _overall_assessment(runs: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = _aggregate_runs(runs)
    if not runs:
        verdict = "no_completed_outputs_available"
    else:
        hard_cap_fraction = aggregate["hard_cap_hit_count"] / aggregate["run_count"]
        decreasing_fraction = aggregate["last_500_validation_decreasing_count"] / aggregate["run_count"]
        oscillatory_count = aggregate["last_500_validation_shape_counts"].get("oscillatory", 0)
        oscillatory_fraction = oscillatory_count / aggregate["run_count"]
        if hard_cap_fraction >= 0.5 and decreasing_fraction >= 0.5 and oscillatory_fraction < 0.25:
            verdict = "fixed_sgd_lr1e2_mostly_still_decreasing_at_hard_cap_schedule_or_more_steps_is_first_suspect"
        elif oscillatory_fraction >= 0.33:
            verdict = "fixed_sgd_lr1e2_often_oscillatory_lr_may_be_too_large"
        elif hard_cap_fraction >= 0.5:
            verdict = "fixed_sgd_lr1e2_hard_cap_limited_with_mixed_or_plateaued_trends"
        else:
            verdict = "fixed_sgd_lr1e2_not_clearly_hard_cap_limited"
    return {
        "verdict": verdict,
        "aggregate": aggregate,
        "same_behavior_across_methods_and_splits": _cross_method_split_behavior(runs),
        "scheduler_or_optimizer_sweep_supported": bool(runs),
        "interpretation_scope": "loss-curve diagnostics only; no new training or benchmark evaluation.",
    }


def _cross_method_split_behavior(runs: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate_by_method_split = _aggregate_by_method_split(runs)
    dominant = {
        key: value["dominant_lr_interpretation"]
        for key, value in aggregate_by_method_split.items()
        if value
    }
    counts = Counter(dominant.values())
    if not dominant:
        status = "unavailable"
        shared = None
    elif len(counts) == 1:
        status = "same_dominant_behavior"
        shared = next(iter(counts))
    elif counts.most_common(1)[0][1] / len(dominant) >= 0.75:
        status = "mostly_same_dominant_behavior"
        shared = counts.most_common(1)[0][0]
    else:
        status = "method_or_split_dependent"
        shared = None
    return {
        "status": status,
        "shared_dominant_interpretation": shared,
        "dominant_by_method_split": dominant,
    }


def _prepared_followup_plan() -> dict[str, Any]:
    return {
        "optimizer_lr_scheduler_pilot": {
            "methods": ["harmonic_prior", "hybrid_harmonic_stiffness_v1_lambda500"],
            "split_type": "random",
            "seed": 0,
            "sgd_learning_rates": [1.0e-2, 3.0e-3, 1.0e-3],
            "adamw_learning_rates": [1.0e-3, 3.0e-4, 1.0e-4],
            "optional_scheduler_variant": "ReduceLROnPlateau on SGD lr=1e-2 for each method",
            "runs_prepared": 14,
            "submit_automatically": False,
        },
        "hybrid_lambda_refinement": {
            "method": "hybrid_harmonic_stiffness_v1",
            "split_type": "random",
            "seed": 0,
            "lambdas": [2.0, 3.0, 5.0, 7.5, 10.0, 20.0, 50.0],
            "residual_variants": False,
            "runs_prepared": 7,
            "submit_automatically": False,
        },
    }


def _mean(values: list[float] | list[int]) -> float | None:
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Full-Database Optimization Diagnostics",
        "",
        "Reporting-only optimization diagnostics from completed formal result JSONs. This is not benchmark evaluation.",
        "",
        "## Scope",
        "",
        f"- Available results: `{summary['coverage']['available_result_count']}` / `{summary['coverage']['expected_result_count']}`",
        f"- Missing results: `{summary['coverage']['missing_result_count']}`",
        f"- Methods: `{summary['methods']}`",
        f"- Splits: `{summary['splits']}`",
        f"- Baseline optimizer/LR in inspected runs: `SGD`, `{FORMAL_10000_LEARNING_RATE}`",
        f"- Overall verdict: `{summary['overall_assessment']['verdict']}`",
        f"- Cross method/split behavior: `{summary['overall_assessment']['same_behavior_across_methods_and_splits']['status']}`",
        "",
        "## Aggregate By Method And Split",
        "",
        "| Split | Method | Runs | Hard cap | Best final | Best <=100 | Best <=500 | Val last500 decreasing | Val shape counts | Dominant LR interpretation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for key, row in sorted(summary["aggregate_by_method_split"].items()):
        split_type, method = key.split(":", 1)
        lines.append(
            f"| {split_type} | {method} | {row['run_count']} | {row['hard_cap_hit_count']} | "
            f"{row['best_at_final_count']} | {row['best_within_last_100_count']} | "
            f"{row['best_within_last_500_count']} | {row['last_500_validation_decreasing_count']} | "
            f"`{row['last_500_validation_shape_counts']}` | `{row['dominant_lr_interpretation']}` |"
        )
    lines.extend(
        [
            "",
            "## Per-Run Diagnostics",
            "",
            "| Split | Method | Seed | Steps | Stop reason | Best step | Gap | Best val | Final val | Reload val | Last100 val delta | Last500 val delta | Last500 val shape | LR interpretation |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for run in sorted(summary["run_diagnostics"], key=lambda item: (item["split_type"], item["method"], item["seed"])):
        last100 = run["window_diagnostics"]["100"]["validation_loss_after_step"]
        last500 = run["window_diagnostics"]["500"]["validation_loss_after_step"]
        lines.append(
            f"| {run['split_type']} | {run['method']} | {run['seed']} | "
            f"{run['executed_train_steps']} | {run['stop_reason']} | "
            f"{run['best_validation_loss_step']} | {_fmt_int(run['best_step_vs_final_gap_steps'])} | "
            f"{_fmt(run['best_validation_loss'])} | {_fmt(run['final_validation_loss'])} | "
            f"{_fmt(run['reloaded_validation_loss'])} | {_fmt(last100.get('delta'))} | "
            f"{_fmt(last500.get('delta'))} | {run['late_stage_curve_shape']['last_500_validation']} | "
            f"`{run['lr_interpretation']}` |"
        )
    lines.extend(
        [
            "",
            "## Prepared Follow-Up Plan",
            "",
            "- Optimizer/LR pilot: harmonic_prior and hybrid_harmonic_stiffness_v1 lambda 5.0 on random seed0; SGD lr 1e-2/3e-3/1e-3, AdamW lr 1e-3/3e-4/1e-4, plus optional ReduceLROnPlateau variants for current SGD lr 1e-2.",
            "- Lambda refinement: hybrid_harmonic_stiffness_v1 only on random seed0 for lambda 2.0, 3.0, 5.0, 7.5, 10.0, 20.0, and 50.0.",
            "- No jobs were submitted by this diagnostics step.",
            "",
            "## Safety",
            "",
            f"- `benchmark_result`: `{summary['benchmark_result']}`",
            f"- `training_executed_by_summary`: `{summary['training_executed_by_summary']}`",
            f"- `sampling_executed_by_summary`: `{summary['sampling_executed_by_summary']}`",
            f"- `raw_data_modified`: `{summary['raw_data_modified']}`",
            f"- `environment_modified`: `{summary['environment_modified']}`",
            "",
            "No PoseBusters, MMFF/UFF benchmark jobs, xTB, training, sampling, preprocessing, scheduler actions, raw-data edits, or environment changes were run.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NA"
    return f"{float(value):.8g}"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "NA"
    return str(int(value))
