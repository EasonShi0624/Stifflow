"""Full-database A/B/D/E attribution-phase summary utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.geom_full_database import FULL_DATABASE_SCOPE
from src.train.formal_10000_training import (
    FORMAL_10000_HARD_MAX_STEPS,
    FORMAL_10000_LEARNING_RATE,
    FORMAL_10000_MAX_STEPS,
    FORMAL_10000_MIN_DELTA,
    FORMAL_10000_PATIENCE,
    FORMAL_10000_SAMPLE_STEP_SIZE,
)
from src.train.formal_full_database_training import (
    FORMAL_FULL_DATABASE_ATTRIBUTION_MODES,
    FORMAL_FULL_DATABASE_CHECKPOINT_SCOPE,
    FORMAL_FULL_DATABASE_RESULT_KIND,
    FORMAL_FULL_DATABASE_RUNTIME_SCOPE,
)


ATTRIBUTION_MODE_LABELS = {
    "isotropic": "isotropic",
    "harmonic_prior": "harmonic",
    "torsion_only": "torsion_only",
    "stiffness_structured": "stiffness",
}
DEFAULT_ATTRIBUTION_SPLIT_TYPES = ("random", "scaffold")
DEFAULT_ATTRIBUTION_SEEDS = (0, 1, 2)
DEFAULT_ATTRIBUTION_SUMMARY_PATH = Path("experiments/results/geom_formal_full_database_attribution_summary.json")
DEFAULT_ATTRIBUTION_REPORT_PATH = Path("reports/tables/geom_formal_full_database_attribution_summary.md")
ATTRIBUTION_SUMMARY_RESULT_KIND = "geom_drugs_full_database_attribution_phase_summary_not_benchmark"
CORE_LOWER_IS_BETTER_METRICS = (
    "best_validation_loss",
    "reloaded_validation_loss",
    "sampled_rmsd_to_target_overall_mean",
    "sampled_rmsd_to_target_train_mean",
    "sampled_rmsd_to_target_val_mean",
    "sampled_rmsd_to_target_test_mean",
)
OPTIONAL_PHYSICAL_LOWER_IS_BETTER_METRICS = ("sampled_mean_abs_bond_length_delta",)
ROTATABLE_BINS = (
    ("0", 0, 0),
    ("1-3", 1, 3),
    ("4-6", 4, 6),
    ("7+", 7, None),
)
TIE_TOLERANCE = 1.0e-12


def default_attribution_result_paths(
    split_types: tuple[str, ...] = DEFAULT_ATTRIBUTION_SPLIT_TYPES,
    seeds: tuple[int, ...] = DEFAULT_ATTRIBUTION_SEEDS,
    modes: tuple[str, ...] = FORMAL_FULL_DATABASE_ATTRIBUTION_MODES,
) -> list[Path]:
    paths = []
    for split_type in split_types:
        split_label = "" if split_type == "random" else f"_{split_type}"
        for seed in seeds:
            for mode in modes:
                paths.append(
                    Path(
                        "experiments/results"
                    )
                    / (
                        "geom_formal_full_database"
                        f"{split_label}_train_{ATTRIBUTION_MODE_LABELS[mode]}_steps{FORMAL_10000_MAX_STEPS}_seed{seed}.json"
                    )
                )
    return paths


def write_formal_attribution_phase_summary(
    result_paths: list[str | Path] | None = None,
    summary_path: str | Path = DEFAULT_ATTRIBUTION_SUMMARY_PATH,
    report_path: str | Path = DEFAULT_ATTRIBUTION_REPORT_PATH,
    expected_split_types: tuple[str, ...] = DEFAULT_ATTRIBUTION_SPLIT_TYPES,
    expected_seeds: tuple[int, ...] = DEFAULT_ATTRIBUTION_SEEDS,
    expected_modes: tuple[str, ...] = FORMAL_FULL_DATABASE_ATTRIBUTION_MODES,
) -> dict[str, Path]:
    paths = [Path(path) for path in (result_paths or default_attribution_result_paths(expected_split_types, expected_seeds, expected_modes))]
    payloads = [_load_result(path) for path in paths]
    rows = [_row_from_result(path, payload) for path, payload in zip(paths, payloads)]
    _validate_attribution_matrix(
        rows=rows,
        expected_split_types=expected_split_types,
        expected_seeds=expected_seeds,
        expected_modes=expected_modes,
    )
    grouped = _group_rows(rows)
    per_split_seed = _per_split_seed_comparisons(grouped, expected_split_types, expected_seeds, expected_modes)
    aggregate = _aggregate_by_split(rows, expected_split_types, expected_modes)
    physical = _physical_summary(rows)
    rotatable = _rotatable_summary(rows)
    summary = {
        "summary_name": "geom_formal_full_database_attribution_summary",
        "result_kind": ATTRIBUTION_SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_full_database_attribution_phase_report",
        "hardware": "reporting_only",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "input_result_paths": [str(path) for path in paths],
        "row_count": len(rows),
        "dataset_scope": FULL_DATABASE_SCOPE,
        "split_types": list(expected_split_types),
        "seed_set": list(expected_seeds),
        "attribution_modes": list(expected_modes),
        "ablation_ids": {
            "A": "isotropic",
            "B": "harmonic_prior",
            "D": "torsion_only",
            "E": "stiffness_structured",
        },
        "dataset": {
            "name": "geom_drugs",
            "kind": rows[0]["dataset_kind"],
            "subset_scope": rows[0]["dataset_subset_scope"],
            "molecule_count": rows[0]["molecule_count"],
            "split_counts_by_type": {
                split_type: grouped[split_type][expected_seeds[0]][expected_modes[0]]["split_counts"]
                for split_type in expected_split_types
            },
            "path_reuse_status": _path_reuse_status(rows),
        },
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
        "per_split_seed_comparisons": per_split_seed,
        "aggregate_metric_comparisons": aggregate,
        "physical_validity_diagnostics": physical,
        "rotatable_bond_stratification": rotatable,
        "directional_verdict": _directional_verdict(per_split_seed, aggregate),
        "safety_labels": {
            "benchmark_result": False,
            "non_benchmark": True,
            "full_database_attribution_phase_summary": True,
            "training_executed_by_summary": False,
            "sampling_executed_by_summary": False,
            "raw_data_modified": False,
            "environment_modified": False,
            "gpu_used_by_summary": False,
            "result_scope": "full_database_A_B_D_E_attribution_phase_reporting_only",
        },
        "notes": [
            "Reporting-only aggregation for the minimum full-database attribution phase: A=isotropic, B=harmonic_prior, D=torsion_only, E=stiffness_structured.",
            "The summary validates that all inputs preserve the frozen model, optimizer, learning rate, checkpoint/reload behavior, validation-selected checkpoint rule, one-step sampling diagnostic, split type, seed set, and hard step cap.",
            "Physical-validity diagnostics are reported only when present in the result JSONs; current support covers finite sampled coordinates and sampled bond-length deltas.",
            "Rotatable-bond stratification is reported only when sampled molecules include rotatable_bond_count metadata.",
            "This is explicitly not a benchmark result and does not run training, sampling, preprocessing, SLURM jobs, GPUs, or environment changes.",
        ],
    }
    summary_path = Path(summary_path)
    report_path = Path(report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def _load_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Attribution input result does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("result_kind") != FORMAL_FULL_DATABASE_RESULT_KIND:
        raise ValueError(f"Result is not a full-database formal output: {path}")
    if payload.get("benchmark_result") is not False:
        raise ValueError(f"Result is not benchmark_result=false: {path}")
    if payload.get("dataset_subset_scope") != FULL_DATABASE_SCOPE:
        raise ValueError(f"Result has unexpected dataset scope: {path}")
    if payload.get("formal_training_run") is not True or payload.get("full_database_formal_training_run") is not True:
        raise ValueError(f"Result is not a full-database formal training output: {path}")
    if payload.get("checkpoint_scope") != FORMAL_FULL_DATABASE_CHECKPOINT_SCOPE:
        raise ValueError(f"Result checkpoint scope mismatch: {path}")
    if payload.get("checkpoint_reusable") is not True or payload.get("checkpoint_selected_by_validation") is not True:
        raise ValueError(f"Result does not use reusable validation-selected checkpoints: {path}")
    if payload.get("runtime_scope") != FORMAL_FULL_DATABASE_RUNTIME_SCOPE:
        raise ValueError(f"Result runtime scope mismatch: {path}")
    if payload.get("runtime_seconds_reliable") is not True:
        raise ValueError(f"Result runtime_seconds is not marked reliable: {path}")
    dataset = payload.get("dataset") or {}
    if dataset.get("subset_scope") != FULL_DATABASE_SCOPE or dataset.get("split_scope") != FULL_DATABASE_SCOPE:
        raise ValueError(f"Result dataset/split scope mismatch: {path}")
    if dataset.get("split_type") not in {"random", "scaffold"}:
        raise ValueError(f"Result split_type must be random or scaffold: {path}")
    split_counts = dataset.get("split_counts")
    if not isinstance(split_counts, dict) or min(int(split_counts[key]) for key in ["train", "val", "test"]) <= 0:
        raise ValueError(f"Result split counts are invalid: {path}")
    metrics = payload.get("metrics") or {}
    for key in ["best_validation_loss", "reloaded_validation_loss", "sampled_molecules"]:
        if key not in metrics:
            raise ValueError(f"Result metrics missing {key}: {path}")
    sampled = metrics["sampled_molecules"]
    if not isinstance(sampled, list) or len(sampled) != int(payload["sample_count"]):
        raise ValueError(f"Result sampled_molecules length does not match sample_count: {path}")
    for item in sampled:
        if item.get("split") not in {"train", "val", "test"}:
            raise ValueError(f"Sampled molecule has invalid split in {path}")
        if not isinstance(item.get("sampled_rmsd_to_target"), (int, float)):
            raise ValueError(f"Sampled molecule missing numeric sampled_rmsd_to_target in {path}")
    return payload


def _row_from_result(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    sampled = metrics["sampled_molecules"]
    physical = _physical_from_sampled(sampled)
    rotatable = _rotatable_from_sampled(sampled)
    by_split = _average_rmsd_by_split(sampled)
    return {
        "run_name": payload["run_name"],
        "source_path": str(path),
        "processed_artifact_path": payload["processed_artifact_path"],
        "split_path": payload["split_path"],
        "corruption_mode": payload["corruption_mode"],
        "random_seed": int(payload["random_seed"]),
        "result_kind": payload["result_kind"],
        "model_name": payload["model_name"],
        "backbone": payload["backbone"],
        "hardware": payload["hardware"],
        "gpu_used": bool(payload["gpu_used"]),
        "scheduler": payload.get("scheduler"),
        "slurm_account": payload.get("slurm_account"),
        "dataset_kind": payload["dataset"]["kind"],
        "dataset_subset_scope": payload["dataset"]["subset_scope"],
        "molecule_count": int(payload["dataset"]["molecule_count"]),
        "split_type": payload["dataset"]["split_type"],
        "split_scope": payload["dataset"]["split_scope"],
        "split_counts": {key: int(payload["dataset"]["split_counts"][key]) for key in ["train", "val", "test"]},
        "max_steps": int(payload["max_steps"]),
        "hard_max_steps": int(payload["hard_max_steps"]),
        "executed_train_steps": int(metrics["executed_train_steps"]),
        "optimizer_step_count": int(payload["optimizer_step_count"]),
        "training_sample_count": int(payload["training_sample_count"]),
        "validation_sample_count": int(payload["validation_sample_count"]),
        "sample_count": int(payload["sample_count"]),
        "optimizer_name": payload["optimizer_name"],
        "learning_rate": float(payload["learning_rate"]),
        "early_stop_monitor": payload["early_stop_monitor"],
        "early_stop_patience": int(payload["early_stop_patience"]),
        "early_stop_min_delta": float(payload["early_stop_min_delta"]),
        "stopped_early": bool(payload["stopped_early"]),
        "stop_reason": payload["early_stop_reason"],
        "best_validation_loss": float(metrics["best_validation_loss"]),
        "best_validation_loss_step": int(metrics["best_validation_loss_step"]),
        "reloaded_validation_loss": float(metrics["reloaded_validation_loss"]),
        "sample_step_count": int(payload["sample_step_count"]),
        "sample_step_size": float(payload["sample_step_size"]),
        "sampled_rmsd_to_target_overall_mean": _mean(item["sampled_rmsd_to_target"] for item in sampled),
        "sampled_rmsd_to_target_by_split_mean": by_split,
        "sampled_rmsd_to_target_train_mean": by_split["train"],
        "sampled_rmsd_to_target_val_mean": by_split["val"],
        "sampled_rmsd_to_target_test_mean": by_split["test"],
        "physical_validity": physical,
        "sampled_mean_abs_bond_length_delta": physical.get("mean_abs_bond_length_delta"),
        "rotatable_bond_stratification": rotatable,
        "runtime_seconds": float(payload["runtime_seconds"]),
        "runtime_scope": payload["runtime_scope"],
        "runtime_seconds_reliable": bool(payload["runtime_seconds_reliable"]),
        "safety_labels": {
            "benchmark_result": payload["benchmark_result"],
            "hardware": payload["hardware"],
            "gpu_used": payload["gpu_used"],
            "scheduler": payload.get("scheduler"),
            "slurm_account": payload.get("slurm_account"),
            "formal_training_run": payload["formal_training_run"],
            "full_database_formal_training_run": payload["full_database_formal_training_run"],
            "checkpoint_reusable": payload["checkpoint_reusable"],
            "checkpoint_selected_by_validation": payload["checkpoint_selected_by_validation"],
            "checkpoint_scope": payload["checkpoint_scope"],
            "sampling_executed": payload["sampling_executed"],
        },
    }


def _validate_attribution_matrix(
    rows: list[dict[str, Any]],
    expected_split_types: tuple[str, ...],
    expected_seeds: tuple[int, ...],
    expected_modes: tuple[str, ...],
) -> None:
    expected = {
        (split_type, int(seed), mode)
        for split_type in expected_split_types
        for seed in expected_seeds
        for mode in expected_modes
    }
    observed = {(row["split_type"], row["random_seed"], row["corruption_mode"]) for row in rows}
    if observed != expected:
        raise ValueError(f"Attribution inputs do not match the expected split/seed/mode matrix: {observed}")
    baseline = rows[0]
    comparable_fields = [
        "dataset_kind",
        "dataset_subset_scope",
        "molecule_count",
        "max_steps",
        "hard_max_steps",
        "optimizer_name",
        "learning_rate",
        "early_stop_monitor",
        "early_stop_patience",
        "early_stop_min_delta",
        "sample_step_count",
        "sample_step_size",
        "model_name",
        "backbone",
    ]
    for row in rows:
        for field in comparable_fields:
            if row[field] != baseline[field]:
                raise ValueError(f"Attribution inputs are not comparable on {field}")
        if row["dataset_subset_scope"] != FULL_DATABASE_SCOPE:
            raise ValueError("Attribution summary requires the full-database scope")
        if row["max_steps"] != FORMAL_10000_MAX_STEPS or row["hard_max_steps"] != FORMAL_10000_HARD_MAX_STEPS:
            raise ValueError("Attribution summary requires the frozen 2000-step horizon")
        if row["optimizer_name"] != "SGD" or row["learning_rate"] != FORMAL_10000_LEARNING_RATE:
            raise ValueError("Attribution summary requires the frozen optimizer settings")
        if row["early_stop_monitor"] != "validation_loss_after_step":
            raise ValueError("Attribution summary requires validation-loss early stopping")
        if row["early_stop_patience"] != FORMAL_10000_PATIENCE:
            raise ValueError("Attribution summary requires the frozen patience")
        if row["early_stop_min_delta"] != FORMAL_10000_MIN_DELTA:
            raise ValueError("Attribution summary requires the frozen min_delta")
        if row["sample_step_count"] != 1 or row["sample_step_size"] != FORMAL_10000_SAMPLE_STEP_SIZE:
            raise ValueError("Attribution summary requires frozen one-step sampling settings")
        if row["safety_labels"]["benchmark_result"] is not False:
            raise ValueError("Attribution inputs must be non-benchmark outputs")


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    grouped: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(row["split_type"], {}).setdefault(row["random_seed"], {})[row["corruption_mode"]] = row
    return grouped


def _per_split_seed_comparisons(
    grouped: dict[str, dict[int, dict[str, dict[str, Any]]]],
    split_types: tuple[str, ...],
    seeds: tuple[int, ...],
    modes: tuple[str, ...],
) -> list[dict[str, Any]]:
    comparisons = []
    for split_type in split_types:
        for seed in seeds:
            by_mode = grouped[split_type][int(seed)]
            if set(by_mode) != set(modes):
                raise ValueError(f"Missing attribution modes for split={split_type} seed={seed}")
            comparisons.append(
                {
                    "split_type": split_type,
                    "seed": int(seed),
                    "rows": by_mode,
                    "stiffness_vs_baselines": _stiffness_vs_baselines(by_mode, include_optional_physical=True),
                    "stop_reasons": {mode: by_mode[mode]["stop_reason"] for mode in modes},
                }
            )
    return comparisons


def _aggregate_by_split(
    rows: list[dict[str, Any]],
    split_types: tuple[str, ...],
    modes: tuple[str, ...],
) -> list[dict[str, Any]]:
    aggregate = []
    for split_type in split_types:
        split_rows = [row for row in rows if row["split_type"] == split_type]
        by_mode = {mode: [row for row in split_rows if row["corruption_mode"] == mode] for mode in modes}
        metric_means: dict[str, dict[str, float]] = {}
        for metric in CORE_LOWER_IS_BETTER_METRICS + OPTIONAL_PHYSICAL_LOWER_IS_BETTER_METRICS:
            mode_means = {}
            for mode, mode_rows in by_mode.items():
                values = [_metric_value(row, metric) for row in mode_rows]
                numeric = [value for value in values if value is not None]
                if len(numeric) == len(mode_rows):
                    mode_means[mode] = _mean(numeric)
            if mode_means:
                metric_means[metric] = mode_means
        aggregate.append(
            {
                "split_type": split_type,
                "metric_means_by_mode": metric_means,
                "stiffness_vs_baselines": _aggregate_stiffness_vs_baselines(metric_means, modes),
            }
        )
    return aggregate


def _stiffness_vs_baselines(
    by_mode: dict[str, dict[str, Any]],
    include_optional_physical: bool,
) -> list[dict[str, Any]]:
    metrics = CORE_LOWER_IS_BETTER_METRICS
    if include_optional_physical:
        metrics = metrics + OPTIONAL_PHYSICAL_LOWER_IS_BETTER_METRICS
    rows = []
    for baseline in [mode for mode in by_mode if mode != "stiffness_structured"]:
        for metric in metrics:
            baseline_value = _metric_value(by_mode[baseline], metric)
            stiffness_value = _metric_value(by_mode["stiffness_structured"], metric)
            if baseline_value is None or stiffness_value is None:
                continue
            rows.append(_lower_is_better(metric, baseline, baseline_value, stiffness_value))
    return rows


def _aggregate_stiffness_vs_baselines(
    metric_means: dict[str, dict[str, float]],
    modes: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    for metric, by_mode in metric_means.items():
        if "stiffness_structured" not in by_mode:
            continue
        for baseline in [mode for mode in modes if mode != "stiffness_structured" and mode in by_mode]:
            rows.append(_lower_is_better(metric, baseline, by_mode[baseline], by_mode["stiffness_structured"]))
    return rows


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    if value is None:
        return None
    return float(value)


def _lower_is_better(metric: str, baseline_mode: str, baseline_value: float, stiffness_value: float) -> dict[str, Any]:
    delta = stiffness_value - baseline_value
    if abs(delta) <= TIE_TOLERANCE:
        outcome = "tied"
    elif delta < 0.0:
        outcome = "better"
    else:
        outcome = "worse"
    return {
        "metric": metric,
        "baseline_mode": baseline_mode,
        "baseline_value": baseline_value,
        "stiffness_structured": stiffness_value,
        "stiffness_minus_baseline": delta,
        "stiffness_outcome": outcome,
        "direction": "lower_is_better",
    }


def _directional_verdict(
    per_split_seed: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
) -> dict[str, Any]:
    core_outcomes = [
        row["stiffness_outcome"]
        for comparison in per_split_seed
        for row in comparison["stiffness_vs_baselines"]
        if row["metric"] in CORE_LOWER_IS_BETTER_METRICS
    ]
    aggregate_core_outcomes = [
        row["stiffness_outcome"]
        for split_row in aggregate
        for row in split_row["stiffness_vs_baselines"]
        if row["metric"] in CORE_LOWER_IS_BETTER_METRICS
    ]
    if core_outcomes and all(outcome == "better" for outcome in core_outcomes):
        verdict = "stiffness_better_than_all_A_B_D_baselines_on_all_core_metrics"
    elif core_outcomes and all(outcome in {"better", "tied"} for outcome in core_outcomes):
        verdict = "stiffness_better_or_tied_than_all_A_B_D_baselines_on_all_core_metrics"
    else:
        verdict = "mixed_or_inconclusive"
    return {
        "verdict": verdict,
        "stiffness_win_count": core_outcomes.count("better"),
        "stiffness_tie_count": core_outcomes.count("tied"),
        "stiffness_loss_count": core_outcomes.count("worse"),
        "tracked_core_comparison_count": len(core_outcomes),
        "aggregate_stiffness_win_count": aggregate_core_outcomes.count("better"),
        "aggregate_stiffness_tie_count": aggregate_core_outcomes.count("tied"),
        "aggregate_stiffness_loss_count": aggregate_core_outcomes.count("worse"),
        "aggregate_core_comparison_count": len(aggregate_core_outcomes),
        "direction": "lower_is_better",
    }


def _physical_from_sampled(sampled: list[dict[str, Any]]) -> dict[str, Any]:
    finite_values = [bool(item["sampled_finite_coordinate"]) for item in sampled if "sampled_finite_coordinate" in item]
    bond_values = [
        float(item["sampled_mean_abs_bond_length_delta"])
        for item in sampled
        if isinstance(item.get("sampled_mean_abs_bond_length_delta"), (int, float))
    ]
    if not finite_values and not bond_values:
        status = "not_available_in_result_json"
    else:
        status = "available_from_sampled_molecules"
    return {
        "status": status,
        "finite_coordinate_rate": None if not finite_values else sum(1 for value in finite_values if value) / len(finite_values),
        "finite_coordinate_count": sum(1 for value in finite_values if value),
        "finite_coordinate_denominator": len(finite_values),
        "mean_abs_bond_length_delta": None if not bond_values else _mean(bond_values),
        "bond_delta_molecule_count": len(bond_values),
        "note": "Closest intramolecular physical-validity diagnostic currently available from sampled result JSONs; not PoseBusters/MMFF/UFF.",
    }


def _physical_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available_rows = [row for row in rows if row["physical_validity"]["status"] == "available_from_sampled_molecules"]
    return {
        "status": "available" if available_rows else "not_available_in_input_results",
        "available_row_count": len(available_rows),
        "row_count": len(rows),
        "metrics": {
            "finite_coordinate_rate": {
                row["run_name"]: row["physical_validity"]["finite_coordinate_rate"]
                for row in available_rows
            },
            "mean_abs_bond_length_delta": {
                row["run_name"]: row["physical_validity"]["mean_abs_bond_length_delta"]
                for row in available_rows
            },
        },
        "note": "Prepared physical-validity reporting path. Values are populated only for result JSONs that include sampled_finite_coordinate and sampled_mean_abs_bond_length_delta.",
    }


def _rotatable_from_sampled(sampled: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in sampled:
        value = item.get("rotatable_bond_count")
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, float) and not value.is_integer():
            continue
        if isinstance(value, (int, float)):
            rows.append(
                {
                    "rotatable_bond_count": int(value),
                    "split": item["split"],
                    "sampled_rmsd_to_target": float(item["sampled_rmsd_to_target"]),
                }
            )
    if not rows:
        return {
            "status": "not_available_in_result_json",
            "bins": {},
            "molecule_count_with_rotatable_bond_count": 0,
            "note": "Rotatable-bond stratification requires sampled_molecules[*].rotatable_bond_count.",
        }
    bins = {}
    for label, lower, upper in ROTATABLE_BINS:
        selected = [
            row
            for row in rows
            if row["rotatable_bond_count"] >= lower and (upper is None or row["rotatable_bond_count"] <= upper)
        ]
        if selected:
            bins[label] = {
                "count": len(selected),
                "mean_sampled_rmsd_to_target": _mean(row["sampled_rmsd_to_target"] for row in selected),
                "mean_sampled_rmsd_to_target_by_split": {
                    split: _mean(row["sampled_rmsd_to_target"] for row in selected if row["split"] == split)
                    if any(row["split"] == split for row in selected)
                    else None
                    for split in ["train", "val", "test"]
                },
            }
    return {
        "status": "available_from_sampled_molecules",
        "bins": bins,
        "molecule_count_with_rotatable_bond_count": len(rows),
        "note": "Non-benchmark rotatable-bond-count stratified RMSD diagnostic.",
    }


def _rotatable_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available_rows = [row for row in rows if row["rotatable_bond_stratification"]["status"] == "available_from_sampled_molecules"]
    by_split_mode_seed = {}
    for row in available_rows:
        by_split_mode_seed.setdefault(row["split_type"], {}).setdefault(row["corruption_mode"], {})[
            str(row["random_seed"])
        ] = row["rotatable_bond_stratification"]["bins"]
    return {
        "status": "available" if available_rows else "not_available_in_input_results",
        "available_row_count": len(available_rows),
        "row_count": len(rows),
        "bins": [label for label, _, _ in ROTATABLE_BINS],
        "by_split_mode_seed": by_split_mode_seed,
        "note": "Prepared rotatable-bond-count stratification path; populated when sampled molecules carry rotatable_bond_count metadata.",
    }


def _path_reuse_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = {}
    for split_type in sorted({row["split_type"] for row in rows}):
        split_rows = [row for row in rows if row["split_type"] == split_type]
        by_split[split_type] = {
            "processed_artifact_paths": sorted({row["processed_artifact_path"] for row in split_rows}),
            "split_paths": sorted({row["split_path"] for row in split_rows}),
        }
    return by_split


def _average_rmsd_by_split(sampled: list[dict[str, Any]]) -> dict[str, float]:
    return {
        split: _mean(item["sampled_rmsd_to_target"] for item in sampled if item["split"] == split)
        for split in ["train", "val", "test"]
    }


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("Cannot compute mean of an empty sequence")
    return sum(items) / len(items)


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Full-Database Attribution Phase Summary",
        "",
        "This is a reporting-only A/B/D/E attribution-phase summary for completed full-database GEOM-Drugs formal outputs. It is not a benchmark result.",
        "",
        "## Scope",
        "",
        f"- Dataset scope: `{summary['dataset_scope']}`",
        f"- Molecule count: `{summary['dataset']['molecule_count']}`",
        f"- Split types: `{summary['split_types']}`",
        f"- Seed set: `{summary['seed_set']}`",
        f"- Modes: `{summary['attribution_modes']}`",
        f"- Max steps / hard cap: `{summary['training_horizon']['max_steps']}` / `{summary['training_horizon']['hard_max_steps']}`",
        f"- Verdict: `{summary['directional_verdict']['verdict']}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Split | Metric | Isotropic | Harmonic | Torsion-only | Stiffness |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split_row in summary["aggregate_metric_comparisons"]:
        metric_means = split_row["metric_means_by_mode"]
        for metric in CORE_LOWER_IS_BETTER_METRICS + OPTIONAL_PHYSICAL_LOWER_IS_BETTER_METRICS:
            if metric not in metric_means:
                continue
            by_mode = metric_means[metric]
            lines.append(
                f"| {split_row['split_type']} | {metric} | {_fmt(by_mode.get('isotropic'))} | "
                f"{_fmt(by_mode.get('harmonic_prior'))} | {_fmt(by_mode.get('torsion_only'))} | "
                f"{_fmt(by_mode.get('stiffness_structured'))} |"
            )
    lines.extend(
        [
            "",
            "## Stiffness Versus Baselines",
            "",
            "| Split | Baseline | Metric | Baseline mean | Stiffness mean | Delta | Outcome |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for split_row in summary["aggregate_metric_comparisons"]:
        for row in split_row["stiffness_vs_baselines"]:
            lines.append(
                f"| {split_row['split_type']} | {row['baseline_mode']} | {row['metric']} | "
                f"{row['baseline_value']:.8g} | {row['stiffness_structured']:.8g} | "
                f"{row['stiffness_minus_baseline']:.8g} | {row['stiffness_outcome']} |"
            )
    lines.extend(
        [
            "",
            "## Physical Diagnostics",
            "",
            f"- Status: `{summary['physical_validity_diagnostics']['status']}`",
            f"- Rows with physical diagnostics: `{summary['physical_validity_diagnostics']['available_row_count']}` / `{summary['physical_validity_diagnostics']['row_count']}`",
            "- Current prepared diagnostic fields: sampled finite-coordinate rate and sampled mean absolute bond-length delta.",
            "- These are intramolecular diagnostics only, not PoseBusters/MMFF/UFF benchmark evaluation.",
            "",
            "## Rotatable-Bond Stratification",
            "",
            f"- Status: `{summary['rotatable_bond_stratification']['status']}`",
            f"- Rows with rotatable metadata: `{summary['rotatable_bond_stratification']['available_row_count']}` / `{summary['rotatable_bond_stratification']['row_count']}`",
            f"- Bins: `{summary['rotatable_bond_stratification']['bins']}`",
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
            "No training, sampling, preprocessing, scheduler action, GPU use, or environment change was performed by this summary.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.8g}"
