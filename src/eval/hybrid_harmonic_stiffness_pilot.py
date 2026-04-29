"""Reporting utilities for the hybrid harmonic + stiffness pilot."""

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
    FORMAL_FULL_DATABASE_CHECKPOINT_SCOPE,
    FORMAL_FULL_DATABASE_RESULT_KIND,
    FORMAL_FULL_DATABASE_RUNTIME_SCOPE,
)


DEFAULT_HYBRID_PILOT_SUMMARY_PATH = Path(
    "experiments/results/geom_full_database_hybrid_harmonic_stiffness_pilot_summary.json"
)
DEFAULT_HYBRID_PILOT_REPORT_PATH = Path(
    "reports/tables/geom_full_database_hybrid_harmonic_stiffness_pilot_summary.md"
)
HYBRID_PILOT_SUMMARY_RESULT_KIND = "geom_drugs_full_database_hybrid_harmonic_stiffness_pilot_summary_not_benchmark"
BASELINE_MODES = ("harmonic_prior", "stiffness_structured", "isotropic", "torsion_only")
HYBRID_MODE_STRENGTHS = (
    ("hybrid_harmonic_stiffness_v1", 0.10),
    ("hybrid_harmonic_stiffness_v1", 0.25),
    ("hybrid_harmonic_stiffness_v1", 0.50),
    ("harmonic_stiffness_residual_v1", 0.10),
    ("harmonic_stiffness_residual_v1", 0.25),
    ("harmonic_stiffness_residual_v1", 0.50),
)
BASELINE_LABELS = {
    "harmonic_prior": "harmonic",
    "stiffness_structured": "stiffness",
    "isotropic": "isotropic",
    "torsion_only": "torsion_only",
}
CORE_LOWER_IS_BETTER_METRICS = (
    "best_validation_loss",
    "reloaded_validation_loss",
    "sampled_rmsd_to_target_overall_mean",
    "sampled_rmsd_to_target_train_mean",
    "sampled_rmsd_to_target_val_mean",
    "sampled_rmsd_to_target_test_mean",
)
OPTIONAL_PHYSICAL_LOWER_IS_BETTER_METRICS = ("sampled_mean_abs_bond_length_delta",)
TIE_TOLERANCE = 1.0e-12


def write_hybrid_harmonic_stiffness_pilot_summary(
    result_paths: list[str | Path] | None = None,
    results_root: str | Path = "experiments/results",
    summary_path: str | Path = DEFAULT_HYBRID_PILOT_SUMMARY_PATH,
    report_path: str | Path = DEFAULT_HYBRID_PILOT_REPORT_PATH,
    split_type: str = "random",
    seed: int = 0,
    require_hybrid_results: bool = False,
) -> dict[str, Path]:
    paths = _resolve_result_paths(
        result_paths=result_paths,
        results_root=Path(results_root),
        split_type=split_type,
        seed=int(seed),
    )
    rows = _load_rows(paths, expected_split_type=split_type, expected_seed=int(seed))
    summary = _build_summary(
        rows=rows,
        input_paths=paths,
        split_type=split_type,
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
    split_type: str,
    seed: int,
) -> list[Path]:
    if result_paths:
        return [Path(path) for path in result_paths]
    paths = []
    split_label = "" if split_type == "random" else f"_{split_type}"
    for mode in BASELINE_MODES:
        paths.append(
            results_root
            / f"geom_formal_full_database{split_label}_train_{BASELINE_LABELS[mode]}_steps{FORMAL_10000_MAX_STEPS}_seed{seed}.json"
        )
    for mode, strength in HYBRID_MODE_STRENGTHS:
        paths.append(
            results_root
            / (
                f"geom_formal_full_database{split_label}_train_{mode}_{_lambda_label(strength)}"
                f"_steps{FORMAL_10000_MAX_STEPS}_seed{seed}.json"
            )
        )
    return paths


def _load_rows(paths: list[Path], expected_split_type: str, expected_seed: int) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        payload = _load_result(path, expected_split_type=expected_split_type, expected_seed=expected_seed)
        rows.append(_row_from_result(path, payload))
    _validate_comparable_rows(rows)
    return rows


def _load_result(path: Path, expected_split_type: str, expected_seed: int) -> dict[str, Any]:
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
    if int(payload.get("random_seed")) != int(expected_seed):
        raise ValueError(f"Result seed mismatch: {path}")
    dataset = payload.get("dataset") or {}
    if dataset.get("split_type") != expected_split_type:
        raise ValueError(f"Result split_type mismatch: {path}")
    metrics = payload.get("metrics") or {}
    for key in ["best_validation_loss", "reloaded_validation_loss", "sampled_molecules"]:
        if key not in metrics:
            raise ValueError(f"Result metrics missing {key}: {path}")
    sampled = metrics["sampled_molecules"]
    if not isinstance(sampled, list) or len(sampled) != int(payload["sample_count"]):
        raise ValueError(f"Result sampled_molecules length does not match sample_count: {path}")
    return payload


def _row_from_result(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    sampled = metrics["sampled_molecules"]
    physical = _physical_from_sampled(sampled)
    by_split = _average_rmsd_by_split(sampled)
    return {
        "run_name": payload["run_name"],
        "source_path": str(path),
        "corruption_mode": payload["corruption_mode"],
        "source_strength": _source_strength(payload),
        "random_seed": int(payload["random_seed"]),
        "split_type": payload["dataset"]["split_type"],
        "dataset_kind": payload["dataset"]["kind"],
        "dataset_subset_scope": payload["dataset"]["subset_scope"],
        "molecule_count": int(payload["dataset"]["molecule_count"]),
        "split_counts": {key: int(payload["dataset"]["split_counts"][key]) for key in ["train", "val", "test"]},
        "model_name": payload["model_name"],
        "backbone": payload["backbone"],
        "max_steps": int(payload["max_steps"]),
        "hard_max_steps": int(payload["hard_max_steps"]),
        "executed_train_steps": int(metrics["executed_train_steps"]),
        "optimizer_name": payload["optimizer_name"],
        "learning_rate": float(payload["learning_rate"]),
        "early_stop_monitor": payload["early_stop_monitor"],
        "early_stop_patience": int(payload["early_stop_patience"]),
        "early_stop_min_delta": float(payload["early_stop_min_delta"]),
        "stopped_early": bool(payload["stopped_early"]),
        "stop_reason": payload["early_stop_reason"],
        "sample_count": int(payload["sample_count"]),
        "sample_step_count": int(payload["sample_step_count"]),
        "sample_step_size": float(payload["sample_step_size"]),
        "best_validation_loss": float(metrics["best_validation_loss"]),
        "best_validation_loss_step": int(metrics["best_validation_loss_step"]),
        "reloaded_validation_loss": float(metrics["reloaded_validation_loss"]),
        "sampled_rmsd_to_target_overall_mean": _mean(item["sampled_rmsd_to_target"] for item in sampled),
        "sampled_rmsd_to_target_by_split_mean": by_split,
        "sampled_rmsd_to_target_train_mean": by_split["train"],
        "sampled_rmsd_to_target_val_mean": by_split["val"],
        "sampled_rmsd_to_target_test_mean": by_split["test"],
        "physical_validity": physical,
        "sampled_mean_abs_bond_length_delta": physical.get("mean_abs_bond_length_delta"),
        "runtime_seconds": float(payload["runtime_seconds"]),
        "runtime_scope": payload["runtime_scope"],
        "runtime_seconds_reliable": bool(payload["runtime_seconds_reliable"]),
        "scheduler": payload.get("scheduler"),
        "slurm_account": payload.get("slurm_account"),
        "hardware": payload.get("hardware"),
        "gpu_used": bool(payload.get("gpu_used")),
    }


def _source_strength(payload: dict[str, Any]) -> float | None:
    if isinstance(payload.get("hybrid_source_strength"), (int, float)):
        return float(payload["hybrid_source_strength"])
    run_name = str(payload.get("run_name", ""))
    marker = "lambda"
    if marker not in run_name:
        return None
    suffix = run_name.split(marker, 1)[1][:3]
    if len(suffix) != 3 or not suffix.isdigit():
        return None
    return int(suffix) / 100.0


def _validate_comparable_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    baseline = rows[0]
    comparable_fields = [
        "dataset_kind",
        "dataset_subset_scope",
        "molecule_count",
        "split_type",
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
                raise ValueError(f"Hybrid pilot inputs are not comparable on {field}")
        if row["max_steps"] != FORMAL_10000_MAX_STEPS or row["hard_max_steps"] != FORMAL_10000_HARD_MAX_STEPS:
            raise ValueError("Hybrid pilot summary requires the frozen 2000-step horizon")
        if row["optimizer_name"] != "SGD" or row["learning_rate"] != FORMAL_10000_LEARNING_RATE:
            raise ValueError("Hybrid pilot summary requires the frozen optimizer settings")
        if row["early_stop_patience"] != FORMAL_10000_PATIENCE:
            raise ValueError("Hybrid pilot summary requires the frozen patience")
        if row["early_stop_min_delta"] != FORMAL_10000_MIN_DELTA:
            raise ValueError("Hybrid pilot summary requires the frozen min_delta")
        if row["sample_step_count"] != 1 or row["sample_step_size"] != FORMAL_10000_SAMPLE_STEP_SIZE:
            raise ValueError("Hybrid pilot summary requires frozen one-step sampling settings")


def _build_summary(
    rows: list[dict[str, Any]],
    input_paths: list[Path],
    split_type: str,
    seed: int,
    require_hybrid_results: bool,
) -> dict[str, Any]:
    by_mode = _group_rows(rows)
    baseline_rows = {mode: by_mode.get((mode, None)) for mode in BASELINE_MODES}
    missing_baselines = [mode for mode, row in baseline_rows.items() if row is None]
    hybrid_rows = {
        (mode, strength): by_mode.get((mode, round(strength, 2)))
        for mode, strength in HYBRID_MODE_STRENGTHS
    }
    missing_hybrids = [
        {"corruption_mode": mode, "source_strength": strength}
        for (mode, strength), row in hybrid_rows.items()
        if row is None
    ]
    if require_hybrid_results and missing_hybrids:
        raise FileNotFoundError(f"Missing expected hybrid pilot result rows: {missing_hybrids}")

    hybrid_comparisons = []
    for mode, strength in HYBRID_MODE_STRENGTHS:
        row = hybrid_rows[(mode, strength)]
        hybrid_comparisons.append(
            _hybrid_comparison(
                mode=mode,
                strength=strength,
                hybrid_row=row,
                baseline_rows=baseline_rows,
            )
        )
    physical_rows = [row for row in rows if row["physical_validity"]["status"] == "available_from_sampled_molecules"]
    summary = {
        "summary_name": "geom_full_database_hybrid_harmonic_stiffness_pilot_summary",
        "result_kind": HYBRID_PILOT_SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_full_database_hybrid_harmonic_stiffness_pilot_report",
        "hardware": "reporting_only",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "split_type": split_type,
        "seed": int(seed),
        "dataset_scope": FULL_DATABASE_SCOPE,
        "input_result_paths": [str(path) for path in input_paths],
        "input_result_paths_found": [row["source_path"] for row in rows],
        "row_count": len(rows),
        "baseline_modes": list(BASELINE_MODES),
        "hybrid_mode_strengths": [
            {"corruption_mode": mode, "source_strength": strength}
            for mode, strength in HYBRID_MODE_STRENGTHS
        ],
        "coverage": {
            "baseline_rows_available": [mode for mode, row in baseline_rows.items() if row is not None],
            "missing_baselines": missing_baselines,
            "hybrid_rows_available": [
                {"corruption_mode": mode, "source_strength": strength}
                for (mode, strength), row in hybrid_rows.items()
                if row is not None
            ],
            "missing_hybrids": missing_hybrids,
            "complete_phase1_hybrid_matrix": not missing_hybrids,
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
        "physical_proxy_diagnostics": {
            "status": "available" if physical_rows else "not_available_in_input_results",
            "available_row_count": len(physical_rows),
            "row_count": len(rows),
            "metrics": {
                row["run_name"]: row["physical_validity"] for row in physical_rows
            },
            "note": "Available physical proxies are sampled finite-coordinate rate and sampled mean absolute bond-length delta only. This is not PoseBusters/MMFF/UFF/xTB.",
        },
        "directional_verdict": _directional_verdict(hybrid_comparisons, missing_baselines),
        "safety_labels": {
            "benchmark_result": False,
            "non_benchmark": True,
            "hybrid_pilot_summary": True,
            "training_executed_by_summary": False,
            "sampling_executed_by_summary": False,
            "raw_data_modified": False,
            "environment_modified": False,
            "gpu_used_by_summary": False,
            "result_scope": "full_database_random_seed0_hybrid_harmonic_stiffness_pilot_reporting_only",
        },
        "notes": [
            "Reporting-only minimal hybrid harmonic + stiffness pilot summary for completed full-database formal outputs. This is not a benchmark result.",
            "Success is not claimed unless at least one hybrid beats harmonic_prior on all tracked core loss/RMSD metrics.",
            "The summary compares hybrids against the existing harmonic_prior, stiffness_structured, isotropic, and torsion_only full-database random seed0 results when available.",
            "Missing hybrid result JSONs are reported as pending rather than treated as wins, losses, or ties.",
            "No training, sampling, preprocessing, scheduler action, GPU use, PoseBusters, MMFF/UFF benchmark, xTB, raw-data mutation, or environment change is performed by this summary.",
        ],
    }
    return summary


def _group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, float | None], dict[str, Any]]:
    grouped = {}
    for row in rows:
        strength = row["source_strength"]
        key = (row["corruption_mode"], None if strength is None else round(float(strength), 2))
        grouped[key] = row
    return grouped


def _hybrid_comparison(
    mode: str,
    strength: float,
    hybrid_row: dict[str, Any] | None,
    baseline_rows: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    if hybrid_row is None:
        return {
            "corruption_mode": mode,
            "source_strength": strength,
            "status": "missing_hybrid_result",
            "result_available": False,
            "core_metrics": {},
            "optional_physical_metrics": {},
            "beats_harmonic_prior_on_all_core_metrics": False,
            "preserves_or_improves_stiffness_advantage_over_isotropic_torsion": False,
            "failure_modes": ["missing_hybrid_result_json"],
        }
    core = _comparison_table(hybrid_row, baseline_rows, CORE_LOWER_IS_BETTER_METRICS)
    optional = _comparison_table(hybrid_row, baseline_rows, OPTIONAL_PHYSICAL_LOWER_IS_BETTER_METRICS)
    harmonic_core = core.get("harmonic_prior", {})
    isotropic_core = core.get("isotropic", {})
    torsion_core = core.get("torsion_only", {})
    beats_harmonic = bool(harmonic_core) and all(item["outcome"] == "better" for item in harmonic_core.values())
    preserves_stiffness_advantage = (
        bool(isotropic_core)
        and bool(torsion_core)
        and all(item["outcome"] in {"better", "tied"} for item in isotropic_core.values())
        and all(item["outcome"] in {"better", "tied"} for item in torsion_core.values())
    )
    failure_modes = []
    if not beats_harmonic:
        failure_modes.append("does_not_beat_harmonic_prior_on_all_tracked_core_metrics")
    if not preserves_stiffness_advantage:
        failure_modes.append("does_not_preserve_stiffness_advantage_over_isotropic_and_torsion_only")
    physical = hybrid_row["physical_validity"]
    if physical.get("finite_coordinate_rate") is not None and physical["finite_coordinate_rate"] < 1.0:
        failure_modes.append("sampled_nonfinite_coordinate_detected")
    if hybrid_row["stopped_early"]:
        failure_modes.append(f"stopped_early:{hybrid_row['stop_reason']}")
    return {
        "corruption_mode": mode,
        "source_strength": strength,
        "run_name": hybrid_row["run_name"],
        "source_path": hybrid_row["source_path"],
        "status": "available",
        "result_available": True,
        "metrics": {metric: hybrid_row.get(metric) for metric in CORE_LOWER_IS_BETTER_METRICS},
        "physical_proxy_diagnostics": physical,
        "core_metrics": core,
        "optional_physical_metrics": optional,
        "beats_harmonic_prior_on_all_core_metrics": beats_harmonic,
        "preserves_or_improves_stiffness_advantage_over_isotropic_torsion": preserves_stiffness_advantage,
        "failure_modes": failure_modes,
    }


def _comparison_table(
    hybrid_row: dict[str, Any],
    baseline_rows: dict[str, dict[str, Any] | None],
    metrics: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    table = {}
    for baseline_mode, baseline_row in baseline_rows.items():
        if baseline_row is None:
            continue
        metric_rows = {}
        for metric in metrics:
            hybrid_value = hybrid_row.get(metric)
            baseline_value = baseline_row.get(metric)
            if hybrid_value is None or baseline_value is None:
                continue
            metric_rows[metric] = _lower_is_better(metric, float(baseline_value), float(hybrid_value))
        if metric_rows:
            table[baseline_mode] = metric_rows
    return table


def _lower_is_better(metric: str, baseline_value: float, hybrid_value: float) -> dict[str, Any]:
    delta = hybrid_value - baseline_value
    if abs(delta) <= TIE_TOLERANCE:
        outcome = "tied"
    elif delta < 0.0:
        outcome = "better"
    else:
        outcome = "worse"
    return {
        "metric": metric,
        "baseline_value": baseline_value,
        "hybrid_value": hybrid_value,
        "hybrid_minus_baseline": delta,
        "outcome": outcome,
        "direction": "lower_is_better",
    }


def _directional_verdict(comparisons: list[dict[str, Any]], missing_baselines: list[str]) -> dict[str, Any]:
    available = [row for row in comparisons if row["result_available"]]
    harmonic_winners = [row for row in available if row["beats_harmonic_prior_on_all_core_metrics"]]
    stiffness_advantage_rows = [
        row for row in available if row["preserves_or_improves_stiffness_advantage_over_isotropic_torsion"]
    ]
    if missing_baselines:
        verdict = "missing_required_baseline_results"
    elif not available:
        verdict = "hybrid_results_not_available"
    elif harmonic_winners:
        verdict = "at_least_one_hybrid_beats_harmonic_prior_on_tracked_core_metrics"
    elif stiffness_advantage_rows:
        verdict = "hybrids_preserve_stiffness_advantage_but_do_not_beat_harmonic_prior"
    else:
        verdict = "hybrid_modulation_mixed_or_harmful"
    return {
        "verdict": verdict,
        "success_claim_supported": bool(harmonic_winners),
        "best_harmonic_beating_hybrids": [
            {"corruption_mode": row["corruption_mode"], "source_strength": row["source_strength"]}
            for row in harmonic_winners
        ],
        "available_hybrid_result_count": len(available),
        "missing_baselines": missing_baselines,
        "direction": "lower_is_better",
    }


def _baseline_metric_table(baseline_rows: dict[str, dict[str, Any] | None]) -> dict[str, dict[str, float]]:
    table = {}
    for mode, row in baseline_rows.items():
        if row is None:
            continue
        table[mode] = {metric: row[metric] for metric in CORE_LOWER_IS_BETTER_METRICS}
        if row.get("sampled_mean_abs_bond_length_delta") is not None:
            table[mode]["sampled_mean_abs_bond_length_delta"] = row["sampled_mean_abs_bond_length_delta"]
    return table


def _dataset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "not_available_until_input_results_exist"}
    row = rows[0]
    return {
        "name": "geom_drugs",
        "kind": row["dataset_kind"],
        "subset_scope": row["dataset_subset_scope"],
        "molecule_count": row["molecule_count"],
        "split_type": row["split_type"],
        "split_counts": row["split_counts"],
    }


def _physical_from_sampled(sampled: list[dict[str, Any]]) -> dict[str, Any]:
    finite_values = [bool(item["sampled_finite_coordinate"]) for item in sampled if "sampled_finite_coordinate" in item]
    bond_values = [
        float(item["sampled_mean_abs_bond_length_delta"])
        for item in sampled
        if isinstance(item.get("sampled_mean_abs_bond_length_delta"), (int, float))
    ]
    bond_count_checked_values = [
        float(item["sampled_bond_count_checked"])
        for item in sampled
        if isinstance(item.get("sampled_bond_count_checked"), (int, float))
    ]
    if not finite_values and not bond_values and not bond_count_checked_values:
        status = "not_available_in_result_json"
    else:
        status = "available_from_sampled_molecules"
    return {
        "status": status,
        "finite_coordinate_rate": None
        if not finite_values
        else sum(1 for value in finite_values if value) / len(finite_values),
        "finite_coordinate_count": sum(1 for value in finite_values if value),
        "finite_coordinate_denominator": len(finite_values),
        "mean_abs_bond_length_delta": None if not bond_values else _mean(bond_values),
        "bond_delta_molecule_count": len(bond_values),
        "mean_bond_count_checked": None if not bond_count_checked_values else _mean(bond_count_checked_values),
        "bond_count_checked_molecule_count": len(bond_count_checked_values),
        "note": "Closest intramolecular physical proxy currently available from sampled result JSONs; not PoseBusters/MMFF/UFF/xTB.",
    }


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


def _lambda_label(value: float) -> str:
    return f"lambda{int(round(float(value) * 100)):03d}"


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Full-Database Hybrid Harmonic + Stiffness Pilot Summary",
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
        f"- Success claim supported: `{summary['directional_verdict']['success_claim_supported']}`",
        "",
        "## Coverage",
        "",
        f"- Baselines available: `{summary['coverage']['baseline_rows_available']}`",
        f"- Missing baselines: `{summary['coverage']['missing_baselines']}`",
        f"- Hybrid rows available: `{summary['coverage']['hybrid_rows_available']}`",
        f"- Missing hybrids: `{summary['coverage']['missing_hybrids']}`",
        "",
        "## Existing Reference Metrics",
        "",
        "| Mode | Best val loss | Reloaded val loss | RMSD overall | RMSD train | RMSD val | RMSD test | Bond delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in BASELINE_MODES:
        row = summary["baseline_metric_table"].get(mode, {})
        lines.append(
            f"| {mode} | {_fmt(row.get('best_validation_loss'))} | {_fmt(row.get('reloaded_validation_loss'))} | "
            f"{_fmt(row.get('sampled_rmsd_to_target_overall_mean'))} | "
            f"{_fmt(row.get('sampled_rmsd_to_target_train_mean'))} | "
            f"{_fmt(row.get('sampled_rmsd_to_target_val_mean'))} | "
            f"{_fmt(row.get('sampled_rmsd_to_target_test_mean'))} | "
            f"{_fmt(row.get('sampled_mean_abs_bond_length_delta'))} |"
        )
    lines.extend(
        [
            "",
            "## Hybrid Comparisons",
            "",
            "| Mode | Lambda | Status | Best val loss | RMSD overall | Beats harmonic? | Preserves A/D advantage? | Failure modes |",
            "|---|---:|---|---:|---:|---|---|---|",
        ]
    )
    for row in summary["hybrid_comparisons"]:
        metrics = row.get("metrics", {})
        lines.append(
            f"| {row['corruption_mode']} | {row['source_strength']} | {row['status']} | "
            f"{_fmt(metrics.get('best_validation_loss'))} | "
            f"{_fmt(metrics.get('sampled_rmsd_to_target_overall_mean'))} | "
            f"{row['beats_harmonic_prior_on_all_core_metrics']} | "
            f"{row['preserves_or_improves_stiffness_advantage_over_isotropic_torsion']} | "
            f"{', '.join(row['failure_modes']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Physical Proxies",
            "",
            f"- Status: `{summary['physical_proxy_diagnostics']['status']}`",
            f"- Rows with physical proxies: `{summary['physical_proxy_diagnostics']['available_row_count']}` / `{summary['physical_proxy_diagnostics']['row_count']}`",
            "- Current fields are sampled finite-coordinate rate and sampled mean absolute bond-length delta when present.",
            "- These are not PoseBusters, MMFF/UFF benchmark jobs, or xTB.",
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
            "No training, sampling, preprocessing, scheduler action, GPU use, physical benchmark job, or environment change was performed by this summary.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.8g}"
