"""Compare bounded micro-pilot checkpoint-and-sampling outputs."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any


DEFAULT_RESULT_PATHS = [
    Path("experiments/results/geom_micro_pilot_ckpt_sample_isotropic_mols6_steps5_seed0.json"),
    Path("experiments/results/geom_micro_pilot_ckpt_sample_stiffness_mols6_steps5_seed0.json"),
]
DEFAULT_MULTI_SEED_RESULT_PATHS = [
    Path(f"experiments/results/geom_micro_pilot_ckpt_sample_isotropic_mols6_steps5_seed{seed}.json")
    for seed in [0, 1, 2]
] + [
    Path(f"experiments/results/geom_micro_pilot_ckpt_sample_stiffness_mols6_steps5_seed{seed}.json")
    for seed in [0, 1, 2]
]
DEFAULT_SUMMARY_PATH = Path("experiments/results/geom_micro_pilot_ckpt_sample_summary.json")
DEFAULT_REPORT_PATH = Path("reports/tables/geom_micro_pilot_ckpt_sample_summary.md")
DEFAULT_MULTI_SEED_SUMMARY_PATH = Path("experiments/results/geom_micro_pilot_ckpt_sample_multiseed_summary.json")
DEFAULT_MULTI_SEED_REPORT_PATH = Path("reports/tables/geom_micro_pilot_ckpt_sample_multiseed_summary.md")
INPUT_RESULT_KIND = "tiny_real_data_bounded_micro_pilot_checkpoint_sampling_not_benchmark"
SUMMARY_RESULT_KIND = "tiny_real_data_bounded_micro_pilot_checkpoint_sampling_comparison_not_benchmark"
MULTI_SEED_SUMMARY_RESULT_KIND = (
    "tiny_real_data_bounded_micro_pilot_checkpoint_sampling_multiseed_summary_not_benchmark"
)
RUNTIME_RATIO_WARNING_THRESHOLD = 5.0
RUNTIME_ABS_WARNING_SECONDS = 1.0
TIE_TOLERANCE = 1.0e-12
COMPARABLE_METRICS = (
    "reloaded_loss",
    "sampled_rmsd_to_target_overall_mean",
    "sampled_rmsd_to_target_train_mean",
    "sampled_rmsd_to_target_val_mean",
    "sampled_rmsd_to_target_test_mean",
)


def write_micro_pilot_checkpoint_sampling_comparison(
    result_paths: list[str | Path] | None = None,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Path]:
    start = time.time()
    paths = [Path(path) for path in (result_paths or DEFAULT_RESULT_PATHS)]
    payloads = [_load_result(path) for path in paths]
    rows = [_row_from_result(path, payload) for path, payload in zip(paths, payloads)]
    by_mode = {row["corruption_mode"]: row for row in rows}
    if set(by_mode) != {"isotropic", "stiffness_structured"}:
        raise ValueError(f"Expected isotropic and stiffness_structured results, got {sorted(by_mode)}")
    _validate_comparable(rows)

    metric_comparisons = _metric_comparisons(by_mode)
    per_molecule = _per_molecule_comparisons(by_mode)
    runtime_audit = _runtime_audit(by_mode)
    summary = {
        "summary_name": "geom_micro_pilot_ckpt_sample_summary",
        "result_kind": SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_cpu_micro_pilot_checkpoint_sampling_report",
        "hardware": "cpu_only",
        "gpu_used": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "larger_real_training_run_started": False,
        "input_result_paths": [str(path) for path in paths],
        "row_count": len(rows),
        "rows": rows,
        "runtime_audit": runtime_audit,
        "metric_comparisons": metric_comparisons,
        "per_molecule_sampled_rmsd": per_molecule,
        "runtime_seconds": time.time() - start,
        "notes": [
            "Reporting-only comparison of two bounded six-molecule micro-pilot checkpoint-and-sampling outputs.",
            "All lower-is-better comparisons are direct diagnostics on the same bounded artifact, split, seed, optimizer, step count, and sampling settings.",
            "These diagnostics are not benchmark results and must not be used as evidence of model quality.",
            "No raw-data modification, environment mutation, GPU execution, SLURM job, training, sampling, checkpointing, or larger real training run was performed by this comparison.",
        ],
    }
    summary_path = Path(summary_path)
    report_path = Path(report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def write_micro_pilot_checkpoint_sampling_multiseed_summary(
    result_paths: list[str | Path] | None = None,
    summary_path: str | Path = DEFAULT_MULTI_SEED_SUMMARY_PATH,
    report_path: str | Path = DEFAULT_MULTI_SEED_REPORT_PATH,
    expected_seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, Path]:
    paths = [Path(path) for path in (result_paths or DEFAULT_MULTI_SEED_RESULT_PATHS)]
    payloads = [_load_result(path) for path in paths]
    rows = [_row_from_result(path, payload) for path, payload in zip(paths, payloads)]
    _validate_multiseed_rows(rows, expected_seeds)
    rows_by_seed = _rows_by_seed(rows)
    per_seed_comparisons = [
        {
            "seed": seed,
            "metric_comparisons": _metric_comparisons(rows_by_seed[seed]),
            "runtime_audit": _runtime_audit(rows_by_seed[seed]),
        }
        for seed in sorted(rows_by_seed)
    ]
    aggregate_metric_comparisons = _aggregate_metric_comparisons(per_seed_comparisons)
    consistency = _stiffness_consistency(aggregate_metric_comparisons)
    runtime_summary = _multiseed_runtime_summary(rows)
    summary = {
        "summary_name": "geom_micro_pilot_ckpt_sample_multiseed_summary",
        "result_kind": MULTI_SEED_SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_cpu_micro_pilot_checkpoint_sampling_multiseed_report",
        "hardware": "cpu_only",
        "gpu_used": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "larger_real_training_run_started": False,
        "seed_set": list(expected_seeds),
        "seed_count": len(expected_seeds),
        "mode_count": 2,
        "input_result_paths": [str(path) for path in paths],
        "row_count": len(rows),
        "rows": rows,
        "per_seed_comparisons": per_seed_comparisons,
        "aggregate_metric_comparisons": aggregate_metric_comparisons,
        "stiffness_consistency": consistency,
        "runtime_summary": runtime_summary,
        "runtime_seconds": None,
        "notes": [
            "Reporting-only aggregation of bounded six-molecule micro-pilot checkpoint-and-sampling outputs across a fixed tiny seed set.",
            "Each input row uses the same bounded artifact, split, model, optimizer, five-step cap, and one-step sampling settings.",
            "Loss and sampled RMSD values are tiny pilot diagnostics only, not benchmark metrics or evidence of model quality.",
            "Runtime metadata is kept for accounting checks only; tiny runtime differences are not interpreted as performance results.",
            "Summary generation runtime is intentionally omitted so the report is deterministic for identical inputs.",
            "No raw-data modification, environment mutation, GPU execution, SLURM job, training, sampling, checkpointing, or larger real training run was performed by this summary.",
        ],
    }
    summary_path = Path(summary_path)
    report_path = Path(report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_multiseed_report(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def _load_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Micro-pilot checkpoint/sampling result does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("result_kind") != INPUT_RESULT_KIND:
        raise ValueError(f"Result is not a bounded micro-pilot checkpoint/sampling output: {path}")
    if payload.get("benchmark_result") is not False:
        raise ValueError(f"Result is not benchmark_result=false: {path}")
    if payload.get("hardware") != "cpu_only" or payload.get("gpu_used") is not False:
        raise ValueError(f"Result is not explicitly CPU-only: {path}")
    if payload.get("scheduler") is not None or payload.get("slurm_account") is not None:
        raise ValueError(f"Result unexpectedly records scheduler use: {path}")
    for key in ["pilot_run", "bounded_pilot", "micro_pilot_run", "training_executed", "checkpoint_written", "checkpoint_reloaded", "sampling_executed"]:
        if payload.get(key) is not True:
            raise ValueError(f"Result missing expected true label {key}: {path}")
    if payload.get("larger_real_training_run_started") is not False:
        raise ValueError(f"Result unexpectedly started a larger real training run: {path}")
    if payload.get("checkpoint_scope") != "disposable_micro_pilot_checkpoint_sampling_only":
        raise ValueError(f"Result checkpoint scope is not disposable micro-pilot only: {path}")
    dataset = payload.get("dataset") or {}
    if dataset.get("subset_scope") != "micro_pilot_processed_dry_run":
        raise ValueError(f"Result is not from the micro-pilot subset: {path}")
    if dataset.get("split_counts") != {"test": 1, "train": 4, "val": 1}:
        raise ValueError(f"Result does not have 4/1/1 train/val/test counts: {path}")
    metrics = payload.get("metrics") or {}
    if not isinstance(metrics.get("reloaded_loss"), (int, float)):
        raise ValueError(f"Result missing numeric reloaded_loss: {path}")
    sampled = metrics.get("sampled_molecules")
    if not isinstance(sampled, list) or len(sampled) != 6:
        raise ValueError(f"Result must contain six sampled molecule diagnostics: {path}")
    for item in sampled:
        if item.get("split") not in {"train", "val", "test"}:
            raise ValueError(f"Sampled molecule has invalid split in {path}")
        if not isinstance(item.get("sampled_rmsd_to_target"), (int, float)):
            raise ValueError(f"Sampled molecule missing numeric sampled_rmsd_to_target in {path}")
    return payload


def _row_from_result(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    sampled = metrics["sampled_molecules"]
    split_averages = _average_rmsd_by_split(sampled)
    return {
        "run_name": payload["run_name"],
        "source_path": str(path),
        "corruption_mode": payload["corruption_mode"],
        "random_seed": payload["random_seed"],
        "result_kind": payload["result_kind"],
        "dataset_kind": payload["dataset"]["kind"],
        "dataset_subset_scope": payload["dataset"]["subset_scope"],
        "split_type": payload["dataset"]["split_type"],
        "split_counts": payload["dataset"]["split_counts"],
        "max_steps": int(payload["max_steps"]),
        "hard_max_steps": int(payload["hard_max_steps"]),
        "optimizer_step_count": int(payload["optimizer_step_count"]),
        "backward_pass_count": int(payload["backward_pass_count"]),
        "training_sample_count": int(payload["training_sample_count"]),
        "sample_count": int(payload["sample_count"]),
        "sample_step_count": int(payload["sample_step_count"]),
        "sample_step_size": float(payload["sample_step_size"]),
        "reloaded_loss": float(metrics["reloaded_loss"]),
        "sampled_rmsd_overall_mean": _mean(item["sampled_rmsd_to_target"] for item in sampled),
        "sampled_rmsd_by_split_mean": split_averages,
        "runtime_seconds": float(payload["runtime_seconds"]),
        "runtime_scope": payload.get("runtime_scope"),
        "runtime_seconds_reliable": bool(payload.get("runtime_seconds_reliable", False)),
        "safety_labels": {
            "benchmark_result": payload["benchmark_result"],
            "hardware": payload["hardware"],
            "gpu_used": payload["gpu_used"],
            "pilot_run": payload["pilot_run"],
            "bounded_pilot": payload["bounded_pilot"],
            "micro_pilot_run": payload["micro_pilot_run"],
            "checkpoint_written": payload["checkpoint_written"],
            "checkpoint_reloaded": payload["checkpoint_reloaded"],
            "checkpoint_scope": payload["checkpoint_scope"],
            "sampling_executed": payload["sampling_executed"],
            "larger_real_training_run_started": payload["larger_real_training_run_started"],
            "scheduler": payload["scheduler"],
            "slurm_account": payload["slurm_account"],
        },
        "sampled_molecules": [
            {
                "molecule_id": item["molecule_id"],
                "split": item["split"],
                "sampled_rmsd_to_target": float(item["sampled_rmsd_to_target"]),
            }
            for item in sampled
        ],
    }


def _validate_comparable(rows: list[dict[str, Any]]) -> None:
    reference = rows[0]
    fields = [
        "random_seed",
        "dataset_kind",
        "dataset_subset_scope",
        "split_type",
        "split_counts",
        "max_steps",
        "hard_max_steps",
        "optimizer_step_count",
        "backward_pass_count",
        "training_sample_count",
        "sample_count",
        "sample_step_count",
        "sample_step_size",
    ]
    for row in rows[1:]:
        for field in fields:
            if row[field] != reference[field]:
                raise ValueError(f"Results are not directly comparable: {field} differs")
        if [item["molecule_id"] for item in row["sampled_molecules"]] != [
            item["molecule_id"] for item in reference["sampled_molecules"]
        ]:
            raise ValueError("Results do not list sampled molecules in the same order")


def _validate_multiseed_rows(rows: list[dict[str, Any]], expected_seeds: tuple[int, ...]) -> None:
    expected_pairs = {
        ("isotropic", int(seed)) for seed in expected_seeds
    } | {
        ("stiffness_structured", int(seed)) for seed in expected_seeds
    }
    observed_pairs = {(row["corruption_mode"], int(row["random_seed"])) for row in rows}
    if observed_pairs != expected_pairs:
        raise ValueError(f"Micro-pilot checkpoint/sampling rows do not match expected mode/seed pairs: {observed_pairs}")
    reference = rows[0]
    fields = [
        "dataset_kind",
        "dataset_subset_scope",
        "split_type",
        "split_counts",
        "max_steps",
        "hard_max_steps",
        "optimizer_step_count",
        "backward_pass_count",
        "training_sample_count",
        "sample_count",
        "sample_step_count",
        "sample_step_size",
        "runtime_scope",
    ]
    for row in rows[1:]:
        for field in fields:
            if row[field] != reference[field]:
                raise ValueError(f"Micro-pilot checkpoint/sampling rows are not comparable: {field} differs")
        if [item["molecule_id"] for item in row["sampled_molecules"]] != [
            item["molecule_id"] for item in reference["sampled_molecules"]
        ]:
            raise ValueError("Micro-pilot checkpoint/sampling rows do not list sampled molecules in the same order")
    if not all(row["runtime_seconds_reliable"] for row in rows):
        raise ValueError("Micro-pilot checkpoint/sampling multi-seed summary requires reliable runtime metadata")


def _rows_by_seed(rows: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["random_seed"]), {})[row["corruption_mode"]] = row
    for seed, by_mode in grouped.items():
        if set(by_mode) != {"isotropic", "stiffness_structured"}:
            raise ValueError(f"Seed {seed} does not have both checkpoint/sampling modes")
        _validate_comparable(list(by_mode.values()))
    return grouped


def _metric_comparisons(by_mode: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    iso = by_mode["isotropic"]
    stiff = by_mode["stiffness_structured"]
    rows = [
        _comparison("reloaded_loss", iso["reloaded_loss"], stiff["reloaded_loss"]),
        _comparison(
            "sampled_rmsd_to_target_overall_mean",
            iso["sampled_rmsd_overall_mean"],
            stiff["sampled_rmsd_overall_mean"],
        ),
    ]
    for split_name in ["train", "val", "test"]:
        rows.append(
            _comparison(
                f"sampled_rmsd_to_target_{split_name}_mean",
                iso["sampled_rmsd_by_split_mean"][split_name],
                stiff["sampled_rmsd_by_split_mean"][split_name],
            )
        )
    return rows


def _aggregate_metric_comparisons(per_seed_comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for metric in COMPARABLE_METRICS:
        metric_rows = [
            row
            for seed_comparison in per_seed_comparisons
            for row in seed_comparison["metric_comparisons"]
            if row["metric"] == metric
        ]
        if len(metric_rows) != len(per_seed_comparisons):
            raise ValueError(f"Missing per-seed comparison for metric {metric}")
        isotropic_values = [row["isotropic"] for row in metric_rows]
        stiffness_values = [row["stiffness_structured"] for row in metric_rows]
        outcomes = [row["stiffness_outcome"] for row in metric_rows]
        rows.append(
            {
                "metric": metric,
                "isotropic_mean": _mean(isotropic_values),
                "stiffness_structured_mean": _mean(stiffness_values),
                "stiffness_minus_isotropic_mean": _mean(stiffness_values) - _mean(isotropic_values),
                "stiffness_outcome_by_seed": outcomes,
                "stiffness_win_count": outcomes.count("better"),
                "stiffness_tie_count": outcomes.count("tied"),
                "stiffness_loss_count": outcomes.count("worse"),
                "seed_count": len(metric_rows),
                "aggregate_outcome": _consistency_label(outcomes),
                "direction": "lower_is_better",
            }
        )
    return rows


def _stiffness_consistency(aggregate_metric_comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = [
        outcome
        for row in aggregate_metric_comparisons
        for outcome in row["stiffness_outcome_by_seed"]
    ]
    win_count = outcomes.count("better")
    tie_count = outcomes.count("tied")
    loss_count = outcomes.count("worse")
    if outcomes and win_count == len(outcomes):
        verdict = "stiffness_wins_consistently"
    elif loss_count == 0 and win_count > 0:
        verdict = "stiffness_nonworse_with_some_wins"
    elif win_count > 0 and loss_count > 0:
        verdict = "mixed"
    elif tie_count == len(outcomes):
        verdict = "tied"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "tracked_metric_seed_comparison_count": len(outcomes),
        "stiffness_win_count": win_count,
        "stiffness_tie_count": tie_count,
        "stiffness_loss_count": loss_count,
    }


def _consistency_label(outcomes: list[str]) -> str:
    if outcomes and all(outcome == "better" for outcome in outcomes):
        return "stiffness_wins_all_seeds"
    if outcomes and all(outcome in {"better", "tied"} for outcome in outcomes) and any(
        outcome == "better" for outcome in outcomes
    ):
        return "stiffness_nonworse_all_seeds"
    if any(outcome == "better" for outcome in outcomes) and any(outcome == "worse" for outcome in outcomes):
        return "mixed"
    if outcomes and all(outcome == "tied" for outcome in outcomes):
        return "tied"
    return "inconclusive"


def _per_molecule_comparisons(by_mode: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    iso_items = by_mode["isotropic"]["sampled_molecules"]
    stiff_items = by_mode["stiffness_structured"]["sampled_molecules"]
    rows = []
    for iso, stiff in zip(iso_items, stiff_items):
        if iso["molecule_id"] != stiff["molecule_id"] or iso["split"] != stiff["split"]:
            raise ValueError("Per-molecule comparison requires matching molecule order and split labels")
        rows.append(
            {
                "molecule_id": iso["molecule_id"],
                "split": iso["split"],
                "isotropic_sampled_rmsd_to_target": iso["sampled_rmsd_to_target"],
                "stiffness_sampled_rmsd_to_target": stiff["sampled_rmsd_to_target"],
                "stiffness_minus_isotropic": stiff["sampled_rmsd_to_target"] - iso["sampled_rmsd_to_target"],
                "stiffness_outcome": _outcome(iso["sampled_rmsd_to_target"], stiff["sampled_rmsd_to_target"]),
                "direction": "lower_is_better",
            }
        )
    return rows


def _multiseed_runtime_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {"isotropic": [], "stiffness_structured": []}
    for row in rows:
        by_mode[row["corruption_mode"]].append(row)
    runtime_seconds_by_mode = {
        mode: [row["runtime_seconds"] for row in mode_rows]
        for mode, mode_rows in by_mode.items()
    }
    runtime_scope_set = sorted({row["runtime_scope"] for row in rows})
    all_reliable = all(row["runtime_seconds_reliable"] for row in rows)
    return {
        "runtime_seconds_by_mode": runtime_seconds_by_mode,
        "runtime_seconds_mean_by_mode": {
            mode: _mean(values) for mode, values in runtime_seconds_by_mode.items()
        },
        "runtime_scope_set": runtime_scope_set,
        "all_runtime_seconds_reliable": all_reliable,
        "note": "Runtime is recorded for accounting only; tiny CPU timing differences are not interpreted as model performance.",
    }


def _runtime_audit(by_mode: dict[str, dict[str, Any]]) -> dict[str, Any]:
    iso = by_mode["isotropic"]
    stiff = by_mode["stiffness_structured"]
    iso_runtime = iso["runtime_seconds"]
    stiff_runtime = stiff["runtime_seconds"]
    smaller = min(iso_runtime, stiff_runtime)
    larger = max(iso_runtime, stiff_runtime)
    ratio = math.inf if smaller == 0.0 else larger / smaller
    abs_diff = abs(iso_runtime - stiff_runtime)
    same_scope = iso.get("runtime_scope") == stiff.get("runtime_scope") and iso.get("runtime_scope") is not None
    both_marked_reliable = iso["runtime_seconds_reliable"] and stiff["runtime_seconds_reliable"]
    suspicious = (not same_scope) or (not both_marked_reliable) or (
        ratio > RUNTIME_RATIO_WARNING_THRESHOLD and abs_diff > RUNTIME_ABS_WARNING_SECONDS
    )
    reasons = []
    if not same_scope:
        reasons.append("runtime_scope_missing_or_mismatched")
    if not both_marked_reliable:
        reasons.append("runtime_seconds_reliable_missing_or_false")
    if ratio > RUNTIME_RATIO_WARNING_THRESHOLD and abs_diff > RUNTIME_ABS_WARNING_SECONDS:
        reasons.append("runtime_ratio_exceeds_threshold")
    return {
        "runtime_seconds_by_mode": {
            "isotropic": iso_runtime,
            "stiffness_structured": stiff_runtime,
        },
        "runtime_scope_by_mode": {
            "isotropic": iso.get("runtime_scope"),
            "stiffness_structured": stiff.get("runtime_scope"),
        },
        "runtime_seconds_reliable_by_mode": {
            "isotropic": iso["runtime_seconds_reliable"],
            "stiffness_structured": stiff["runtime_seconds_reliable"],
        },
        "absolute_difference_seconds": abs_diff,
        "max_min_ratio": ratio,
        "suspicious_timing_inconsistency": suspicious,
        "runtime_seconds_comparable": not suspicious,
        "reasons": reasons,
        "recommendation": (
            "Regenerate the two micro-pilot checkpoint/sampling outputs with the fixed timing scope before comparing runtime."
            if suspicious
            else "Runtime fields share the fixed timing scope and are suitable for this bounded pilot comparison."
        ),
    }


def _average_rmsd_by_split(sampled: list[dict[str, Any]]) -> dict[str, float]:
    return {
        split: _mean(item["sampled_rmsd_to_target"] for item in sampled if item["split"] == split)
        for split in ["train", "val", "test"]
    }


def _comparison(metric: str, isotropic_value: float, stiffness_value: float) -> dict[str, Any]:
    return {
        "metric": metric,
        "isotropic": isotropic_value,
        "stiffness_structured": stiffness_value,
        "stiffness_minus_isotropic": stiffness_value - isotropic_value,
        "stiffness_outcome": _outcome(isotropic_value, stiffness_value),
        "direction": "lower_is_better",
    }


def _outcome(isotropic_value: float, stiffness_value: float) -> str:
    delta = stiffness_value - isotropic_value
    if abs(delta) <= TIE_TOLERANCE:
        return "tied"
    if delta < 0.0:
        return "better"
    return "worse"


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("Cannot average an empty value list")
    return sum(items) / len(items)


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Bounded Micro-Pilot Checkpoint/Sampling Comparison",
        "",
        "This report compares two existing six-molecule micro-pilot checkpoint-and-sampling outputs. It is bounded pilot-only and not a benchmark result.",
        "",
        "## Safety Labels",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| result_kind | `{summary['result_kind']}` |",
        f"| benchmark_result | `{summary['benchmark_result']}` |",
        f"| hardware | `{summary['hardware']}` |",
        f"| gpu_used | `{summary['gpu_used']}` |",
        f"| training_executed_by_summary | `{summary['training_executed_by_summary']}` |",
        f"| sampling_executed_by_summary | `{summary['sampling_executed_by_summary']}` |",
        "",
        "## Runtime Audit",
        "",
    ]
    audit = summary["runtime_audit"]
    lines.extend(
        [
            "| Mode | runtime_seconds | runtime_scope | reliable |",
            "|---|---:|---|---:|",
            (
                "| isotropic | "
                f"{_fmt(audit['runtime_seconds_by_mode']['isotropic'])} | "
                f"`{audit['runtime_scope_by_mode']['isotropic']}` | "
                f"`{audit['runtime_seconds_reliable_by_mode']['isotropic']}` |"
            ),
            (
                "| stiffness_structured | "
                f"{_fmt(audit['runtime_seconds_by_mode']['stiffness_structured'])} | "
                f"`{audit['runtime_scope_by_mode']['stiffness_structured']}` | "
                f"`{audit['runtime_seconds_reliable_by_mode']['stiffness_structured']}` |"
            ),
            "",
            f"- suspicious_timing_inconsistency: `{audit['suspicious_timing_inconsistency']}`",
            f"- max_min_ratio: `{_fmt(audit['max_min_ratio'])}`",
            f"- absolute_difference_seconds: `{_fmt(audit['absolute_difference_seconds'])}`",
            f"- reasons: `{', '.join(audit['reasons']) if audit['reasons'] else 'none'}`",
            f"- recommendation: {audit['recommendation']}",
            "",
            "## Direct Metrics",
            "",
            "| Metric | Isotropic | Stiffness | Stiffness - Isotropic | Outcome |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in summary["metric_comparisons"]:
        lines.append(
            f"| `{row['metric']}` | {_fmt(row['isotropic'])} | {_fmt(row['stiffness_structured'])} | "
            f"{_fmt(row['stiffness_minus_isotropic'])} | {row['stiffness_outcome']} |"
        )
    lines.extend(
        [
            "",
            "## Per-Molecule Sampled RMSD",
            "",
            "| Split | Molecule ID | Isotropic | Stiffness | Stiffness - Isotropic | Outcome |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in summary["per_molecule_sampled_rmsd"]:
        lines.append(
            f"| {row['split']} | `{row['molecule_id']}` | "
            f"{_fmt(row['isotropic_sampled_rmsd_to_target'])} | "
            f"{_fmt(row['stiffness_sampled_rmsd_to_target'])} | "
            f"{_fmt(row['stiffness_minus_isotropic'])} | {row['stiffness_outcome']} |"
        )
    lines.extend(
        [
            "",
            "Lower values are marked better for `reloaded_loss` and sampled RMSD diagnostics. These are tiny bounded pilot diagnostics only.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_multiseed_report(summary: dict[str, Any]) -> str:
    consistency = summary["stiffness_consistency"]
    lines = [
        "# Bounded Micro-Pilot Checkpoint/Sampling Multi-Seed Summary",
        "",
        "This report aggregates bounded six-molecule checkpoint-and-sampling pilots across a fixed tiny seed set. It is bounded pilot-only and not a benchmark result.",
        "",
        "## Safety Labels",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| result_kind | `{summary['result_kind']}` |",
        f"| benchmark_result | `{summary['benchmark_result']}` |",
        f"| seed_set | `{summary['seed_set']}` |",
        f"| hardware | `{summary['hardware']}` |",
        f"| gpu_used | `{summary['gpu_used']}` |",
        f"| training_executed_by_summary | `{summary['training_executed_by_summary']}` |",
        f"| sampling_executed_by_summary | `{summary['sampling_executed_by_summary']}` |",
        "",
        "## Stability Verdict",
        "",
        f"- verdict: `{consistency['verdict']}`",
        f"- stiffness wins: `{consistency['stiffness_win_count']}`",
        f"- stiffness ties: `{consistency['stiffness_tie_count']}`",
        f"- stiffness losses: `{consistency['stiffness_loss_count']}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Isotropic mean | Stiffness mean | Stiffness - Isotropic | Seed outcomes | Aggregate outcome |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in summary["aggregate_metric_comparisons"]:
        lines.append(
            f"| `{row['metric']}` | {_fmt(row['isotropic_mean'])} | "
            f"{_fmt(row['stiffness_structured_mean'])} | "
            f"{_fmt(row['stiffness_minus_isotropic_mean'])} | "
            f"`{row['stiffness_outcome_by_seed']}` | {row['aggregate_outcome']} |"
        )
    lines.extend(
        [
            "",
            "## Per-Seed Metrics",
            "",
            "| Seed | Metric | Isotropic | Stiffness | Stiffness - Isotropic | Outcome |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for seed_comparison in summary["per_seed_comparisons"]:
        for row in seed_comparison["metric_comparisons"]:
            lines.append(
                f"| {seed_comparison['seed']} | `{row['metric']}` | {_fmt(row['isotropic'])} | "
                f"{_fmt(row['stiffness_structured'])} | {_fmt(row['stiffness_minus_isotropic'])} | "
                f"{row['stiffness_outcome']} |"
            )
    runtime = summary["runtime_summary"]
    lines.extend(
        [
            "",
            "## Runtime Accounting",
            "",
            "| Mode | runtimes_seconds | mean_seconds |",
            "|---|---|---:|",
            (
                "| isotropic | "
                f"`{[_fmt(value) for value in runtime['runtime_seconds_by_mode']['isotropic']]}` | "
                f"{_fmt(runtime['runtime_seconds_mean_by_mode']['isotropic'])} |"
            ),
            (
                "| stiffness_structured | "
                f"`{[_fmt(value) for value in runtime['runtime_seconds_by_mode']['stiffness_structured']]}` | "
                f"{_fmt(runtime['runtime_seconds_mean_by_mode']['stiffness_structured'])} |"
            ),
            "",
            f"- runtime_scope_set: `{runtime['runtime_scope_set']}`",
            f"- all_runtime_seconds_reliable: `{runtime['all_runtime_seconds_reliable']}`",
            f"- note: {runtime['note']}",
            "",
            "Lower values are marked better for `reloaded_loss` and sampled RMSD diagnostics. These are tiny bounded pilot diagnostics only.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{float(value):.6g}"
