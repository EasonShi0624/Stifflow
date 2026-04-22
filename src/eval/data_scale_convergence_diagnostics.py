"""Convergence diagnostics for bounded 10000-molecule data-scale pilots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RESULT_KIND = "tiny_real_data_bounded_data_scale_check_convergence_diagnostics_not_benchmark"
INPUT_RESULT_KIND = "tiny_real_data_bounded_data_scale_check_checkpoint_sampling_not_benchmark"
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_MAX_MOLECULES = 10_000
DEFAULT_TRAIN_COUNT = 9_000
DEFAULT_MAX_STEPS = 200
DEFAULT_HARD_MAX_STEPS = 200
DEFAULT_TAIL_WINDOW = 20
DEFAULT_RESULT_PATHS = tuple(
    Path(f"experiments/results/geom_data_scale_check_ckpt_sample_{mode}_mols10000_steps200_seed{seed}.json")
    for mode in ("isotropic", "stiffness")
    for seed in DEFAULT_SEEDS
)
DEFAULT_SUMMARY_PATH = Path(
    "experiments/results/geom_data_scale_check_ckpt_sample_convergence_diagnostics_mols10000_steps200.json"
)
DEFAULT_REPORT_PATH = Path(
    "reports/tables/geom_data_scale_check_ckpt_sample_convergence_diagnostics_mols10000_steps200.md"
)


def write_data_scale_check_convergence_diagnostics(
    result_paths: list[str | Path] | tuple[str | Path, ...] = DEFAULT_RESULT_PATHS,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    expected_seeds: tuple[int, ...] = DEFAULT_SEEDS,
    max_molecules: int = DEFAULT_MAX_MOLECULES,
    max_steps: int = DEFAULT_MAX_STEPS,
    tail_window: int = DEFAULT_TAIL_WINDOW,
) -> dict[str, Path]:
    rows = [_convergence_row(Path(path), tail_window=tail_window) for path in result_paths]
    _validate_rows(rows, expected_seeds=expected_seeds, max_molecules=max_molecules, max_steps=max_steps)
    aggregate_by_mode = _aggregate_by_mode(rows)
    verdict = _diagnostic_verdict(rows)
    summary = {
        "summary_name": "geom_data_scale_check_ckpt_sample_convergence_diagnostics_mols10000_steps200",
        "result_kind": RESULT_KIND,
        "benchmark_result": False,
        "dataset_name": "geom_drugs",
        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
        "dataset_subset_scope": "data_scale_check_10000_processed_dry_run",
        "hardware": "cpu_only",
        "gpu_used": False,
        "scheduler": None,
        "slurm_account": None,
        "pilot_run": True,
        "bounded_pilot": True,
        "data_scale_check_run": True,
        "data_scale_check_10000_run": True,
        "data_scale_check_10000_convergence_run": True,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "raw_data_modified": False,
        "environment_modified": False,
        "larger_real_training_run_started": False,
        "seed_set": list(expected_seeds),
        "max_molecules": int(max_molecules),
        "max_steps": int(max_steps),
        "tail_window": int(tail_window),
        "input_result_paths": [str(path) for path in result_paths],
        "row_count": len(rows),
        "rows": rows,
        "aggregate_by_mode": aggregate_by_mode,
        "diagnostic_verdict": verdict,
        "notes": [
            "Reporting-only convergence diagnostics over existing bounded 10000-molecule steps200 checkpoint/sampling outputs.",
            "The summary reads recorded per-step loss traces only; it does not run training, sampling, preprocessing, or checkpoint loading.",
            "Loss-trend diagnostics are bounded pilot diagnostics only, not benchmark metrics or model-quality claims.",
            "No raw-data modification, environment mutation, GPU execution, SLURM job, or larger real training run was performed by this summary.",
        ],
    }
    summary_path = Path(summary_path)
    report_path = Path(report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_diagnostics_markdown(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def _convergence_row(path: Path, tail_window: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    losses = [float(step["loss_before_step"]) for step in metrics["train_steps"]]
    gradients = [float(step["gradient_l2_norm"]) for step in metrics["train_steps"] if "gradient_l2_norm" in step]
    if not losses:
        raise ValueError(f"Convergence diagnostics require non-empty train_steps: {path}")
    if tail_window < 2 or tail_window > len(losses):
        raise ValueError(f"Invalid tail_window={tail_window} for {len(losses)} train steps")
    tail_losses = losses[-tail_window:]
    min_delta = float(payload.get("early_stop_min_delta") or metrics.get("early_stop_min_delta") or 0.0)
    best_index = min(range(len(losses)), key=lambda index: losses[index])
    tail_transition_count = len(tail_losses) - 1
    tail_improvement_count = sum(1 for previous, current in zip(tail_losses, tail_losses[1:]) if current < previous)
    tail_min_delta_improvement_count = sum(
        1 for previous, current in zip(tail_losses, tail_losses[1:]) if previous - current > min_delta
    )
    tail_delta = tail_losses[-1] - tail_losses[0]
    tail_relative_delta = tail_delta / tail_losses[0] if tail_losses[0] else 0.0
    tail_slope = _linear_slope(tail_losses)
    return {
        "source_path": str(path),
        "run_name": payload["run_name"],
        "corruption_mode": payload["corruption_mode"],
        "random_seed": int(payload["random_seed"]),
        "result_kind": payload["result_kind"],
        "benchmark_result": payload["benchmark_result"],
        "hardware": payload["hardware"],
        "gpu_used": payload["gpu_used"],
        "scheduler": payload.get("scheduler"),
        "slurm_account": payload.get("slurm_account"),
        "dataset_subset_scope": payload["dataset"]["subset_scope"],
        "sample_count": int(payload["sample_count"]),
        "training_sample_count": int(payload["training_sample_count"]),
        "split_counts": payload["dataset"]["split_counts"],
        "max_steps": int(payload["max_steps"]),
        "hard_max_steps": int(payload["hard_max_steps"]),
        "optimizer_step_count": int(payload["optimizer_step_count"]),
        "early_stop_enabled": bool(payload.get("early_stop_enabled", False)),
        "early_stop_patience": int(payload["early_stop_patience"]),
        "early_stop_min_delta": min_delta,
        "early_stop_monitor": payload["early_stop_monitor"],
        "stopped_early": bool(payload["stopped_early"]),
        "early_stop_reason": payload["early_stop_reason"],
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "best_loss": losses[best_index],
        "best_loss_step": best_index + 1,
        "best_at_final_step": best_index == len(losses) - 1,
        "total_loss_delta": losses[-1] - losses[0],
        "total_loss_relative_delta": (losses[-1] - losses[0]) / losses[0] if losses[0] else 0.0,
        "tail_window": int(tail_window),
        "tail_start_step": len(losses) - tail_window + 1,
        "tail_start_loss": tail_losses[0],
        "tail_final_loss": tail_losses[-1],
        "tail_loss_delta": tail_delta,
        "tail_loss_relative_delta": tail_relative_delta,
        "tail_slope_per_step": tail_slope,
        "tail_transition_count": tail_transition_count,
        "tail_improvement_count": tail_improvement_count,
        "tail_min_delta_improvement_count": tail_min_delta_improvement_count,
        "tail_monotonic_nonincreasing": all(current <= previous for previous, current in zip(tail_losses, tail_losses[1:])),
        "tail_all_transitions_improve_more_than_min_delta": tail_min_delta_improvement_count == tail_transition_count,
        "tail_trend": _tail_trend_label(
            tail_delta=tail_delta,
            tail_slope=tail_slope,
            best_at_final_step=best_index == len(losses) - 1,
            tail_min_delta_improvement_count=tail_min_delta_improvement_count,
            tail_transition_count=tail_transition_count,
        ),
        "final_gradient_l2_norm": gradients[-1] if gradients else None,
        "tail_gradient_l2_norm_mean": _mean(gradients[-tail_window:]) if gradients else None,
    }


def _validate_rows(
    rows: list[dict[str, Any]],
    expected_seeds: tuple[int, ...],
    max_molecules: int,
    max_steps: int,
) -> None:
    expected_pairs = {
        ("isotropic", int(seed)) for seed in expected_seeds
    } | {
        ("stiffness_structured", int(seed)) for seed in expected_seeds
    }
    observed_pairs = {(row["corruption_mode"], row["random_seed"]) for row in rows}
    if observed_pairs != expected_pairs:
        raise ValueError(f"Convergence diagnostics rows do not match expected mode/seed pairs: {observed_pairs}")
    for row in rows:
        if row["result_kind"] != INPUT_RESULT_KIND:
            raise ValueError(f"Unexpected result kind in convergence diagnostics input: {row['source_path']}")
        if row["benchmark_result"] is not False:
            raise ValueError(f"Convergence diagnostics input must be benchmark_result=false: {row['source_path']}")
        if row["hardware"] != "cpu_only" or row["gpu_used"] is not False:
            raise ValueError(f"Convergence diagnostics input must be CPU-only: {row['source_path']}")
        if row["dataset_subset_scope"] != "data_scale_check_10000_processed_dry_run":
            raise ValueError(f"Convergence diagnostics input subset mismatch: {row['source_path']}")
        if row["sample_count"] != max_molecules:
            raise ValueError(f"Convergence diagnostics input molecule count mismatch: {row['source_path']}")
        if row["training_sample_count"] != DEFAULT_TRAIN_COUNT:
            raise ValueError(f"Convergence diagnostics input train count mismatch: {row['source_path']}")
        if row["max_steps"] != max_steps or row["hard_max_steps"] != DEFAULT_HARD_MAX_STEPS:
            raise ValueError(f"Convergence diagnostics input step cap mismatch: {row['source_path']}")
        if row["early_stop_enabled"] is not True:
            raise ValueError(f"Convergence diagnostics input must have early_stop enabled: {row['source_path']}")
        if row["early_stop_monitor"] != "train_loss_before_step":
            raise ValueError(f"Convergence diagnostics input early-stop monitor mismatch: {row['source_path']}")


def _aggregate_by_mode(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregate = {}
    for mode in ["isotropic", "stiffness_structured"]:
        mode_rows = sorted([row for row in rows if row["corruption_mode"] == mode], key=lambda row: row["random_seed"])
        aggregate[mode] = {
            "seed_count": len(mode_rows),
            "hit_hard_cap_count": sum(1 for row in mode_rows if not row["stopped_early"] and row["optimizer_step_count"] == row["max_steps"]),
            "stopped_early_count": sum(1 for row in mode_rows if row["stopped_early"]),
            "best_at_final_step_count": sum(1 for row in mode_rows if row["best_at_final_step"]),
            "tail_monotonic_nonincreasing_count": sum(1 for row in mode_rows if row["tail_monotonic_nonincreasing"]),
            "tail_all_transitions_improve_more_than_min_delta_count": sum(
                1 for row in mode_rows if row["tail_all_transitions_improve_more_than_min_delta"]
            ),
            "initial_loss_mean": _mean([row["initial_loss"] for row in mode_rows]),
            "final_loss_mean": _mean([row["final_loss"] for row in mode_rows]),
            "total_loss_relative_delta_mean": _mean([row["total_loss_relative_delta"] for row in mode_rows]),
            "tail_loss_delta_mean": _mean([row["tail_loss_delta"] for row in mode_rows]),
            "tail_loss_relative_delta_mean": _mean([row["tail_loss_relative_delta"] for row in mode_rows]),
            "tail_slope_per_step_mean": _mean([row["tail_slope_per_step"] for row in mode_rows]),
            "final_gradient_l2_norm_mean": _mean(
                [row["final_gradient_l2_norm"] for row in mode_rows if row["final_gradient_l2_norm"] is not None]
            ),
            "tail_trends": [row["tail_trend"] for row in mode_rows],
        }
    return aggregate


def _diagnostic_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_hit_hard_cap = all(not row["stopped_early"] and row["optimizer_step_count"] == row["max_steps"] for row in rows)
    all_best_at_final = all(row["best_at_final_step"] for row in rows)
    all_tail_decreasing = all(row["tail_monotonic_nonincreasing"] for row in rows)
    all_tail_improves_by_rule = all(row["tail_all_transitions_improve_more_than_min_delta"] for row in rows)
    if all_hit_hard_cap and all_best_at_final and all_tail_improves_by_rule:
        verdict = "still_downward_at_hard_cap"
        assessment = (
            "All runs hit the strict 200-step hard cap, but every run's best loss is at step 200 and every "
            "tail-window transition improves by more than the configured early-stop min_delta. The current "
            "evidence points to continuing downward trends near the cap rather than an early-stop rule that is too strict."
        )
    elif all_hit_hard_cap and not all_tail_decreasing:
        verdict = "early_stop_rule_needs_review"
        assessment = (
            "All runs hit the hard cap and at least one tail-window trace is not monotonically decreasing. "
            "The patience or min_delta settings may need review before increasing the horizon."
        )
    else:
        verdict = "mixed_convergence_signals"
        assessment = (
            "The convergence traces show mixed stopping and tail-trend behavior. Treat this as diagnostic only "
            "and inspect per-run rows before changing any training settings."
        )
    return {
        "verdict": verdict,
        "all_hit_hard_cap": all_hit_hard_cap,
        "all_best_at_final_step": all_best_at_final,
        "all_tail_monotonic_nonincreasing": all_tail_decreasing,
        "all_tail_transitions_improve_more_than_min_delta": all_tail_improves_by_rule,
        "assessment": assessment,
    }


def _tail_trend_label(
    tail_delta: float,
    tail_slope: float,
    best_at_final_step: bool,
    tail_min_delta_improvement_count: int,
    tail_transition_count: int,
) -> str:
    if best_at_final_step and tail_delta < 0.0 and tail_slope < 0.0 and tail_min_delta_improvement_count == tail_transition_count:
        return "still_decreasing_through_hard_cap"
    if tail_delta < 0.0 and tail_slope < 0.0:
        return "downward_near_hard_cap"
    if abs(tail_delta) <= 1.0e-12:
        return "flat_tail"
    return "nonmonotonic_or_increasing_tail"


def _linear_slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = _mean(values)
    denominator = sum((index - mean_x) ** 2 for index in range(n))
    if denominator == 0.0:
        return 0.0
    numerator = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
    return numerator / denominator


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute mean over an empty value list")
    return sum(float(value) for value in values) / len(values)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


def _diagnostics_markdown(summary: dict[str, Any]) -> str:
    verdict = summary["diagnostic_verdict"]
    lines = [
        "# Bounded 10000-Molecule Convergence Diagnostics",
        "",
        "This report analyzes recorded loss traces from existing bounded steps200 checkpoint-and-sampling pilots. It is bounded pilot-only and not a benchmark result.",
        "",
        "## Safety Labels",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| result_kind | `{summary['result_kind']}` |",
        f"| benchmark_result | `{summary['benchmark_result']}` |",
        f"| max_molecules | `{summary['max_molecules']}` |",
        f"| max_steps | `{summary['max_steps']}` |",
        f"| tail_window | `{summary['tail_window']}` |",
        f"| hardware | `{summary['hardware']}` |",
        f"| gpu_used | `{summary['gpu_used']}` |",
        f"| training_executed_by_summary | `{summary['training_executed_by_summary']}` |",
        f"| sampling_executed_by_summary | `{summary['sampling_executed_by_summary']}` |",
        f"| larger_real_training_run_started | `{summary['larger_real_training_run_started']}` |",
        "",
        "## Verdict",
        "",
        f"- verdict: `{verdict['verdict']}`",
        f"- all_hit_hard_cap: `{verdict['all_hit_hard_cap']}`",
        f"- all_best_at_final_step: `{verdict['all_best_at_final_step']}`",
        f"- all_tail_monotonic_nonincreasing: `{verdict['all_tail_monotonic_nonincreasing']}`",
        f"- all_tail_transitions_improve_more_than_min_delta: `{verdict['all_tail_transitions_improve_more_than_min_delta']}`",
        f"- assessment: {verdict['assessment']}",
        "",
        "## Aggregate By Mode",
        "",
        "| Mode | hard-cap hits | best-at-final | tail monotonic | tail all > min_delta | final loss mean | tail delta mean | tail relative delta mean | tail slope mean | final grad mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ["isotropic", "stiffness_structured"]:
        row = summary["aggregate_by_mode"][mode]
        lines.append(
            f"| {mode} | {row['hit_hard_cap_count']} | {row['best_at_final_step_count']} | "
            f"{row['tail_monotonic_nonincreasing_count']} | "
            f"{row['tail_all_transitions_improve_more_than_min_delta_count']} | "
            f"{_fmt(row['final_loss_mean'])} | {_fmt(row['tail_loss_delta_mean'])} | "
            f"{_fmt(row['tail_loss_relative_delta_mean'])} | {_fmt(row['tail_slope_per_step_mean'])} | "
            f"{_fmt(row['final_gradient_l2_norm_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-Run Tail Diagnostics",
            "",
            "| Mode | Seed | Stop reason | Steps | Initial loss | Final loss | Best step | Tail delta | Tail relative delta | Tail min-delta improvements | Tail trend |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(summary["rows"], key=lambda item: (item["corruption_mode"], item["random_seed"])):
        lines.append(
            f"| {row['corruption_mode']} | {row['random_seed']} | {row['early_stop_reason']} | "
            f"{row['optimizer_step_count']} | {_fmt(row['initial_loss'])} | {_fmt(row['final_loss'])} | "
            f"{row['best_loss_step']} | {_fmt(row['tail_loss_delta'])} | "
            f"{_fmt(row['tail_loss_relative_delta'])} | {row['tail_min_delta_improvement_count']}/{row['tail_transition_count']} | "
            f"{row['tail_trend']} |"
        )
    lines.extend(
        [
            "",
            "These diagnostics do not change the early-stop rule or any training setting. They only summarize recorded loss traces from existing bounded non-benchmark outputs.",
            "",
        ]
    )
    return "\n".join(lines)
