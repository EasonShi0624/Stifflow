"""Aggregate bounded tiny real-data pilot outputs into a non-benchmark report."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


DEFAULT_RESULT_PATHS = [
    Path("experiments/results/geom_tiny_real_pilot_isotropic_steps6_seed0.json"),
    Path("experiments/results/geom_tiny_real_pilot_stiffness_steps6_seed0.json"),
]
MICRO_PILOT_RESULT_PATHS = [
    Path("experiments/results/geom_micro_pilot_isotropic_mols6_steps6_seed0.json"),
    Path("experiments/results/geom_micro_pilot_stiffness_mols6_steps6_seed0.json"),
]
MICRO_PILOT_MULTI_SEED_RESULT_PATHS = [
    Path(f"experiments/results/geom_micro_pilot_isotropic_mols6_steps6_seed{seed}.json")
    for seed in [0, 1]
] + [
    Path(f"experiments/results/geom_micro_pilot_stiffness_mols6_steps6_seed{seed}.json")
    for seed in [0, 1]
]
SUMMARY_RESULT_KIND = "tiny_real_data_bounded_pilot_comparison_not_benchmark"
MICRO_PILOT_SUMMARY_RESULT_KIND = "tiny_real_data_bounded_micro_pilot_comparison_not_benchmark"
MICRO_PILOT_MULTI_SEED_SUMMARY_RESULT_KIND = (
    "tiny_real_data_bounded_micro_pilot_multiseed_summary_not_benchmark"
)
PILOT_RESULT_KIND = "tiny_real_data_bounded_pilot_not_benchmark"
VALID_SCOPES = {"tiny_real_pilot", "micro_pilot"}


def write_tiny_real_pilot_comparison(
    result_paths: list[str | Path] | None = None,
    summary_path: str | Path | None = None,
    report_path: str | Path | None = None,
    comparison_scope: str = "tiny_real_pilot",
) -> dict[str, Path]:
    if comparison_scope not in VALID_SCOPES:
        raise ValueError(f"Unsupported pilot comparison scope: {comparison_scope}")
    start = time.time()
    default_paths = MICRO_PILOT_RESULT_PATHS if comparison_scope == "micro_pilot" else DEFAULT_RESULT_PATHS
    paths = [Path(path) for path in (result_paths or default_paths)]
    payloads = [_load_pilot_result(path, comparison_scope=comparison_scope) for path in paths]
    rows = [_row_from_result(path, payload) for path, payload in zip(paths, payloads)]
    modes = sorted(row["corruption_mode"] for row in rows)
    if modes != ["isotropic", "stiffness_structured"]:
        raise ValueError(f"Pilot comparison expects isotropic and stiffness_structured rows, got {modes}")
    if comparison_scope == "micro_pilot":
        summary_name = "geom_micro_pilot_summary"
        result_kind = MICRO_PILOT_SUMMARY_RESULT_KIND
        default_summary_path = "experiments/results/geom_micro_pilot_summary.json"
        default_report_path = "reports/tables/geom_micro_pilot_summary.md"
        title = "Tiny Real-Data Bounded Micro-Pilot Summary"
        scope_note = "two bounded six-molecule GEOM micro-pilot outputs only"
        artifact_note = (
            "The input pilots each used the bounded six-molecule GEOM micro-pilot artifact "
            "with four train, one validation, and one test molecule."
        )
    else:
        summary_name = "geom_tiny_real_pilot_summary"
        result_kind = SUMMARY_RESULT_KIND
        default_summary_path = "experiments/results/geom_tiny_real_pilot_summary.json"
        default_report_path = "reports/tables/geom_tiny_real_pilot_summary.md"
        title = "Tiny Real-Data Bounded Pilot Summary"
        scope_note = "two bounded three-molecule GEOM tiny real-data pilot outputs only"
        artifact_note = (
            "The input pilots each used the existing three-molecule GEOM tiny real artifact "
            "and one train/val/test molecule."
        )

    summary = {
        "summary_name": summary_name,
        "comparison_scope": comparison_scope,
        "result_kind": result_kind,
        "benchmark_result": False,
        "execution_mode": "local_cpu_tiny_real_bounded_pilot_report",
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
        "runtime_seconds": time.time() - start,
        "notes": [
            f"This is a reporting-only comparison of {scope_note}.",
            artifact_note,
            "Loss traces and final losses are pilot diagnostics only, not benchmark metrics or evidence of model quality.",
            "No raw-data modification, environment mutation, GPU execution, SLURM job, sampling, checkpointing, or larger real training run was used by this summary.",
        ],
    }

    summary_path = Path(summary_path or default_summary_path)
    report_path = Path(report_path or default_report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(summary, title=title, scope_note=scope_note), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def write_micro_pilot_multiseed_summary(
    result_paths: list[str | Path] | None = None,
    summary_path: str | Path = "experiments/results/geom_micro_pilot_multiseed_summary.json",
    report_path: str | Path = "reports/tables/geom_micro_pilot_multiseed_summary.md",
    expected_seeds: tuple[int, ...] = (0, 1),
) -> dict[str, Path]:
    start = time.time()
    paths = [Path(path) for path in (result_paths or MICRO_PILOT_MULTI_SEED_RESULT_PATHS)]
    payloads = [_load_pilot_result(path, comparison_scope="micro_pilot") for path in paths]
    rows = [_row_from_result(path, payload) for path, payload in zip(paths, payloads)]
    _validate_multiseed_rows(rows, expected_seeds)
    mode_statistics = _mode_statistics(rows)
    summary = {
        "summary_name": "geom_micro_pilot_multiseed_summary",
        "comparison_scope": "micro_pilot_multiseed",
        "result_kind": MICRO_PILOT_MULTI_SEED_SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_cpu_tiny_real_bounded_micro_pilot_multiseed_report",
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
        "mode_statistics": mode_statistics,
        "rows": rows,
        "runtime_seconds": time.time() - start,
        "notes": [
            "This is a reporting-only aggregation of bounded six-molecule micro-pilot runs across a tiny fixed seed set.",
            "The input rows each use the same bounded micro-pilot artifact and split with four train, one validation, and one test molecule.",
            "Loss traces and final-loss statistics are pilot diagnostics only, not benchmark metrics or evidence of model quality.",
            "No raw-data modification, environment mutation, GPU execution, SLURM job, sampling, checkpointing, or larger real training run was used by this summary.",
        ],
    }
    summary_path = Path(summary_path)
    report_path = Path(report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_multiseed_report(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def _load_pilot_result(path: Path, comparison_scope: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Tiny real pilot result does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("result_kind") != PILOT_RESULT_KIND:
        raise ValueError(f"Result is not a bounded tiny real pilot output: {path}")
    if payload.get("benchmark_result") is not False:
        raise ValueError(f"Pilot result is not explicitly benchmark_result=false: {path}")
    if payload.get("hardware") != "cpu_only":
        raise ValueError(f"Pilot result is not explicitly hardware=cpu_only: {path}")
    if payload.get("gpu_used") is not False:
        raise ValueError(f"Pilot result unexpectedly used GPU: {path}")
    if payload.get("scheduler") is not None:
        raise ValueError(f"Pilot result unexpectedly records scheduler use: {path}")
    if payload.get("slurm_account") is not None:
        raise ValueError(f"Pilot result unexpectedly records a SLURM account: {path}")
    if payload.get("pilot_run") is not True or payload.get("bounded_pilot") is not True:
        raise ValueError(f"Pilot result is not labeled as bounded pilot: {path}")
    if payload.get("larger_real_training_run_started") is not False:
        raise ValueError(f"Pilot result unexpectedly started a larger real training run: {path}")
    if payload.get("checkpoint_written") is not False:
        raise ValueError(f"Pilot result unexpectedly wrote a checkpoint: {path}")
    if payload.get("sampling_executed") is not False:
        raise ValueError(f"Pilot result unexpectedly ran sampling: {path}")
    dataset = payload.get("dataset") or {}
    if dataset.get("kind") != "geom_drugs_tiny_processed_dry_run":
        raise ValueError(f"Pilot result is not from the tiny real processed artifact: {path}")
    metrics = payload.get("metrics") or {}
    split_counts = dataset.get("split_counts")
    metric_split_counts = metrics.get("split_sample_counts")
    if comparison_scope == "micro_pilot":
        if payload.get("micro_pilot_run") is not True:
            raise ValueError(f"Pilot result is not labeled micro_pilot_run=true: {path}")
        if dataset.get("subset_scope") != "micro_pilot_processed_dry_run":
            raise ValueError(f"Pilot result is not from the micro-pilot subset: {path}")
        if split_counts != {"test": 1, "train": 4, "val": 1}:
            raise ValueError(f"Micro-pilot result does not have 4/1/1 train/val/test counts: {path}")
        if metric_split_counts != {"test": 1, "train": 4, "val": 1}:
            raise ValueError(f"Micro-pilot metrics do not have 4/1/1 train/val/test samples: {path}")
    else:
        if payload.get("micro_pilot_run") is True:
            raise ValueError(f"Tiny pilot comparison received a micro-pilot result: {path}")
        if split_counts != {"test": 1, "train": 1, "val": 1}:
            raise ValueError(f"Pilot result does not have one train/val/test molecule: {path}")
        if metric_split_counts != {"test": 1, "train": 1, "val": 1}:
            raise ValueError(f"Pilot metrics do not have one train/val/test sample: {path}")
    if not metrics.get("step_records"):
        raise ValueError(f"Pilot result has no per-step records: {path}")
    return payload


def _row_from_result(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    initial_losses = metrics["initial_split_losses"]
    final_losses = metrics["final_split_losses"]
    step_records = metrics["step_records"]
    max_steps = int(payload["max_steps"])
    if len(step_records) != max_steps:
        raise ValueError(f"Pilot step count mismatch for {path}")
    return {
        "run_name": payload["run_name"],
        "random_seed": payload.get("random_seed"),
        "corruption_mode": payload["corruption_mode"],
        "source_path": str(path),
        "result_kind": payload["result_kind"],
        "dataset_kind": payload["dataset"]["kind"],
        "dataset_subset_scope": payload["dataset"].get("subset_scope"),
        "micro_pilot_run": payload.get("micro_pilot_run", False),
        "split_type": payload["dataset"]["split_type"],
        "split_counts": payload["dataset"]["split_counts"],
        "sample_count": payload["sample_count"],
        "trainable_parameter_count": metrics["trainable_parameter_count"],
        "optimizer_name": metrics["optimizer_name"],
        "learning_rate": metrics["learning_rate"],
        "max_steps": max_steps,
        "hard_max_steps": int(payload["hard_max_steps"]),
        "optimizer_step_count": int(payload["optimizer_step_count"]),
        "backward_pass_count": int(payload["backward_pass_count"]),
        "initial_split_losses": initial_losses,
        "final_split_losses": final_losses,
        "train_loss_delta": final_losses["train"] - initial_losses["train"],
        "val_loss_delta": final_losses["val"] - initial_losses["val"],
        "test_loss_delta": final_losses["test"] - initial_losses["test"],
        "step_loss_trace": [
            {
                "optimizer_step": int(record["optimizer_step"]),
                "train_loss_before_step": record["train_loss_before_step"],
                "train_loss_after_step": record["split_losses_after_step"]["train"],
                "val_loss_after_step": record["split_losses_after_step"]["val"],
                "test_loss_after_step": record["split_losses_after_step"]["test"],
                "gradient_l2_norm": record["gradient_l2_norm"],
            }
            for record in step_records
        ],
        "safety_labels": {
            "benchmark_result": payload["benchmark_result"],
            "hardware": payload["hardware"],
            "gpu_used": payload["gpu_used"],
            "pilot_run": payload["pilot_run"],
            "bounded_pilot": payload["bounded_pilot"],
            "micro_pilot_run": payload.get("micro_pilot_run", False),
            "larger_real_training_run_started": payload["larger_real_training_run_started"],
            "checkpoint_written": payload["checkpoint_written"],
            "sampling_executed": payload["sampling_executed"],
            "scheduler": payload["scheduler"],
            "slurm_account": payload["slurm_account"],
        },
        "note": "Bounded tiny real-data pilot diagnostics only; not benchmark results.",
    }


def _validate_multiseed_rows(rows: list[dict[str, Any]], expected_seeds: tuple[int, ...]) -> None:
    expected_modes = {"isotropic", "stiffness_structured"}
    expected_pairs = {(mode, seed) for mode in expected_modes for seed in expected_seeds}
    observed_pairs = {(row["corruption_mode"], int(row["random_seed"])) for row in rows}
    if observed_pairs != expected_pairs:
        raise ValueError(f"Micro-pilot multi-seed rows do not match expected mode/seed pairs: {observed_pairs}")
    for row in rows:
        if row["dataset_subset_scope"] != "micro_pilot_processed_dry_run":
            raise ValueError("Micro-pilot multi-seed summary requires micro-pilot subset rows")
        if row["split_counts"] != {"test": 1, "train": 4, "val": 1}:
            raise ValueError("Micro-pilot multi-seed summary requires 4/1/1 split counts")
        if row["safety_labels"]["checkpoint_written"] is not False:
            raise ValueError("Micro-pilot multi-seed summary does not allow checkpointed inputs")
        if row["safety_labels"]["sampling_executed"] is not False:
            raise ValueError("Micro-pilot multi-seed summary does not allow sampled inputs")


def _mode_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = {}
    for mode in ["isotropic", "stiffness_structured"]:
        mode_rows = sorted(
            [row for row in rows if row["corruption_mode"] == mode],
            key=lambda row: int(row["random_seed"]),
        )
        stats[mode] = {
            "seed_count": len(mode_rows),
            "seeds": [int(row["random_seed"]) for row in mode_rows],
            "split_counts": mode_rows[0]["split_counts"] if mode_rows else None,
            "sample_count": mode_rows[0]["sample_count"] if mode_rows else None,
            "optimizer_step_count": mode_rows[0]["optimizer_step_count"] if mode_rows else None,
            "final_train_loss": _series_stats([row["final_split_losses"]["train"] for row in mode_rows]),
            "final_val_loss": _series_stats([row["final_split_losses"]["val"] for row in mode_rows]),
            "final_test_loss": _series_stats([row["final_split_losses"]["test"] for row in mode_rows]),
            "train_loss_delta": _series_stats([row["train_loss_delta"] for row in mode_rows]),
            "val_loss_delta": _series_stats([row["val_loss_delta"] for row in mode_rows]),
            "test_loss_delta": _series_stats([row["test_loss_delta"] for row in mode_rows]),
        }
    return stats


def _series_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _markdown_report(summary: dict[str, Any], title: str, scope_note: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"**Scope:** {scope_note}. This is not a benchmark result.",
        "",
        f"- `result_kind`: `{summary['result_kind']}`",
        f"- `comparison_scope`: `{summary['comparison_scope']}`",
        f"- `benchmark_result`: `{str(summary['benchmark_result']).lower()}`",
        f"- `hardware`: `{summary['hardware']}`",
        f"- `gpu_used`: `{str(summary['gpu_used']).lower()}`",
        f"- `training_executed_by_summary`: `{str(summary['training_executed_by_summary']).lower()}`",
        f"- `larger_real_training_run_started`: `{str(summary['larger_real_training_run_started']).lower()}`",
        f"- `slurm_account`: `{summary['slurm_account']}`",
        "",
        "| Mode | Run | Subset | Count | Train/Val/Test | Steps | LR | Train Loss Initial | Train Loss Final | Val Loss Final | Test Loss Final | Checkpoint | Sampling |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {mode} | {run} | {subset} | {count} | {splits} | {steps} | {lr} | {train_i} | {train_f} | "
            "{val_f} | {test_f} | {checkpoint} | {sampling} |".format(
                mode=row["corruption_mode"],
                run=row["run_name"],
                subset=row["dataset_subset_scope"] or "tiny_real_processed_dry_run",
                count=row["sample_count"],
                splits=_split_counts_cell(row["split_counts"]),
                steps=row["optimizer_step_count"],
                lr=_format_value(row["learning_rate"]),
                train_i=_format_value(row["initial_split_losses"]["train"]),
                train_f=_format_value(row["final_split_losses"]["train"]),
                val_f=_format_value(row["final_split_losses"]["val"]),
                test_f=_format_value(row["final_split_losses"]["test"]),
                checkpoint=_bool_cell(row["safety_labels"]["checkpoint_written"]),
                sampling=_bool_cell(row["safety_labels"]["sampling_executed"]),
            )
        )
    lines.extend(
        [
            "",
            "Per-step train loss trace:",
            "",
            "| Mode | Step | Train Loss Before | Train Loss After | Val Loss After | Test Loss After | Grad L2 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        for step in row["step_loss_trace"]:
            lines.append(
                "| {mode} | {step} | {train_before} | {train_after} | {val_after} | {test_after} | {grad} |".format(
                    mode=row["corruption_mode"],
                    step=step["optimizer_step"],
                    train_before=_format_value(step["train_loss_before_step"]),
                    train_after=_format_value(step["train_loss_after_step"]),
                    val_after=_format_value(step["val_loss_after_step"]),
                    test_after=_format_value(step["test_loss_after_step"]),
                    grad=_format_value(step["gradient_l2_norm"]),
                )
            )
    lines.extend(
        [
            "",
            "Notes:",
            "- The summary reads existing pilot result JSON files only; it does not run training.",
            "- Each input pilot is bounded and records train/validation/test losses as diagnostics only.",
            "- Losses are tiny pilot diagnostics and must not be reported as GEOM-Drugs benchmark metrics.",
            "- No raw data was modified, and no GPU execution, SLURM job, environment mutation, checkpointing, sampling, or larger real training run was used.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_multiseed_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Tiny Real-Data Bounded Micro-Pilot Multi-Seed Summary",
        "",
        "**Scope:** bounded six-molecule GEOM micro-pilot outputs across a tiny fixed seed set. This is not a benchmark result.",
        "",
        f"- `result_kind`: `{summary['result_kind']}`",
        f"- `comparison_scope`: `{summary['comparison_scope']}`",
        f"- `benchmark_result`: `{str(summary['benchmark_result']).lower()}`",
        f"- `hardware`: `{summary['hardware']}`",
        f"- `gpu_used`: `{str(summary['gpu_used']).lower()}`",
        f"- `seed_set`: `{summary['seed_set']}`",
        f"- `training_executed_by_summary`: `{str(summary['training_executed_by_summary']).lower()}`",
        f"- `larger_real_training_run_started`: `{str(summary['larger_real_training_run_started']).lower()}`",
        f"- `slurm_account`: `{summary['slurm_account']}`",
        "",
        "Final-loss statistics by mode:",
        "",
        "| Mode | Seeds | Count | Train/Val/Test | Steps | Train Final Mean | Val Final Mean | Test Final Mean | Train Delta Mean | Checkpoint | Sampling |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for mode in ["isotropic", "stiffness_structured"]:
        stats = summary["mode_statistics"][mode]
        lines.append(
            "| {mode} | {seeds} | {count} | {splits} | {steps} | {train} | {val} | {test} | {delta} | no | no |".format(
                mode=mode,
                seeds=",".join(str(seed) for seed in stats["seeds"]),
                count=stats["sample_count"],
                splits=_split_counts_cell(stats["split_counts"]),
                steps=stats["optimizer_step_count"],
                train=_format_value(stats["final_train_loss"]["mean"]),
                val=_format_value(stats["final_val_loss"]["mean"]),
                test=_format_value(stats["final_test_loss"]["mean"]),
                delta=_format_value(stats["train_loss_delta"]["mean"]),
            )
        )

    lines.extend(
        [
            "",
            "Per-seed final losses:",
            "",
            "| Mode | Seed | Run | Train Final | Val Final | Test Final | Train Delta | Step Records |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(summary["rows"], key=lambda item: (item["corruption_mode"], int(item["random_seed"]))):
        lines.append(
            "| {mode} | {seed} | {run} | {train} | {val} | {test} | {delta} | {steps} |".format(
                mode=row["corruption_mode"],
                seed=row["random_seed"],
                run=row["run_name"],
                train=_format_value(row["final_split_losses"]["train"]),
                val=_format_value(row["final_split_losses"]["val"]),
                test=_format_value(row["final_split_losses"]["test"]),
                delta=_format_value(row["train_loss_delta"]),
                steps=len(row["step_loss_trace"]),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- The summary reads bounded micro-pilot result JSON files after the multi-seed runner completes.",
            "- Each input row is capped to the six-molecule artifact, CPU-only execution, no checkpointing, and no sampling.",
            "- Losses are multi-seed micro-pilot diagnostics and must not be reported as GEOM-Drugs benchmark metrics.",
            "- No raw data was modified, and no GPU execution, SLURM job, environment mutation, checkpointing, sampling, or larger real training run was used.",
            "",
        ]
    )
    return "\n".join(lines)


def _split_counts_cell(split_counts: dict[str, int]) -> str:
    return f"{split_counts['train']}/{split_counts['val']}/{split_counts['test']}"


def _bool_cell(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "NA"


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return _bool_cell(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
