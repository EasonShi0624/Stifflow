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
SUMMARY_RESULT_KIND = "tiny_real_data_bounded_pilot_comparison_not_benchmark"
PILOT_RESULT_KIND = "tiny_real_data_bounded_pilot_not_benchmark"


def write_tiny_real_pilot_comparison(
    result_paths: list[str | Path] | None = None,
    summary_path: str | Path = "experiments/results/geom_tiny_real_pilot_summary.json",
    report_path: str | Path = "reports/tables/geom_tiny_real_pilot_summary.md",
) -> dict[str, Path]:
    start = time.time()
    paths = [Path(path) for path in (result_paths or DEFAULT_RESULT_PATHS)]
    payloads = [_load_pilot_result(path) for path in paths]
    rows = [_row_from_result(path, payload) for path, payload in zip(paths, payloads)]
    modes = sorted(row["corruption_mode"] for row in rows)
    if modes != ["isotropic", "stiffness_structured"]:
        raise ValueError(f"Pilot comparison expects isotropic and stiffness_structured rows, got {modes}")

    summary = {
        "summary_name": "geom_tiny_real_pilot_summary",
        "result_kind": SUMMARY_RESULT_KIND,
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
            "This is a reporting-only comparison of two bounded tiny real-data pilot outputs.",
            "The input pilots each used the existing three-molecule GEOM tiny real artifact and one train/val/test molecule.",
            "Loss traces and final losses are pilot diagnostics only, not benchmark metrics or evidence of model quality.",
            "No raw-data modification, environment mutation, GPU execution, SLURM job, sampling, checkpointing, or larger real training run was used by this summary.",
        ],
    }

    summary_path = Path(summary_path)
    report_path = Path(report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def _load_pilot_result(path: Path) -> dict[str, Any]:
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
    if dataset.get("split_counts") != {"test": 1, "train": 1, "val": 1}:
        raise ValueError(f"Pilot result does not have one train/val/test molecule: {path}")
    metrics = payload.get("metrics") or {}
    if metrics.get("split_sample_counts") != {"test": 1, "train": 1, "val": 1}:
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
        "corruption_mode": payload["corruption_mode"],
        "source_path": str(path),
        "result_kind": payload["result_kind"],
        "dataset_kind": payload["dataset"]["kind"],
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
            "larger_real_training_run_started": payload["larger_real_training_run_started"],
            "checkpoint_written": payload["checkpoint_written"],
            "sampling_executed": payload["sampling_executed"],
            "scheduler": payload["scheduler"],
            "slurm_account": payload["slurm_account"],
        },
        "note": "Bounded tiny real-data pilot diagnostics only; not benchmark results.",
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Tiny Real-Data Bounded Pilot Summary",
        "",
        "**Scope:** two bounded three-molecule GEOM tiny real-data pilot outputs only. This is not a benchmark result.",
        "",
        f"- `result_kind`: `{summary['result_kind']}`",
        f"- `benchmark_result`: `{str(summary['benchmark_result']).lower()}`",
        f"- `hardware`: `{summary['hardware']}`",
        f"- `gpu_used`: `{str(summary['gpu_used']).lower()}`",
        f"- `training_executed_by_summary`: `{str(summary['training_executed_by_summary']).lower()}`",
        f"- `larger_real_training_run_started`: `{str(summary['larger_real_training_run_started']).lower()}`",
        f"- `slurm_account`: `{summary['slurm_account']}`",
        "",
        "| Mode | Run | Count | Train/Val/Test | Steps | LR | Train Loss Initial | Train Loss Final | Val Loss Final | Test Loss Final | Checkpoint | Sampling |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {mode} | {run} | {count} | {splits} | {steps} | {lr} | {train_i} | {train_f} | "
            "{val_f} | {test_f} | {checkpoint} | {sampling} |".format(
                mode=row["corruption_mode"],
                run=row["run_name"],
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
            "- Each input pilot is bounded to the existing three-molecule artifact with one train, one validation, and one test molecule.",
            "- Losses are tiny pilot diagnostics and must not be reported as GEOM-Drugs benchmark metrics.",
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
