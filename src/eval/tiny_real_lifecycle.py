"""Aggregate the tiny real-data smoke lifecycle into one report."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


DEFAULT_RESULT_PATHS = [
    Path("experiments/results/geom_tiny_real_rdkit_etkdg_mmff_seed0.json"),
    Path("experiments/results/geom_tiny_real_model_isotropic_seed0.json"),
    Path("experiments/results/geom_tiny_real_model_stiffness_seed0.json"),
    Path("experiments/results/geom_tiny_real_execution_isotropic_seed0.json"),
    Path("experiments/results/geom_tiny_real_execution_stiffness_seed0.json"),
    Path("experiments/results/geom_tiny_real_single_step_isotropic_seed0.json"),
    Path("experiments/results/geom_tiny_real_single_step_stiffness_seed0.json"),
    Path("experiments/results/geom_tiny_real_multistep_isotropic_steps3_seed0.json"),
    Path("experiments/results/geom_tiny_real_multistep_stiffness_steps3_seed0.json"),
    Path("experiments/results/geom_tiny_real_ckpt_sample_isotropic_steps3_seed0.json"),
    Path("experiments/results/geom_tiny_real_ckpt_sample_stiffness_steps3_seed0.json"),
]

SUMMARY_RESULT_KIND = "tiny_real_data_lifecycle_smoke_summary_not_benchmark"
SUPPORTED_RESULT_KINDS = {
    "tiny_real_data_dry_run_not_benchmark": "baseline",
    "tiny_real_data_model_dry_run_not_benchmark": "model_dry_run",
    "tiny_real_data_model_execution_smoke_not_benchmark": "execution_smoke",
    "tiny_real_data_single_step_training_smoke_not_benchmark": "single_step_training_smoke",
    "tiny_real_data_bounded_multistep_training_smoke_not_benchmark": "bounded_multistep_training_smoke",
    "tiny_real_data_checkpoint_sampling_smoke_not_benchmark": "checkpoint_sampling_smoke",
}


def write_tiny_real_lifecycle_summary(
    result_paths: list[str | Path] | None = None,
    summary_path: str | Path = "experiments/results/geom_tiny_real_lifecycle_summary.json",
    report_path: str | Path = "reports/tables/geom_tiny_real_lifecycle_summary.md",
) -> dict[str, Path]:
    start = time.time()
    paths = [Path(path) for path in (result_paths or DEFAULT_RESULT_PATHS)]
    payloads = [_load_result(path) for path in paths]
    rows = [_row_from_result(path, payload) for path, payload in zip(paths, payloads)]
    stage_counts: dict[str, int] = {}
    for row in rows:
        stage_counts[row["stage"]] = stage_counts.get(row["stage"], 0) + 1

    summary = {
        "summary_name": "geom_tiny_real_lifecycle_summary",
        "result_kind": SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_cpu_tiny_real_lifecycle_smoke_report",
        "hardware": "cpu_only",
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "larger_real_training_run_started": False,
        "input_result_paths": [str(path) for path in paths],
        "row_count": len(rows),
        "stage_counts": stage_counts,
        "rows": rows,
        "runtime_seconds": time.time() - start,
        "notes": [
            "This is a tiny real-data smoke lifecycle summary over existing result JSON files.",
            "It aggregates baseline, dry-run, execution, training-smoke, and checkpoint/sampling-smoke stages.",
            "Rows are plumbing smoke diagnostics only and are not benchmark results.",
            "No raw data modification, environment mutation, GPU execution, SLURM job, or larger real training run was used.",
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
        raise FileNotFoundError(f"Tiny real lifecycle input is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    result_kind = payload.get("result_kind")
    if result_kind not in SUPPORTED_RESULT_KINDS:
        raise ValueError(f"Unsupported tiny real lifecycle result_kind for {path}: {result_kind}")
    if payload.get("benchmark_result") is not False:
        raise ValueError(f"Input must be labeled benchmark_result=false: {path}")
    if payload.get("hardware") != "cpu_only":
        raise ValueError(f"Input must be labeled hardware=cpu_only: {path}")
    if payload.get("gpu_used") is True:
        raise ValueError(f"Input unexpectedly used GPU: {path}")
    if payload.get("scheduler") is not None:
        raise ValueError(f"Input unexpectedly records scheduler use: {path}")
    if payload.get("slurm_account") is not None:
        raise ValueError(f"Input unexpectedly records SLURM account use: {path}")
    if payload.get("larger_real_training_run_started") is True:
        raise ValueError(f"Input unexpectedly started a larger real training run: {path}")
    dataset = payload.get("dataset") or {}
    if dataset and dataset.get("kind") != "geom_drugs_tiny_processed_dry_run":
        raise ValueError(f"Input is not from the tiny real processed artifact: {path}")
    return payload


def _row_from_result(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    result_kind = payload["result_kind"]
    metrics = payload.get("metrics") or {}
    stage = SUPPORTED_RESULT_KINDS[result_kind]
    row = {
        "stage": stage,
        "run_name": payload.get("run_name"),
        "method": _method(payload),
        "result_kind": result_kind,
        "source_path": str(path),
        "dataset_kind": (payload.get("dataset") or {}).get("kind", payload.get("dataset_kind")),
        "sample_or_molecule_count": _count(payload, metrics),
        "training_executed": payload.get("training_executed"),
        "sampling_executed": payload.get("sampling_executed"),
        "checkpoint_written": payload.get("checkpoint_written", False),
        "checkpoint_reloaded": payload.get("checkpoint_reloaded", False),
        "checkpoint_scope": payload.get("checkpoint_scope"),
        "max_steps": payload.get("max_steps"),
        "optimizer_step_count": payload.get("optimizer_step_count", metrics.get("optimizer_step_count")),
        "backward_pass_count": payload.get("backward_pass_count", metrics.get("backward_pass_count")),
        "success_rate": metrics.get("success_rate"),
        "rmsd_proxy": _rmsd_proxy(stage, metrics),
        "loss_before": _loss_before(stage, metrics),
        "loss_after": _loss_after(stage, metrics),
        "step_record_count": len(metrics.get("step_records", metrics.get("train_steps", []))),
        "sampled_molecule_count": len(metrics.get("sampled_molecules", [])),
        "mean_abs_bond_length_delta": _bond_delta(stage, metrics),
        "mean_mmff_energy": metrics.get("mean_mmff_energy"),
        "note": _stage_note(stage),
    }
    return row


def _method(payload: dict[str, Any]) -> str:
    if payload.get("baseline_name"):
        return str(payload["baseline_name"])
    return str(payload.get("corruption_mode") or payload.get("model_name") or "unknown")


def _count(payload: dict[str, Any], metrics: dict[str, Any]) -> int | None:
    return (
        metrics.get("attempted_molecule_count")
        or metrics.get("sample_count")
        or payload.get("sample_count")
        or (payload.get("dataset") or {}).get("molecule_count")
    )


def _rmsd_proxy(stage: str, metrics: dict[str, Any]) -> float | None:
    if stage == "baseline":
        return metrics.get("mean_aligned_rmsd_to_tiny_real_conformer")
    if stage == "model_dry_run":
        return (metrics.get("cov_mat") or {}).get("mat_r_smoke")
    sampled = metrics.get("sampled_molecules") or []
    if sampled:
        values = [item["sampled_rmsd_to_target"] for item in sampled if item.get("sampled_rmsd_to_target") is not None]
        return sum(values) / len(values) if values else None
    return None


def _loss_before(stage: str, metrics: dict[str, Any]) -> float | None:
    if stage == "single_step_training_smoke":
        return metrics.get("loss_before_step")
    return metrics.get("initial_loss")


def _loss_after(stage: str, metrics: dict[str, Any]) -> float | None:
    if stage == "single_step_training_smoke":
        return metrics.get("loss_after_step")
    if stage == "checkpoint_sampling_smoke":
        return metrics.get("reloaded_loss")
    return metrics.get("final_loss") or metrics.get("mean_velocity_mse")


def _bond_delta(stage: str, metrics: dict[str, Any]) -> float | None:
    if stage == "baseline":
        return metrics.get("mean_abs_bond_length_delta")
    if stage == "model_dry_run":
        return (metrics.get("validity") or {}).get("mean_abs_bond_length_delta")
    return None


def _stage_note(stage: str) -> str:
    notes = {
        "baseline": "RDKit ETKDG/MMFF tiny real dry-run; not a benchmark baseline.",
        "model_dry_run": "Model-side source construction dry-run; no model execution.",
        "execution_smoke": "Forward/loss execution smoke; no optimizer or backward pass.",
        "single_step_training_smoke": "Exactly one optimizer step; not a real training run.",
        "bounded_multistep_training_smoke": "Hard-capped tiny optimizer sequence; not larger training.",
        "checkpoint_sampling_smoke": "Disposable checkpoint reload plus one bounded sampling update.",
    }
    return notes[stage]


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Tiny Real-Data End-to-End Smoke Lifecycle Summary",
        "",
        "**Scope:** existing three-molecule GEOM tiny real-data smoke outputs only. This is not a benchmark result.",
        "",
        f"- `result_kind`: `{summary['result_kind']}`",
        f"- `benchmark_result`: `{str(summary['benchmark_result']).lower()}`",
        f"- `hardware`: `{summary['hardware']}`",
        f"- `larger_real_training_run_started`: `{str(summary['larger_real_training_run_started']).lower()}`",
        f"- `slurm_account`: `{summary['slurm_account']}`",
        "",
        "Stage counts:",
    ]
    for stage, count in sorted(summary["stage_counts"].items()):
        lines.append(f"- `{stage}`: {count}")
    lines.extend(
        [
            "",
            "| Stage | Method | Run | Count | Train | Sample | Checkpoint | Steps | Loss Before | Loss After | RMSD Proxy | Bond Delta | MMFF Energy |",
            "|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            "| {stage} | {method} | {run} | {count} | {train} | {sample} | {checkpoint} | {steps} | "
            "{loss_before} | {loss_after} | {rmsd} | {bond} | {energy} |".format(
                stage=row["stage"],
                method=row["method"],
                run=row["run_name"],
                count=_format_value(row["sample_or_molecule_count"]),
                train=_bool_cell(row["training_executed"]),
                sample=_bool_cell(row["sampling_executed"]),
                checkpoint=_checkpoint_cell(row),
                steps=_format_value(row["optimizer_step_count"]),
                loss_before=_format_value(row["loss_before"]),
                loss_after=_format_value(row["loss_after"]),
                rmsd=_format_value(row["rmsd_proxy"]),
                bond=_format_value(row["mean_abs_bond_length_delta"]),
                energy=_format_value(row["mean_mmff_energy"]),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Dry-run rows validate data/source plumbing only.",
            "- Execution rows validate forward/loss plumbing without optimization.",
            "- Training-smoke rows are deliberately tiny and bounded; they are not real training runs.",
            "- Checkpoint/sampling rows use disposable smoke checkpoints and one bounded sampling update.",
            "- These rows must not be reported as GEOM-Drugs benchmark results.",
            "",
        ]
    )
    return "\n".join(lines)


def _checkpoint_cell(row: dict[str, Any]) -> str:
    if row["checkpoint_written"]:
        return "written/reloaded" if row["checkpoint_reloaded"] else "written"
    return "no"


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
