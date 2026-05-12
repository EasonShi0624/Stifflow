#!/usr/bin/env python
"""Summarize completed Stage 1 low-lambda AdamW hybrid training results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_hybrid_low_lambda_adamw_plan.json")
DEFAULT_SUMMARY_PATH = Path("experiments/results/geom_full_database_hybrid_low_lambda_adamw_summary.json")
DEFAULT_REPORT_PATH = Path("reports/tables/geom_full_database_hybrid_low_lambda_adamw_summary.md")
METRICS = (
    "best_validation_loss",
    "reloaded_validation_loss",
    "sampled_rmsd_to_target_overall_mean",
    "sampled_rmsd_to_target_train_mean",
    "sampled_rmsd_to_target_val_mean",
    "sampled_rmsd_to_target_test_mean",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-path", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--require-results", action="store_true")
    args = parser.parse_args()

    outputs = summarize(args)
    for key, path in outputs.items():
        print(f"{key}: {path}")
    print("benchmark_result: false")
    print("note: Low-lambda AdamW training summary reads existing result JSONs only; no training, evaluation, scheduler submission, raw-data mutation, or environment mutation.")


def summarize(args: argparse.Namespace) -> dict[str, Path]:
    plan_path = Path(args.plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows, missing = _load_rows(plan)
    if args.require_results and missing:
        raise FileNotFoundError("Missing low-lambda AdamW result JSONs:\n" + json.dumps(missing, indent=2, sort_keys=True))
    summary = {
        "summary_name": "geom_full_database_hybrid_low_lambda_adamw_summary",
        "result_kind": "geom_drugs_full_database_hybrid_low_lambda_adamw_summary_not_benchmark",
        "benchmark_result": False,
        "execution_mode": "local_reporting_only_low_lambda_adamw_training_summary",
        "hardware": "reporting_only",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "plan_path": str(plan_path),
        "available_result_count": len(rows),
        "expected_result_count": len(plan["training_jobs"]),
        "missing_result_count": len(missing),
        "missing_results": missing,
        "seed_level_rows": rows,
        "aggregate_by_split_lambda": _aggregate(rows, ("split_type", "lambda_value")),
        "rankings": _rankings(rows),
        "references": plan["references"],
        "notes": [
            "Old SGD lambda runs are not included.",
            "This summary covers Stage 1 seed0 random/scaffold low-lambda AdamW training only.",
        ],
    }
    summary_path = Path(args.summary_path)
    report_path = Path(args.report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def _load_rows(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    missing = []
    for job in plan["training_jobs"]:
        path = Path(job["expected_result_path"])
        if not path.exists():
            missing.append(job)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = _row_from_result(job, path, payload)
        rows.append(row)
    return rows, missing


def _row_from_result(job: dict[str, Any], path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("benchmark_result") is not False:
        raise ValueError(f"Result must be benchmark_result=false: {path}")
    if payload.get("corruption_mode") != "hybrid_harmonic_stiffness_v1":
        raise ValueError(f"Unexpected corruption mode in {path}")
    if str(payload.get("optimizer_name")) != "AdamW":
        raise ValueError(f"Expected AdamW result: {path}")
    if abs(float(payload.get("learning_rate")) - 1.0e-3) > 1.0e-12:
        raise ValueError(f"Expected lr=1e-3 result: {path}")
    if abs(float(payload.get("hybrid_source_strength")) - float(job["lambda_value"])) > 1.0e-12:
        raise ValueError(f"Lambda mismatch in {path}")
    metrics = payload["metrics"]
    sampled = metrics["sampled_molecules"]
    by_split = {
        split: _safe_mean([float(row["sampled_rmsd_to_target"]) for row in sampled if row.get("split") == split])
        for split in ("train", "val", "test")
    }
    return {
        "split_type": job["split_type"],
        "seed": int(job["seed"]),
        "lambda_label": job["lambda_label"],
        "lambda_value": float(job["lambda_value"]),
        "result_path": str(path),
        "run_name": payload.get("run_name"),
        "optimizer_name": payload.get("optimizer_name"),
        "learning_rate": float(payload.get("learning_rate")),
        "best_validation_loss": float(metrics["best_validation_loss"]),
        "best_validation_loss_step": int(metrics["best_validation_loss_step"]),
        "reloaded_validation_loss": float(metrics["reloaded_validation_loss"]),
        "executed_train_steps": int(metrics["executed_train_steps"]),
        "stop_reason": str(metrics.get("stop_reason") or payload.get("early_stop_reason")),
        "sampled_rmsd_to_target_overall_mean": _safe_mean(
            [float(row["sampled_rmsd_to_target"]) for row in sampled]
        ),
        "sampled_rmsd_to_target_train_mean": by_split["train"],
        "sampled_rmsd_to_target_val_mean": by_split["val"],
        "sampled_rmsd_to_target_test_mean": by_split["test"],
    }


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = {}
    for key, group in grouped.items():
        name = "/".join(str(part) for part in key)
        output[name] = {
            **{keys[index]: key[index] for index in range(len(keys))},
            "count": len(group),
            "metrics": {
                metric: _safe_mean([row[metric] for row in group if row.get(metric) is not None]) for metric in METRICS
            },
        }
    return output


def _rankings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for split_type in sorted({row["split_type"] for row in rows}):
        split_rows = [row for row in rows if row["split_type"] == split_type]
        output[split_type] = {
            metric: [
                {"lambda_value": row["lambda_value"], "value": row[metric], "rank": index + 1}
                for index, row in enumerate(sorted(split_rows, key=lambda item: item[metric]))
            ]
            for metric in METRICS
        }
    return output


def _safe_mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Low-Lambda AdamW Training Summary",
        "",
        "Reads completed Stage 1 low-lambda AdamW result JSONs only. This is not a benchmark.",
        "",
        f"- Available results: `{summary['available_result_count']}` / `{summary['expected_result_count']}`",
        f"- Missing results: `{summary['missing_result_count']}`",
        "",
        "| Split | Lambda | Best val loss | Reloaded val loss | RMSD overall | Stop reason | Steps |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in sorted(summary["seed_level_rows"], key=lambda item: (item["split_type"], item["lambda_value"])):
        lines.append(
            f"| {row['split_type']} | {row['lambda_value']} | {row['best_validation_loss']:.8g} | "
            f"{row['reloaded_validation_loss']:.8g} | {row['sampled_rmsd_to_target_overall_mean']:.8g} | "
            f"{row['stop_reason']} | {row['executed_train_steps']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
