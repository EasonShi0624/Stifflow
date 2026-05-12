#!/usr/bin/env python
"""Merge hybrid-lambda complete val/test COV/AMR chunk JSONLs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.merge_geom_full_database_final_cov_amr_complete_valtest as base  # noqa: E402
from src.data.geom_full_database import FULL_DATABASE_SCOPE  # noqa: E402


DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_hybrid_lambda_complete_cov_amr_plan.json")
DEFAULT_SUMMARY_PATH = Path("experiments/results/geom_full_database_hybrid_lambda_complete_cov_amr_summary.json")
DEFAULT_REPORT_PATH = Path("reports/tables/geom_full_database_hybrid_lambda_complete_cov_amr_summary.md")
PLAN_RESULT_KIND = "geom_drugs_full_database_hybrid_lambda_complete_cov_amr_plan_not_benchmark"
SUMMARY_RESULT_KIND = "geom_drugs_full_database_hybrid_lambda_complete_cov_amr_summary_not_benchmark"
METRICS = base.METRICS
METRIC_DISPLAY = base.METRIC_DISPLAY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-path", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--chunks-dir", default=None)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    outputs = merge(args)
    for key, value in outputs.items():
        print(f"{key}: {value}")
    print("benchmark_result: false")
    print("note: Merged streamed hybrid-lambda complete val/test COV/AMR outputs; no training, sampling, scheduler submission, raw-data mutation, or environment mutation.")


def merge(args: argparse.Namespace) -> dict[str, Path]:
    start = time.time()
    plan_path = Path(args.plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _validate_plan(plan)
    if args.chunks_dir is not None:
        base._override_chunks_dir(plan, Path(args.chunks_dir))

    task_scan = base._scan_tasks(plan)
    if args.require_complete and not task_scan["complete"]:
        missing_text = json.dumps(task_scan["incomplete_tasks"][:20], indent=2, sort_keys=True)
        raise RuntimeError(f"Hybrid-lambda COV/AMR merge requires all chunks. First incomplete tasks:\n{missing_text}")

    complete_entries = [entry["entry_label"] for entry in plan["complete_entries"]]
    hybrid_entries = [entry for entry in plan["complete_entries"] if entry["role"] == "hybrid_candidate"]
    stream = base._stream_aggregates(plan)
    status_counts = stream["status_counts"]
    aggregate_by_entry = stream["aggregates"]["by_entry"]
    aggregate_by_split_entry = stream["aggregates"]["by_split_entry"]
    aggregate_by_split_partition_entry = stream["aggregates"]["by_split_partition_entry"]
    aggregate_by_split_seed_entry = stream["aggregates"]["by_split_seed_entry"]
    aggregate_by_split_partition_seed_entry = stream["aggregates"]["by_split_partition_seed_entry"]
    aggregate_by_partition_entry = stream["aggregates"]["by_partition_entry"]
    rankings = {
        "cross_split": base._rankings(aggregate_by_entry, tuple(), complete_entries),
        "by_split": base._rankings(aggregate_by_split_entry, ("split_type",), complete_entries),
        "by_split_partition": base._rankings(aggregate_by_split_partition_entry, ("split_type", "partition"), complete_entries),
        "by_split_seed": base._rankings(aggregate_by_split_seed_entry, ("split_type", "seed"), complete_entries),
    }
    comparisons = base._hybrids_vs_harmonic(
        aggregates={
            "cross_split": (aggregate_by_entry, tuple()),
            "by_split": (aggregate_by_split_entry, ("split_type",)),
            "by_split_partition": (aggregate_by_split_partition_entry, ("split_type", "partition")),
            "by_split_seed": (aggregate_by_split_seed_entry, ("split_type", "seed")),
            "by_split_partition_seed": (
                aggregate_by_split_partition_seed_entry,
                ("split_type", "partition", "seed"),
            ),
        },
        hybrid_entries=hybrid_entries,
    )
    lambda_selection = base._lambda_selection(aggregate_by_entry, hybrid_entries)
    decision = base._decision_support(
        comparisons=comparisons,
        lambda_selection=lambda_selection,
        complete=task_scan["complete"],
    )
    summary = {
        "summary_name": "geom_full_database_hybrid_lambda_complete_cov_amr_summary",
        "result_kind": SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "hybrid_lambda_complete_cov_amr_chunk_merge_reporting_only",
        "hardware": "reporting_only",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "dataset_scope": FULL_DATABASE_SCOPE,
        "plan_path": str(plan_path),
        "plan_result_kind": plan["result_kind"],
        "splits": plan["splits"],
        "partitions": plan["partitions"],
        "train_partition_included": plan["train_partition_included"],
        "entries": plan["complete_entries"],
        "eligible_hybrid_lambdas": plan["entry_readiness"]["eligible_hybrid_lambdas"],
        "evaluation": plan["evaluation"],
        "coverage": {
            "expected_task_count": task_scan["expected_task_count"],
            "completed_task_count": task_scan["completed_task_count"],
            "missing_status_count": task_scan["missing_status_count"],
            "missing_jsonl_count": task_scan["missing_jsonl_count"],
            "incomplete_task_count": task_scan["incomplete_task_count"],
            "complete": task_scan["complete"],
            "planned_molecule_slots": plan["coverage"]["total_molecule_slots"],
            "planned_method_seed_molecule_records": plan["coverage"]["estimated_method_seed_molecule_records"],
            "evaluated_method_seed_molecule_records": stream["evaluated_record_count"],
            "skipped_record_count": stream["skipped_record_count"],
            "failed_record_count": stream["failed_record_count"],
            "json_decode_error_count": status_counts["json_decode_error"],
            "row_status_counts": dict(status_counts),
        },
        "task_completeness": task_scan,
        "aggregate_by_entry": aggregate_by_entry,
        "aggregate_by_split_entry": aggregate_by_split_entry,
        "aggregate_by_split_partition_entry": aggregate_by_split_partition_entry,
        "aggregate_by_split_seed_entry": aggregate_by_split_seed_entry,
        "aggregate_by_split_partition_seed_entry": aggregate_by_split_partition_seed_entry,
        "aggregate_by_partition_entry": aggregate_by_partition_entry,
        "rankings": rankings,
        "hybrids_vs_harmonic": comparisons,
        "lambda_selection": lambda_selection,
        "seed_level_consistency": base._seed_level_consistency(comparisons),
        "decision_support": decision,
        "skipped_examples": stream["skipped_examples"],
        "failed_examples": stream["failed_examples"],
        "chunk_jsonl_dir": plan["chunking"]["chunks_dir"],
        "runtime_seconds": time.time() - start,
        "notes": [
            "This merge reads per-chunk JSONLs and status files only.",
            "COV-R and COV-P are higher-is-better; AMR-R and AMR-P are lower-is-better.",
            "Old SGD lambda results are excluded upstream by the plan.",
            "If no hybrid dominates all four COV/AMR metrics, report the Pareto tradeoff rather than forcing one universal lambda.",
        ],
    }
    summary_path = Path(args.summary_path)
    report_path = Path(args.report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("result_kind") != PLAN_RESULT_KIND:
        raise ValueError("Unexpected hybrid-lambda complete COV/AMR plan result_kind")
    if plan.get("benchmark_result") is not False:
        raise ValueError("Hybrid-lambda complete COV/AMR plan must be benchmark_result=false")
    if plan.get("dataset_scope") != FULL_DATABASE_SCOPE:
        raise ValueError("Hybrid-lambda complete COV/AMR plan has wrong dataset scope")
    if plan.get("train_partition_included") is not False:
        raise ValueError("Default hybrid-lambda complete COV/AMR run must exclude train")
    if plan.get("chunking", {}).get("slurm_array_used") is not False:
        raise ValueError("Array-style plans are disallowed")
    if not plan.get("evaluation_ready") and plan.get("tasks"):
        raise ValueError("Plan has tasks even though evaluation_ready=false")


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Lambda Complete Val/Test COV/AMR Summary",
        "",
        "Merged complete val/test COV/AMR chunk outputs for hybrid lambda comparison. This remains non-benchmark reporting.",
        "",
        "## Completeness",
        "",
        f"- Completed tasks: `{summary['coverage']['completed_task_count']}` / `{summary['coverage']['expected_task_count']}`",
        f"- Evaluated records: `{summary['coverage']['evaluated_method_seed_molecule_records']}` / `{summary['coverage']['planned_method_seed_molecule_records']}`",
        f"- Skipped records: `{summary['coverage']['skipped_record_count']}`",
        f"- Failed records: `{summary['coverage']['failed_record_count']}`",
        f"- Complete: `{summary['coverage']['complete']}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Entry | Records | COV-R mean | COV-R median | COV-P mean | COV-P median | AMR-R mean | AMR-R median | AMR-P mean | AMR-P median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in [item["entry_label"] for item in summary["entries"]]:
        row = summary["aggregate_by_entry"].get(entry)
        if row is None:
            lines.append(f"| {entry} | 0 | NA | NA | NA | NA | NA | NA | NA | NA |")
            continue
        metrics = row["metric_summary"]
        lines.append(
            f"| {entry} | {row['record_count']} | "
            f"{base._fmt(metrics['cov_r']['mean'])} | {base._fmt(metrics['cov_r']['median'])} | "
            f"{base._fmt(metrics['cov_p']['mean'])} | {base._fmt(metrics['cov_p']['median'])} | "
            f"{base._fmt(metrics['amr_r']['mean'])} | {base._fmt(metrics['amr_r']['median'])} | "
            f"{base._fmt(metrics['amr_p']['mean'])} | {base._fmt(metrics['amr_p']['median'])} |"
        )
    lines.extend(["", "## Cross-Split Rankings", "", "| Metric | Ranking |", "|---|---|"])
    rankings = summary["rankings"]["cross_split"].get("all", {}).get("metric_rankings", {})
    for metric in METRICS:
        ranking = rankings.get(metric, [])
        text = " > ".join(f"{row['entry_label']} ({base._fmt(row['mean'])})" for row in ranking)
        lines.append(f"| {METRIC_DISPLAY[metric]} | {text or 'NA'} |")
    selection = summary["lambda_selection"]
    decision = summary["decision_support"]
    lines.extend(
        [
            "",
            "## Lambda Selection",
            "",
            f"- Precision-optimal lambda: `{selection.get('precision_optimal_lambda')}`",
            f"- Recall-optimal lambda: `{selection.get('recall_optimal_lambda')}`",
            f"- Balanced lambda: `{selection.get('balanced_lambda')}`",
            f"- Pareto frontier: `{selection.get('pareto_frontier')}`",
            f"- Smaller-lambda comparison available: `{selection.get('low_lambda_comparison_available')}`",
            "",
            "## Decision Support",
            "",
            f"- Conclusion: `{decision.get('conclusion') or decision.get('reason')}`",
            f"- Hybrid improves both precision and recall: `{bool(decision.get('hybrids_improving_both_precision_and_recall'))}`",
            f"- Hybrid shows precision-recall tradeoff: `{bool(decision.get('hybrids_with_precision_recall_tradeoff'))}`",
            f"- Harmonic better on recall for: `{decision.get('hybrids_where_harmonic_is_better_on_recall')}`",
            f"- Supports final COV/AMR claim: `{decision['supports_final_cov_amr_claim']}`",
            f"- Supports benchmark-style physical validation: `{decision['supports_physical_benchmark_validation']}`",
            "",
            "## Safety",
            "",
            f"- `benchmark_result`: `{summary['benchmark_result']}`",
            f"- `training_executed_by_summary`: `{summary['training_executed_by_summary']}`",
            f"- `sampling_executed_by_summary`: `{summary['sampling_executed_by_summary']}`",
            f"- `raw_data_modified`: `{summary['raw_data_modified']}`",
            f"- `environment_modified`: `{summary['environment_modified']}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
