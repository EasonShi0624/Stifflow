#!/usr/bin/env python
"""Merge full final-optimized AMR/COV chunk JSONLs into aggregate summaries."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.geom_full_database import FULL_DATABASE_SCOPE  # noqa: E402
from src.eval.final_optimized_all_methods import SEEDS  # noqa: E402


DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_final_optimized_amr_cov_full_plan.json")
DEFAULT_SUMMARY_PATH = Path("experiments/results/geom_full_database_final_optimized_amr_cov_full_summary.json")
DEFAULT_REPORT_PATH = Path("reports/tables/geom_full_database_final_optimized_amr_cov_full_summary.md")
METHODS = (
    "hybrid_harmonic_stiffness_v1_lambda500",
    "harmonic_prior",
    "stiffness_structured",
    "isotropic",
    "torsion_only",
)
METRICS = ("cov_r", "cov_p", "amr_r", "amr_p")
METRIC_DISPLAY = {"cov_r": "COV-R", "cov_p": "COV-P", "amr_r": "AMR-R", "amr_p": "AMR-P"}
HIGHER_IS_BETTER = {"cov_r", "cov_p"}
TIE_TOLERANCE = 1.0e-12


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
    print("note: Merged streamed full AMR/COV chunk outputs; no training, sampling, raw-data mutation, environment mutation, or scheduler submission.")


def merge(args: argparse.Namespace) -> dict[str, Path]:
    start = time.time()
    plan_path = Path(args.plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _validate_plan(plan)
    if args.chunks_dir is not None:
        _override_chunks_dir(plan, Path(args.chunks_dir))

    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    task_scan = _scan_tasks(plan)
    for task in plan["tasks"]:
        jsonl_path = Path(task["jsonl_path"])
        if not jsonl_path.exists():
            continue
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    status_counts["json_decode_error"] += 1
                    rows.append(
                        {
                            "status": "failed",
                            "task_id": int(task["task_id"]),
                            "split_type": task["split_type"],
                            "partition": task["partition"],
                            "method": task["method"],
                            "seed": int(task["seed"]),
                            "molecule_id": None,
                            "reason": "json_decode_error",
                            "detail": f"{jsonl_path}:{line_number}: {exc}",
                        }
                    )
                    continue
                status = str(row.get("status") or "evaluated")
                status_counts[status] += 1
                rows.append(row)

    if args.require_complete and not task_scan["complete"]:
        missing_text = json.dumps(task_scan["incomplete_tasks"][:20], indent=2, sort_keys=True)
        raise RuntimeError(f"Full AMR/COV chunk merge requires complete task set. First incomplete tasks:\n{missing_text}")

    evaluated_rows = [row for row in rows if row.get("status") == "evaluated"]
    skipped_rows = [row for row in rows if row.get("status") == "skipped"]
    failed_rows = [row for row in rows if row.get("status") == "failed"]
    aggregate_by_method = _aggregate(evaluated_rows, ("method",))
    aggregate_by_split_method = _aggregate(evaluated_rows, ("split_type", "method"))
    aggregate_by_split_partition_method = _aggregate(evaluated_rows, ("split_type", "partition", "method"))
    aggregate_by_split_partition_seed_method = _aggregate(
        evaluated_rows,
        ("split_type", "partition", "seed", "method"),
    )
    aggregate_by_split_seed_method = _aggregate(evaluated_rows, ("split_type", "seed", "method"))
    aggregate_by_partition_method = _aggregate(evaluated_rows, ("partition", "method"))

    comparisons = {
        "by_split": _hybrid_vs_harmonic(aggregate_by_split_method, ("split_type",)),
        "by_split_partition": _hybrid_vs_harmonic(aggregate_by_split_partition_method, ("split_type", "partition")),
        "by_split_partition_seed": _hybrid_vs_harmonic(
            aggregate_by_split_partition_seed_method,
            ("split_type", "partition", "seed"),
        ),
        "cross_split": _hybrid_vs_harmonic(aggregate_by_method, tuple()),
    }
    rankings = {
        "by_split": _rankings(aggregate_by_split_method, ("split_type",)),
        "by_split_partition": _rankings(aggregate_by_split_partition_method, ("split_type", "partition")),
        "cross_split": _rankings(aggregate_by_method, tuple()),
    }
    summary = {
        "summary_name": "geom_full_database_final_optimized_amr_cov_full_summary",
        "result_kind": "geom_drugs_full_database_final_optimized_amr_cov_full_summary_not_benchmark",
        "benchmark_result": False,
        "execution_mode": "full_amr_cov_chunk_merge_reporting_only",
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
        "methods": list(METHODS),
        "seeds": list(SEEDS),
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
            "evaluated_method_seed_molecule_records": len(evaluated_rows),
            "skipped_record_count": len(skipped_rows),
            "failed_record_count": len(failed_rows),
            "json_decode_error_count": status_counts["json_decode_error"],
            "row_status_counts": dict(status_counts),
        },
        "task_completeness": task_scan,
        "aggregate_by_method": aggregate_by_method,
        "aggregate_by_split_method": aggregate_by_split_method,
        "aggregate_by_split_partition_method": aggregate_by_split_partition_method,
        "aggregate_by_split_partition_seed_method": aggregate_by_split_partition_seed_method,
        "aggregate_by_split_seed_method": aggregate_by_split_seed_method,
        "aggregate_by_partition_method": aggregate_by_partition_method,
        "hybrid_lambda500_vs_harmonic": comparisons,
        "method_rankings": rankings,
        "decision_support": _decision_support(comparisons, complete=task_scan["complete"]),
        "skipped_examples": skipped_rows[:500],
        "failed_examples": failed_rows[:500],
        "chunk_jsonl_dir": plan["chunking"]["chunks_dir"],
        "runtime_seconds": time.time() - start,
        "notes": [
            "This merge reads per-chunk JSONLs and status files only; it does not run checkpoint sampling.",
            "COV-R and COV-P are higher-is-better; AMR-R and AMR-P are lower-is-better.",
            "Use --require-complete after all standalone chunk jobs finish to fail loudly on missing chunks.",
            "This remains non-benchmark until the full evaluation has actually been run and reviewed.",
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
    if plan.get("result_kind") != "geom_drugs_full_database_final_optimized_amr_cov_full_plan_not_benchmark":
        raise ValueError("Unexpected full AMR/COV plan result_kind")
    if plan.get("benchmark_result") is not False:
        raise ValueError("Full AMR/COV plan must be benchmark_result=false")
    if plan.get("dataset_scope") != FULL_DATABASE_SCOPE:
        raise ValueError("Full AMR/COV plan has wrong dataset scope")
    if tuple(plan.get("methods", [])) != METHODS:
        raise ValueError("Full AMR/COV plan method order changed unexpectedly")
    if tuple(plan.get("seeds", [])) != SEEDS:
        raise ValueError("Full AMR/COV plan seed set does not match final optimized matrix")


def _override_chunks_dir(plan: dict[str, Any], chunks_dir: Path) -> None:
    plan["chunking"]["chunks_dir"] = str(chunks_dir)
    for task in plan["tasks"]:
        output_prefix = f"chunk_{int(task['task_id']):05d}"
        task["jsonl_path"] = str(chunks_dir / f"{output_prefix}.jsonl")
        task["manifest_path"] = str(chunks_dir / f"{output_prefix}_manifest.json")
        task["status_path"] = str(chunks_dir / f"{output_prefix}_status.json")


def _scan_tasks(plan: dict[str, Any]) -> dict[str, Any]:
    incomplete = []
    completed = 0
    missing_status = 0
    missing_jsonl = 0
    for task in plan["tasks"]:
        status_path = Path(task["status_path"])
        jsonl_path = Path(task["jsonl_path"])
        task_status = None
        if status_path.exists():
            try:
                task_status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
            except json.JSONDecodeError:
                task_status = "invalid_status_json"
        else:
            missing_status += 1
        if not jsonl_path.exists():
            missing_jsonl += 1
        if task_status == "completed" and jsonl_path.exists():
            completed += 1
            continue
        incomplete.append(
            {
                "task_id": int(task["task_id"]),
                "split_type": task["split_type"],
                "partition": task["partition"],
                "method": task["method"],
                "seed": int(task["seed"]),
                "chunk_index": int(task["chunk_index"]),
                "status_path": str(status_path),
                "jsonl_path": str(jsonl_path),
                "status": task_status,
                "jsonl_exists": jsonl_path.exists(),
            }
        )
    expected = len(plan["tasks"])
    return {
        "expected_task_count": expected,
        "completed_task_count": completed,
        "missing_status_count": missing_status,
        "missing_jsonl_count": missing_jsonl,
        "incomplete_task_count": len(incomplete),
        "complete": len(incomplete) == 0 and completed == expected,
        "incomplete_tasks": incomplete[:1000],
        "incomplete_task_truncated": len(incomplete) > 1000,
    }


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = {}
    for key, group in grouped.items():
        name = "/".join(str(part) for part in key) if key else "all"
        output[name] = {
            **{keys[index]: key[index] for index in range(len(keys))},
            "record_count": len(group),
            "metric_summary": {
                metric: _numeric_summary([float(row["metrics"][metric]) for row in group]) for metric in METRICS
            },
            "reference_conformers_per_molecule": _count_summary(
                [int(row["reference_conformer_count_used"]) for row in group]
            ),
            "generated_conformers_per_molecule": _count_summary(
                [int(row["generated_conformer_count"]) for row in group]
            ),
        }
    return output


def _hybrid_vs_harmonic(aggregate: dict[str, Any], comparison_keys: tuple[str, ...]) -> dict[str, Any]:
    buckets: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in aggregate.values():
        key = tuple(row.get(name) for name in comparison_keys)
        if row.get("method") in {"hybrid_harmonic_stiffness_v1_lambda500", "harmonic_prior"}:
            buckets[key][str(row["method"])] = row
    output = {}
    for key, methods in sorted(buckets.items(), key=lambda item: tuple(str(part) for part in item[0])):
        name = "/".join(str(part) for part in key) if key else "all"
        hybrid = methods.get("hybrid_harmonic_stiffness_v1_lambda500")
        harmonic = methods.get("harmonic_prior")
        if hybrid is None or harmonic is None:
            output[name] = {"available": False, **{comparison_keys[index]: key[index] for index in range(len(key))}}
            continue
        metric_outcomes = {}
        counts = {"better": 0, "tied": 0, "worse": 0, "missing": 0}
        for metric in METRICS:
            candidate = hybrid["metric_summary"][metric]["mean"]
            reference = harmonic["metric_summary"][metric]["mean"]
            outcome = _metric_outcome(metric, candidate, reference)
            counts[outcome] += 1
            metric_outcomes[metric] = {
                "hybrid_lambda500": candidate,
                "harmonic_prior": reference,
                "outcome": outcome,
            }
        output[name] = {
            "available": True,
            **{comparison_keys[index]: key[index] for index in range(len(key))},
            "metric_outcomes": metric_outcomes,
            "counts": counts,
            "hybrid_beats_harmonic_on_all_amr_cov_metrics": counts["better"] == len(METRICS),
            "hybrid_improves_amr_but_worsens_cov_r": (
                metric_outcomes["amr_r"]["outcome"] == "better"
                and metric_outcomes["amr_p"]["outcome"] == "better"
                and metric_outcomes["cov_r"]["outcome"] == "worse"
            ),
            "hybrid_improves_cov_p_but_worsens_cov_r": (
                metric_outcomes["cov_p"]["outcome"] == "better"
                and metric_outcomes["cov_r"]["outcome"] == "worse"
            ),
        }
    return output


def _rankings(aggregate: dict[str, Any], grouping_keys: tuple[str, ...]) -> dict[str, Any]:
    buckets: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in aggregate.values():
        key = tuple(row.get(name) for name in grouping_keys)
        buckets[key][str(row["method"])] = row
    output = {}
    for key, methods in sorted(buckets.items(), key=lambda item: tuple(str(part) for part in item[0])):
        name = "/".join(str(part) for part in key) if key else "all"
        metric_rankings = {}
        for metric in METRICS:
            values = []
            for method in METHODS:
                row = methods.get(method)
                if row is not None:
                    values.append((method, row["metric_summary"][metric]["mean"]))
            reverse = metric in HIGHER_IS_BETTER
            metric_rankings[metric] = [
                {"method": method, "mean": value, "rank": index + 1}
                for index, (method, value) in enumerate(sorted(values, key=lambda item: item[1], reverse=reverse))
            ]
        output[name] = {
            **{grouping_keys[index]: key[index] for index in range(len(key))},
            "metric_rankings": metric_rankings,
        }
    return output


def _decision_support(comparisons: dict[str, Any], complete: bool) -> dict[str, Any]:
    if not complete:
        return {
            "available": False,
            "reason": "chunk_set_incomplete",
            "supports_physical_benchmark_validation": False,
        }
    by_split = comparisons["by_split"]
    split_results = {split: by_split.get(split) for split in ("random", "scaffold")}
    if any(row is None or not row.get("available") for row in split_results.values()):
        return {
            "available": False,
            "reason": "missing_random_or_scaffold_comparison",
            "supports_physical_benchmark_validation": False,
        }
    hybrid_beats_all = all(
        row["hybrid_beats_harmonic_on_all_amr_cov_metrics"] for row in split_results.values() if row is not None
    )
    improves_amr_worsens_cov_r = any(
        row["hybrid_improves_amr_but_worsens_cov_r"] for row in split_results.values() if row is not None
    )
    improves_cov_p_worsens_cov_r = any(
        row["hybrid_improves_cov_p_but_worsens_cov_r"] for row in split_results.values() if row is not None
    )
    harmonic_better_any = any(
        outcome["outcome"] == "worse"
        for row in split_results.values()
        if row is not None
        for outcome in row["metric_outcomes"].values()
    )
    if hybrid_beats_all:
        conclusion = "hybrid_improves_amr_and_cov_over_harmonic_on_random_and_scaffold"
    elif improves_amr_worsens_cov_r:
        conclusion = "hybrid_amr_gain_with_possible_diversity_loss_or_mode_collapse"
    elif improves_cov_p_worsens_cov_r:
        conclusion = "hybrid_precision_recall_tradeoff"
    elif harmonic_better_any:
        conclusion = "harmonic_remains_better_on_at_least_one_amr_cov_metric"
    else:
        conclusion = "mixed_or_tied_full_amr_cov_result"
    return {
        "available": True,
        "conclusion": conclusion,
        "supports_physical_benchmark_validation": hybrid_beats_all,
        "hybrid_improves_amr_but_worsens_cov_r": improves_amr_worsens_cov_r,
        "hybrid_improves_cov_p_but_worsens_cov_r": improves_cov_p_worsens_cov_r,
        "harmonic_better_on_any_metric": harmonic_better_any,
    }


def _metric_outcome(metric: str, candidate: float | None, reference: float | None) -> str:
    if candidate is None or reference is None:
        return "missing"
    if metric in HIGHER_IS_BETTER:
        if candidate > reference + TIE_TOLERANCE:
            return "better"
        if candidate < reference - TIE_TOLERANCE:
            return "worse"
        return "tied"
    if candidate < reference - TIE_TOLERANCE:
        return "better"
    if candidate > reference + TIE_TOLERANCE:
        return "worse"
    return "tied"


def _numeric_summary(values: list[float]) -> dict[str, float | None | int]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": min(values),
        "max": max(values),
    }


def _count_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": min(values),
        "max": max(values),
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Final Optimized Full Val/Test AMR/COV Summary",
        "",
        "Merged full val/test AMR/COV chunk outputs from existing optimized checkpoints. This is not a physical benchmark evaluation.",
        "",
        "## Completeness",
        "",
        f"- Completed tasks: `{summary['coverage']['completed_task_count']}` / `{summary['coverage']['expected_task_count']}`",
        f"- Evaluated records: `{summary['coverage']['evaluated_method_seed_molecule_records']}` / `{summary['coverage']['planned_method_seed_molecule_records']}`",
        f"- Skipped records: `{summary['coverage']['skipped_record_count']}`",
        f"- Failed records: `{summary['coverage']['failed_record_count']}`",
        f"- Complete: `{summary['coverage']['complete']}`",
        "",
        "## Hybrid Lambda500 vs Harmonic",
        "",
        "| Split | COV-R | COV-P | AMR-R | AMR-P | Better | Tied | Worse |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for split_type in ("random", "scaffold"):
        row = summary["hybrid_lambda500_vs_harmonic"]["by_split"].get(split_type, {"available": False})
        if not row.get("available"):
            lines.append(f"| {split_type} | NA | NA | NA | NA | 0 | 0 | 0 |")
            continue
        outcomes = row["metric_outcomes"]
        counts = row["counts"]
        lines.append(
            f"| {split_type} | {outcomes['cov_r']['outcome']} | {outcomes['cov_p']['outcome']} | "
            f"{outcomes['amr_r']['outcome']} | {outcomes['amr_p']['outcome']} | "
            f"{counts['better']} | {counts['tied']} | {counts['worse']} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            "| Split | Partition | Method | Records | COV-R mean | COV-P mean | AMR-R mean | AMR-P mean |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split_type in summary["splits"]:
        for partition in summary["partitions"]:
            for method in METHODS:
                key = f"{split_type}/{partition}/{method}"
                row = summary["aggregate_by_split_partition_method"].get(key)
                if row is None:
                    lines.append(f"| {split_type} | {partition} | {method} | 0 | NA | NA | NA | NA |")
                    continue
                metrics = row["metric_summary"]
                lines.append(
                    f"| {split_type} | {partition} | {method} | {row['record_count']} | "
                    f"{_fmt(metrics['cov_r']['mean'])} | {_fmt(metrics['cov_p']['mean'])} | "
                    f"{_fmt(metrics['amr_r']['mean'])} | {_fmt(metrics['amr_p']['mean'])} |"
                )

    decision = summary["decision_support"]
    lines.extend(
        [
            "",
            "## Decision Support",
            "",
            f"- Available: `{decision['available']}`",
            f"- Conclusion: `{decision.get('conclusion') or decision.get('reason')}`",
            f"- Supports physical benchmark validation: `{decision['supports_physical_benchmark_validation']}`",
            "",
            "## Commands",
            "",
            "Rerun the merge after all standalone chunk jobs finish:",
            "",
            "```bash",
            "python scripts/merge_geom_full_database_final_optimized_amr_cov_full.py "
            "--plan-path experiments/results/geom_full_database_final_optimized_amr_cov_full_plan.json "
            "--summary-path experiments/results/geom_full_database_final_optimized_amr_cov_full_summary.json "
            "--report-path reports/tables/geom_full_database_final_optimized_amr_cov_full_summary.md "
            "--require-complete",
            "```",
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


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.8g}"


if __name__ == "__main__":
    main()
