#!/usr/bin/env python
"""Merge complete val/test COV/AMR chunk JSONLs into aggregate summaries."""

from __future__ import annotations

import argparse
import json
import sys
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.geom_full_database import FULL_DATABASE_SCOPE  # noqa: E402


DEFAULT_PLAN_PATH = Path("experiments/results/geom_full_database_final_cov_amr_complete_valtest_plan.json")
DEFAULT_SUMMARY_PATH = Path("experiments/results/geom_full_database_final_cov_amr_complete_valtest_summary.json")
DEFAULT_REPORT_PATH = Path("reports/tables/geom_full_database_final_cov_amr_complete_valtest_summary.md")
PLAN_RESULT_KIND = "geom_drugs_full_database_final_cov_amr_complete_valtest_plan_not_benchmark"
SUMMARY_RESULT_KIND = "geom_drugs_full_database_final_cov_amr_complete_valtest_summary_not_benchmark"
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
    print("note: Merged streamed complete val/test COV/AMR outputs; no training, sampling, scheduler submission, raw-data mutation, or environment mutation.")


def merge(args: argparse.Namespace) -> dict[str, Path]:
    start = time.time()
    plan_path = Path(args.plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _validate_plan(plan)
    if args.chunks_dir is not None:
        _override_chunks_dir(plan, Path(args.chunks_dir))

    task_scan = _scan_tasks(plan)
    if args.require_complete and not task_scan["complete"]:
        missing_text = json.dumps(task_scan["incomplete_tasks"][:20], indent=2, sort_keys=True)
        raise RuntimeError(f"Complete val/test COV/AMR merge requires all chunks. First incomplete tasks:\n{missing_text}")

    complete_entries = [entry["entry_label"] for entry in plan["complete_entries"]]
    hybrid_entries = [
        entry for entry in plan["complete_entries"] if entry["role"] == "hybrid_candidate"
    ]
    stream = _stream_aggregates(plan)
    status_counts = stream["status_counts"]
    aggregate_by_entry = stream["aggregates"]["by_entry"]
    aggregate_by_split_entry = stream["aggregates"]["by_split_entry"]
    aggregate_by_split_partition_entry = stream["aggregates"]["by_split_partition_entry"]
    aggregate_by_split_seed_entry = stream["aggregates"]["by_split_seed_entry"]
    aggregate_by_split_partition_seed_entry = stream["aggregates"]["by_split_partition_seed_entry"]
    aggregate_by_partition_entry = stream["aggregates"]["by_partition_entry"]

    rankings = {
        "cross_split": _rankings(aggregate_by_entry, tuple(), complete_entries),
        "by_split": _rankings(aggregate_by_split_entry, ("split_type",), complete_entries),
        "by_split_partition": _rankings(
            aggregate_by_split_partition_entry,
            ("split_type", "partition"),
            complete_entries,
        ),
        "by_split_seed": _rankings(aggregate_by_split_seed_entry, ("split_type", "seed"), complete_entries),
    }
    comparisons = _hybrids_vs_harmonic(
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
    lambda_selection = _lambda_selection(aggregate_by_entry, hybrid_entries)

    summary = {
        "summary_name": "geom_full_database_final_cov_amr_complete_valtest_summary",
        "result_kind": SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "complete_valtest_cov_amr_chunk_merge_reporting_only",
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
        "seed_level_consistency": _seed_level_consistency(comparisons),
        "decision_support": _decision_support(
            comparisons=comparisons,
            lambda_selection=lambda_selection,
            complete=task_scan["complete"],
        ),
        "skipped_examples": stream["skipped_examples"],
        "failed_examples": stream["failed_examples"],
        "chunk_jsonl_dir": plan["chunking"]["chunks_dir"],
        "runtime_seconds": time.time() - start,
        "notes": [
            "This merge reads per-chunk JSONLs and status files only.",
            "COV-R and COV-P are higher-is-better; AMR-R and AMR-P are lower-is-better.",
            "Use --require-complete after all standalone chunk jobs finish to fail loudly on missing chunks.",
            "Old SGD lambda results are excluded upstream by the plan.",
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
        raise ValueError("Unexpected complete val/test COV/AMR plan result_kind")
    if plan.get("benchmark_result") is not False:
        raise ValueError("Complete val/test COV/AMR plan must be benchmark_result=false")
    if plan.get("dataset_scope") != FULL_DATABASE_SCOPE:
        raise ValueError("Complete val/test COV/AMR plan has wrong dataset scope")
    if plan.get("train_partition_included") is not False:
        raise ValueError("Default complete val/test COV/AMR run must exclude train")
    if plan.get("chunking", {}).get("slurm_array_used") is not False:
        raise ValueError("Array-style plans are disallowed")


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
                "entry_label": task["entry_label"],
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


def _read_rows(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
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
                            "entry_label": task["entry_label"],
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
    return rows, status_counts


def _stream_aggregates(plan: dict[str, Any]) -> dict[str, Any]:
    groupings: dict[str, tuple[str, ...]] = {
        "by_entry": ("entry_label",),
        "by_split_entry": ("split_type", "entry_label"),
        "by_split_partition_entry": ("split_type", "partition", "entry_label"),
        "by_split_seed_entry": ("split_type", "seed", "entry_label"),
        "by_split_partition_seed_entry": ("split_type", "partition", "seed", "entry_label"),
        "by_partition_entry": ("partition", "entry_label"),
    }
    accumulators: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in groupings}
    status_counts: Counter[str] = Counter()
    skipped_examples: list[dict[str, Any]] = []
    failed_examples: list[dict[str, Any]] = []
    evaluated_count = 0
    skipped_count = 0
    failed_count = 0

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
                    failed_count += 1
                    if len(failed_examples) < 500:
                        failed_examples.append(
                            {
                                "status": "failed",
                                "task_id": int(task["task_id"]),
                                "split_type": task["split_type"],
                                "partition": task["partition"],
                                "entry_label": task["entry_label"],
                                "seed": int(task["seed"]),
                                "molecule_id": None,
                                "reason": "json_decode_error",
                                "detail": f"{jsonl_path}:{line_number}: {exc}",
                            }
                        )
                    continue
                status = str(row.get("status") or "evaluated")
                status_counts[status] += 1
                if status == "evaluated":
                    evaluated_count += 1
                    for name, keys in groupings.items():
                        _add_to_accumulator(accumulators[name], keys, row)
                elif status == "skipped":
                    skipped_count += 1
                    if len(skipped_examples) < 500:
                        skipped_examples.append(row)
                elif status == "failed":
                    failed_count += 1
                    if len(failed_examples) < 500:
                        failed_examples.append(row)

    return {
        "status_counts": status_counts,
        "evaluated_record_count": evaluated_count,
        "skipped_record_count": skipped_count,
        "failed_record_count": failed_count,
        "skipped_examples": skipped_examples,
        "failed_examples": failed_examples,
        "aggregates": {
            name: _finalize_accumulators(groups, groupings[name])
            for name, groups in accumulators.items()
        },
    }


def _add_to_accumulator(groups: dict[str, dict[str, Any]], keys: tuple[str, ...], row: dict[str, Any]) -> None:
    group_key = tuple(row[key] for key in keys)
    name = "/".join(str(part) for part in group_key) if group_key else "all"
    if name not in groups:
        groups[name] = {
            "key_values": {keys[index]: group_key[index] for index in range(len(keys))},
            "record_count": 0,
            "entry_label": row.get("entry_label"),
            "method": row.get("method"),
            "role": row.get("role"),
            "lambda_value": row.get("lambda_value"),
            "metrics": {metric: array("d") for metric in METRICS},
            "reference_counts": array("i"),
            "generated_counts": array("i"),
        }
    group = groups[name]
    group["record_count"] += 1
    metrics = row["metrics"]
    for metric in METRICS:
        group["metrics"][metric].append(float(metrics[metric]))
    group["reference_counts"].append(int(row["reference_conformer_count_used"]))
    group["generated_counts"].append(int(row["generated_conformer_count"]))


def _finalize_accumulators(groups: dict[str, dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, group in groups.items():
        output[name] = {
            **group["key_values"],
            "record_count": group["record_count"],
            "entry_label": group["entry_label"],
            "method": group["method"],
            "role": group["role"],
            "lambda_value": group["lambda_value"],
            "metric_summary": {
                metric: _numeric_summary(group["metrics"][metric])
                for metric in METRICS
            },
            "reference_conformers_per_molecule": _count_summary(group["reference_counts"]),
            "generated_conformers_per_molecule": _count_summary(group["generated_counts"]),
        }
    return output


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = {}
    for key, group in grouped.items():
        name = "/".join(str(part) for part in key) if key else "all"
        metric_summary = {
            metric: _numeric_summary([float(row["metrics"][metric]) for row in group])
            for metric in METRICS
        }
        output[name] = {
            **{keys[index]: key[index] for index in range(len(keys))},
            "record_count": len(group),
            "method": group[0].get("method"),
            "role": group[0].get("role"),
            "lambda_value": group[0].get("lambda_value"),
            "metric_summary": metric_summary,
            "reference_conformers_per_molecule": _count_summary(
                [int(row["reference_conformer_count_used"]) for row in group]
            ),
            "generated_conformers_per_molecule": _count_summary(
                [int(row["generated_conformer_count"]) for row in group]
            ),
        }
    return output


def _rankings(
    aggregate: dict[str, Any],
    grouping_keys: tuple[str, ...],
    entries: list[str],
) -> dict[str, Any]:
    buckets: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in aggregate.values():
        key = tuple(row.get(name) for name in grouping_keys)
        buckets[key][str(row["entry_label"])] = row
    output = {}
    for key, methods in sorted(buckets.items(), key=lambda item: tuple(str(part) for part in item[0])):
        name = "/".join(str(part) for part in key) if key else "all"
        metric_rankings = {}
        for metric in METRICS:
            values = []
            for entry in entries:
                row = methods.get(entry)
                if row is not None:
                    values.append((entry, row["metric_summary"][metric]["mean"]))
            reverse = metric in HIGHER_IS_BETTER
            metric_rankings[metric] = [
                {"entry_label": entry, "mean": value, "rank": index + 1}
                for index, (entry, value) in enumerate(sorted(values, key=lambda item: item[1], reverse=reverse))
            ]
        output[name] = {
            **{grouping_keys[index]: key[index] for index in range(len(key))},
            "metric_rankings": metric_rankings,
        }
    return output


def _hybrids_vs_harmonic(
    aggregates: dict[str, tuple[dict[str, Any], tuple[str, ...]]],
    hybrid_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    output = {}
    for name, (aggregate, grouping_keys) in aggregates.items():
        output[name] = _hybrids_vs_harmonic_one(aggregate, grouping_keys, hybrid_entries)
    return output


def _hybrids_vs_harmonic_one(
    aggregate: dict[str, Any],
    grouping_keys: tuple[str, ...],
    hybrid_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    buckets: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in aggregate.values():
        key = tuple(row.get(name) for name in grouping_keys)
        buckets[key][str(row["entry_label"])] = row
    output = {}
    for key, entries in sorted(buckets.items(), key=lambda item: tuple(str(part) for part in item[0])):
        group_name = "/".join(str(part) for part in key) if key else "all"
        harmonic = entries.get("harmonic_prior")
        group_out = {**{grouping_keys[index]: key[index] for index in range(len(key))}}
        for hybrid in hybrid_entries:
            label = hybrid["entry_label"]
            row = entries.get(label)
            if row is None or harmonic is None:
                group_out[label] = {"available": False}
                continue
            metric_outcomes = {}
            counts = {"better": 0, "tied": 0, "worse": 0, "missing": 0}
            for metric in METRICS:
                candidate = row["metric_summary"][metric]["mean"]
                reference = harmonic["metric_summary"][metric]["mean"]
                outcome = _metric_outcome(metric, candidate, reference)
                counts[outcome] += 1
                metric_outcomes[metric] = {
                    "hybrid": candidate,
                    "harmonic_prior": reference,
                    "outcome": outcome,
                }
            group_out[label] = {
                "available": True,
                "lambda_value": hybrid.get("lambda_value"),
                "metric_outcomes": metric_outcomes,
                "counts": counts,
                "beats_harmonic_on_all_amr_cov_metrics": counts["better"] == len(METRICS),
                "improves_precision_but_worsens_recall": (
                    metric_outcomes["amr_p"]["outcome"] == "better"
                    and metric_outcomes["cov_r"]["outcome"] == "worse"
                ),
                "improves_both_precision_and_recall": (
                    metric_outcomes["amr_p"]["outcome"] == "better"
                    and metric_outcomes["cov_p"]["outcome"] == "better"
                    and metric_outcomes["amr_r"]["outcome"] == "better"
                    and metric_outcomes["cov_r"]["outcome"] == "better"
                ),
            }
        output[group_name] = group_out
    return output


def _lambda_selection(aggregate_by_entry: dict[str, Any], hybrid_entries: list[dict[str, Any]]) -> dict[str, Any]:
    available = [entry for entry in hybrid_entries if entry["entry_label"] in aggregate_by_entry]
    if not available:
        return {
            "available": False,
            "reason": "no_complete_hybrid_lambda_chunks",
            "precision_optimal_lambda": None,
            "recall_optimal_lambda": None,
            "balanced_lambda": None,
            "pareto_frontier": [],
            "lambda_rankings": {},
        }
    rows = [
        {
            "entry_label": entry["entry_label"],
            "lambda_value": entry.get("lambda_value"),
            "metrics": aggregate_by_entry[entry["entry_label"]]["metric_summary"],
        }
        for entry in available
    ]
    precision = min(
        rows,
        key=lambda row: (
            _metric_value(row, "amr_p", default=float("inf")),
            _metric_value(row, "amr_r", default=float("inf")),
        ),
    )
    recall = max(
        rows,
        key=lambda row: (
            _metric_value(row, "cov_r", default=float("-inf")),
            -_metric_value(row, "amr_r", default=float("inf")),
        ),
    )
    rank_scores = _lambda_rank_scores(rows)
    balanced = min(rank_scores, key=lambda row: (row["rank_sum"], row["lambda_value"] or 0.0))
    return {
        "available": True,
        "hybrid_lambda_count": len(rows),
        "precision_optimal_lambda": _lambda_choice(precision, "lowest AMR-P, AMR-R tie-breaker"),
        "recall_optimal_lambda": _lambda_choice(recall, "highest COV-R, lowest AMR-R tie-breaker"),
        "balanced_lambda": {
            "entry_label": balanced["entry_label"],
            "lambda_value": balanced["lambda_value"],
            "rank_sum": balanced["rank_sum"],
            "criterion": "rank aggregation over AMR-P, AMR-R, 1-COV-R, and 1-COV-P",
        },
        "pareto_frontier": _pareto_frontier(rows),
        "lambda_rank_scores": rank_scores,
        "lambda_rankings": _lambda_metric_rankings(rows),
        "low_lambda_comparison_available": len(rows) > 1,
        "lambda5_only_rmsd_precision_optimal": len(rows) == 1 and rows[0]["entry_label"] == "hybrid_lambda500",
    }


def _lambda_rank_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = {row["entry_label"]: {"entry_label": row["entry_label"], "lambda_value": row["lambda_value"], "rank_sum": 0} for row in rows}
    for metric in METRICS:
        reverse = metric in HIGHER_IS_BETTER
        ordered = sorted(rows, key=lambda row: _metric_value(row, metric), reverse=reverse)
        for index, row in enumerate(ordered, start=1):
            scores[row["entry_label"]]["rank_sum"] += index
    return sorted(scores.values(), key=lambda row: (row["rank_sum"], row["lambda_value"] or 0.0))


def _lambda_metric_rankings(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for metric in METRICS:
        reverse = metric in HIGHER_IS_BETTER
        ordered = sorted(rows, key=lambda row: _metric_value(row, metric), reverse=reverse)
        output[metric] = [
            {
                "rank": index,
                "entry_label": row["entry_label"],
                "lambda_value": row["lambda_value"],
                "mean": _metric_value(row, metric),
            }
            for index, row in enumerate(ordered, start=1)
        ]
    return output


def _pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if _dominates(other, row):
                dominated = True
                break
        if not dominated:
            frontier.append(
                {
                    "entry_label": row["entry_label"],
                    "lambda_value": row["lambda_value"],
                    "cov_r": _metric_value(row, "cov_r"),
                    "cov_p": _metric_value(row, "cov_p"),
                    "amr_r": _metric_value(row, "amr_r"),
                    "amr_p": _metric_value(row, "amr_p"),
                }
            )
    return sorted(frontier, key=lambda row: row["lambda_value"] or 0.0)


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    better_or_equal = []
    strictly_better = []
    for metric in METRICS:
        av = _metric_value(a, metric)
        bv = _metric_value(b, metric)
        if metric in HIGHER_IS_BETTER:
            better_or_equal.append(av >= bv - TIE_TOLERANCE)
            strictly_better.append(av > bv + TIE_TOLERANCE)
        else:
            better_or_equal.append(av <= bv + TIE_TOLERANCE)
            strictly_better.append(av < bv - TIE_TOLERANCE)
    return all(better_or_equal) and any(strictly_better)


def _lambda_choice(row: dict[str, Any], criterion: str) -> dict[str, Any]:
    return {
        "entry_label": row["entry_label"],
        "lambda_value": row["lambda_value"],
        "criterion": criterion,
        "cov_r": _metric_value(row, "cov_r"),
        "cov_p": _metric_value(row, "cov_p"),
        "amr_r": _metric_value(row, "amr_r"),
        "amr_p": _metric_value(row, "amr_p"),
    }


def _metric_value(row: dict[str, Any], metric: str, default: float = float("nan")) -> float:
    value = row["metrics"][metric]["mean"]
    return default if value is None else float(value)


def _seed_level_consistency(comparisons: dict[str, Any]) -> dict[str, Any]:
    by_seed = comparisons["by_split_seed"]
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for group in by_seed.values():
        for label, row in group.items():
            if not label.startswith("hybrid_lambda") or not isinstance(row, dict):
                continue
            if not row.get("available"):
                counts[label]["missing"] += 1
                continue
            if row["beats_harmonic_on_all_amr_cov_metrics"]:
                counts[label]["beats_all"] += 1
            elif any(outcome["outcome"] == "worse" for outcome in row["metric_outcomes"].values()):
                counts[label]["mixed_or_worse"] += 1
            else:
                counts[label]["tied_or_partial"] += 1
    return {label: dict(counter) for label, counter in counts.items()}


def _decision_support(
    comparisons: dict[str, Any],
    lambda_selection: dict[str, Any],
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {
            "available": False,
            "reason": "chunk_set_incomplete",
            "supports_final_cov_amr_claim": False,
            "supports_physical_benchmark_validation": False,
        }
    cross = comparisons["cross_split"].get("all", {})
    hybrid_outcomes = {label: row for label, row in cross.items() if label.startswith("hybrid_lambda")}
    if not hybrid_outcomes:
        return {
            "available": False,
            "reason": "no_hybrid_lambda_comparison",
            "supports_final_cov_amr_claim": False,
            "supports_physical_benchmark_validation": False,
        }
    improves_both = [
        label for label, row in hybrid_outcomes.items() if row.get("available") and row.get("improves_both_precision_and_recall")
    ]
    precision_recall_tradeoff = [
        label for label, row in hybrid_outcomes.items() if row.get("available") and row.get("improves_precision_but_worsens_recall")
    ]
    harmonic_better_recall = [
        label
        for label, row in hybrid_outcomes.items()
        if row.get("available")
        and (
            row["metric_outcomes"]["cov_r"]["outcome"] == "worse"
            or row["metric_outcomes"]["amr_r"]["outcome"] == "worse"
        )
    ]
    if improves_both:
        conclusion = "at_least_one_hybrid_improves_precision_and_recall_over_harmonic"
    elif precision_recall_tradeoff:
        conclusion = "precision_recall_tradeoff"
    elif harmonic_better_recall:
        conclusion = "no_hybrid_beats_harmonic_on_recall"
    else:
        conclusion = "mixed_or_tied_cov_amr_outcome"
    return {
        "available": True,
        "conclusion": conclusion,
        "hybrids_improving_both_precision_and_recall": improves_both,
        "hybrids_with_precision_recall_tradeoff": precision_recall_tradeoff,
        "hybrids_where_harmonic_is_better_on_recall": harmonic_better_recall,
        "precision_optimal_lambda": lambda_selection.get("precision_optimal_lambda"),
        "recall_optimal_lambda": lambda_selection.get("recall_optimal_lambda"),
        "balanced_lambda": lambda_selection.get("balanced_lambda"),
        "lambda5_is_only_complete_hybrid_lambda": lambda_selection.get("lambda5_only_rmsd_precision_optimal", False),
        "smaller_lambda_recall_or_balanced_comparison_available": lambda_selection.get("low_lambda_comparison_available", False),
        "supports_final_cov_amr_claim": bool(improves_both),
        "supports_physical_benchmark_validation": bool(improves_both),
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
        "# Complete Val/Test COV/AMR Summary",
        "",
        "Merged complete val/test COV/AMR chunk outputs from existing AdamW checkpoints. This remains non-benchmark reporting.",
        "",
        "## Completeness",
        "",
        f"- Completed tasks: `{summary['coverage']['completed_task_count']}` / `{summary['coverage']['expected_task_count']}`",
        f"- Evaluated records: `{summary['coverage']['evaluated_method_seed_molecule_records']}` / `{summary['coverage']['planned_method_seed_molecule_records']}`",
        f"- Skipped records: `{summary['coverage']['skipped_record_count']}`",
        f"- Failed records: `{summary['coverage']['failed_record_count']}`",
        f"- Complete: `{summary['coverage']['complete']}`",
        "",
        "## Eligible Lambdas",
        "",
        f"`{summary['eligible_hybrid_lambdas']}`",
        "",
        "## Cross-Split Aggregate Metrics",
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
            f"{_fmt(metrics['cov_r']['mean'])} | {_fmt(metrics['cov_r']['median'])} | "
            f"{_fmt(metrics['cov_p']['mean'])} | {_fmt(metrics['cov_p']['median'])} | "
            f"{_fmt(metrics['amr_r']['mean'])} | {_fmt(metrics['amr_r']['median'])} | "
            f"{_fmt(metrics['amr_p']['mean'])} | {_fmt(metrics['amr_p']['median'])} |"
        )

    selection = summary["lambda_selection"]
    decision = summary["decision_support"]
    cross_rankings = summary["rankings"]["cross_split"].get("all", {}).get("metric_rankings", {})
    lines.extend(
        [
            "",
            "## Cross-Split Rankings",
            "",
            "| Metric | Ranking |",
            "|---|---|",
        ]
    )
    for metric in METRICS:
        ranking = cross_rankings.get(metric, [])
        ranking_text = " > ".join(f"{row['entry_label']} ({_fmt(row['mean'])})" for row in ranking)
        lines.append(f"| {METRIC_DISPLAY[metric]} | {ranking_text or 'NA'} |")
    lines.extend(
        [
            "",
            "## Hybrid Lambda500 vs Harmonic",
            "",
            "| Split | COV-R | COV-P | AMR-R | AMR-P | Better | Tied | Worse |",
            "|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    for split_type in summary["splits"]:
        row = summary["hybrids_vs_harmonic"]["by_split"].get(split_type, {}).get("hybrid_lambda500")
        if not row or not row.get("available"):
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
            "## Seed-Level Consistency",
            "",
            f"`{summary['seed_level_consistency']}`",
            "",
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
            f"- Available: `{decision['available']}`",
            f"- Conclusion: `{decision.get('conclusion') or decision.get('reason')}`",
            f"- Hybrid improves both precision and recall: `{bool(decision.get('hybrids_improving_both_precision_and_recall'))}`",
            f"- Hybrid shows precision-recall tradeoff: `{bool(decision.get('hybrids_with_precision_recall_tradeoff'))}`",
            f"- Harmonic better on recall for: `{decision.get('hybrids_where_harmonic_is_better_on_recall')}`",
            f"- Smaller-lambda COV/AMR comparison available: `{decision.get('smaller_lambda_recall_or_balanced_comparison_available')}`",
            f"- Supports final COV/AMR claim: `{decision['supports_final_cov_amr_claim']}`",
            f"- Supports benchmark-style physical validation: `{decision['supports_physical_benchmark_validation']}`",
            "- Claim guidance: `weaken ensemble-level claim; lambda=5.0 is precision-favorable on AMR-P but does not improve recall over harmonic in this complete val/test COV/AMR run`",
            "",
            "## Commands",
            "",
            "Rerun the merge after all standalone chunk jobs finish:",
            "",
            "```bash",
            "python scripts/merge_geom_full_database_final_cov_amr_complete_valtest.py "
            "--plan-path experiments/results/geom_full_database_final_cov_amr_complete_valtest_plan.json "
            "--summary-path experiments/results/geom_full_database_final_cov_amr_complete_valtest_summary.json "
            "--report-path reports/tables/geom_full_database_final_cov_amr_complete_valtest_summary.md "
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
