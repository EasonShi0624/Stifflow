"""Reporting utilities for fixed lambda-5 hybrid harmonic/stiffness multiseed follow-up."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.geom_full_database import FULL_DATABASE_SCOPE
from src.eval.hybrid_harmonic_stiffness_pilot import (
    BASELINE_LABELS,
    BASELINE_MODES,
    CORE_LOWER_IS_BETTER_METRICS,
    _best_hybrid_by_core_rank,
    _dataset_summary,
    _fmt,
    _fmt_int,
    _group_rows,
    _hybrid_comparison,
    _load_rows,
)
from src.train.formal_10000_training import (
    FORMAL_10000_HARD_MAX_STEPS,
    FORMAL_10000_LEARNING_RATE,
    FORMAL_10000_MAX_STEPS,
    FORMAL_10000_MIN_DELTA,
    FORMAL_10000_PATIENCE,
    FORMAL_10000_SAMPLE_STEP_SIZE,
)


HYBRID_LAMBDA500_SUMMARY_RESULT_KIND = (
    "geom_drugs_full_database_hybrid_harmonic_stiffness_lambda500_multiseed_summary_not_benchmark"
)
HYBRID_LAMBDA500_MASTER_RESULT_KIND = (
    "geom_drugs_full_database_hybrid_harmonic_stiffness_lambda500_cross_split_master_summary_not_benchmark"
)
HYBRID_MODE = "hybrid_harmonic_stiffness_v1"
HYBRID_STRENGTH = 5.0
HYBRID_LABEL = "hybrid_harmonic_stiffness_v1_lambda500"
SEEDS = (0, 1, 2)
SPLITS = ("random", "scaffold")
METHOD_ORDER = (HYBRID_LABEL, *BASELINE_MODES)

DEFAULT_RANDOM_SUMMARY_PATH = Path(
    "experiments/results/geom_full_database_hybrid_harmonic_stiffness_lambda500_random_multiseed_summary.json"
)
DEFAULT_RANDOM_REPORT_PATH = Path(
    "reports/tables/geom_full_database_hybrid_harmonic_stiffness_lambda500_random_multiseed_summary.md"
)
DEFAULT_SCAFFOLD_SUMMARY_PATH = Path(
    "experiments/results/geom_full_database_hybrid_harmonic_stiffness_lambda500_scaffold_multiseed_summary.json"
)
DEFAULT_SCAFFOLD_REPORT_PATH = Path(
    "reports/tables/geom_full_database_hybrid_harmonic_stiffness_lambda500_scaffold_multiseed_summary.md"
)
DEFAULT_MASTER_SUMMARY_PATH = Path(
    "experiments/results/geom_full_database_hybrid_harmonic_stiffness_lambda500_master_summary.json"
)
DEFAULT_MASTER_REPORT_PATH = Path(
    "reports/tables/geom_full_database_hybrid_harmonic_stiffness_lambda500_master_summary.md"
)


def write_hybrid_lambda500_multiseed_summaries(
    results_root: str | Path = "experiments/results",
    random_summary_path: str | Path = DEFAULT_RANDOM_SUMMARY_PATH,
    random_report_path: str | Path = DEFAULT_RANDOM_REPORT_PATH,
    scaffold_summary_path: str | Path = DEFAULT_SCAFFOLD_SUMMARY_PATH,
    scaffold_report_path: str | Path = DEFAULT_SCAFFOLD_REPORT_PATH,
    master_summary_path: str | Path = DEFAULT_MASTER_SUMMARY_PATH,
    master_report_path: str | Path = DEFAULT_MASTER_REPORT_PATH,
    require_hybrid_results: bool = False,
) -> dict[str, Path]:
    results_root = Path(results_root)
    split_summaries = {
        split_type: _build_split_summary(
            split_type=split_type,
            results_root=results_root,
            require_hybrid_results=require_hybrid_results,
        )
        for split_type in SPLITS
    }
    master_summary = _build_master_summary(split_summaries)
    outputs = {
        "random_summary": Path(random_summary_path),
        "random_report": Path(random_report_path),
        "scaffold_summary": Path(scaffold_summary_path),
        "scaffold_report": Path(scaffold_report_path),
        "master_summary": Path(master_summary_path),
        "master_report": Path(master_report_path),
    }
    _write_json(outputs["random_summary"], split_summaries["random"])
    _write_text(outputs["random_report"], _split_markdown_report(split_summaries["random"]))
    _write_json(outputs["scaffold_summary"], split_summaries["scaffold"])
    _write_text(outputs["scaffold_report"], _split_markdown_report(split_summaries["scaffold"]))
    _write_json(outputs["master_summary"], master_summary)
    _write_text(outputs["master_report"], _master_markdown_report(master_summary))
    return outputs


def _build_split_summary(
    split_type: str,
    results_root: Path,
    require_hybrid_results: bool,
) -> dict[str, Any]:
    seed_summaries = [
        _build_seed_summary(split_type=split_type, seed=seed, results_root=results_root)
        for seed in SEEDS
    ]
    missing_hybrid_results = [
        {"split_type": split_type, "seed": row["seed"], "expected_path": row["expected_hybrid_result_path"]}
        for row in seed_summaries
        if not row["hybrid_result_available"]
    ]
    if require_hybrid_results and missing_hybrid_results:
        raise FileNotFoundError(f"Missing expected lambda-5 hybrid results: {missing_hybrid_results}")

    comparisons = [row["hybrid_comparison"] for row in seed_summaries]
    available_comparisons = [row for row in comparisons if row["result_available"]]
    baseline_counts = {
        baseline_mode: _win_tie_loss_counts(available_comparisons, baseline_mode)
        for baseline_mode in BASELINE_MODES
    }
    aggregate_metrics = _aggregate_metrics(seed_summaries)
    qualitative_ordering = _qualitative_ordering(aggregate_metrics)
    verdict = _split_verdict(
        missing_hybrid_results=missing_hybrid_results,
        baseline_counts=baseline_counts,
        available_comparisons=available_comparisons,
    )
    return {
        "summary_name": f"geom_full_database_{split_type}_hybrid_harmonic_stiffness_lambda500_multiseed_summary",
        "result_kind": HYBRID_LAMBDA500_SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_full_database_hybrid_harmonic_stiffness_lambda500_multiseed_report",
        "hardware": "reporting_only",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "split_type": split_type,
        "seeds": list(SEEDS),
        "dataset_scope": FULL_DATABASE_SCOPE,
        "hybrid_method": {
            "corruption_mode": HYBRID_MODE,
            "source_strength": HYBRID_STRENGTH,
            "locked_for_followup": True,
        },
        "baseline_modes": list(BASELINE_MODES),
        "coverage": {
            "complete_hybrid_seed_set": not missing_hybrid_results,
            "available_hybrid_seeds": [
                row["seed"] for row in seed_summaries if row["hybrid_result_available"]
            ],
            "missing_hybrid_results": missing_hybrid_results,
            "missing_baselines": [
                {"seed": row["seed"], "missing_baselines": row["missing_baselines"]}
                for row in seed_summaries
                if row["missing_baselines"]
            ],
        },
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
        "seed_summaries": seed_summaries,
        "aggregate_metrics": aggregate_metrics,
        "win_tie_loss_counts": baseline_counts,
        "qualitative_ordering": qualitative_ordering,
        "hybrid_beats_harmonic_prior_on_all_tracked_metrics_all_available_seeds": _all_available_win(
            available_comparisons, "harmonic_prior"
        ),
        "hybrid_beats_harmonic_prior_on_all_tracked_metrics_complete_seed_set": (
            not missing_hybrid_results and _all_available_win(available_comparisons, "harmonic_prior")
        ),
        "hybrid_beats_stiffness_structured_on_all_tracked_metrics_complete_seed_set": (
            not missing_hybrid_results and _all_available_win(available_comparisons, "stiffness_structured")
        ),
        "hybrid_beats_isotropic_on_all_tracked_metrics_complete_seed_set": (
            not missing_hybrid_results and _all_available_win(available_comparisons, "isotropic")
        ),
        "hybrid_beats_torsion_only_on_all_tracked_metrics_complete_seed_set": (
            not missing_hybrid_results and _all_available_win(available_comparisons, "torsion_only")
        ),
        "over_modulation_risk_assessment": _over_modulation_risk_assessment(
            missing_hybrid_results=missing_hybrid_results,
            baseline_counts=baseline_counts,
            qualitative_ordering=qualitative_ordering,
        ),
        "directional_verdict": verdict,
        "safety_labels": {
            "benchmark_result": False,
            "non_benchmark": True,
            "lambda500_multiseed_summary": True,
            "training_executed_by_summary": False,
            "sampling_executed_by_summary": False,
            "raw_data_modified": False,
            "environment_modified": False,
            "gpu_used_by_summary": False,
        },
        "notes": [
            "Reporting-only fixed lambda-5 hybrid harmonic + stiffness multiseed summary. This is not a benchmark result.",
            "The summary compares hybrid_harmonic_stiffness_v1 lambda 5.0 against existing harmonic_prior, stiffness_structured, isotropic, and torsion_only references for the same split and seeds.",
            "Missing lambda-5 hybrid result JSONs are reported as pending rather than treated as wins, losses, or ties.",
            "No training, sampling, preprocessing, scheduler action, GPU use, physical benchmark job, xTB, raw-data mutation, or environment change is performed by this summary.",
        ],
    }


def _build_seed_summary(split_type: str, seed: int, results_root: Path) -> dict[str, Any]:
    paths = _result_paths_for_seed(split_type=split_type, seed=seed, results_root=results_root)
    rows = _load_rows(list(paths.values()), expected_split_type=split_type, expected_seed=seed)
    grouped = _group_rows(rows)
    baseline_rows = {mode: grouped.get((mode, None)) for mode in BASELINE_MODES}
    hybrid_row = grouped.get((HYBRID_MODE, HYBRID_STRENGTH))
    comparison = _hybrid_comparison(
        mode=HYBRID_MODE,
        strength=HYBRID_STRENGTH,
        hybrid_row=hybrid_row,
        baseline_rows=baseline_rows,
    )
    return {
        "seed": seed,
        "split_type": split_type,
        "input_result_paths": {key: str(path) for key, path in paths.items()},
        "input_result_paths_found": [row["source_path"] for row in rows],
        "expected_hybrid_result_path": str(paths[HYBRID_LABEL]),
        "row_count": len(rows),
        "dataset": _dataset_summary(rows),
        "baseline_rows_available": [mode for mode, row in baseline_rows.items() if row is not None],
        "missing_baselines": [mode for mode, row in baseline_rows.items() if row is None],
        "hybrid_result_available": hybrid_row is not None,
        "baseline_metric_table": _seed_baseline_metric_table(baseline_rows),
        "hybrid_comparison": comparison,
    }


def _result_paths_for_seed(split_type: str, seed: int, results_root: Path) -> dict[str, Path]:
    split_label = "" if split_type == "random" else "_scaffold"
    paths = {
        mode: results_root
        / (
            f"geom_formal_full_database{split_label}_train_{BASELINE_LABELS[mode]}"
            f"_steps{FORMAL_10000_MAX_STEPS}_seed{seed}.json"
        )
        for mode in BASELINE_MODES
    }
    paths[HYBRID_LABEL] = results_root / (
        f"geom_formal_full_database{split_label}_train_{HYBRID_LABEL}"
        f"_steps{FORMAL_10000_MAX_STEPS}_seed{seed}.json"
    )
    return paths


def _seed_baseline_metric_table(baseline_rows: dict[str, dict[str, Any] | None]) -> dict[str, dict[str, Any]]:
    table = {}
    for mode, row in baseline_rows.items():
        if row is None:
            continue
        table[mode] = _metric_snapshot(row)
    return table


def _metric_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": row["run_name"],
        "source_path": row["source_path"],
        "executed_train_steps": row["executed_train_steps"],
        "stop_reason": row["stop_reason"],
        **{metric: row[metric] for metric in CORE_LOWER_IS_BETTER_METRICS},
    }


def _win_tie_loss_counts(
    comparisons: list[dict[str, Any]],
    baseline_mode: str,
) -> dict[str, Any]:
    counts = {"better": 0, "tied": 0, "worse": 0, "missing": 0}
    seed_all_core_better = 0
    seed_all_core_better_or_tied = 0
    per_seed = []
    for comparison in comparisons:
        baseline_summary = comparison["baseline_core_summary"][baseline_mode]
        metric_rows = comparison["core_metrics"].get(baseline_mode, {})
        seed_counts = {"better": 0, "tied": 0, "worse": 0, "missing": 0}
        for metric in CORE_LOWER_IS_BETTER_METRICS:
            outcome = metric_rows.get(metric, {}).get("outcome", "missing")
            if outcome not in seed_counts:
                outcome = "missing"
            counts[outcome] += 1
            seed_counts[outcome] += 1
        if baseline_summary["all_core_better"]:
            seed_all_core_better += 1
        if baseline_summary["all_core_better_or_tied"]:
            seed_all_core_better_or_tied += 1
        per_seed.append(
            {
                "seed": _seed_from_run_name(comparison.get("run_name")),
                "all_core_better": baseline_summary["all_core_better"],
                "all_core_better_or_tied": baseline_summary["all_core_better_or_tied"],
                "counts": seed_counts,
            }
        )
    return {
        "baseline_mode": baseline_mode,
        "metric_counts_over_available_seeds": counts,
        "available_seed_count": len(comparisons),
        "required_seed_count": len(SEEDS),
        "seed_all_core_better_count": seed_all_core_better,
        "seed_all_core_better_or_tied_count": seed_all_core_better_or_tied,
        "hybrid_beats_baseline_on_all_core_metrics_all_available_seeds": (
            bool(comparisons) and seed_all_core_better == len(comparisons)
        ),
        "hybrid_beats_or_ties_baseline_on_all_core_metrics_all_available_seeds": (
            bool(comparisons) and seed_all_core_better_or_tied == len(comparisons)
        ),
        "per_seed": per_seed,
    }


def _aggregate_metrics(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[dict[str, Any]]] = {method: [] for method in METHOD_ORDER}
    for seed_summary in seed_summaries:
        for mode, row in seed_summary["baseline_metric_table"].items():
            values[mode].append({"seed": seed_summary["seed"], **row})
        comparison = seed_summary["hybrid_comparison"]
        if comparison["result_available"]:
            values[HYBRID_LABEL].append(
                {
                    "seed": seed_summary["seed"],
                    "run_name": comparison["run_name"],
                    "source_path": comparison["source_path"],
                    "executed_train_steps": comparison["executed_train_steps"],
                    "stop_reason": comparison["stop_reason"],
                    **comparison["metrics"],
                }
            )
    return {
        method: _aggregate_method_metrics(rows)
        for method, rows in values.items()
        if rows
    }


def _aggregate_method_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "available_seed_count": len(rows),
        "seeds": [row["seed"] for row in rows],
        "per_seed": rows,
        "mean": {
            metric: sum(float(row[metric]) for row in rows) / len(rows)
            for metric in CORE_LOWER_IS_BETTER_METRICS
        },
    }


def _qualitative_ordering(aggregate_metrics: dict[str, Any]) -> dict[str, Any]:
    complete_methods = {
        method: metrics
        for method, metrics in aggregate_metrics.items()
        if metrics["available_seed_count"] == len(SEEDS)
    }
    if len(complete_methods) < len(METHOD_ORDER):
        return {
            "status": "pending_complete_seed_set",
            "ordering": [],
            "missing_complete_methods": [
                method for method in METHOD_ORDER if method not in complete_methods
            ],
        }
    ranks = {method: 0.0 for method in complete_methods}
    for metric in CORE_LOWER_IS_BETTER_METRICS:
        sorted_methods = sorted(
            complete_methods,
            key=lambda method: float(complete_methods[method]["mean"][metric]),
        )
        for rank, method in enumerate(sorted_methods, start=1):
            ranks[method] += float(rank)
    ordering = sorted(
        complete_methods,
        key=lambda method: (
            ranks[method] / len(CORE_LOWER_IS_BETTER_METRICS),
            complete_methods[method]["mean"]["best_validation_loss"],
            complete_methods[method]["mean"]["sampled_rmsd_to_target_overall_mean"],
        ),
    )
    return {
        "status": "available",
        "ordering": ordering,
        "average_core_metric_rank": {
            method: ranks[method] / len(CORE_LOWER_IS_BETTER_METRICS)
            for method in ordering
        },
        "direction": "lower_is_better",
    }


def _split_verdict(
    missing_hybrid_results: list[dict[str, Any]],
    baseline_counts: dict[str, dict[str, Any]],
    available_comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    if not available_comparisons:
        verdict = "hybrid_lambda500_results_not_available"
        decision_rule = "await_missing_hybrid_lambda500_jobs"
    elif missing_hybrid_results:
        verdict = "partial_hybrid_lambda500_seed_set_available"
        decision_rule = "complete_missing_hybrid_lambda500_jobs_before_claim"
    elif baseline_counts["harmonic_prior"]["hybrid_beats_baseline_on_all_core_metrics_all_available_seeds"]:
        verdict = "hybrid_lambda500_beats_harmonic_prior_on_all_tracked_metrics_all_seeds"
        decision_rule = "eligible_for_cross_split_interpretation"
    elif baseline_counts["stiffness_structured"]["hybrid_beats_baseline_on_all_core_metrics_all_available_seeds"]:
        verdict = "hybrid_lambda500_beats_stiffness_but_not_harmonic_all_seeds"
        decision_rule = "partial_transfer_do_not_overclaim"
    else:
        verdict = "hybrid_lambda500_does_not_beat_stiffness_all_seeds"
        decision_rule = "stop_or_reassess_over_modulation"
    return {
        "verdict": verdict,
        "decision_rule": decision_rule,
        "available_hybrid_seed_count": len(available_comparisons),
        "required_seed_count": len(SEEDS),
        "missing_hybrid_results": missing_hybrid_results,
        "success_claim_supported": (
            not missing_hybrid_results
            and baseline_counts["harmonic_prior"]["hybrid_beats_baseline_on_all_core_metrics_all_available_seeds"]
        ),
    }


def _over_modulation_risk_assessment(
    missing_hybrid_results: list[dict[str, Any]],
    baseline_counts: dict[str, dict[str, Any]],
    qualitative_ordering: dict[str, Any],
) -> dict[str, Any]:
    wins_all_baselines = all(
        baseline_counts[baseline]["hybrid_beats_baseline_on_all_core_metrics_all_available_seeds"]
        for baseline in BASELINE_MODES
    )
    hybrid_best = (
        qualitative_ordering.get("status") == "available"
        and qualitative_ordering.get("ordering", [None])[0] == HYBRID_LABEL
    )
    if missing_hybrid_results:
        status = "pending_missing_fixed_hybrid_seeds"
        supported = None
    elif wins_all_baselines and hybrid_best:
        status = "not_supported_by_observed_tracked_metrics"
        supported = False
    elif wins_all_baselines:
        status = "not_supported_on_pairwise_metrics_but_ordering_mixed"
        supported = False
    else:
        status = "supported_or_inconclusive_from_observed_tracked_metrics"
        supported = True
    return {
        "status": status,
        "over_modulation_risk_supported": supported,
        "basis": "tracked core loss/RMSD metrics only; no physical benchmark jobs, PoseBusters, MMFF/UFF, or xTB.",
    }


def _build_master_summary(split_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    random_summary = split_summaries["random"]
    scaffold_summary = split_summaries["scaffold"]
    ordering_same = _ordering_same(random_summary, scaffold_summary)
    cross_split_complete = all(
        summary["coverage"]["complete_hybrid_seed_set"]
        for summary in split_summaries.values()
    )
    hybrid_beats_harmonic = all(
        summary["hybrid_beats_harmonic_prior_on_all_tracked_metrics_complete_seed_set"]
        for summary in split_summaries.values()
    )
    hybrid_beats_all_baselines = all(
        summary[f"hybrid_beats_{baseline}_on_all_tracked_metrics_complete_seed_set"]
        for summary in split_summaries.values()
        for baseline in ["harmonic_prior", "stiffness_structured", "isotropic", "torsion_only"]
    )
    over_modulation = _master_over_modulation_risk(
        split_summaries=split_summaries,
        cross_split_complete=cross_split_complete,
        hybrid_beats_all_baselines=hybrid_beats_all_baselines,
        ordering_same=ordering_same,
    )
    if not cross_split_complete:
        verdict = "pending_missing_hybrid_lambda500_multiseed_results"
        decision_rule = "run_only_the_prepared_missing_fixed_hybrid_jobs_after_approval"
    elif hybrid_beats_harmonic and hybrid_beats_all_baselines:
        verdict = "hybrid_lambda500_new_strongest_source_design_on_current_tracked_metrics"
        decision_rule = "report_new_strongest_source_design_current_tracked_metrics_non_benchmark"
    elif hybrid_beats_harmonic:
        verdict = "hybrid_lambda500_beats_harmonic_prior_across_random_and_scaffold"
        decision_rule = "fixed_lambda500_supported_against_harmonic_but_compare_other_baselines"
    else:
        verdict = "hybrid_lambda500_does_not_beat_harmonic_prior_across_splits"
        decision_rule = "do_not_claim_hybrid_success"
    return {
        "summary_name": "geom_full_database_hybrid_harmonic_stiffness_lambda500_cross_split_master_summary",
        "result_kind": HYBRID_LAMBDA500_MASTER_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_full_database_hybrid_harmonic_stiffness_lambda500_cross_split_master_report",
        "hardware": "reporting_only",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "dataset_scope": FULL_DATABASE_SCOPE,
        "hybrid_method": {
            "corruption_mode": HYBRID_MODE,
            "source_strength": HYBRID_STRENGTH,
            "locked_for_followup": True,
        },
        "splits": {split: _master_split_snapshot(summary) for split, summary in split_summaries.items()},
        "baseline_modes": list(BASELINE_MODES),
        "cross_split_complete_hybrid_seed_set": cross_split_complete,
        "hybrid_beats_harmonic_prior_on_all_tracked_metrics_both_splits": (
            cross_split_complete and hybrid_beats_harmonic
        ),
        "hybrid_beats_all_baselines_on_all_tracked_metrics_both_splits": (
            cross_split_complete and hybrid_beats_all_baselines
        ),
        "random_and_scaffold_same_qualitative_ordering": ordering_same,
        "over_modulation_risk_assessment": over_modulation,
        "directional_verdict": {
            "verdict": verdict,
            "decision_rule": decision_rule,
            "success_claim_supported": cross_split_complete and hybrid_beats_harmonic,
            "requires_user_approval_before_any_scheduler_submission": True,
        },
        "safety_labels": {
            "benchmark_result": False,
            "non_benchmark": True,
            "lambda500_cross_split_master_summary": True,
            "training_executed_by_summary": False,
            "sampling_executed_by_summary": False,
            "raw_data_modified": False,
            "environment_modified": False,
            "gpu_used_by_summary": False,
        },
        "notes": [
            "Reporting-only fixed lambda-5 cross-split master summary. This is not a benchmark result.",
            "The master summary compares hybrid_harmonic_stiffness_v1 lambda 5.0 against harmonic_prior, stiffness_structured, isotropic, and torsion_only over random and scaffold splits.",
            "No training, sampling, preprocessing, scheduler action, GPU use, physical benchmark job, xTB, raw-data mutation, or environment change is performed by this summary.",
        ],
    }


def _master_split_snapshot(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "split_type": summary["split_type"],
        "complete_hybrid_seed_set": summary["coverage"]["complete_hybrid_seed_set"],
        "available_hybrid_seeds": summary["coverage"]["available_hybrid_seeds"],
        "missing_hybrid_results": summary["coverage"]["missing_hybrid_results"],
        "qualitative_ordering": summary["qualitative_ordering"],
        "win_tie_loss_counts": summary["win_tie_loss_counts"],
        "over_modulation_risk_assessment": summary["over_modulation_risk_assessment"],
        "directional_verdict": summary["directional_verdict"],
    }


def _ordering_same(random_summary: dict[str, Any], scaffold_summary: dict[str, Any]) -> dict[str, Any]:
    random_ordering = random_summary["qualitative_ordering"]
    scaffold_ordering = scaffold_summary["qualitative_ordering"]
    if random_ordering["status"] != "available" or scaffold_ordering["status"] != "available":
        return {
            "status": "pending_complete_seed_set",
            "same_ordering": None,
            "random_ordering": random_ordering.get("ordering", []),
            "scaffold_ordering": scaffold_ordering.get("ordering", []),
        }
    return {
        "status": "available",
        "same_ordering": random_ordering["ordering"] == scaffold_ordering["ordering"],
        "random_ordering": random_ordering["ordering"],
        "scaffold_ordering": scaffold_ordering["ordering"],
    }


def _master_over_modulation_risk(
    split_summaries: dict[str, dict[str, Any]],
    cross_split_complete: bool,
    hybrid_beats_all_baselines: bool,
    ordering_same: dict[str, Any],
) -> dict[str, Any]:
    if not cross_split_complete:
        status = "pending_missing_fixed_hybrid_seeds"
        supported = None
    elif hybrid_beats_all_baselines and ordering_same.get("same_ordering") is True:
        status = "not_supported_by_observed_tracked_metrics"
        supported = False
    elif hybrid_beats_all_baselines:
        status = "not_supported_pairwise_but_cross_split_ordering_differs"
        supported = False
    else:
        status = "supported_or_inconclusive_from_observed_tracked_metrics"
        supported = True
    return {
        "status": status,
        "over_modulation_risk_supported": supported,
        "split_assessments": {
            split: summary["over_modulation_risk_assessment"]
            for split, summary in split_summaries.items()
        },
        "basis": "tracked core loss/RMSD metrics only; no physical benchmark jobs, PoseBusters, MMFF/UFF, or xTB.",
    }


def _all_available_win(comparisons: list[dict[str, Any]], baseline_mode: str) -> bool:
    return bool(comparisons) and all(
        row["baseline_core_summary"][baseline_mode]["all_core_better"]
        for row in comparisons
    )


def _seed_from_run_name(run_name: str | None) -> int | None:
    if not run_name or "_seed" not in run_name:
        return None
    suffix = run_name.rsplit("_seed", 1)[1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _split_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        f"# Full-Database {summary['split_type'].title()} Hybrid Lambda 5.0 Multiseed Summary",
        "",
        summary["notes"][0],
        "",
        "## Scope",
        "",
        f"- Split type: `{summary['split_type']}`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Hybrid: `{HYBRID_MODE}` lambda `{HYBRID_STRENGTH}`",
        f"- Complete hybrid seed set: `{summary['coverage']['complete_hybrid_seed_set']}`",
        f"- Available hybrid seeds: `{summary['coverage']['available_hybrid_seeds']}`",
        f"- Missing hybrid results: `{summary['coverage']['missing_hybrid_results']}`",
        f"- Verdict: `{summary['directional_verdict']['verdict']}`",
        f"- Decision rule: `{summary['directional_verdict']['decision_rule']}`",
        f"- Over-modulation risk: `{summary['over_modulation_risk_assessment']['status']}`",
        "",
        "## Aggregate Ordering",
        "",
        f"- Status: `{summary['qualitative_ordering']['status']}`",
        f"- Ordering: `{summary['qualitative_ordering'].get('ordering', [])}`",
        "",
        "## Aggregate Means",
        "",
        "| Method | Seeds | Best val loss | Reloaded val loss | RMSD overall | RMSD train | RMSD val | RMSD test |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        row = summary["aggregate_metrics"].get(method)
        if row is None:
            continue
        mean = row["mean"]
        lines.append(
            f"| {method} | {row['available_seed_count']} | "
            f"{_fmt(mean.get('best_validation_loss'))} | "
            f"{_fmt(mean.get('reloaded_validation_loss'))} | "
            f"{_fmt(mean.get('sampled_rmsd_to_target_overall_mean'))} | "
            f"{_fmt(mean.get('sampled_rmsd_to_target_train_mean'))} | "
            f"{_fmt(mean.get('sampled_rmsd_to_target_val_mean'))} | "
            f"{_fmt(mean.get('sampled_rmsd_to_target_test_mean'))} |"
        )
    lines.extend(
        [
            "",
        "## Hybrid Per-Seed Metrics",
        "",
        "| Seed | Status | Steps | Stop reason | Best val loss | Reloaded val loss | RMSD overall | RMSD train | RMSD val | RMSD test | Beats harmonic | Beats stiffness | Beats isotropic | Beats torsion |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for seed_summary in summary["seed_summaries"]:
        comparison = seed_summary["hybrid_comparison"]
        metrics = comparison.get("metrics", {})
        lines.append(
            f"| {seed_summary['seed']} | {comparison['status']} | "
            f"{_fmt_int(comparison.get('executed_train_steps'))} | {comparison.get('stop_reason') or 'NA'} | "
            f"{_fmt(metrics.get('best_validation_loss'))} | "
            f"{_fmt(metrics.get('reloaded_validation_loss'))} | "
            f"{_fmt(metrics.get('sampled_rmsd_to_target_overall_mean'))} | "
            f"{_fmt(metrics.get('sampled_rmsd_to_target_train_mean'))} | "
            f"{_fmt(metrics.get('sampled_rmsd_to_target_val_mean'))} | "
            f"{_fmt(metrics.get('sampled_rmsd_to_target_test_mean'))} | "
            f"{comparison['beats_harmonic_prior_on_all_core_metrics']} | "
            f"{comparison['beats_stiffness_structured_on_all_core_metrics']} | "
            f"{comparison['beats_isotropic_on_all_core_metrics']} | "
            f"{comparison['beats_torsion_only_on_all_core_metrics']} |"
        )
    lines.extend(
        [
            "",
            "## Win/Tie/Loss Counts",
            "",
            "| Baseline | Available seeds | Better | Tied | Worse | Missing metric cells | Seeds all-core better |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for baseline in BASELINE_MODES:
        counts = summary["win_tie_loss_counts"][baseline]
        metric_counts = counts["metric_counts_over_available_seeds"]
        lines.append(
            f"| {baseline} | {counts['available_seed_count']} | "
            f"{metric_counts['better']} | {metric_counts['tied']} | {metric_counts['worse']} | "
            f"{metric_counts['missing']} | {counts['seed_all_core_better_count']} |"
        )
    lines.extend(
        [
            "",
            "## Safety Labels",
            "",
            f"- `benchmark_result`: `{summary['benchmark_result']}`",
            f"- `training_executed_by_summary`: `{summary['training_executed_by_summary']}`",
            f"- `sampling_executed_by_summary`: `{summary['sampling_executed_by_summary']}`",
            f"- `raw_data_modified`: `{summary['raw_data_modified']}`",
            f"- `environment_modified`: `{summary['environment_modified']}`",
            "",
            "No training, sampling, preprocessing, scheduler action, GPU use, physical benchmark job, xTB, or environment change was performed by this summary.",
            "",
        ]
    )
    return "\n".join(lines)


def _master_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Full-Database Hybrid Lambda 5.0 Cross-Split Master Summary",
        "",
        summary["notes"][0],
        "",
        "## Scope",
        "",
        f"- Hybrid: `{HYBRID_MODE}` lambda `{HYBRID_STRENGTH}`",
        f"- Cross-split complete hybrid seed set: `{summary['cross_split_complete_hybrid_seed_set']}`",
        f"- Beats harmonic on all tracked metrics both splits: `{summary['hybrid_beats_harmonic_prior_on_all_tracked_metrics_both_splits']}`",
        f"- Beats all baselines on all tracked metrics both splits: `{summary['hybrid_beats_all_baselines_on_all_tracked_metrics_both_splits']}`",
        f"- Same qualitative ordering: `{summary['random_and_scaffold_same_qualitative_ordering']}`",
        f"- Over-modulation risk: `{summary['over_modulation_risk_assessment']['status']}`",
        f"- Verdict: `{summary['directional_verdict']['verdict']}`",
        f"- Decision rule: `{summary['directional_verdict']['decision_rule']}`",
        "",
        "## Split Coverage",
        "",
        "| Split | Complete hybrid seeds | Available hybrid seeds | Missing hybrid results | Ordering status | Ordering |",
        "|---|---|---|---|---|---|",
    ]
    for split in SPLITS:
        split_summary = summary["splits"][split]
        lines.append(
            f"| {split} | {split_summary['complete_hybrid_seed_set']} | "
            f"{split_summary['available_hybrid_seeds']} | "
            f"{split_summary['missing_hybrid_results']} | "
            f"{split_summary['qualitative_ordering']['status']} | "
            f"{split_summary['qualitative_ordering'].get('ordering', [])} |"
        )
    lines.extend(
        [
            "",
            "## Safety Labels",
            "",
            f"- `benchmark_result`: `{summary['benchmark_result']}`",
            f"- `training_executed_by_summary`: `{summary['training_executed_by_summary']}`",
            f"- `sampling_executed_by_summary`: `{summary['sampling_executed_by_summary']}`",
            f"- `raw_data_modified`: `{summary['raw_data_modified']}`",
            f"- `environment_modified`: `{summary['environment_modified']}`",
            "",
            "No training, sampling, preprocessing, scheduler action, GPU use, physical benchmark job, xTB, or environment change was performed by this summary.",
            "",
        ]
    )
    return "\n".join(lines)
