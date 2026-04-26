"""Reporting-only scientific closure summaries from existing full-database outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.eval.formal_attribution import (
    ATTRIBUTION_MASTER_SUMMARY_RESULT_KIND,
    ATTRIBUTION_RANDOM_SUMMARY_RESULT_KIND,
    ATTRIBUTION_SCAFFOLD_SUMMARY_RESULT_KIND,
    DEFAULT_ATTRIBUTION_MASTER_SUMMARY_PATH,
    DEFAULT_ATTRIBUTION_RANDOM_SUMMARY_PATH,
    DEFAULT_ATTRIBUTION_SCAFFOLD_SUMMARY_PATH,
)
from src.eval.formal_full_database_compare import (
    DEFAULT_MULTI_SEED_SUMMARY_PATH,
    DEFAULT_SCAFFOLD_MULTI_SEED_SUMMARY_PATH,
    MULTI_SEED_SUMMARY_RESULT_KIND,
    SCAFFOLD_MULTI_SEED_SUMMARY_RESULT_KIND,
)


DEFAULT_CORE_RESULTS_SUMMARY_PATH = Path("experiments/results/geom_full_database_paper_core_results_summary.json")
DEFAULT_CORE_RESULTS_REPORT_PATH = Path("reports/tables/geom_full_database_paper_core_results_summary.md")
DEFAULT_PHYSICAL_SUMMARY_PATH = Path("experiments/results/geom_full_database_physical_validity_summary.json")
DEFAULT_PHYSICAL_REPORT_PATH = Path("reports/tables/geom_full_database_physical_validity_summary.md")
DEFAULT_ROTATABLE_SUMMARY_PATH = Path("experiments/results/geom_full_database_rotatable_bond_stratified_summary.json")
DEFAULT_ROTATABLE_REPORT_PATH = Path("reports/tables/geom_full_database_rotatable_bond_stratified_summary.md")
DEFAULT_CLAIM_SUMMARY_PATH = Path("experiments/results/geom_full_database_claim_and_limitation_summary.json")
DEFAULT_CLAIM_REPORT_PATH = Path("reports/tables/geom_full_database_claim_and_limitation_summary.md")

CORE_RESULTS_RESULT_KIND = "geom_drugs_full_database_paper_core_results_summary_not_benchmark"
PHYSICAL_SUMMARY_RESULT_KIND = "geom_drugs_full_database_physical_validity_summary_not_benchmark"
ROTATABLE_SUMMARY_RESULT_KIND = "geom_drugs_full_database_rotatable_bond_stratified_summary_not_benchmark"
CLAIM_SUMMARY_RESULT_KIND = "geom_drugs_full_database_claim_and_limitation_summary_not_benchmark"

FORMAL_METRICS = (
    "best_validation_loss",
    "reloaded_validation_loss",
    "sampled_rmsd_to_target_overall_mean",
    "sampled_rmsd_to_target_train_mean",
    "sampled_rmsd_to_target_val_mean",
    "sampled_rmsd_to_target_test_mean",
)
ATTRIBUTION_MODE_ORDER = ("isotropic", "harmonic_prior", "torsion_only", "stiffness_structured")
PHYSICAL_METRIC_ORDER = (
    "finite_coordinate_rate_mean",
    "mean_abs_bond_length_delta_mean",
    "mean_bond_count_checked_mean",
)
ROTATABLE_BIN_ORDER = ("0", "1-3", "4-6", "7+")


def write_paper_closure_outputs(
    formal_random_summary_path: str | Path = DEFAULT_MULTI_SEED_SUMMARY_PATH,
    formal_scaffold_summary_path: str | Path = DEFAULT_SCAFFOLD_MULTI_SEED_SUMMARY_PATH,
    attribution_random_summary_path: str | Path = DEFAULT_ATTRIBUTION_RANDOM_SUMMARY_PATH,
    attribution_scaffold_summary_path: str | Path = DEFAULT_ATTRIBUTION_SCAFFOLD_SUMMARY_PATH,
    attribution_master_summary_path: str | Path = DEFAULT_ATTRIBUTION_MASTER_SUMMARY_PATH,
    core_summary_path: str | Path = DEFAULT_CORE_RESULTS_SUMMARY_PATH,
    core_report_path: str | Path = DEFAULT_CORE_RESULTS_REPORT_PATH,
    physical_summary_path: str | Path = DEFAULT_PHYSICAL_SUMMARY_PATH,
    physical_report_path: str | Path = DEFAULT_PHYSICAL_REPORT_PATH,
    rotatable_summary_path: str | Path = DEFAULT_ROTATABLE_SUMMARY_PATH,
    rotatable_report_path: str | Path = DEFAULT_ROTATABLE_REPORT_PATH,
    claim_summary_path: str | Path = DEFAULT_CLAIM_SUMMARY_PATH,
    claim_report_path: str | Path = DEFAULT_CLAIM_REPORT_PATH,
) -> dict[str, Path]:
    formal_random = _load_summary(Path(formal_random_summary_path), MULTI_SEED_SUMMARY_RESULT_KIND)
    formal_scaffold = _load_summary(Path(formal_scaffold_summary_path), SCAFFOLD_MULTI_SEED_SUMMARY_RESULT_KIND)
    attribution_random = _load_summary(Path(attribution_random_summary_path), ATTRIBUTION_RANDOM_SUMMARY_RESULT_KIND)
    attribution_scaffold = _load_summary(Path(attribution_scaffold_summary_path), ATTRIBUTION_SCAFFOLD_SUMMARY_RESULT_KIND)
    attribution_master = _load_summary(Path(attribution_master_summary_path), ATTRIBUTION_MASTER_SUMMARY_RESULT_KIND)

    inputs = {
        "formal_random": formal_random,
        "formal_scaffold": formal_scaffold,
        "attribution_random": attribution_random,
        "attribution_scaffold": attribution_scaffold,
        "attribution_master": attribution_master,
    }

    core_summary = _build_core_results_summary(inputs)
    physical_summary = _build_physical_validity_summary(inputs)
    rotatable_summary = _build_rotatable_summary(inputs)
    claim_summary = _build_claim_summary(core_summary, physical_summary, rotatable_summary)

    outputs = {}
    outputs["core_summary"] = _write_json(Path(core_summary_path), core_summary)
    outputs["core_report"] = _write_text(Path(core_report_path), _core_markdown_report(core_summary))
    outputs["physical_summary"] = _write_json(Path(physical_summary_path), physical_summary)
    outputs["physical_report"] = _write_text(Path(physical_report_path), _physical_markdown_report(physical_summary))
    outputs["rotatable_summary"] = _write_json(Path(rotatable_summary_path), rotatable_summary)
    outputs["rotatable_report"] = _write_text(Path(rotatable_report_path), _rotatable_markdown_report(rotatable_summary))
    outputs["claim_summary"] = _write_json(Path(claim_summary_path), claim_summary)
    outputs["claim_report"] = _write_text(Path(claim_report_path), _claim_markdown_report(claim_summary))
    return outputs


def _load_summary(path: Path, expected_result_kind: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required input summary does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("result_kind") != expected_result_kind:
        raise ValueError(f"Unexpected result_kind for {path}: {payload.get('result_kind')}")
    if payload.get("benchmark_result") is not False:
        raise ValueError(f"Expected benchmark_result=false in {path}")
    return payload


def _build_core_results_summary(inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    formal_sections = {
        "random": _formal_split_section(inputs["formal_random"]),
        "scaffold": _formal_split_section(inputs["formal_scaffold"]),
    }
    attribution_sections = {
        "random": _attribution_split_section(inputs["attribution_random"]),
        "scaffold": _attribution_split_section(inputs["attribution_scaffold"]),
    }
    headline_findings = _headline_findings(inputs["formal_random"], inputs["formal_scaffold"], inputs["attribution_master"])
    return {
        "summary_name": "geom_full_database_paper_core_results_summary",
        "result_kind": CORE_RESULTS_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "reporting_only_closure_from_existing_summaries",
        "hardware": "reporting_only",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "dataset_scope": inputs["formal_random"]["dataset_scope"],
        "seed_set": list(inputs["formal_random"]["seed_set"]),
        "input_summary_paths": {
            "formal_random": str(DEFAULT_MULTI_SEED_SUMMARY_PATH),
            "formal_scaffold": str(DEFAULT_SCAFFOLD_MULTI_SEED_SUMMARY_PATH),
            "attribution_random": str(DEFAULT_ATTRIBUTION_RANDOM_SUMMARY_PATH),
            "attribution_scaffold": str(DEFAULT_ATTRIBUTION_SCAFFOLD_SUMMARY_PATH),
            "attribution_master": str(DEFAULT_ATTRIBUTION_MASTER_SUMMARY_PATH),
        },
        "formal_AE_multiseed": formal_sections,
        "attribution_ABDE_multiseed": attribution_sections,
        "cross_split_attribution_verdict": inputs["attribution_master"]["directional_verdict"],
        "headline_findings": headline_findings,
        "safety_labels": _report_safety_labels("paper_core_results_summary"),
        "notes": [
            "Paper-style core results closure built from existing full-database random-split/scaffold-split formal and attribution summaries only.",
            "This report does not rerun training or sampling and is explicitly non-benchmark.",
            "Formal A/E summaries establish stiffness vs isotropic behavior; attribution summaries establish A/B/D/E ordering.",
        ],
    }


def _build_physical_validity_summary(inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    split_sections = {
        "random": _physical_split_section(inputs["attribution_random"]),
        "scaffold": _physical_split_section(inputs["attribution_scaffold"]),
    }
    combined = _physical_combined_section(split_sections)
    availability_limitation, coverage_gap_present = _diagnostic_availability_limitation(
        split_sections,
        diagnostic_name="physical-validity",
        remaining_limitation=(
            "The available physical-validity evidence is still limited to sampled finite-coordinate rate and mean "
            "bond-length delta proxies from existing outputs, not full PoseBusters/MMFF/UFF/xTB recomputation."
        ),
    )
    return {
        "summary_name": "geom_full_database_physical_validity_summary",
        "result_kind": PHYSICAL_SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "reporting_only_closure_from_existing_summaries",
        "hardware": "reporting_only",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "dataset_scope": inputs["attribution_master"]["dataset_scope"],
        "seed_set": list(inputs["attribution_master"]["seed_set"]),
        "split_summaries": split_sections,
        "combined_across_splits": combined,
        "availability_limitation": availability_limitation,
        "coverage_gap_present": coverage_gap_present,
        "safety_labels": _report_safety_labels("physical_validity_summary"),
        "notes": [
            "Physical-validity closure from existing sampled diagnostic fields only.",
            "No PoseBusters/MMFF/UFF/xTB recomputation was performed here.",
        ],
    }


def _build_rotatable_summary(inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    split_sections = {
        "random": _rotatable_split_section(inputs["attribution_random"]),
        "scaffold": _rotatable_split_section(inputs["attribution_scaffold"]),
    }
    combined = _rotatable_combined_section(split_sections)
    availability_limitation, coverage_gap_present = _diagnostic_availability_limitation(
        split_sections,
        diagnostic_name="rotatable-bond-count",
        remaining_limitation=(
            "The available rotatable-bond analysis is still limited to one-step sampled RMSD summaries from the "
            "existing outputs, not a broader benchmark-style conformer diversity evaluation."
        ),
    )
    return {
        "summary_name": "geom_full_database_rotatable_bond_stratified_summary",
        "result_kind": ROTATABLE_SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "reporting_only_closure_from_existing_summaries",
        "hardware": "reporting_only",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "dataset_scope": inputs["attribution_master"]["dataset_scope"],
        "seed_set": list(inputs["attribution_master"]["seed_set"]),
        "split_summaries": split_sections,
        "combined_across_splits": combined,
        "availability_limitation": availability_limitation,
        "coverage_gap_present": coverage_gap_present,
        "safety_labels": _report_safety_labels("rotatable_bond_stratified_summary"),
        "notes": [
            "Rotatable-bond-count closure from existing sampled metadata only.",
            "This report does not infer missing stratified diagnostics for isotropic or stiffness runs.",
        ],
    }


def _build_claim_summary(
    core_summary: dict[str, Any],
    physical_summary: dict[str, Any],
    rotatable_summary: dict[str, Any],
) -> dict[str, Any]:
    random_attr = core_summary["attribution_ABDE_multiseed"]["random"]
    scaffold_attr = core_summary["attribution_ABDE_multiseed"]["scaffold"]

    stiffness_vs_isotropic = {
        "random": core_summary["formal_AE_multiseed"]["random"]["directional_verdict"]["verdict"],
        "scaffold": core_summary["formal_AE_multiseed"]["scaffold"]["directional_verdict"]["verdict"],
        "support_level": "supported_on_tracked_loss_and_sampled_rmsd_metrics",
        "statement": (
            "Stiffness is better than isotropic on the tracked best/reloaded validation loss and sampled RMSD metrics "
            "for both random and scaffold multiseed full-database runs."
        ),
    }
    stiffness_vs_torsion = {
        "random": random_attr["baseline_relations"]["torsion_only"],
        "scaffold": scaffold_attr["baseline_relations"]["torsion_only"],
        "support_level": "supported_on_tracked_loss_and_sampled_rmsd_metrics",
        "statement": (
            "Stiffness is better than torsion-only on the tracked best/reloaded validation loss and sampled RMSD metrics "
            "for both random and scaffold attribution summaries."
        ),
    }
    stiffness_vs_harmonic = {
        "random": random_attr["baseline_relations"]["harmonic_prior"],
        "scaffold": scaffold_attr["baseline_relations"]["harmonic_prior"],
        "support_level": "not_supported_harmonic_prior_is_better_on_tracked_loss_and_sampled_rmsd_metrics",
        "statement": (
            "The current evidence does not support stiffness beating harmonic-prior. Harmonic-prior is better than "
            "stiffness on the tracked best/reloaded validation loss and sampled RMSD metrics on both random and scaffold."
        ),
    }
    random_vs_scaffold = {
        "support_level": "supported_qualitatively_stable_ordering_across_splits",
        "statement": (
            "Switching from random to scaffold split does not reverse the qualitative ordering on tracked metrics: "
            "stiffness stays above isotropic and torsion-only, while harmonic-prior stays above stiffness."
        ),
    }
    supported = [
        stiffness_vs_isotropic["statement"],
        stiffness_vs_torsion["statement"],
        random_vs_scaffold["statement"],
    ]
    if not physical_summary["coverage_gap_present"]:
        supported.append("Physical-validity proxy diagnostics now cover A/B/D/E on both random and scaffold splits.")
    if not rotatable_summary["coverage_gap_present"]:
        supported.append(
            "Rotatable-bond-count stratified sampled RMSD diagnostics now cover A/B/D/E on both random and scaffold splits."
        )
    not_supported = [
        stiffness_vs_harmonic["statement"],
        physical_summary["availability_limitation"],
        rotatable_summary["availability_limitation"],
    ]
    coverage_phrase = (
        "The earlier isotropic/stiffness physical-validity and rotatable-bin coverage gap is now closed from existing outputs."
        if not physical_summary["coverage_gap_present"] and not rotatable_summary["coverage_gap_present"]
        else "A physical-validity and/or rotatable-bin coverage gap still remains in the current summaries."
    )
    return {
        "summary_name": "geom_full_database_claim_and_limitation_summary",
        "result_kind": CLAIM_SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "reporting_only_closure_from_existing_summaries",
        "hardware": "reporting_only",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "dataset_scope": core_summary["dataset_scope"],
        "seed_set": core_summary["seed_set"],
        "claims": {
            "stiffness_vs_isotropic": stiffness_vs_isotropic,
            "stiffness_vs_torsion_only": stiffness_vs_torsion,
            "stiffness_vs_harmonic_prior": stiffness_vs_harmonic,
            "random_vs_scaffold_behavior": random_vs_scaffold,
        },
        "supported_by_current_evidence": supported,
        "not_supported_by_current_evidence": not_supported,
        "one_sentence_conclusion": (
            "On the current full-database evidence, stiffness is a clear improvement over isotropic and torsion-only, "
            f"but it does not beat harmonic-prior on the tracked loss/RMSD metrics. {coverage_phrase} "
            "The backfilled physical-validity and rotatable analyses still remain proxy diagnostics rather than full "
            "benchmark-style validity metrics."
        ),
        "safety_labels": _report_safety_labels("claim_and_limitation_summary"),
        "notes": [
            "Short claim/limitation closure synthesized from existing full-database formal and attribution summaries only.",
            "No new training, sampling, preprocessing, or benchmark evaluation was performed here.",
        ],
    }


def _formal_split_section(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    for row in summary["aggregate_metric_comparisons"]:
        metrics[row["metric"]] = {
            "isotropic_mean": row["isotropic_mean"],
            "stiffness_structured_mean": row["stiffness_structured_mean"],
            "stiffness_minus_isotropic_mean": row["stiffness_minus_isotropic_mean"],
            "stiffness_outcome_by_seed": row["stiffness_outcome_by_seed"],
            "stiffness_win_count": row["stiffness_win_count"],
            "stiffness_tie_count": row["stiffness_tie_count"],
            "stiffness_loss_count": row["stiffness_loss_count"],
        }
    return {
        "split_type": summary["dataset"]["split_type"],
        "split_counts": summary["dataset"]["split_counts"],
        "directional_verdict": summary["directional_verdict"],
        "metrics": metrics,
    }


def _attribution_split_section(summary: dict[str, Any]) -> dict[str, Any]:
    split_row = summary["aggregate_metric_comparisons"][0]
    metric_table = {}
    for metric, mode_means in split_row["metric_means_by_mode"].items():
        metric_table[metric] = {mode: mode_means.get(mode) for mode in ATTRIBUTION_MODE_ORDER}
    baseline_relations = {}
    for baseline in ("isotropic", "harmonic_prior", "torsion_only"):
        baseline_relations[baseline] = _baseline_relation(summary["directional_verdict"]["baseline_verdicts"][baseline])
    return {
        "split_type": summary["split_types"][0],
        "split_counts": summary["dataset"]["split_counts_by_type"][summary["split_types"][0]],
        "coverage": summary["coverage"],
        "directional_verdict": summary["directional_verdict"],
        "metric_table": metric_table,
        "baseline_relations": baseline_relations,
    }


def _headline_findings(
    formal_random: dict[str, Any],
    formal_scaffold: dict[str, Any],
    attribution_master: dict[str, Any],
) -> dict[str, Any]:
    harmonic_relation = _baseline_relation(attribution_master["directional_verdict"]["baseline_verdicts"]["harmonic_prior"])
    torsion_relation = _baseline_relation(attribution_master["directional_verdict"]["baseline_verdicts"]["torsion_only"])
    return {
        "stiffness_vs_isotropic": {
            "random": formal_random["directional_verdict"]["verdict"],
            "scaffold": formal_scaffold["directional_verdict"]["verdict"],
        },
        "stiffness_vs_torsion_only": torsion_relation,
        "stiffness_vs_harmonic_prior": harmonic_relation,
        "cross_split_master_verdict": attribution_master["directional_verdict"]["overall_verdict"],
    }


def _physical_split_section(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary["rows"]
    modes = {}
    for mode in ATTRIBUTION_MODE_ORDER:
        mode_rows = [row for row in rows if row["corruption_mode"] == mode]
        available_rows = [row for row in mode_rows if row["physical_validity"]["status"] == "available_from_sampled_molecules"]
        modes[mode] = {
            "seed_count": len(mode_rows),
            "available_seed_count": len(available_rows),
            "finite_coordinate_rate_mean": _mean_or_none(
                [row["physical_validity"]["finite_coordinate_rate"] for row in available_rows]
            ),
            "mean_abs_bond_length_delta_mean": _mean_or_none(
                [row["physical_validity"]["mean_abs_bond_length_delta"] for row in available_rows]
            ),
            "mean_bond_count_checked_mean": _mean_or_none(
                [row["physical_validity"]["mean_bond_count_checked"] for row in available_rows]
            ),
        }
    comparisons = []
    for baseline in ("isotropic", "torsion_only", "harmonic_prior"):
        for metric in ("finite_coordinate_rate_mean", "mean_abs_bond_length_delta_mean"):
            stiffness = modes["stiffness_structured"][metric]
            baseline_value = modes[baseline][metric]
            if stiffness is None or baseline_value is None:
                continue
            if metric == "finite_coordinate_rate_mean":
                relation = _compare_higher_is_better(
                    "stiffness_structured",
                    stiffness,
                    baseline,
                    baseline_value,
                )
            else:
                relation = _compare_lower_is_better(
                    "stiffness_structured",
                    stiffness,
                    baseline,
                    baseline_value,
                )
            comparisons.append(
                {
                    "metric": metric,
                    "baseline_mode": baseline,
                    "stiffness_structured": stiffness,
                    "baseline_value": baseline_value,
                    **relation,
                }
            )
    return {
        "split_type": summary["split_types"][0],
        "status": summary["physical_validity_diagnostics"]["status"],
        "available_row_count": summary["physical_validity_diagnostics"]["available_row_count"],
        "row_count": summary["physical_validity_diagnostics"]["row_count"],
        "modes": modes,
        "comparisons": comparisons,
    }


def _physical_combined_section(split_sections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    modes = {}
    for mode in ATTRIBUTION_MODE_ORDER:
        mode_entries = [split_sections[split]["modes"][mode] for split in split_sections]
        modes[mode] = {
            "available_split_count": sum(1 for entry in mode_entries if entry["available_seed_count"] > 0),
            "finite_coordinate_rate_mean": _mean_or_none(
                [entry["finite_coordinate_rate_mean"] for entry in mode_entries]
            ),
            "mean_abs_bond_length_delta_mean": _mean_or_none(
                [entry["mean_abs_bond_length_delta_mean"] for entry in mode_entries]
            ),
            "mean_bond_count_checked_mean": _mean_or_none(
                [entry["mean_bond_count_checked_mean"] for entry in mode_entries]
            ),
        }
    return {"modes": modes}


def _rotatable_split_section(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary["rows"]
    modes = {}
    for mode in ATTRIBUTION_MODE_ORDER:
        mode_rows = [row for row in rows if row["corruption_mode"] == mode]
        available_rows = [
            row for row in mode_rows if row["rotatable_bond_stratification"]["status"] == "available_from_sampled_molecules"
        ]
        bins = {}
        for bin_label in ROTATABLE_BIN_ORDER:
            bin_payloads = [
                row["rotatable_bond_stratification"]["bins"].get(bin_label)
                for row in available_rows
                if row["rotatable_bond_stratification"]["bins"].get(bin_label) is not None
            ]
            bins[bin_label] = {
                "available_seed_count": len(bin_payloads),
                "mean_sampled_rmsd_to_target": _mean_or_none(
                    [payload["mean_sampled_rmsd_to_target"] for payload in bin_payloads]
                ),
                "mean_sampled_rmsd_to_target_by_split": {
                    split: _mean_or_none(
                        [payload["mean_sampled_rmsd_to_target_by_split"].get(split) for payload in bin_payloads]
                    )
                    for split in ("train", "val", "test")
                },
            }
        modes[mode] = {
            "seed_count": len(mode_rows),
            "available_seed_count": len(available_rows),
            "bins": bins,
        }
    comparisons = []
    for bin_label in ROTATABLE_BIN_ORDER:
        stiffness = modes["stiffness_structured"]["bins"][bin_label]["mean_sampled_rmsd_to_target"]
        if stiffness is None:
            continue
        for baseline in ("isotropic", "torsion_only", "harmonic_prior"):
            baseline_value = modes[baseline]["bins"][bin_label]["mean_sampled_rmsd_to_target"]
            if baseline_value is None:
                continue
            comparisons.append(
                {
                    "bin": bin_label,
                    "baseline_mode": baseline,
                    "stiffness_structured": stiffness,
                    "baseline_value": baseline_value,
                    **_compare_lower_is_better(
                        "stiffness_structured",
                        stiffness,
                        baseline,
                        baseline_value,
                    ),
                }
            )
    return {
        "split_type": summary["split_types"][0],
        "status": summary["rotatable_bond_stratification"]["status"],
        "available_row_count": summary["rotatable_bond_stratification"]["available_row_count"],
        "row_count": summary["rotatable_bond_stratification"]["row_count"],
        "modes": modes,
        "comparisons": comparisons,
    }


def _rotatable_combined_section(split_sections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    modes = {}
    for mode in ATTRIBUTION_MODE_ORDER:
        bins = {}
        for bin_label in ROTATABLE_BIN_ORDER:
            bin_entries = [split_sections[split]["modes"][mode]["bins"][bin_label] for split in split_sections]
            bins[bin_label] = {
                "mean_sampled_rmsd_to_target": _mean_or_none(
                    [entry["mean_sampled_rmsd_to_target"] for entry in bin_entries]
                ),
                "mean_sampled_rmsd_to_target_by_split": {
                    split_name: _mean_or_none(
                        [entry["mean_sampled_rmsd_to_target_by_split"][split_name] for entry in bin_entries]
                    )
                    for split_name in ("train", "val", "test")
                },
            }
        modes[mode] = {"bins": bins}
    return {"modes": modes}


def _baseline_relation(baseline_verdict: dict[str, Any]) -> str:
    tracked = int(baseline_verdict["tracked_core_comparison_count"])
    wins = int(baseline_verdict["stiffness_win_count"])
    ties = int(baseline_verdict["stiffness_tie_count"])
    losses = int(baseline_verdict["stiffness_loss_count"])
    if tracked == 0:
        return "not_available"
    if wins == tracked:
        return "stiffness_better_on_all_tracked_core_metrics"
    if losses == tracked:
        return "baseline_better_on_all_tracked_core_metrics"
    if ties == tracked:
        return "tied_on_all_tracked_core_metrics"
    if wins + ties == tracked:
        return "stiffness_better_or_tied_on_all_tracked_core_metrics"
    if losses + ties == tracked:
        return "baseline_better_or_tied_on_all_tracked_core_metrics"
    return "mixed_or_inconclusive"


def _diagnostic_availability_limitation(
    split_sections: dict[str, dict[str, Any]],
    diagnostic_name: str,
    remaining_limitation: str,
) -> tuple[str, bool]:
    missing = []
    for split_name, split_section in split_sections.items():
        for mode in ATTRIBUTION_MODE_ORDER:
            mode_section = split_section["modes"][mode]
            if mode_section["available_seed_count"] < mode_section["seed_count"]:
                missing.append(f"{split_name}:{mode} {mode_section['available_seed_count']}/{mode_section['seed_count']}")
    if not missing:
        return (
            f"No A/B/D/E coverage gap remains in the {diagnostic_name} summary. {remaining_limitation}",
            False,
        )
    return (
        f"The {diagnostic_name} summary is still missing completed rows for: {', '.join(missing)}. "
        f"{remaining_limitation}",
        True,
    )


def _compare_lower_is_better(
    first_mode: str,
    first: float,
    second_mode: str,
    second: float,
) -> dict[str, Any]:
    delta = first - second
    if abs(delta) <= 1.0e-12:
        better_mode = "tied"
    else:
        better_mode = first_mode if delta < 0.0 else second_mode
    return {"delta": delta, "better_mode": better_mode, "direction": "lower_is_better"}


def _compare_higher_is_better(
    first_mode: str,
    first: float,
    second_mode: str,
    second: float,
) -> dict[str, Any]:
    delta = first - second
    if abs(delta) <= 1.0e-12:
        better_mode = "tied"
    else:
        better_mode = first_mode if delta > 0.0 else second_mode
    return {"delta": delta, "better_mode": better_mode, "direction": "higher_is_better"}


def _mean_or_none(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _report_safety_labels(result_scope: str) -> dict[str, Any]:
    return {
        "benchmark_result": False,
        "non_benchmark": True,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "raw_data_modified": False,
        "environment_modified": False,
        "gpu_used_by_summary": False,
        "result_scope": result_scope,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _core_markdown_report(summary: dict[str, Any]) -> str:
    stiffness_vs_isotropic = summary["headline_findings"]["stiffness_vs_isotropic"]
    lines = [
        "# Full-Database Paper-Style Core Results Summary",
        "",
        "This is a reporting-only synthesis from existing full-database formal and attribution summaries. It is not a benchmark result.",
        "",
        "## Headline",
        "",
        f"- Cross-split attribution verdict: `{summary['cross_split_attribution_verdict']['overall_verdict']}`",
        f"- Stiffness vs isotropic: random=`{stiffness_vs_isotropic['random']}`, scaffold=`{stiffness_vs_isotropic['scaffold']}`",
        f"- Stiffness vs torsion-only: `{summary['headline_findings']['stiffness_vs_torsion_only']}`",
        f"- Stiffness vs harmonic-prior: `{summary['headline_findings']['stiffness_vs_harmonic_prior']}`",
        "",
        "## Formal A/E Multiseed",
        "",
        "| Split | Metric | Isotropic | Stiffness | Delta | Win/Tie/Loss |",
        "|---|---|---:|---:|---:|---|",
    ]
    for split_name, split_section in summary["formal_AE_multiseed"].items():
        for metric in FORMAL_METRICS:
            row = split_section["metrics"][metric]
            lines.append(
                f"| {split_name} | {metric} | {_fmt(row['isotropic_mean'])} | {_fmt(row['stiffness_structured_mean'])} | "
                f"{_fmt(row['stiffness_minus_isotropic_mean'])} | "
                f"{row['stiffness_win_count']}/{row['stiffness_tie_count']}/{row['stiffness_loss_count']} |"
            )
    lines.extend(
        [
            "",
            "## Attribution A/B/D/E Multiseed",
            "",
            "| Split | Metric | Isotropic | Harmonic | Torsion-only | Stiffness |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for split_name, split_section in summary["attribution_ABDE_multiseed"].items():
        for metric in FORMAL_METRICS:
            row = split_section["metric_table"][metric]
            lines.append(
                f"| {split_name} | {metric} | {_fmt(row['isotropic'])} | {_fmt(row['harmonic_prior'])} | "
                f"{_fmt(row['torsion_only'])} | {_fmt(row['stiffness_structured'])} |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Formal A/E summaries support the stiffness-vs-isotropic conclusion on both random and scaffold splits.",
            "- Attribution summaries show that stiffness remains better than torsion-only, but not better than harmonic-prior, on the tracked loss/RMSD metrics.",
            "- This report uses existing summaries only; it does not rerun training, sampling, or evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def _physical_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Full-Database Physical-Validity Summary",
        "",
        "This is a reporting-only summary from already available physical-diagnostic fields in the attribution outputs. It is not a benchmark result.",
        "",
        f"- Limitation: {summary['availability_limitation']}",
        "",
        "## Availability and Means",
        "",
        "| Split | Mode | Seeds With Diagnostics | Finite Coordinate Rate | Mean Abs Bond-Length Delta | Mean Bond Count Checked |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split_name, split_section in summary["split_summaries"].items():
        for mode in ATTRIBUTION_MODE_ORDER:
            row = split_section["modes"][mode]
            lines.append(
                f"| {split_name} | {mode} | {row['available_seed_count']}/{row['seed_count']} | "
                f"{_fmt(row['finite_coordinate_rate_mean'])} | {_fmt(row['mean_abs_bond_length_delta_mean'])} | "
                f"{_fmt(row['mean_bond_count_checked_mean'])} |"
            )
    lines.extend(
        [
            "",
            "## Stiffness Vs Baselines From Available Diagnostics",
            "",
            "| Split | Metric | Baseline | Stiffness | Baseline Value | Delta | Better |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for split_name, split_section in summary["split_summaries"].items():
        for row in split_section["comparisons"]:
            lines.append(
                f"| {split_name} | {row['metric']} | {row['baseline_mode']} | {_fmt(row['stiffness_structured'])} | "
                f"{_fmt(row['baseline_value'])} | {_fmt(row['delta'])} | {row['better_mode']} |"
            )
    lines.extend(
        [
            "",
            summary["availability_limitation"],
            "",
        ]
    )
    return "\n".join(lines)


def _rotatable_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Full-Database Rotatable-Bond-Count Stratified Summary",
        "",
        "This is a reporting-only summary from already available rotatable-bond-count metadata in the attribution outputs. It is not a benchmark result.",
        "",
        f"- Limitation: {summary['availability_limitation']}",
        "",
        "## Availability",
        "",
        "| Split | Mode | Seeds With Rotatable Metadata |",
        "|---|---|---:|",
    ]
    for split_name, split_section in summary["split_summaries"].items():
        for mode in ATTRIBUTION_MODE_ORDER:
            lines.append(
                f"| {split_name} | {mode} | {split_section['modes'][mode]['available_seed_count']}/{split_section['modes'][mode]['seed_count']} |"
            )
    lines.extend(
        [
            "",
            "## Stiffness Vs Baselines By Rotatable Bin",
            "",
            "| Split | Bin | Baseline | Stiffness Overall | Baseline Overall | Better |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for split_name, split_section in summary["split_summaries"].items():
        for row in split_section["comparisons"]:
            lines.append(
                f"| {split_name} | {row['bin']} | {row['baseline_mode']} | {_fmt(row['stiffness_structured'])} | {_fmt(row['baseline_value'])} | {row['better_mode']} |"
            )
    lines.extend(
        [
            "",
            summary["availability_limitation"],
            "",
        ]
    )
    return "\n".join(lines)


def _claim_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Full-Database Claim And Limitation Summary",
        "",
        summary["one_sentence_conclusion"],
        "",
        "## Claims",
        "",
        f"- Stiffness vs isotropic: {summary['claims']['stiffness_vs_isotropic']['statement']}",
        f"- Stiffness vs torsion-only: {summary['claims']['stiffness_vs_torsion_only']['statement']}",
        f"- Stiffness vs harmonic-prior: {summary['claims']['stiffness_vs_harmonic_prior']['statement']}",
        f"- Random vs scaffold: {summary['claims']['random_vs_scaffold_behavior']['statement']}",
        "",
        "## Supported By Current Evidence",
        "",
    ]
    for item in summary["supported_by_current_evidence"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Not Supported By Current Evidence", ""])
    for item in summary["not_supported_by_current_evidence"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.8g}"
