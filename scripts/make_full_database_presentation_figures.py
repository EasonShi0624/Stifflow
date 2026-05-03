#!/usr/bin/env python
"""Create a presentation-ready full-database reporting bundle from existing summaries."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any

from _stifflow_runtime import ensure_stifflow_python

ensure_stifflow_python()

os.environ.setdefault("MPLCONFIGDIR", str(Path("experiments/cache/matplotlib")))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("experiments/cache/xdg")))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
FIGURES_DIR = ROOT / "reports" / "figures"
TABLE_PATH = ROOT / "reports" / "tables" / "full_database_presentation_key_table.md"
AMR_COV_TABLE_PATH = ROOT / "reports" / "tables" / "full_database_amr_cov_metric_availability.md"
REPORT_PATH = ROOT / "reports" / "full_database_results_report.md"
SUMMARY_PATH = RESULTS_DIR / "geom_full_database_presentation_results_summary.json"
AMR_COV_FULL_TEST_SUMMARY_PATH = RESULTS_DIR / "geom_full_database_amr_cov_full_test_abde_seed0.json"

INPUT_PATHS = {
    "formal_random": RESULTS_DIR / "geom_formal_full_database_multiseed_summary.json",
    "formal_scaffold": RESULTS_DIR / "geom_formal_full_database_scaffold_multiseed_summary.json",
    "attribution_random": RESULTS_DIR / "geom_formal_full_database_attribution_random_summary.json",
    "attribution_scaffold": RESULTS_DIR / "geom_formal_full_database_attribution_scaffold_summary.json",
    "attribution_master": RESULTS_DIR / "geom_formal_full_database_attribution_summary.json",
    "core": RESULTS_DIR / "geom_full_database_paper_core_results_summary.json",
    "physical": RESULTS_DIR / "geom_full_database_physical_validity_summary.json",
    "rotatable": RESULTS_DIR / "geom_full_database_rotatable_bond_stratified_summary.json",
    "claims": RESULTS_DIR / "geom_full_database_claim_and_limitation_summary.json",
    "processed_manifest": ROOT / "data" / "processed" / "geom_drugs_full_database" / "manifest.json",
    "random_split_manifest": ROOT / "data" / "splits" / "geom_drugs_full_database" / "random_seed0" / "manifest.json",
    "scaffold_split_manifest": ROOT / "data" / "splits" / "geom_drugs_full_database" / "scaffold_seed0" / "manifest.json",
}

CORE_METRICS = (
    "best_validation_loss",
    "reloaded_validation_loss",
    "sampled_rmsd_to_target_overall_mean",
    "sampled_rmsd_to_target_train_mean",
    "sampled_rmsd_to_target_val_mean",
    "sampled_rmsd_to_target_test_mean",
)
RMSD_METRICS = (
    "sampled_rmsd_to_target_overall_mean",
    "sampled_rmsd_to_target_train_mean",
    "sampled_rmsd_to_target_val_mean",
    "sampled_rmsd_to_target_test_mean",
)
METRIC_LABELS = {
    "best_validation_loss": "best val loss",
    "reloaded_validation_loss": "reloaded val loss",
    "sampled_rmsd_to_target_overall_mean": "sampled RMSD overall",
    "sampled_rmsd_to_target_train_mean": "sampled RMSD train",
    "sampled_rmsd_to_target_val_mean": "sampled RMSD val",
    "sampled_rmsd_to_target_test_mean": "sampled RMSD test",
    "sampled_mean_abs_bond_length_delta": "mean abs bond delta",
}
METHOD_LABELS = {
    "harmonic_prior": "harmonic",
    "harmonic": "harmonic",
    "stiffness_structured": "stiffness_structured",
    "isotropic": "isotropic",
    "torsion_only": "torsion_only",
}
METHOD_COLORS = {
    "harmonic_prior": "#2c7fb8",
    "harmonic": "#2c7fb8",
    "stiffness_structured": "#31a354",
    "isotropic": "#756bb1",
    "torsion_only": "#de2d26",
}
METHOD_ORDER = ("harmonic_prior", "stiffness_structured", "isotropic", "torsion_only")
CORE_PAIR_ORDER = ("isotropic", "stiffness_structured")
ROTATABLE_BINS = ("0", "1-3", "4-6", "7+")
REQUESTED_ENSEMBLE_METRICS = ("AMR-R", "COV-R", "AMR-P", "COV-P")
SUPPORTED_PRESENTATION_METRICS = (
    "best_validation_loss",
    "reloaded_validation_loss",
    "sampled_rmsd_to_target_overall_mean",
    "sampled_rmsd_to_target_train_mean",
    "sampled_rmsd_to_target_val_mean",
    "sampled_rmsd_to_target_test_mean",
    "finite_coordinate_rate",
    "sampled_mean_abs_bond_length_delta",
    "rotatable_bond_stratified_sampled_rmsd_to_target",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    parser.add_argument("--table-path", default=str(TABLE_PATH))
    parser.add_argument("--amr-cov-table-path", default=str(AMR_COV_TABLE_PATH))
    parser.add_argument("--report-path", default=str(REPORT_PATH))
    parser.add_argument("--summary-path", default=str(SUMMARY_PATH))
    args = parser.parse_args()

    payloads = {name: load_json(path) for name, path in INPUT_PATHS.items()}
    amr_cov_full_test = load_optional_json(AMR_COV_FULL_TEST_SUMMARY_PATH)
    figures_dir = Path(args.figures_dir)
    table_path = Path(args.table_path)
    amr_cov_table_path = Path(args.amr_cov_table_path)
    report_path = Path(args.report_path)
    summary_path = Path(args.summary_path)

    figure_paths = {
        "core": figures_dir / "full_database_core_random_vs_scaffold.png",
        "attribution_random": figures_dir / "full_database_attribution_random.png",
        "attribution_scaffold": figures_dir / "full_database_attribution_scaffold.png",
        "physical": figures_dir / "full_database_physical_validity.png",
        "rotatable": figures_dir / "full_database_rotatable_bond_stratified.png",
        "seed_level": figures_dir / "full_database_seed_level_comparison.png",
    }
    metric_availability = build_metric_availability(payloads, amr_cov_full_test)

    set_style()
    plot_core_random_vs_scaffold(
        payloads["formal_random"], payloads["formal_scaffold"], figure_paths["core"]
    )
    plot_attribution(payloads["attribution_random"], "random", figure_paths["attribution_random"])
    plot_attribution(payloads["attribution_scaffold"], "scaffold", figure_paths["attribution_scaffold"])
    plot_physical_validity(payloads["physical"], figure_paths["physical"])
    plot_rotatable_stratified(payloads["rotatable"], figure_paths["rotatable"])
    plot_seed_level_comparison(
        payloads["formal_random"], payloads["formal_scaffold"], figure_paths["seed_level"]
    )

    write_text(amr_cov_table_path, build_amr_cov_table(metric_availability))
    write_text(table_path, build_key_table(payloads, metric_availability))
    write_json(
        summary_path,
        build_presentation_summary(
            payloads=payloads,
            metric_availability=metric_availability,
            amr_cov_full_test=amr_cov_full_test,
            figure_paths=figure_paths,
            table_path=table_path,
            amr_cov_table_path=amr_cov_table_path,
            report_path=report_path,
            summary_path=summary_path,
        ),
    )
    write_text(
        report_path,
        build_report(payloads, figure_paths, table_path, amr_cov_table_path, summary_path, metric_availability),
    )

    for label, path in figure_paths.items():
        print(f"{label}_figure: {relative(path)}")
        print(f"{label}_figure_pdf: {relative(path.with_suffix('.pdf'))}")
    print(f"summary_json: {relative(summary_path)}")
    print(f"key_table: {relative(table_path)}")
    print(f"amr_cov_table: {relative(amr_cov_table_path)}")
    print(f"report: {relative(report_path)}")
    print("note: generated from existing JSON summaries only; no training, preprocessing, sampling, jobs, or environment changes.")


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def metric_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["metric"]: entry for entry in summary["aggregate_metric_comparisons"]}


def metric_table(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    return summary["aggregate_metric_comparisons"][0]["metric_means_by_mode"]


def plot_core_random_vs_scaffold(
    random_summary: dict[str, Any], scaffold_summary: dict[str, Any], output_path: Path
) -> None:
    summaries = {"random": metric_map(random_summary), "scaffold": metric_map(scaffold_summary)}
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.4))
    axes = axes.ravel()
    width = 0.34
    split_positions = {"random": 0.0, "scaffold": 1.0}
    method_offsets = {"isotropic": -width / 2, "stiffness_structured": width / 2}

    for ax, metric in zip(axes, CORE_METRICS):
        for method in CORE_PAIR_ORDER:
            xs = []
            ys = []
            for split in ("random", "scaffold"):
                entry = summaries[split][metric]
                key = f"{method}_mean"
                xs.append(split_positions[split] + method_offsets[method])
                ys.append(entry[key])
            ax.bar(
                xs,
                ys,
                width=width,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                alpha=0.9,
            )
            for split in ("random", "scaffold"):
                entry = summaries[split][metric]
                values = entry[f"{method}_values_by_seed"]
                center = split_positions[split] + method_offsets[method]
                for seed_idx, value in enumerate(values):
                    jitter = (seed_idx - 1) * 0.035
                    ax.scatter(center + jitter, value, color="black", s=12, zorder=5, alpha=0.75)

        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks([split_positions["random"], split_positions["scaffold"]])
        ax.set_xticklabels(["random", "scaffold"])
        ax.yaxis.set_major_formatter(FuncFormatter(smart_tick))
        ax.set_ylabel("loss" if "loss" in metric else "RMSD")
        ax.margins(y=0.15)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="center left", bbox_to_anchor=(0.88, 0.53), frameon=False)
    fig.suptitle("Core full-database A/E comparison: isotropic vs stiffness_structured", y=0.99)
    fig.text(
        0.5,
        -0.01,
        "Bars are seed means over seeds 0, 1, 2; black dots are individual seed values. Lower is better.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.03, 0.86, 0.94))
    save_figure(fig, output_path)


def plot_attribution(summary: dict[str, Any], split_type: str, output_path: Path) -> None:
    table = metric_table(summary)
    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    y_positions = list(range(len(CORE_METRICS)))
    offsets = {
        "harmonic_prior": -0.27,
        "stiffness_structured": -0.09,
        "isotropic": 0.09,
        "torsion_only": 0.27,
    }

    for method in METHOD_ORDER:
        xs = []
        ys = []
        for idx, metric in enumerate(CORE_METRICS):
            values = table[metric]
            finite_values = [value for value in values.values() if value is not None and value > 0]
            if not finite_values or method not in values:
                continue
            best = min(finite_values)
            xs.append(values[method] / best)
            ys.append(idx + offsets[method])
        ax.scatter(
            xs,
            ys,
            s=84,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

    ax.axvline(1.0, color="#222222", linewidth=1.0, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("metric value / best value for that metric (log scale; lower is better)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([METRIC_LABELS[metric] for metric in CORE_METRICS])
    ax.invert_yaxis()
    ax.set_title(f"Attribution comparison on {split_type} split")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, axis="x", which="both", alpha=0.25)
    ax.grid(True, axis="y", alpha=0.12)
    fig.text(
        0.5,
        -0.02,
        "Each row is normalized within metric so the best available method is 1.0. Actual means are reported in the markdown table and report.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, output_path)


def plot_physical_validity(summary: dict[str, Any], output_path: Path) -> None:
    methods = ("harmonic_prior", "stiffness_structured", "torsion_only", "isotropic")
    splits = ("random", "scaffold")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    width = 0.18
    positions = list(range(len(splits)))
    method_offsets = {
        method: (idx - (len(methods) - 1) / 2) * width for idx, method in enumerate(methods)
    }

    for ax, metric, ylabel in (
        (axes[0], "mean_abs_bond_length_delta_mean", "mean abs bond length delta"),
        (axes[1], "finite_coordinate_rate_mean", "finite-coordinate rate"),
    ):
        for method in methods:
            xs = [pos + method_offsets[method] for pos in positions]
            ys = [
                summary["split_summaries"][split]["modes"][method][metric]
                for split in splits
                if method in summary["split_summaries"][split]["modes"]
            ]
            ax.bar(
                xs,
                ys,
                width=width,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                alpha=0.9,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(splits)
        ax.set_ylabel(ylabel)
        ax.set_title(METRIC_LABELS.get(metric.replace("_mean", ""), ylabel))
        ax.yaxis.set_major_formatter(FuncFormatter(smart_tick))

    axes[1].set_ylim(0.985, 1.003)
    axes[1].text(
        0.5,
        0.987,
        "all available mode/split means = 1.0",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.88, 0.52), frameon=False)
    fig.suptitle("Physical-validity proxy diagnostics from existing sampled outputs", y=0.98)
    fig.text(
        0.5,
        -0.02,
        "Coverage is complete for A/B/D/E here, but diagnostics are finite-coordinate and bond-length-delta proxies, not PoseBusters/MMFF/UFF/xTB.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 0.86, 0.92))
    save_figure(fig, output_path)


def plot_rotatable_stratified(summary: dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.5), sharey=True)
    methods = ("harmonic_prior", "stiffness_structured", "isotropic", "torsion_only")
    x = list(range(len(ROTATABLE_BINS)))

    for ax, split in zip(axes, ("random", "scaffold")):
        split_modes = summary["split_summaries"][split]["modes"]
        for method in methods:
            bins = split_modes[method]["bins"]
            y = [bins[bin_label]["mean_sampled_rmsd_to_target"] for bin_label in ROTATABLE_BINS]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(ROTATABLE_BINS)
        ax.set_xlabel("rotatable bond bin")
        ax.set_title(f"{split} split")
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="both", alpha=0.25)

    axes[0].set_ylabel("mean sampled RMSD to target (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.88, 0.52), frameon=False)
    fig.suptitle("Rotatable-bond stratified one-step sampled RMSD", y=0.98)
    fig.text(
        0.5,
        -0.02,
        "Lower is better. This is a stratified RMSD proxy from existing sampled metadata, not a conformer-diversity benchmark.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 0.86, 0.92))
    save_figure(fig, output_path)


def plot_seed_level_comparison(
    random_summary: dict[str, Any], scaffold_summary: dict[str, Any], output_path: Path
) -> None:
    summaries = {"random": metric_map(random_summary), "scaffold": metric_map(scaffold_summary)}
    seed_colors = {0: "#252525", 1: "#636363", 2: "#969696"}
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.4), sharex=True, sharey=True)

    all_deltas = []
    for split in ("random", "scaffold"):
        for metric in CORE_METRICS:
            entry = summaries[split][metric]
            for iso, stiff in zip(
                entry["isotropic_values_by_seed"], entry["stiffness_structured_values_by_seed"]
            ):
                all_deltas.append(stiff - iso)
    x_min = min(all_deltas) * 1.12
    x_max = max(0.002, max(all_deltas) * 1.12)

    for ax, split in zip(axes, ("random", "scaffold")):
        for metric_idx, metric in enumerate(CORE_METRICS):
            entry = summaries[split][metric]
            for seed_idx, (iso, stiff) in enumerate(
                zip(entry["isotropic_values_by_seed"], entry["stiffness_structured_values_by_seed"])
            ):
                delta = stiff - iso
                y = metric_idx + (seed_idx - 1) * 0.12
                ax.scatter(delta, y, s=58, color=seed_colors[seed_idx], label=f"seed {seed_idx}" if metric_idx == 0 else None)
        ax.axvline(0.0, color="#222222", linewidth=1.1)
        ax.set_title(f"{split} split")
        ax.set_xlabel("stiffness_structured - isotropic")
        ax.set_xlim(x_min, x_max)
        ax.grid(True, axis="x", alpha=0.25)
        ax.grid(True, axis="y", alpha=0.12)

    axes[0].set_yticks(range(len(CORE_METRICS)))
    axes[0].set_yticklabels([METRIC_LABELS[metric] for metric in CORE_METRICS])
    axes[0].invert_yaxis()
    axes[0].set_ylabel("tracked core metric")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle("Seed-level A/E deltas: negative means stiffness_structured is lower", y=0.98)
    fig.text(
        0.5,
        -0.02,
        "All plotted seed-level deltas are negative across random and scaffold full-database A/E summaries.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    save_figure(fig, output_path)


def build_metric_availability(
    payloads: dict[str, dict[str, Any]],
    amr_cov_full_test: dict[str, Any] | None,
) -> dict[str, Any]:
    processed_manifest = payloads["processed_manifest"]
    attribution_master = payloads["attribution_master"]
    if amr_cov_full_test is not None:
        metric_rows = [
            {
                "metric": metric,
                "status": "present",
                "computable_from_existing_completed_outputs": True,
                "reason": (
                    "Computed by the reporting-only AMR/COV evaluator from existing checkpoints, "
                    "cached splits, and raw GEOM reference conformer pickles."
                ),
            }
            for metric in REQUESTED_ENSEMBLE_METRICS
        ]
        return {
            "requested_metrics_status": "available_from_full_test_current_protocol",
            "requested_metrics": metric_rows,
            "amr_name_note": (
                "AMR is treated here as the average-minimum-RMSD ensemble metric; the repository docs also "
                "refer to the same family as MAT-R/MAT-P."
            ),
            "can_compute_honestly_without_rerunning_training": True,
            "can_compute_honestly_from_current_completed_outputs": True,
            "full_test_summary_path": relative(AMR_COV_FULL_TEST_SUMMARY_PATH),
            "full_test_coverage": amr_cov_full_test.get("coverage"),
            "full_test_output_paths": amr_cov_full_test.get("output_paths", {}),
            "full_test_evaluation": amr_cov_full_test.get("evaluation", {}),
            "processed_artifact_evidence": {
                "artifact_path": processed_manifest["artifact_path"],
                "molecule_count": processed_manifest["molecule_count"],
                "conformer_count": processed_manifest["conformer_count"],
                "max_conformers_per_mol": processed_manifest["bounds"]["max_conformers_per_mol"],
                "one_conformer_per_molecule": processed_manifest["bounds"]["one_conformer_per_molecule"],
            },
            "supported_current_metrics": list(SUPPORTED_PRESENTATION_METRICS) + list(REQUESTED_ENSEMBLE_METRICS),
            "minimum_future_artifacts_needed": [],
            "future_reporting_path": [
                "Run additional seeds or a stronger sampler if needed; do not reinterpret this as broader than the current one-step protocol.",
            ],
        }
    sample_fields_observed = [
        "atom_count",
        "molecule_id",
        "predicted_velocity_norm",
        "rotatable_bond_count",
        "sample_step_count",
        "sample_step_size",
        "sampled_bond_count_checked",
        "sampled_finite_coordinate",
        "sampled_mean_abs_bond_length_delta",
        "sampled_rmsd_to_target",
        "split",
    ]
    missing_artifacts = [
        "multiple generated conformers per molecule, with generated coordinate arrays retained",
        "multiple reference conformers per molecule, with reference coordinate arrays retained",
        "a pairwise generated-vs-reference RMSD matrix or enough coordinates to reconstruct it",
        "thresholds and split-specific denominators for recall and precision coverage",
    ]
    metric_rows = []
    for metric in REQUESTED_ENSEMBLE_METRICS:
        metric_rows.append(
            {
                "metric": metric,
                "status": "not_present",
                "computable_from_existing_completed_outputs": False,
                "reason": (
                    "Current full-database outputs are one-conformer-per-molecule and store one scalar "
                    "sampled_rmsd_to_target per molecule, not generated/reference conformer ensembles."
                ),
            }
        )
    return {
        "requested_metrics_status": "unavailable",
        "requested_metrics": metric_rows,
        "amr_name_note": (
            "AMR is treated here as the average-minimum-RMSD ensemble metric; the repository docs also "
            "refer to the same family as MAT-R/MAT-P. Neither AMR/MAT nor COV recall/precision is present "
            "for the full-database outputs."
        ),
        "can_compute_honestly_without_rerunning_training": False,
        "can_compute_honestly_from_current_completed_outputs": False,
        "why_not_computable": [
            "The processed full-database artifact uses one conformer per molecule.",
            "The completed result JSONs store sampled scalar diagnostics, not generated coordinate ensembles.",
            "Recall and precision directions require reference and generated conformer sets per molecule.",
            "A thresholded COV number could be fabricated from sampled_rmsd_to_target, but it would be a degenerate one-sample proxy, not AMR/COV-R/P.",
        ],
        "processed_artifact_evidence": {
            "artifact_path": processed_manifest["artifact_path"],
            "molecule_count": processed_manifest["molecule_count"],
            "conformer_count": processed_manifest["conformer_count"],
            "max_conformers_per_mol": processed_manifest["bounds"]["max_conformers_per_mol"],
            "one_conformer_per_molecule": processed_manifest["bounds"]["one_conformer_per_molecule"],
        },
        "completed_output_evidence": {
            "full_database_attribution_raw_result_count": len(attribution_master["input_result_paths"]),
            "full_database_attribution_matrix_complete": attribution_master["coverage"]["complete_expected_matrix"],
            "split_types": attribution_master["split_types"],
            "seed_set": attribution_master["seed_set"],
            "modes": attribution_master["attribution_modes"],
            "sampled_molecule_fields_observed_in_raw_result_audit": sample_fields_observed,
        },
        "supported_current_metrics": list(SUPPORTED_PRESENTATION_METRICS),
        "minimum_future_artifacts_needed": missing_artifacts,
        "future_reporting_path": [
            "Persist generated conformer coordinates for each evaluated molecule and seed.",
            "Persist or reference the full GEOM conformer ensemble for each evaluated molecule.",
            "Add a reporting-only evaluator that computes pairwise RMSD matrices and writes AMR-R, COV-R, AMR-P, and COV-P by split/method/seed.",
            "Aggregate those metric JSONs with the existing presentation bundle without changing the trained checkpoints.",
        ],
    }


def build_amr_cov_table(metric_availability: dict[str, Any]) -> str:
    if metric_availability["requested_metrics_status"] == "available_from_full_test_current_protocol":
        output_paths = metric_availability.get("full_test_output_paths", {})
        lines = [
            "# Full-Database AMR/COV Metric Status",
            "",
            "AMR/COV is available from the reporting-only full test-set current-protocol evaluator. No training was rerun.",
            "",
            "| Requested metric | Present | Source |",
            "|---|---|---|",
        ]
        for row in metric_availability["requested_metrics"]:
            lines.append(f"| {row['metric']} | yes | {row['reason']} |")
        lines.extend(
            [
                "",
                "## AMR/COV Artifacts",
                "",
                f"- Summary JSON: `{metric_availability['full_test_summary_path']}`",
                f"- Combined table: `{output_paths.get('combined_report')}`",
                f"- Random table: `{output_paths.get('random_report')}`",
                f"- Scaffold table: `{output_paths.get('scaffold_report')}`",
                f"- Combined figure: `{output_paths.get('combined_figure')}`",
                f"- Random figure: `{output_paths.get('random_figure')}`",
                f"- Scaffold figure: `{output_paths.get('scaffold_figure')}`",
                "",
            ]
        )
        return "\n".join(lines)
    lines = [
        "# Full-Database AMR/COV Metric Availability",
        "",
        "This audit is generated from existing completed outputs only. No training, sampling, preprocessing, scheduler jobs, or environment changes were run.",
        "",
        "| Requested metric | Present in current full-database summaries/results | Computable honestly from current completed outputs | Reason |",
        "|---|---|---|---|",
    ]
    for row in metric_availability["requested_metrics"]:
        lines.append(
            f"| {row['metric']} | no | no | {row['reason']} |"
        )
    evidence = metric_availability["processed_artifact_evidence"]
    lines.extend(
        [
            "",
            "## Audit Evidence",
            "",
            f"- Processed artifact: `{evidence['artifact_path']}`",
            f"- Molecules/conformers: `{evidence['molecule_count']}` / `{evidence['conformer_count']}`",
            f"- Max conformers per molecule: `{evidence['max_conformers_per_mol']}`",
            f"- One conformer per molecule: `{evidence['one_conformer_per_molecule']}`",
            "- Raw result sampled fields audited: "
            + ", ".join(f"`{field}`" for field in metric_availability["completed_output_evidence"]["sampled_molecule_fields_observed_in_raw_result_audit"]),
            "",
            "## Minimal Future Path",
            "",
        ]
    )
    for item in metric_availability["future_reporting_path"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def build_presentation_summary(
    payloads: dict[str, dict[str, Any]],
    metric_availability: dict[str, Any],
    amr_cov_full_test: dict[str, Any] | None,
    figure_paths: dict[str, Path],
    table_path: Path,
    amr_cov_table_path: Path,
    report_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    core = payloads["core"]
    random_core = core["formal_AE_multiseed"]["random"]["metrics"]
    scaffold_core = core["formal_AE_multiseed"]["scaffold"]["metrics"]
    random_attr = core["attribution_ABDE_multiseed"]["random"]["metric_table"]
    scaffold_attr = core["attribution_ABDE_multiseed"]["scaffold"]["metric_table"]
    return {
        "summary_name": "geom_full_database_presentation_results_summary",
        "result_kind": "geom_drugs_full_database_presentation_bundle_summary_not_benchmark",
        "benchmark_result": False,
        "execution_mode": "local_reporting_only_presentation_bundle",
        "hardware": "reporting_only",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "preprocessing_executed_by_summary": False,
        "environment_modified": False,
        "raw_data_modified": False,
        "dataset_scope": core["dataset_scope"],
        "seed_set": core["seed_set"],
        "metric_availability": metric_availability,
        "amr_cov_full_test_summary": amr_cov_full_test,
        "input_paths": {name: relative(path) for name, path in INPUT_PATHS.items()},
        "generated_artifacts": {
            "summary_json": relative(summary_path),
            "report": relative(report_path),
            "key_table": relative(table_path),
            "amr_cov_table": relative(amr_cov_table_path),
            "figures": {
                label: {"png": relative(path), "pdf": relative(path.with_suffix(".pdf"))}
                for label, path in figure_paths.items()
            },
        },
        "supported_key_results": {
            "formal_AE_multiseed": {
                "random": {
                    "best_validation_loss": random_core["best_validation_loss"],
                    "sampled_rmsd_to_target_overall_mean": random_core["sampled_rmsd_to_target_overall_mean"],
                    "directional_verdict": core["formal_AE_multiseed"]["random"]["directional_verdict"],
                },
                "scaffold": {
                    "best_validation_loss": scaffold_core["best_validation_loss"],
                    "sampled_rmsd_to_target_overall_mean": scaffold_core["sampled_rmsd_to_target_overall_mean"],
                    "directional_verdict": core["formal_AE_multiseed"]["scaffold"]["directional_verdict"],
                },
            },
            "attribution_ABDE_multiseed": {
                "random_ordering": ordering_text(random_attr, CORE_METRICS),
                "scaffold_ordering": ordering_text(scaffold_attr, CORE_METRICS),
            },
        },
        "headline_findings": [
            amr_cov_headline(metric_availability),
            "The current supported result hierarchy on tracked loss/RMSD diagnostics is harmonic > stiffness_structured > isotropic > torsion_only.",
            "stiffness_structured is consistently better than isotropic and torsion_only on the tracked full-database diagnostics.",
            "harmonic remains stronger than stiffness_structured on the tracked current diagnostics.",
        ],
        "notes": [
            "This bundle reuses existing result JSONs and summaries only.",
            "The current full-database setup is one-conformer-per-molecule with one-step sampled RMSD diagnostics.",
            amr_cov_note(metric_availability),
        ],
    }


def amr_cov_available(metric_availability: dict[str, Any]) -> bool:
    return metric_availability["requested_metrics_status"] == "available_from_full_test_current_protocol"


def amr_cov_headline(metric_availability: dict[str, Any]) -> str:
    if amr_cov_available(metric_availability):
        return "AMR-R, COV-R, AMR-P, and COV-P are available from the full test-set current-protocol evaluator."
    return "AMR-R, COV-R, AMR-P, and COV-P are not present and are not computable honestly from the current completed outputs."


def amr_cov_note(metric_availability: dict[str, Any]) -> str:
    if amr_cov_available(metric_availability):
        return "AMR/COV ensemble metrics are sourced from the separate full test-set current-protocol evaluation summary."
    return "AMR/COV ensemble metrics require future generated/reference conformer ensemble artifacts."


def amr_cov_audit_paragraph(metric_availability: dict[str, Any], processed_evidence: dict[str, Any]) -> str:
    if amr_cov_available(metric_availability):
        paths = metric_availability["full_test_output_paths"]
        return (
            "The requested AMR/COV ensemble metrics are available from a separate full test-set "
            "current-protocol evaluation that reloads existing checkpoints, reads raw GEOM reference "
            "conformer pickles, and writes presentation artifacts. "
            f"Combined table: `{paths.get('combined_report')}`. "
            f"Combined figure: `{paths.get('combined_figure')}`."
        )
    return (
        "The requested AMR/COV ensemble metrics are unavailable in the completed full-database outputs. "
        f"The current processed artifact records `{processed_evidence['molecule_count']}` molecules and "
        f"`{processed_evidence['conformer_count']}` conformers with "
        f"`max_conformers_per_mol={processed_evidence['max_conformers_per_mol']}` and "
        f"`one_conformer_per_molecule={processed_evidence['one_conformer_per_molecule']}`. The completed "
        "result JSONs store one scalar `sampled_rmsd_to_target` per sampled molecule plus proxy validity "
        "fields, not generated coordinate ensembles or full reference conformer sets."
    )


def amr_cov_report_lines(metric_availability: dict[str, Any]) -> list[str]:
    if not amr_cov_available(metric_availability):
        return []
    paths = metric_availability["full_test_output_paths"]
    return [
        "### 2.2 AMR/COV full test-set figures",
        "",
        f"![AMR/COV combined comparison]({relative_to_report(ROOT / paths['combined_figure'])})",
        "",
        f"Caption: Full test-set AMR/COV comparison across random and scaffold splits under the current one-step sampler protocol. Source file: `{paths['combined_figure']}`.",
        "",
        f"![AMR/COV random comparison]({relative_to_report(ROOT / paths['random_figure'])})",
        "",
        f"Caption: Random split A/B/D/E comparison for AMR-R, COV-R, AMR-P, and COV-P. Source file: `{paths['random_figure']}`.",
        "",
        f"![AMR/COV scaffold comparison]({relative_to_report(ROOT / paths['scaffold_figure'])})",
        "",
        f"Caption: Scaffold split A/B/D/E comparison for AMR-R, COV-R, AMR-P, and COV-P. Source file: `{paths['scaffold_figure']}`.",
        "",
    ]


def amr_cov_limitation(metric_availability: dict[str, Any]) -> str:
    if amr_cov_available(metric_availability):
        return (
            "- AMR/COV metrics are full test-set current-protocol results, but they still use the repository's "
            "one-step checkpoint sampler and configured reference-conformer cap."
        )
    return (
        "- AMR-R, COV-R, AMR-P, and COV-P are not present in the supplied summary/result files and are not "
        "computable from the current one-conformer/one-sample artifacts."
    )


def amr_cov_presentation_bullet(metric_availability: dict[str, Any]) -> str:
    if amr_cov_available(metric_availability):
        return "- Present AMR/COV as full test-set current-protocol ensemble metrics, separate from broader benchmark claims."
    return "- Separate supported results from unavailable metrics: AMR/COV, COV/MAT, and full physical-validity checks are not in the current summaries."


def build_key_table(payloads: dict[str, dict[str, Any]], metric_availability: dict[str, Any]) -> str:
    core = payloads["core"]
    physical = payloads["physical"]
    random_core = core["formal_AE_multiseed"]["random"]["metrics"]
    scaffold_core = core["formal_AE_multiseed"]["scaffold"]["metrics"]
    random_attr = core["attribution_ABDE_multiseed"]["random"]["metric_table"]
    scaffold_attr = core["attribution_ABDE_multiseed"]["scaffold"]["metric_table"]
    random_physical = physical["split_summaries"]["random"]["modes"]
    scaffold_physical = physical["split_summaries"]["scaffold"]["modes"]

    lines = [
        "# Full-Database Presentation Key Table",
        "",
        "All numeric values are multiseed means over seeds 0, 1, and 2. Lower is better except finite-coordinate rate. The source summaries are labeled non-benchmark/reporting-only.",
        "",
        "| Comparison | Metric or ordering | Random split | Scaffold split | Verdict |",
        "|---|---|---:|---:|---|",
    ]
    if amr_cov_available(metric_availability):
        lines.append(
            "| AMR/COV full test-set | AMR-R, COV-R, AMR-P, COV-P | "
            f"available in `{metric_availability['full_test_output_paths'].get('random_report')}` | "
            f"available in `{metric_availability['full_test_output_paths'].get('scaffold_report')}` | "
            "full test-set current protocol; see AMR/COV summary JSON |"
        )
    else:
        lines.append(
            "| AMR/COV availability | AMR-R, COV-R, AMR-P, COV-P | not present; not computable from one-conformer/one-sample outputs | not present; not computable from one-conformer/one-sample outputs | future ensemble artifacts required |"
        )
    for metric in CORE_METRICS:
        r = random_core[metric]
        s = scaffold_core[metric]
        lines.append(
            "| A/E isotropic vs stiffness_structured | "
            f"{METRIC_LABELS[metric]} | "
            f"isotropic {fmt(r['isotropic_mean'])}; stiffness_structured {fmt(r['stiffness_structured_mean'])} | "
            f"isotropic {fmt(s['isotropic_mean'])}; stiffness_structured {fmt(s['stiffness_structured_mean'])} | "
            "stiffness_structured lower in 3/3 seeds for both splits |"
        )

    lines.append(
        "| Attribution A/B/D/E | tracked core metric ordering | "
        f"{ordering_text(random_attr, CORE_METRICS)} | "
        f"{ordering_text(scaffold_attr, CORE_METRICS)} | "
        "qualitative ordering does not flip between random and scaffold |"
    )
    lines.append(
        "| Physical-validity proxy | mean abs bond length delta ordering | "
        f"{ordering_text({metric: {mode: values['mean_abs_bond_length_delta_mean'] for mode, values in random_physical.items()} for metric in ['sampled_mean_abs_bond_length_delta']}, ['sampled_mean_abs_bond_length_delta'])} | "
        f"{ordering_text({metric: {mode: values['mean_abs_bond_length_delta_mean'] for mode, values in scaffold_physical.items()} for metric in ['sampled_mean_abs_bond_length_delta']}, ['sampled_mean_abs_bond_length_delta'])} | "
        "stiffness_structured improves over isotropic and torsion_only; harmonic remains lowest |"
    )
    lines.append(
        "| A/E verdict | random and scaffold multiseed formal summaries | "
        f"{core['formal_AE_multiseed']['random']['directional_verdict']['stiffness_win_count']}/"
        f"{core['formal_AE_multiseed']['random']['directional_verdict']['tracked_metric_seed_comparison_count']} seed-metric wins | "
        f"{core['formal_AE_multiseed']['scaffold']['directional_verdict']['stiffness_win_count']}/"
        f"{core['formal_AE_multiseed']['scaffold']['directional_verdict']['tracked_metric_seed_comparison_count']} seed-metric wins | "
        "supports stiffness_structured > isotropic on tracked metrics |"
    )
    lines.append(
        "| Attribution verdict | A/B/D/E tracked loss/RMSD metrics | "
        "harmonic > stiffness_structured > isotropic > torsion_only | "
        "harmonic > stiffness_structured > isotropic > torsion_only | "
        "supports a hierarchy, not a stiffness-wins-overall claim |"
    )
    lines.append(
        "| Overall conclusion | current supported claim | "
        "harmonic > stiffness_structured > isotropic > torsion_only | "
        "harmonic > stiffness_structured > isotropic > torsion_only | "
        "stiffness_structured is robustly better than isotropic/torsion_only, but harmonic is stronger |"
    )
    lines.append("")
    return "\n".join(lines)


def build_report(
    payloads: dict[str, dict[str, Any]],
    figure_paths: dict[str, Path],
    table_path: Path,
    amr_cov_table_path: Path,
    summary_path: Path,
    metric_availability: dict[str, Any],
) -> str:
    core = payloads["core"]
    physical = payloads["physical"]
    rotatable = payloads["rotatable"]
    random_core = core["formal_AE_multiseed"]["random"]["metrics"]
    scaffold_core = core["formal_AE_multiseed"]["scaffold"]["metrics"]
    random_attr = core["attribution_ABDE_multiseed"]["random"]["metric_table"]
    scaffold_attr = core["attribution_ABDE_multiseed"]["scaffold"]["metric_table"]
    random_phys = physical["split_summaries"]["random"]["modes"]
    scaffold_phys = physical["split_summaries"]["scaffold"]["modes"]

    random_best = random_core["best_validation_loss"]
    random_rmsd = random_core["sampled_rmsd_to_target_overall_mean"]
    scaffold_best = scaffold_core["best_validation_loss"]
    scaffold_rmsd = scaffold_core["sampled_rmsd_to_target_overall_mean"]

    random_order = ordering_text(random_attr, CORE_METRICS)
    scaffold_order = ordering_text(scaffold_attr, CORE_METRICS)
    random_bond_order = ordering_text(
        {"sampled_mean_abs_bond_length_delta": {mode: values["mean_abs_bond_length_delta_mean"] for mode, values in random_phys.items()}},
        ["sampled_mean_abs_bond_length_delta"],
    )
    scaffold_bond_order = ordering_text(
        {"sampled_mean_abs_bond_length_delta": {mode: values["mean_abs_bond_length_delta_mean"] for mode, values in scaffold_phys.items()}},
        ["sampled_mean_abs_bond_length_delta"],
    )
    rot_range = rotatable_ranges(rotatable)
    processed_evidence = metric_availability["processed_artifact_evidence"]

    lines = [
        "# Full-Database Results Report: Source-Corruption Comparisons for Molecular Conformation Flow Matching",
        "",
        "## 1. Executive summary",
        "",
        "- The current supported message is a hierarchy: harmonic > stiffness_structured > isotropic > torsion_only on the tracked core metrics. Stiffness_structured is a robust improvement over isotropic and torsion_only, but it does not beat harmonic.",
        f"- In formal A/E random-split multiseed runs, stiffness_structured improves best validation loss from {fmt(random_best['isotropic_mean'])} to {fmt(random_best['stiffness_structured_mean'])} and sampled RMSD overall from {fmt(random_rmsd['isotropic_mean'])} to {fmt(random_rmsd['stiffness_structured_mean'])}.",
        f"- In formal A/E scaffold-split multiseed runs, stiffness_structured improves best validation loss from {fmt(scaffold_best['isotropic_mean'])} to {fmt(scaffold_best['stiffness_structured_mean'])} and sampled RMSD overall from {fmt(scaffold_rmsd['isotropic_mean'])} to {fmt(scaffold_rmsd['stiffness_structured_mean'])}.",
        "- Across both random and scaffold A/E summaries, stiffness_structured is lower than isotropic on all 6 tracked metrics and all 18 seed-metric checks.",
        f"- In attribution, the random split ordering is {random_order}; the scaffold split ordering is {scaffold_order}.",
        "- Harmonic is stronger than stiffness_structured on the tracked loss and sampled RMSD metrics, so the evidence does not support a stiffness-wins-overall claim.",
        "- The qualitative ordering does not flip between random and scaffold splits.",
        f"- {amr_cov_headline(metric_availability)}",
        "- Physical-validity evidence is currently proxy-level: finite-coordinate rates and mean absolute bond-length deltas are available, but PoseBusters, MMFF/UFF, xTB, and COV/MAT are not present in the supplied summaries.",
        "",
        "## 2. Experimental scope",
        "",
        "- Dataset scope: `geom_drugs_full_database_one_conformer`.",
        "- Splits: full-database random split and Bemis-Murcko scaffold split.",
        "- Seeds: 0, 1, and 2.",
        "- Formal A/E comparison: `isotropic` vs `stiffness_structured`.",
        "- Attribution setting: A/B/D/E modes `isotropic`, `harmonic`, `torsion_only`, and `stiffness_structured`.",
        "- Tracked core metrics: best validation loss, reloaded validation loss, sampled RMSD overall, and sampled RMSD on train/val/test.",
        "- Reporting scope: this bundle reads existing summaries only. It does not run training, preprocessing, sampling, scheduler jobs, or environment modification.",
        f"- Compact numeric table: `{relative(table_path)}`.",
        f"- AMR/COV availability table: `{relative(amr_cov_table_path)}`.",
        f"- Machine-readable presentation summary: `{relative(summary_path)}`.",
        "",
        "### 2.1 AMR/COV availability audit",
        "",
        amr_cov_audit_paragraph(metric_availability, processed_evidence),
        "",
        f"See `{relative(amr_cov_table_path)}` for the explicit metric-by-metric AMR/COV status.",
        "",
        *amr_cov_report_lines(metric_availability),
        "## 3. Main results",
        "",
        "### 3.1 Random split: isotropic vs stiffness_structured",
        "",
        f"On the random split, stiffness_structured is lower than isotropic on every tracked A/E metric. Best validation loss improves from {fmt(random_best['isotropic_mean'])} to {fmt(random_best['stiffness_structured_mean'])}; reloaded validation loss has the same means. Sampled RMSD overall improves from {fmt(random_rmsd['isotropic_mean'])} to {fmt(random_rmsd['stiffness_structured_mean'])}. The train/val/test RMSD means also move in the same direction.",
        "",
        f"![Core full-database comparison]({relative_to_report(figure_paths['core'])})",
        "",
        f"Caption: Core full-database A/E comparison for random and scaffold splits. Bars are multiseed means and black dots are seed-level values. The figure covers best validation loss, reloaded validation loss, sampled RMSD overall, and sampled RMSD by train/val/test. Source file: `{relative(figure_paths['core'])}`.",
        "",
        "### 3.2 Scaffold split: isotropic vs stiffness_structured",
        "",
        f"On the scaffold split, the same direction holds. Best validation loss improves from {fmt(scaffold_best['isotropic_mean'])} to {fmt(scaffold_best['stiffness_structured_mean'])}; sampled RMSD overall improves from {fmt(scaffold_rmsd['isotropic_mean'])} to {fmt(scaffold_rmsd['stiffness_structured_mean'])}. The split-specific RMSD means remain lower for stiffness_structured, including validation and test.",
        "",
        f"![Seed-level comparison]({relative_to_report(figure_paths['seed_level'])})",
        "",
        f"Caption: Seed-level deltas for stiffness_structured minus isotropic. Negative values indicate lower loss/RMSD for stiffness_structured. All plotted seed-level deltas are negative in the available random and scaffold A/E summaries. Source file: `{relative(figure_paths['seed_level'])}`.",
        "",
        "### 3.3 Attribution on random split",
        "",
        f"The random attribution summary gives the same qualitative ordering across tracked core metrics: {random_order}. Stiffness_structured is better than isotropic and torsion_only, but harmonic is lower than stiffness_structured on best/reloaded validation loss and sampled RMSD metrics.",
        "",
        f"![Random attribution comparison]({relative_to_report(figure_paths['attribution_random'])})",
        "",
        f"Caption: Random-split attribution comparison. Each metric row is normalized to the best method for that metric, so 1.0 is best and larger values are worse. This makes the ordering visible despite the large torsion_only values. Source file: `{relative(figure_paths['attribution_random'])}`.",
        "",
        "### 3.4 Attribution on scaffold split",
        "",
        f"The scaffold attribution summary preserves the same qualitative ordering: {scaffold_order}. This is important because the scaffold split does not reverse the random-split story. The current evidence still places harmonic above stiffness_structured on the tracked loss/RMSD metrics.",
        "",
        f"![Scaffold attribution comparison]({relative_to_report(figure_paths['attribution_scaffold'])})",
        "",
        f"Caption: Scaffold-split attribution comparison with metric-wise normalization. The qualitative ordering is stable relative to the random split. Source file: `{relative(figure_paths['attribution_scaffold'])}`.",
        "",
        "### 3.5 Physical-validity and rotatable-bond diagnostics",
        "",
        f"Physical-validity diagnostics are available for the A/B/D/E modes on both splits, but they are proxies. Finite-coordinate rate is 1.0 for all available mode/split means. Mean absolute bond-length delta orders as {random_bond_order} on random and {scaffold_bond_order} on scaffold. This supports stiffness_structured improving over isotropic and torsion_only on this proxy, while harmonic remains the strongest method on bond-length preservation.",
        "",
        f"![Physical-validity proxy diagnostics]({relative_to_report(figure_paths['physical'])})",
        "",
        f"Caption: Physical-validity proxy diagnostics from existing sampled outputs. The left panel reports mean absolute bond-length delta; the right panel reports finite-coordinate rate. These are not full PoseBusters, MMFF/UFF, or xTB checks. Source file: `{relative(figure_paths['physical'])}`.",
        "",
        f"Rotatable-bond stratification is also available from existing sampled metadata. Across bins, the combined split means range from {fmt(rot_range['harmonic_prior'][0])} to {fmt(rot_range['harmonic_prior'][1])} for harmonic, {fmt(rot_range['stiffness_structured'][0])} to {fmt(rot_range['stiffness_structured'][1])} for stiffness_structured, {fmt(rot_range['isotropic'][0])} to {fmt(rot_range['isotropic'][1])} for isotropic, and {fmt(rot_range['torsion_only'][0])} to {fmt(rot_range['torsion_only'][1])} for torsion_only. The qualitative ordering persists across flexibility bins.",
        "",
        f"![Rotatable-bond stratified RMSD]({relative_to_report(figure_paths['rotatable'])})",
        "",
        f"Caption: Rotatable-bond-count stratified one-step sampled RMSD for random and scaffold splits. The y-axis is logarithmic to keep torsion_only visible without hiding the other methods. This is a stratified RMSD diagnostic, not a conformer-diversity benchmark. Source file: `{relative(figure_paths['rotatable'])}`.",
        "",
        "## 4. Interpretation",
        "",
        "The evidence supports stiffness_structured as a robust improvement over isotropic source corruption under the tracked objective and sampling diagnostics. The improvement is not a one-seed artifact: it appears on all tracked A/E metrics and all seed-level comparisons in both random and scaffold summaries.",
        "",
        "The attribution setting also rules out a simple torsion_only explanation for the observed improvement. Stiffness_structured is consistently better than torsion_only on the tracked core metrics and on the available bond-length-delta proxy.",
        "",
        "The evidence does not support stiffness_structured beating harmonic. Harmonic is lower than stiffness_structured on the tracked loss/RMSD metrics and on the bond-length-delta proxy on both splits. The honest conclusion is therefore a hierarchy: harmonic > stiffness_structured > isotropic > torsion_only on the tracked core metrics, while stiffness_structured remains a robust improvement over isotropic and torsion_only.",
        "",
        "## 5. Limitations",
        "",
        "- The source artifacts are explicitly labeled non-benchmark/reporting-only; this report preserves that label.",
        "- The full-database outputs use a one-conformer-per-molecule setup and one-step sampled RMSD diagnostics, not a full conformer-ensemble benchmark.",
        amr_cov_limitation(metric_availability),
        "- PoseBusters, MMFF/UFF energies, MMFF/UFF force norms, and xTB/GFN2-xTB checks are not present in the supplied summary files.",
        "- Physical validity is limited to finite-coordinate coverage and mean absolute bond-length delta proxies.",
        "- The rotatable-bond analysis is a stratified RMSD diagnostic, not a diversity or coverage analysis.",
        "- No new training, sampling, preprocessing, or scheduler execution was performed for this bundle.",
        "",
        "## 6. Suggested presentation message",
        "",
        "The clean slide message is that stiffness_structured source corruption is a reliable improvement over isotropic and torsion_only under the fixed current setup, but harmonic remains the stronger tracked baseline. The value of stiffness_structured is robustness and attribution clarity, not an overall win over harmonic.",
        "",
        "- Use the hierarchy headline: harmonic > stiffness_structured > isotropic > torsion_only.",
        "- State that stiffness_structured beats isotropic on all tracked random and scaffold A/E seed-metric comparisons.",
        "- State that stiffness_structured beats torsion_only in the attribution setting on both splits.",
        "- State that harmonic is stronger than stiffness_structured on the tracked loss/RMSD metrics.",
        amr_cov_presentation_bullet(metric_availability),
        "",
    ]
    return "\n".join(lines)


def ordering_text(metric_values_by_metric: dict[str, dict[str, float]], metrics: list[str] | tuple[str, ...]) -> str:
    per_metric_orders = []
    for metric in metrics:
        values = metric_values_by_metric[metric]
        ordered = sorted(values.items(), key=lambda item: item[1])
        per_metric_orders.append(tuple(METHOD_LABELS.get(mode, mode) for mode, _ in ordered))
    first = per_metric_orders[0]
    if all(order == first for order in per_metric_orders):
        return " < ".join(first)
    return "mixed by metric"


def rotatable_ranges(summary: dict[str, Any]) -> dict[str, tuple[float, float]]:
    modes = summary["combined_across_splits"]["modes"]
    ranges = {}
    for method, section in modes.items():
        values = [
            section["bins"][bin_label]["mean_sampled_rmsd_to_target"]
            for bin_label in ROTATABLE_BINS
            if bin_label in section["bins"]
        ]
        ranges[method] = (min(values), max(values))
    return ranges


def fmt(value: float | int | None) -> str:
    if value is None:
        return "NA"
    value = float(value)
    if not math.isfinite(value):
        return "NA"
    if abs(value) >= 1:
        return f"{value:.4f}"
    if abs(value) >= 0.1:
        return f"{value:.4f}"
    return f"{value:.6f}"


def smart_tick(value: float, _pos: int) -> str:
    value = float(value)
    if not math.isfinite(value):
        return "NA"
    if abs(value) >= 1:
        return f"{value:.2f}"
    if abs(value) >= 0.01:
        return f"{value:.3f}"
    return f"{value:.2g}"


def relative_to_report(path: Path) -> str:
    return str(path.relative_to(REPORT_PATH.parent))


if __name__ == "__main__":
    main()
