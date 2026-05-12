#!/usr/bin/env python
"""Build the final non-benchmark reporting bundle for hybrid lambda COV/AMR results."""

from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
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


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"

LAMBDA_COV_AMR_SUMMARY_PATH = RESULTS_DIR / "geom_full_database_hybrid_lambda_complete_cov_amr_summary.json"
ALL_METHOD_COV_AMR_SUMMARY_PATH = RESULTS_DIR / "geom_full_database_final_cov_amr_complete_valtest_summary.json"
ALL_METHOD_OPTIMIZED_SUMMARY_PATH = RESULTS_DIR / "geom_full_database_final_optimized_all_methods_summary.json"

INTERPRETATION_SUMMARY_PATH = (
    RESULTS_DIR / "geom_full_database_final_hybrid_lambda_cov_amr_interpretation_summary.json"
)
INTERPRETATION_REPORT_PATH = (
    TABLES_DIR / "geom_full_database_final_hybrid_lambda_cov_amr_interpretation_summary.md"
)
CLAIMS_SUMMARY_PATH = RESULTS_DIR / "geom_full_database_final_claims_and_limitations_summary.json"
CLAIMS_REPORT_PATH = TABLES_DIR / "geom_full_database_final_claims_and_limitations_summary.md"
NARRATIVE_PATH = TABLES_DIR / "geom_full_database_final_results_discussion_narrative.md"

TRADEOFF_FIGURE_PATH = FIGURES_DIR / "geom_full_database_hybrid_lambda_cov_amr_tradeoff.png"
PARETO_FIGURE_PATH = FIGURES_DIR / "geom_full_database_hybrid_lambda_precision_recall_pareto.png"
METHOD_COMPARISON_FIGURE_PATH = (
    FIGURES_DIR / "geom_full_database_final_method_comparison_rmsd_cov_amr.png"
)
CLAIM_TABLE_FIGURE_PATH = FIGURES_DIR / "geom_full_database_final_claim_summary_table.png"

LAMBDA_ENTRY_ORDER = (
    "harmonic_prior",
    "hybrid_lambda0025",
    "hybrid_lambda010",
    "hybrid_lambda100",
    "hybrid_lambda300",
    "hybrid_lambda500",
)
LAMBDA_VALUES = {
    "harmonic_prior": 0.0,
    "hybrid_lambda0025": 0.025,
    "hybrid_lambda010": 0.1,
    "hybrid_lambda100": 1.0,
    "hybrid_lambda300": 3.0,
    "hybrid_lambda500": 5.0,
}
LAMBDA_LABELS = {
    "harmonic_prior": "0\nharmonic",
    "hybrid_lambda0025": "0.025",
    "hybrid_lambda010": "0.1",
    "hybrid_lambda100": "1.0",
    "hybrid_lambda300": "3.0",
    "hybrid_lambda500": "5.0",
}
METHOD_ORDER = (
    "harmonic_prior",
    "hybrid_lambda0025",
    "hybrid_lambda500",
    "stiffness_structured",
    "isotropic",
    "torsion_only",
)
METHOD_LABELS = {
    "harmonic_prior": "harmonic",
    "hybrid_lambda0025": "hybrid\nlambda 0.025",
    "hybrid_lambda500": "hybrid\nlambda 5.0",
    "stiffness_structured": "stiffness",
    "isotropic": "isotropic",
    "torsion_only": "torsion\nonly",
}
METHOD_TO_OPTIMIZED_KEY = {
    "harmonic_prior": "harmonic_prior",
    "hybrid_lambda500": "hybrid_harmonic_stiffness_v1_lambda500",
    "stiffness_structured": "stiffness_structured",
    "isotropic": "isotropic",
    "torsion_only": "torsion_only",
}
METRIC_DISPLAY = {
    "cov_r": "COV-R",
    "cov_p": "COV-P",
    "amr_r": "AMR-R",
    "amr_p": "AMR-P",
}
LOWER_IS_BETTER = {"amr_p", "amr_r", "rmsd"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-summary-path", default=str(LAMBDA_COV_AMR_SUMMARY_PATH))
    parser.add_argument("--all-method-cov-amr-summary-path", default=str(ALL_METHOD_COV_AMR_SUMMARY_PATH))
    parser.add_argument("--all-method-optimized-summary-path", default=str(ALL_METHOD_OPTIMIZED_SUMMARY_PATH))
    parser.add_argument("--interpretation-summary-path", default=str(INTERPRETATION_SUMMARY_PATH))
    parser.add_argument("--interpretation-report-path", default=str(INTERPRETATION_REPORT_PATH))
    parser.add_argument("--claims-summary-path", default=str(CLAIMS_SUMMARY_PATH))
    parser.add_argument("--claims-report-path", default=str(CLAIMS_REPORT_PATH))
    parser.add_argument("--narrative-path", default=str(NARRATIVE_PATH))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    set_style()
    lambda_summary = load_json(Path(args.lambda_summary_path))
    all_method_cov_amr = load_json(Path(args.all_method_cov_amr_summary_path))
    all_method_optimized = load_json(Path(args.all_method_optimized_summary_path))

    lambda_points = build_lambda_points(lambda_summary)
    lambda0025_training = aggregate_lambda0025_training_metrics()
    method_comparison = build_method_comparison(
        lambda_summary=lambda_summary,
        all_method_cov_amr=all_method_cov_amr,
        all_method_optimized=all_method_optimized,
        lambda0025_training=lambda0025_training,
    )
    interpretation_summary = build_interpretation_summary(
        lambda_summary=lambda_summary,
        lambda_points=lambda_points,
        method_comparison=method_comparison,
        lambda0025_training=lambda0025_training,
    )
    claims_summary = build_claims_summary(interpretation_summary, method_comparison)

    figures_dir = Path(args.figures_dir)
    figure_paths = {
        "lambda_tradeoff": figures_dir / TRADEOFF_FIGURE_PATH.name,
        "precision_recall_pareto": figures_dir / PARETO_FIGURE_PATH.name,
        "method_comparison": figures_dir / METHOD_COMPARISON_FIGURE_PATH.name,
        "claim_summary_table": figures_dir / CLAIM_TABLE_FIGURE_PATH.name,
    }
    plot_lambda_tradeoff(lambda_points, figure_paths["lambda_tradeoff"])
    plot_pareto(lambda_points, lambda_summary["lambda_selection"], figure_paths["precision_recall_pareto"])
    plot_method_comparison(method_comparison, figure_paths["method_comparison"])
    plot_claim_table(claims_summary["claim_summary_rows"], figure_paths["claim_summary_table"])

    interpretation_summary["figure_paths"] = {
        key: relative(path) for key, path in figure_paths.items()
    }
    claims_summary["figure_paths"] = interpretation_summary["figure_paths"]

    write_json(Path(args.interpretation_summary_path), interpretation_summary)
    write_text(Path(args.interpretation_report_path), interpretation_markdown(interpretation_summary))
    write_json(Path(args.claims_summary_path), claims_summary)
    write_text(Path(args.claims_report_path), claims_markdown(claims_summary))
    write_text(Path(args.narrative_path), narrative_markdown(interpretation_summary, claims_summary))

    print(f"interpretation_summary: {relative(Path(args.interpretation_summary_path))}")
    print(f"interpretation_report: {relative(Path(args.interpretation_report_path))}")
    print(f"claims_summary: {relative(Path(args.claims_summary_path))}")
    print(f"claims_report: {relative(Path(args.claims_report_path))}")
    print(f"narrative: {relative(Path(args.narrative_path))}")
    for label, path in figure_paths.items():
        print(f"{label}_figure: {relative(path)}")
        print(f"{label}_figure_pdf: {relative(path.with_suffix('.pdf'))}")
    print("benchmark_result: false")
    print("note: reporting-only bundle from existing summaries/results; no jobs, training, raw-data mutation, or environment mutation.")


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 240,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def metric_means(summary: dict[str, Any], entry: str) -> dict[str, float]:
    row = summary["aggregate_by_entry"][entry]["metric_summary"]
    return {metric: float(row[metric]["mean"]) for metric in ("cov_r", "cov_p", "amr_r", "amr_p")}


def metric_medians(summary: dict[str, Any], entry: str) -> dict[str, float]:
    row = summary["aggregate_by_entry"][entry]["metric_summary"]
    return {metric: float(row[metric]["median"]) for metric in ("cov_r", "cov_p", "amr_r", "amr_p")}


def build_lambda_points(lambda_summary: dict[str, Any]) -> list[dict[str, Any]]:
    points = []
    for entry in LAMBDA_ENTRY_ORDER:
        points.append(
            {
                "entry_label": entry,
                "lambda_value": LAMBDA_VALUES[entry],
                "plot_label": LAMBDA_LABELS[entry],
                "metrics_mean": metric_means(lambda_summary, entry),
                "metrics_median": metric_medians(lambda_summary, entry),
                "record_count": lambda_summary["aggregate_by_entry"][entry]["record_count"],
                "is_harmonic_reference": entry == "harmonic_prior",
                "is_balanced_or_recall": entry == "hybrid_lambda0025",
                "is_precision_or_rmsd": entry == "hybrid_lambda500",
            }
        )
    return points


def aggregate_lambda0025_training_metrics() -> dict[str, Any]:
    rows = []
    for split in ("random", "scaffold"):
        split_prefix = "geom_formal_full_database" + ("" if split == "random" else "_scaffold")
        for seed in (0, 1, 2):
            path = (
                RESULTS_DIR
                / f"{split_prefix}_train_hybrid_harmonic_stiffness_v1_lambda0025_steps2000_seed{seed}_low_lambda_adamw_lr1e3.json"
            )
            if not path.exists():
                raise FileNotFoundError(f"Missing lambda=0.025 AdamW result needed for RMSD: {path}")
            payload = load_json(path)
            metrics = payload["metrics"]
            rmsd_values: list[float] = []
            split_rmsd: dict[str, list[float]] = {"train": [], "val": [], "test": []}
            for item in metrics.get("sampled_molecules", []):
                value = item.get("sampled_rmsd_to_target")
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    continue
                value_f = float(value)
                rmsd_values.append(value_f)
                item_split = item.get("split")
                if item_split in split_rmsd:
                    split_rmsd[item_split].append(value_f)
            rows.append(
                {
                    "split_type": split,
                    "seed": seed,
                    "source_path": relative(path),
                    "best_validation_loss": scalar(metrics.get("best_validation_loss")),
                    "reloaded_validation_loss": scalar(metrics.get("reloaded_validation_loss")),
                    "sampled_rmsd_to_target_overall_mean": mean_or_none(rmsd_values),
                    "sampled_rmsd_to_target_train_mean": mean_or_none(split_rmsd["train"]),
                    "sampled_rmsd_to_target_val_mean": mean_or_none(split_rmsd["val"]),
                    "sampled_rmsd_to_target_test_mean": mean_or_none(split_rmsd["test"]),
                    "sampled_record_count": len(rmsd_values),
                    "optimizer_name": payload.get("optimizer_name"),
                    "learning_rate": payload.get("learning_rate"),
                    "stop_reason": payload.get("early_stop_reason"),
                    "executed_train_steps": metrics.get("executed_train_steps"),
                }
            )
    return {
        "available_seed_split_count": len(rows),
        "per_seed_split": rows,
        "mean": {
            key: mean_or_none([row[key] for row in rows if row.get(key) is not None])
            for key in (
                "best_validation_loss",
                "reloaded_validation_loss",
                "sampled_rmsd_to_target_overall_mean",
                "sampled_rmsd_to_target_train_mean",
                "sampled_rmsd_to_target_val_mean",
                "sampled_rmsd_to_target_test_mean",
            )
        },
    }


def scalar(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def mean_or_none(values: list[float | None]) -> float | None:
    cleaned = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return mean(cleaned) if cleaned else None


def build_method_comparison(
    lambda_summary: dict[str, Any],
    all_method_cov_amr: dict[str, Any],
    all_method_optimized: dict[str, Any],
    lambda0025_training: dict[str, Any],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    optimized_metrics = all_method_optimized["cross_split_summary"]["aggregate_metrics"]
    for method in METHOD_ORDER:
        if method == "hybrid_lambda0025":
            cov_source = lambda_summary
            cov_entry = method
            training_mean = lambda0025_training["mean"]
        else:
            cov_source = all_method_cov_amr
            cov_entry = method
            training_key = METHOD_TO_OPTIMIZED_KEY[method]
            training_mean = optimized_metrics[training_key]["mean"]
        rows[method] = {
            "display_label": METHOD_LABELS[method].replace("\n", " "),
            "training_metrics": {
                "best_validation_loss": training_mean.get("best_validation_loss"),
                "reloaded_validation_loss": training_mean.get("reloaded_validation_loss"),
                "sampled_rmsd_to_target_overall_mean": training_mean.get(
                    "sampled_rmsd_to_target_overall_mean"
                ),
                "sampled_rmsd_to_target_train_mean": training_mean.get(
                    "sampled_rmsd_to_target_train_mean"
                ),
                "sampled_rmsd_to_target_val_mean": training_mean.get("sampled_rmsd_to_target_val_mean"),
                "sampled_rmsd_to_target_test_mean": training_mean.get(
                    "sampled_rmsd_to_target_test_mean"
                ),
            },
            "cov_amr_metrics_mean": metric_means(cov_source, cov_entry),
            "cov_amr_metrics_median": metric_medians(cov_source, cov_entry),
            "record_count": cov_source["aggregate_by_entry"][cov_entry]["record_count"],
        }
    return {
        "method_order": list(METHOD_ORDER),
        "rows": rows,
        "notes": [
            "lambda=0.025 RMSD values are aggregated directly from its AdamW result JSON sampled_molecules fields.",
            "COV/AMR values for harmonic and lambda=5 are taken from the complete lambda comparison for lambda-specific plots and from the all-method COV/AMR summary for baseline comparison; values match within numerical noise.",
            "torsion_only has strong recall-like AMR-R/COV-R but poor precision and tracked RMSD, so it is interpreted as recall-biased rather than balanced.",
        ],
    }


def build_interpretation_summary(
    lambda_summary: dict[str, Any],
    lambda_points: list[dict[str, Any]],
    method_comparison: dict[str, Any],
    lambda0025_training: dict[str, Any],
) -> dict[str, Any]:
    selection = lambda_summary["lambda_selection"]
    decision = lambda_summary["decision_support"]
    harmonic = next(point for point in lambda_points if point["entry_label"] == "harmonic_prior")
    lambda0025 = next(point for point in lambda_points if point["entry_label"] == "hybrid_lambda0025")
    lambda500 = next(point for point in lambda_points if point["entry_label"] == "hybrid_lambda500")
    return {
        "summary_name": "geom_full_database_final_hybrid_lambda_cov_amr_interpretation_summary",
        "result_kind": "geom_drugs_full_database_final_hybrid_lambda_cov_amr_interpretation_not_benchmark",
        "benchmark_result": False,
        "execution_mode": "reporting_only_from_existing_summaries",
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "scheduler": None,
        "source_paths": {
            "lambda_cov_amr_summary": relative(LAMBDA_COV_AMR_SUMMARY_PATH),
            "all_method_cov_amr_summary": relative(ALL_METHOD_COV_AMR_SUMMARY_PATH),
            "all_method_optimized_summary": relative(ALL_METHOD_OPTIMIZED_SUMMARY_PATH),
        },
        "coverage": lambda_summary["coverage"],
        "lambda_points": lambda_points,
        "lambda_selection": selection,
        "decision_support": decision,
        "key_numeric_findings": {
            "lambda5_precision_optimal_amr_p_mean": lambda500["metrics_mean"]["amr_p"],
            "lambda5_cov_p_mean": lambda500["metrics_mean"]["cov_p"],
            "lambda0025_best_hybrid_cov_r_mean": lambda0025["metrics_mean"]["cov_r"],
            "lambda0025_best_hybrid_amr_r_mean": lambda0025["metrics_mean"]["amr_r"],
            "harmonic_amr_r_mean": harmonic["metrics_mean"]["amr_r"],
            "harmonic_cov_r_mean": harmonic["metrics_mean"]["cov_r"],
            "lambda0025_sampled_rmsd_overall_mean": lambda0025_training["mean"][
                "sampled_rmsd_to_target_overall_mean"
            ],
            "lambda5_sampled_rmsd_overall_mean": method_comparison["rows"]["hybrid_lambda500"][
                "training_metrics"
            ]["sampled_rmsd_to_target_overall_mean"],
            "harmonic_sampled_rmsd_overall_mean": method_comparison["rows"]["harmonic_prior"][
                "training_metrics"
            ]["sampled_rmsd_to_target_overall_mean"],
        },
        "interpretation": {
            "primary_conclusion": "stiffness-modulated harmonic priors create a tunable precision-recall frontier rather than one universally best lambda",
            "lambda5_role": "RMSD/validation-loss and AMR-P precision-favorable; not ensemble-optimal",
            "lambda0025_role": "best hybrid recall/balanced COV/AMR lambda by COV-R, AMR-R, and rank aggregation",
            "harmonic_role": "best AMR-R recall anchor within the harmonic-vs-hybrid lambda comparison",
            "dominance_statement": "No tested hybrid lambda beats harmonic_prior on both precision and recall COV/AMR metrics.",
            "paper_claim_guidance": "Present a precision-recall Pareto frontier; do not claim lambda=5.0 or any single hybrid lambda is universally best.",
        },
        "method_comparison": method_comparison,
    }


def build_claims_summary(
    interpretation_summary: dict[str, Any],
    method_comparison: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        {
            "method_or_lambda": "harmonic prior",
            "validation_rmsd_role": "Strong global-prior baseline; weaker than hybrid lambda=5.0 on optimized loss/RMSD.",
            "precision_role": "AMR-P weaker than all tested hybrid lambdas in the lambda comparison.",
            "recall_role": "Best AMR-R within the harmonic-vs-hybrid lambda comparison.",
            "final_interpretation": "Recall anchor for the harmonic/hybrid family; not dominated by hybrid on ensemble metrics.",
        },
        {
            "method_or_lambda": "hybrid lambda=0.025",
            "validation_rmsd_role": "Low-modulation hybrid; not selected by RMSD/validation loss.",
            "precision_role": "Small AMR-P improvement over harmonic, far weaker than lambda=5.0.",
            "recall_role": "Best hybrid COV-R and AMR-R; balanced by rank aggregation.",
            "final_interpretation": "Best hybrid recall/balanced setting, but not a full dominance result over harmonic.",
        },
        {
            "method_or_lambda": "hybrid lambda=5.0",
            "validation_rmsd_role": "Strongest optimized validation loss and single-sample RMSD among harmonic/hybrid settings.",
            "precision_role": "Precision-optimal by AMR-P; COV-P tied at 1.0.",
            "recall_role": "Worse COV-R and AMR-R than harmonic and low-lambda hybrids.",
            "final_interpretation": "RMSD/precision-favorable setting; not ensemble-optimal.",
        },
        {
            "method_or_lambda": "stiffness structured",
            "validation_rmsd_role": "Better than isotropic/torsion-only on tracked RMSD, weaker than harmonic/hybrid.",
            "precision_role": "Improves AMR-P over harmonic but not as strongly as hybrid lambda=5.0.",
            "recall_role": "Does not beat harmonic on AMR-R in the all-method COV/AMR summary.",
            "final_interpretation": "Local stiffness alone helps precision but does not replace harmonic/global prior.",
        },
        {
            "method_or_lambda": "isotropic",
            "validation_rmsd_role": "Weak optimized baseline on loss/RMSD.",
            "precision_role": "Poor AMR-P relative to harmonic and hybrids.",
            "recall_role": "Not recall-leading in the final comparison.",
            "final_interpretation": "Naive Cartesian source baseline remains weaker.",
        },
        {
            "method_or_lambda": "torsion only",
            "validation_rmsd_role": "Very weak optimized loss/RMSD.",
            "precision_role": "Poor AMR-P and lower COV-P, indicating weak precision.",
            "recall_role": "Recall-biased COV-R/AMR-R behavior, but not balanced.",
            "final_interpretation": "Diversity/recall-favorable but precision-poor; useful as a tradeoff contrast.",
        },
    ]
    return {
        "summary_name": "geom_full_database_final_claims_and_limitations_summary",
        "result_kind": "geom_drugs_full_database_final_claims_and_limitations_not_benchmark",
        "benchmark_result": False,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "scheduler": None,
        "claim_summary_rows": rows,
        "claim_strength": {
            "supported": [
                "hybrid_harmonic_stiffness_v1 lambda=5.0 improves validation loss, single-sample RMSD, and AMR-P precision relative to harmonic_prior.",
                "hybrid_harmonic_stiffness_v1 lambda=0.025 is the best tested hybrid lambda for recall/balanced COV/AMR.",
                "The hybrid lambda sweep forms a precision-recall Pareto frontier.",
            ],
            "not_supported": [
                "A single hybrid lambda is universally best.",
                "lambda=5.0 is ensemble-optimal.",
                "Any tested hybrid lambda dominates harmonic_prior on both precision and recall COV/AMR metrics.",
            ],
            "limitations": [
                "This is still labeled non-benchmark reporting.",
                "COV/AMR evaluates ensemble matching, while the training selection originally used validation loss and single-sample RMSD.",
                "The final narrative must distinguish harmonic-vs-hybrid conclusions from torsion_only's recall-biased but precision-poor behavior.",
            ],
        },
        "method_comparison": method_comparison,
        "interpretation_source": interpretation_summary["summary_name"],
    }


def plot_lambda_tradeoff(lambda_points: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), sharex=True)
    axes = axes.ravel()
    metrics = ("amr_p", "amr_r", "cov_r", "cov_p")
    positions = list(range(len(lambda_points)))
    labels = [point["plot_label"] for point in lambda_points]
    for ax, metric in zip(axes, metrics):
        ys = [point["metrics_mean"][metric] for point in lambda_points]
        ax.plot(positions, ys, color="#2f6f8f", marker="o", linewidth=2.0)
        for pos, point, y in zip(positions, lambda_points, ys):
            if point["entry_label"] == "hybrid_lambda0025":
                ax.scatter(pos, y, s=95, color="#2ca25f", edgecolor="black", linewidth=0.8, zorder=4)
            elif point["entry_label"] == "hybrid_lambda500":
                ax.scatter(pos, y, s=95, color="#e6550d", edgecolor="black", linewidth=0.8, zorder=4)
            elif point["entry_label"] == "harmonic_prior":
                ax.scatter(pos, y, s=90, color="#6b6b6b", marker="*", zorder=4)
        ax.axvline(1, color="#2ca25f", alpha=0.2, linewidth=5)
        ax.axvline(5, color="#e6550d", alpha=0.18, linewidth=5)
        ax.set_title(f"{METRIC_DISPLAY[metric]} mean")
        ax.set_ylabel("lower is better" if metric in LOWER_IS_BETTER else "higher is better")
        if metric == "cov_p":
            ax.set_ylim(0.995, 1.001)
        if metric == "cov_r":
            ax.set_ylim(min(ys) - 0.0002, max(ys) + 0.0002)
    for ax in axes[-2:]:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel("lambda")
    fig.suptitle("Hybrid lambda COV/AMR tradeoff: low lambda preserves recall, high lambda improves precision", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_pareto(lambda_points: list[dict[str, Any]], selection: dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    hybrid_points = [point for point in lambda_points if not point["is_harmonic_reference"]]
    harmonic = next(point for point in lambda_points if point["is_harmonic_reference"])
    plot_specs = (
        ("amr_p", "amr_r", "AMR-P precision (lower)", "AMR-R recall (lower)"),
        ("amr_p", "cov_r", "AMR-P precision (lower)", "COV-R recall (higher)"),
    )
    for ax, (x_metric, y_metric, x_label, y_label) in zip(axes, plot_specs):
        xs = [point["metrics_mean"][x_metric] for point in hybrid_points]
        ys = [point["metrics_mean"][y_metric] for point in hybrid_points]
        ax.plot(xs, ys, color="#2f6f8f", linewidth=2, alpha=0.75, label="hybrid Pareto frontier")
        for point in hybrid_points:
            x = point["metrics_mean"][x_metric]
            y = point["metrics_mean"][y_metric]
            color = "#2ca25f" if point["entry_label"] == "hybrid_lambda0025" else "#e6550d" if point["entry_label"] == "hybrid_lambda500" else "#3182bd"
            ax.scatter(x, y, s=82, color=color, edgecolor="black", linewidth=0.7, zorder=3)
            ax.annotate(str(point["lambda_value"]), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.scatter(
            harmonic["metrics_mean"][x_metric],
            harmonic["metrics_mean"][y_metric],
            color="#6b6b6b",
            marker="*",
            s=160,
            label="harmonic",
            zorder=4,
        )
        ax.annotate(
            "harmonic",
            (harmonic["metrics_mean"][x_metric], harmonic["metrics_mean"][y_metric]),
            textcoords="offset points",
            xytext=(6, -13),
            fontsize=8,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(loc="best", frameon=False)
    axes[0].set_title("Precision vs AMR-R recall")
    axes[1].set_title("Precision vs COV-R recall")
    fig.suptitle("No hybrid lambda dominates harmonic on both precision and recall", y=1.03)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_method_comparison(method_comparison: dict[str, Any], output_path: Path) -> None:
    rows = method_comparison["rows"]
    methods = list(METHOD_ORDER)
    labels = [METHOD_LABELS[method] for method in methods]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    axes = axes.ravel()
    specs = (
        ("sampled_rmsd_to_target_overall_mean", "Sampled RMSD overall", "training_metrics", True),
        ("amr_p", "AMR-P", "cov_amr_metrics_mean", False),
        ("amr_r", "AMR-R", "cov_amr_metrics_mean", False),
        ("cov_r", "COV-R", "cov_amr_metrics_mean", False),
    )
    colors = ["#6b6b6b", "#2ca25f", "#e6550d", "#31a354", "#756bb1", "#de2d26"]
    for ax, (metric, title, section, log_scale) in zip(axes, specs):
        values = [rows[method][section].get(metric) for method in methods]
        bars = ax.bar(range(len(methods)), values, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_title(title)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels)
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("lower is better (log)")
            finite_values = [value for value in values if value is not None and value > 0]
            if finite_values:
                ax.set_ylim(min(finite_values) * 0.75, max(finite_values) * 1.65)
        else:
            ax.set_ylabel("lower is better" if metric in LOWER_IS_BETTER else "higher is better")
            finite_values = [value for value in values if value is not None]
            if finite_values:
                bottom = 0.0 if min(finite_values) >= 0.0 else min(finite_values) * 1.1
                ax.set_ylim(bottom, max(finite_values) * 1.18)
        for bar, value in zip(bars, values):
            if value is None:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                format_number(value),
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7,
            )
    fig.suptitle("Final method comparison: RMSD/precision gains do not imply universal recall dominance", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_claim_table(rows: list[dict[str, str]], output_path: Path) -> None:
    columns = ["Method / lambda", "validation/RMSD role", "precision role", "recall role", "final interpretation"]
    wrap_widths = [18, 34, 32, 32, 34]
    cell_text = [
        [
            textwrap.fill(row["method_or_lambda"], wrap_widths[0]),
            textwrap.fill(row["validation_rmsd_role"], wrap_widths[1]),
            textwrap.fill(row["precision_role"], wrap_widths[2]),
            textwrap.fill(row["recall_role"], wrap_widths[3]),
            textwrap.fill(row["final_interpretation"], wrap_widths[4]),
        ]
        for row in rows
    ]
    fig, ax = plt.subplots(figsize=(18.5, 9.8))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.0, 3.25)
    widths = [0.13, 0.22, 0.2, 0.2, 0.25]
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.PAD = 0.1
        if col < len(widths):
            cell.set_width(widths[col])
        if row == 0:
            cell.set_facecolor("#e9eef3")
            cell.set_text_props(weight="bold")
        elif row in (2, 3):
            cell.set_facecolor("#f7fbf7" if row == 2 else "#fff5eb")
    fig.suptitle("Claim summary: precision-recall frontier, not a single universal winner", y=0.98)
    save_figure(fig, output_path)


def interpretation_markdown(summary: dict[str, Any]) -> str:
    rows = summary["lambda_points"]
    lines = [
        "# Final Hybrid Lambda COV/AMR Interpretation",
        "",
        "This report integrates the complete val/test COV/AMR lambda evaluation. It is non-benchmark reporting.",
        "",
        "## Key Findings",
        "",
        f"- Primary conclusion: {summary['interpretation']['primary_conclusion']}.",
        f"- lambda=5.0 role: {summary['interpretation']['lambda5_role']}.",
        f"- lambda=0.025 role: {summary['interpretation']['lambda0025_role']}.",
        f"- Harmonic role: {summary['interpretation']['harmonic_role']}.",
        f"- Dominance statement: {summary['interpretation']['dominance_statement']}",
        "",
        "## Lambda Metrics",
        "",
        "| Entry | Lambda | COV-R mean | COV-P mean | AMR-R mean | AMR-P mean | Interpretation |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        metrics = row["metrics_mean"]
        if row["entry_label"] == "hybrid_lambda0025":
            role = "balanced / recall-favorable"
        elif row["entry_label"] == "hybrid_lambda500":
            role = "precision / RMSD-favorable"
        elif row["entry_label"] == "harmonic_prior":
            role = "lambda=0 harmonic reference"
        else:
            role = "Pareto frontier point"
        lines.append(
            f"| {row['entry_label']} | {row['lambda_value']} | {format_number(metrics['cov_r'])} | "
            f"{format_number(metrics['cov_p'])} | {format_number(metrics['amr_r'])} | "
            f"{format_number(metrics['amr_p'])} | {role} |"
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Precision-optimal lambda: `{summary['lambda_selection']['precision_optimal_lambda']}`",
            f"- Recall-optimal lambda: `{summary['lambda_selection']['recall_optimal_lambda']}`",
            f"- Balanced lambda: `{summary['lambda_selection']['balanced_lambda']}`",
            f"- Pareto frontier: `{summary['lambda_selection']['pareto_frontier']}`",
            "",
            "## Figures",
            "",
        ]
    )
    for label, path in summary["figure_paths"].items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            f"- `benchmark_result`: `{summary['benchmark_result']}`",
            f"- `training_executed_by_summary`: `{summary['training_executed_by_summary']}`",
            f"- `raw_data_modified`: `{summary['raw_data_modified']}`",
            f"- `environment_modified`: `{summary['environment_modified']}`",
            "",
        ]
    )
    return "\n".join(lines)


def claims_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Final Claims And Limitations",
        "",
        "This summary states the paper/presentation claim boundary implied by the complete COV/AMR lambda evaluation.",
        "",
        "## Claim Summary Table",
        "",
        "| Method / lambda | validation/RMSD role | precision role | recall role | final interpretation |",
        "|---|---|---|---|---|",
    ]
    for row in summary["claim_summary_rows"]:
        lines.append(
            f"| {row['method_or_lambda']} | {row['validation_rmsd_role']} | {row['precision_role']} | "
            f"{row['recall_role']} | {row['final_interpretation']} |"
        )
    lines.extend(["", "## Supported Claims", ""])
    for item in summary["claim_strength"]["supported"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Not Supported", ""])
    for item in summary["claim_strength"]["not_supported"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Limitations", ""])
    for item in summary["claim_strength"]["limitations"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            f"- `benchmark_result`: `{summary['benchmark_result']}`",
            f"- `training_executed_by_summary`: `{summary['training_executed_by_summary']}`",
            f"- `raw_data_modified`: `{summary['raw_data_modified']}`",
            f"- `environment_modified`: `{summary['environment_modified']}`",
            "",
        ]
    )
    return "\n".join(lines)


def narrative_markdown(interpretation: dict[str, Any], claims: dict[str, Any]) -> str:
    key = interpretation["key_numeric_findings"]
    lines = [
        "# Concise Results And Discussion Narrative",
        "",
        "## Results",
        "",
        (
            "Under the optimized AdamW lr=1e-3 protocol, the high-modulation hybrid "
            "`hybrid_harmonic_stiffness_v1` with `lambda=5.0` remains the strongest setting "
            "on validation loss and single-sample RMSD among the harmonic/hybrid source designs. "
            f"Its cross-split sampled RMSD mean is {format_number(key['lambda5_sampled_rmsd_overall_mean'])}, "
            f"compared with {format_number(key['harmonic_sampled_rmsd_overall_mean'])} for harmonic_prior."
        ),
        "",
        (
            "The complete val/test COV/AMR lambda evaluation changes the interpretation. "
            f"`lambda=5.0` is precision-optimal, with AMR-P mean {format_number(key['lambda5_precision_optimal_amr_p_mean'])} "
            "and COV-P tied at 1.0, but it is not ensemble-optimal because its COV-R and AMR-R are worse than lower-lambda settings."
        ),
        "",
        (
            f"The best hybrid recall/balanced setting is `lambda=0.025`, with COV-R mean "
            f"{format_number(key['lambda0025_best_hybrid_cov_r_mean'])} and AMR-R mean "
            f"{format_number(key['lambda0025_best_hybrid_amr_r_mean'])}. Harmonic_prior remains the "
            f"AMR-R recall anchor within the harmonic-vs-hybrid lambda comparison at "
            f"{format_number(key['harmonic_amr_r_mean'])}. No tested hybrid lambda beats harmonic_prior "
            "on both precision and recall COV/AMR metrics."
        ),
        "",
        "## Discussion",
        "",
        (
            "The honest conclusion is not that one hybrid lambda is universally best. "
            "Instead, stiffness modulation turns the harmonic prior into a tunable precision-recall frontier: "
            "larger lambda improves AMR-P precision, while smaller lambda preserves more recall by COV-R/AMR-R."
        ),
        "",
        (
            "For presentation and paper framing, `lambda=5.0` should be described as RMSD/precision-favorable, "
            "not COV/AMR-optimal. `lambda=0.025` should be described as the best tested hybrid for recall/balanced "
            "ensemble metrics. The final claim should emphasize controllable tradeoffs and should explicitly state "
            "that harmonic_prior is not dominated in ensemble-level COV/AMR."
        ),
        "",
        (
            "The all-method comparison adds context: torsion_only can look recall-favorable on AMR-R/COV-R, "
            "but its precision and tracked RMSD are poor. That supports reporting a Pareto frontier rather than "
            "collapsing the story to a single scalar winner."
        ),
        "",
        "## Recommended Claim Wording",
        "",
        (
            "Stiffness-modulated harmonic source priors improve precision-oriented metrics and create a tunable "
            "precision-recall frontier, but the optimized hybrid does not uniformly dominate harmonic_prior on "
            "ensemble COV/AMR metrics."
        ),
        "",
    ]
    return "\n".join(lines)


def format_number(value: Any) -> str:
    if value is None:
        return "NA"
    value_f = float(value)
    if abs(value_f) >= 100:
        return f"{value_f:.1f}"
    if abs(value_f) >= 10:
        return f"{value_f:.3f}"
    if abs(value_f) >= 1:
        return f"{value_f:.6f}"
    return f"{value_f:.6f}"


if __name__ == "__main__":
    main()
