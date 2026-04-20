"""Aggregate tiny real-data dry-run outputs into a non-benchmark report."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


DEFAULT_RESULT_PATHS = [
    Path("experiments/results/geom_tiny_real_rdkit_etkdg_mmff_seed0.json"),
    Path("experiments/results/geom_tiny_real_model_isotropic_seed0.json"),
    Path("experiments/results/geom_tiny_real_model_stiffness_seed0.json"),
]
SUMMARY_RESULT_KIND = "tiny_real_data_comparison_dry_run_not_benchmark"
RDKIT_RESULT_KIND = "tiny_real_data_dry_run_not_benchmark"
MODEL_RESULT_KIND = "tiny_real_data_model_dry_run_not_benchmark"


def write_tiny_real_comparison(
    result_paths: list[str | Path] | None = None,
    summary_path: str | Path = "experiments/results/geom_tiny_real_dry_run_summary.json",
    report_path: str | Path = "reports/tables/geom_tiny_real_dry_run_summary.md",
) -> dict[str, Path]:
    start = time.time()
    paths = [Path(path) for path in (result_paths or DEFAULT_RESULT_PATHS)]
    payloads = [_load_tiny_real_result(path) for path in paths]
    rows = [_row_from_result(path, payload) for path, payload in zip(paths, payloads)]

    summary = {
        "summary_name": "geom_tiny_real_dry_run_summary",
        "result_kind": SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_cpu_tiny_real_dry_run_report",
        "hardware": "cpu_only",
        "scheduler": None,
        "slurm_account": None,
        "training_executed": False,
        "model_training_started": False,
        "sampling_executed": False,
        "input_result_paths": [str(path) for path in paths],
        "row_count": len(rows),
        "rows": rows,
        "runtime_seconds": time.time() - start,
        "notes": [
            "This is a tiny real-data dry-run summary over bounded GEOM artifacts.",
            "Rows mix an RDKit baseline dry-run and model-side source-corruption dry-runs; values are not benchmark metrics.",
            "No raw-data modification, model training, model sampling, GPU execution, SLURM job, or environment mutation was used.",
        ],
    }

    summary_path = Path(summary_path)
    report_path = Path(report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def _load_tiny_real_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Tiny real dry-run result does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    result_kind = payload.get("result_kind")
    if result_kind not in {RDKIT_RESULT_KIND, MODEL_RESULT_KIND}:
        raise ValueError(f"Result is not a supported tiny real dry-run output: {path}")
    if payload.get("benchmark_result") is not False:
        raise ValueError(f"Result is not explicitly benchmark_result=false: {path}")
    if payload.get("hardware") != "cpu_only":
        raise ValueError(f"Result is not explicitly hardware=cpu_only: {path}")
    if payload.get("scheduler") is not None:
        raise ValueError(f"Tiny real result unexpectedly records a scheduler: {path}")
    if payload.get("slurm_account") is not None:
        raise ValueError(f"Tiny real result unexpectedly records a SLURM account: {path}")
    if payload.get("training_executed") is not False:
        raise ValueError(f"Tiny real result unexpectedly records training execution: {path}")
    if payload.get("sampling_executed") is not False:
        raise ValueError(f"Tiny real result unexpectedly records model sampling execution: {path}")
    dataset = payload.get("dataset") or {}
    if dataset.get("kind") != "geom_drugs_tiny_processed_dry_run":
        raise ValueError(f"Result is not from the tiny real GEOM dry-run artifact: {path}")
    return payload


def _row_from_result(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("baseline_name") == "rdkit_etkdg_v3_mmff":
        metrics = payload["metrics"]
        return {
            "run_name": payload["run_name"],
            "method": "rdkit_etkdg_v3_mmff",
            "dry_run_role": "baseline_dry_run",
            "source_path": str(path),
            "dataset_kind": payload["dataset"]["kind"],
            "split_counts": payload["dataset"]["split_counts"],
            "sample_or_molecule_count": metrics["attempted_molecule_count"],
            "success_rate": metrics["success_rate"],
            "rmsd_proxy": metrics["mean_aligned_rmsd_to_tiny_real_conformer"],
            "cov_r_dry_run": None,
            "mat_r_dry_run": None,
            "mean_abs_bond_length_delta": metrics["mean_abs_bond_length_delta"],
            "mean_mmff_energy": metrics["mean_mmff_energy"],
            "execution_status": payload.get("execution_status"),
            "rdkit_version": payload.get("rdkit_version"),
            "training_executed": payload.get("training_executed"),
            "sampling_executed": payload.get("sampling_executed"),
            "note": "RDKit ETKDGv3 + MMFF tiny real-data dry-run; not a GEOM benchmark.",
        }

    metrics = payload["metrics"]
    return {
        "run_name": payload["run_name"],
        "method": payload["corruption_mode"],
        "dry_run_role": "model_side_source_dry_run",
        "source_path": str(path),
        "dataset_kind": payload["dataset"]["kind"],
        "split_counts": payload["dataset"]["split_counts"],
        "sample_or_molecule_count": payload["sample_count"],
        "success_rate": metrics["validity"]["finite_coordinate_rate"],
        "rmsd_proxy": metrics["cov_mat"]["mat_r_smoke"],
        "cov_r_dry_run": metrics["cov_mat"]["cov_r_smoke"],
        "mat_r_dry_run": metrics["cov_mat"]["mat_r_smoke"],
        "mean_abs_bond_length_delta": metrics["validity"]["mean_abs_bond_length_delta"],
        "mean_mmff_energy": None,
        "execution_status": "completed",
        "rdkit_version": None,
        "training_executed": payload.get("training_executed"),
        "sampling_executed": payload.get("sampling_executed"),
        "note": "Model-side source-vs-target dry-run; no neural training or model sampling.",
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Tiny Real-Data Dry-Run Summary",
        "",
        "**Scope:** bounded three-molecule GEOM dry-run report. This is not a benchmark result.",
        "",
        f"- `result_kind`: `{summary['result_kind']}`",
        f"- `benchmark_result`: `{str(summary['benchmark_result']).lower()}`",
        f"- `hardware`: `{summary['hardware']}`",
        f"- `training_executed`: `{str(summary['training_executed']).lower()}`",
        f"- `model_training_started`: `{str(summary['model_training_started']).lower()}`",
        f"- `sampling_executed`: `{str(summary['sampling_executed']).lower()}`",
        f"- `slurm_account`: `{summary['slurm_account']}`",
        "",
        "| Method | Role | Run | Count | Success Rate | RMSD Proxy | COV-R Dry-Run | MAT-R Dry-Run | Mean Abs Bond Delta | Mean MMFF Energy | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {method} | {role} | {run_name} | {count} | {success_rate} | {rmsd_proxy} | "
            "{cov} | {mat} | {bond_delta} | {energy} | {status} |".format(
                method=row["method"],
                role=row["dry_run_role"],
                run_name=row["run_name"],
                count=row["sample_or_molecule_count"],
                success_rate=_format_value(row["success_rate"]),
                rmsd_proxy=_format_value(row["rmsd_proxy"]),
                cov=_format_value(row["cov_r_dry_run"]),
                mat=_format_value(row["mat_r_dry_run"]),
                bond_delta=_format_value(row["mean_abs_bond_length_delta"]),
                energy=_format_value(row["mean_mmff_energy"]),
                status=row["execution_status"],
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- The RDKit row summarizes the current tiny real-data ETKDGv3 + MMFF dry-run.",
            "- The isotropic and stiffness rows summarize source-vs-target model-side plumbing dry-runs.",
            "- These values must not be reported as GEOM-Drugs benchmark results.",
            "- No raw data was modified, and no training, model sampling, GPU execution, SLURM job, or environment mutation was run.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
