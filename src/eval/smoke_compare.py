"""Aggregate existing smoke outputs into a smoke-only comparison report."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


DEFAULT_RESULT_PATHS = [
    Path("experiments/results/smoke_random_isotropic_seed0.json"),
    Path("experiments/results/smoke_random_stiffness_seed0.json"),
    Path("experiments/results/smoke_random_rdkit_etkdg_mmff_seed0.json"),
]


def write_smoke_comparison(
    result_paths: list[str | Path] | None = None,
    summary_path: str | Path = "experiments/results/smoke_comparison_summary.json",
    report_path: str | Path = "reports/tables/smoke_comparison_summary.md",
) -> dict[str, Path]:
    start = time.time()
    paths = [Path(path) for path in (result_paths or DEFAULT_RESULT_PATHS)]
    payloads = [_load_smoke_result(path) for path in paths]
    rows = [_row_from_result(path, payload) for path, payload in zip(paths, payloads)]

    summary = {
        "summary_name": "smoke_comparison_summary",
        "result_kind": "smoke_only_not_benchmark",
        "benchmark_result": False,
        "execution_mode": "local_smoke_report",
        "hardware": "cpu_only",
        "scheduler": None,
        "slurm_account": None,
        "training_executed": False,
        "input_result_paths": [str(path) for path in paths],
        "row_count": len(rows),
        "rows": rows,
        "runtime_seconds": time.time() - start,
        "notes": [
            "This is a smoke-only comparison over synthetic fixture outputs.",
            "Rows mix source-corruption wiring checks and an RDKit fixture baseline; values are not benchmark metrics.",
            "No training, GPU execution, SLURM job, or environment mutation was used.",
        ],
    }

    summary_path = Path(summary_path)
    report_path = Path(report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def _load_smoke_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Smoke result does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("result_kind") != "smoke_only_not_benchmark":
        raise ValueError(f"Result is not labeled smoke_only_not_benchmark: {path}")
    if payload.get("benchmark_result") is not False:
        raise ValueError(f"Result is not explicitly benchmark_result=false: {path}")
    if payload.get("hardware") != "cpu_only":
        raise ValueError(f"Result is not explicitly hardware=cpu_only: {path}")
    if payload.get("slurm_account") is not None:
        raise ValueError(f"Smoke result unexpectedly records a SLURM account: {path}")
    if payload.get("training_executed") is not False:
        raise ValueError(f"Smoke result unexpectedly records training execution: {path}")
    return payload


def _row_from_result(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("baseline_name") == "rdkit_etkdg_v3_mmff":
        metrics = payload["metrics"]
        return {
            "run_name": payload["run_name"],
            "method": "rdkit_etkdg_v3_mmff",
            "source_path": str(path),
            "dataset_kind": payload["dataset"]["kind"],
            "sample_or_molecule_count": metrics["attempted_molecule_count"],
            "success_rate": metrics["success_rate"],
            "rmsd_proxy": metrics["mean_aligned_rmsd_to_fixture"],
            "cov_r_smoke": None,
            "mat_r_smoke": None,
            "mean_abs_bond_length_delta": metrics["mean_abs_bond_length_delta"],
            "mean_mmff_energy": metrics["mean_mmff_energy"],
            "execution_status": payload.get("execution_status"),
            "rdkit_version": payload.get("rdkit_version"),
            "note": "RDKit fixture baseline; not a generated-conformer benchmark.",
        }

    metrics = payload["metrics"]
    return {
        "run_name": payload["run_name"],
        "method": payload["corruption_mode"],
        "source_path": str(path),
        "dataset_kind": payload["dataset"]["kind"],
        "sample_or_molecule_count": payload["sample_count"],
        "success_rate": metrics["validity"]["finite_coordinate_rate"],
        "rmsd_proxy": metrics["cov_mat"]["mat_r_smoke"],
        "cov_r_smoke": metrics["cov_mat"]["cov_r_smoke"],
        "mat_r_smoke": metrics["cov_mat"]["mat_r_smoke"],
        "mean_abs_bond_length_delta": metrics["validity"]["mean_abs_bond_length_delta"],
        "mean_mmff_energy": None,
        "execution_status": "completed",
        "rdkit_version": None,
        "note": "Source-vs-target corruption wiring check; not model generation.",
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Smoke Comparison Summary",
        "",
        "**Scope:** smoke-only synthetic fixture report. This is not a benchmark result.",
        "",
        f"- `result_kind`: `{summary['result_kind']}`",
        f"- `benchmark_result`: `{str(summary['benchmark_result']).lower()}`",
        f"- `hardware`: `{summary['hardware']}`",
        f"- `training_executed`: `{str(summary['training_executed']).lower()}`",
        f"- `slurm_account`: `{summary['slurm_account']}`",
        "",
        "| Method | Run | Count | Success Rate | RMSD Proxy | COV-R Smoke | MAT-R Smoke | Mean Abs Bond Delta | Mean MMFF Energy | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {method} | {run_name} | {count} | {success_rate} | {rmsd_proxy} | "
            "{cov} | {mat} | {bond_delta} | {energy} | {status} |".format(
                method=row["method"],
                run_name=row["run_name"],
                count=row["sample_or_molecule_count"],
                success_rate=_format_value(row["success_rate"]),
                rmsd_proxy=_format_value(row["rmsd_proxy"]),
                cov=_format_value(row["cov_r_smoke"]),
                mat=_format_value(row["mat_r_smoke"]),
                bond_delta=_format_value(row["mean_abs_bond_length_delta"]),
                energy=_format_value(row["mean_mmff_energy"]),
                status=row["execution_status"],
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Isotropic and stiffness rows compare corrupted source `x0` to target `x1` on the synthetic fixture.",
            "- The RDKit row compares ETKDGv3 + MMFF heavy-atom coordinates to the fixture conformer.",
            "- These values are smoke wiring diagnostics only and must not be reported as GEOM-Drugs benchmark results.",
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
