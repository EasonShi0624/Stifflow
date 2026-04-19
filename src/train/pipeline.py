"""Local CPU-only smoke pipeline orchestration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.train.smoke import run_smoke
from src.utils.config import load_config


DEFAULT_SMOKE_CONFIGS = [
    Path("configs/train/tiny_smoke.yaml"),
    Path("configs/train/tiny_smoke_stiffness.yaml"),
]


def run_smoke_pipeline(
    config_paths: list[str | Path] | None = None,
    output_root: str | Path | None = None,
    pipeline_name: str = "smoke_pair_synthetic_seed0",
) -> dict[str, Path]:
    start = time.time()
    paths = [Path(path) for path in (config_paths or DEFAULT_SMOKE_CONFIGS)]
    if len(paths) < 2:
        raise ValueError("Smoke pipeline expects at least two configs to compare wiring paths")

    resolved_output_root = Path(output_root) if output_root is not None else _default_output_root(paths[0])
    run_outputs = []
    run_results = []

    for config_path in paths:
        run_paths = run_smoke(
            config_path,
            dry_run=True,
            output_root_override=resolved_output_root,
        )
        result = json.loads(run_paths["results"].read_text(encoding="utf-8"))
        manifest = json.loads(run_paths["manifest"].read_text(encoding="utf-8"))
        _validate_smoke_only_result(result, manifest)
        run_outputs.append({key: str(value) for key, value in run_paths.items()})
        run_results.append(result)

    pipeline_manifest_path = resolved_output_root / "manifests" / f"{pipeline_name}.json"
    pipeline_results_path = resolved_output_root / "results" / f"{pipeline_name}.json"
    pipeline_log_path = resolved_output_root / "logs" / f"{pipeline_name}.log"

    pipeline_manifest = {
        "pipeline_name": pipeline_name,
        "config_paths": [str(path) for path in paths],
        "output_root": str(resolved_output_root),
        "execution_mode": "local_smoke_pipeline",
        "result_kind": "smoke_only_not_benchmark",
        "benchmark_result": False,
        "hardware": "cpu_only",
        "scheduler": None,
        "slurm_account": None,
        "training_executed": False,
        "run_outputs": run_outputs,
        "notes": [
            "Runs both synthetic smoke configs locally.",
            "No neural model training, GPU execution, SLURM job, or benchmark evaluation was run.",
        ],
    }

    pipeline_results = {
        "pipeline_name": pipeline_name,
        "execution_mode": "local_smoke_pipeline",
        "result_kind": "smoke_only_not_benchmark",
        "benchmark_result": False,
        "hardware": "cpu_only",
        "scheduler": None,
        "slurm_account": None,
        "training_executed": False,
        "dataset_kind": "synthetic_smoke",
        "run_count": len(run_results),
        "runs": _summarize_runs(run_results),
        "runtime_seconds": time.time() - start,
        "notes": [
            "Aggregate smoke-only wiring report.",
            "Values compare corrupted source x0 to target x1 on a synthetic fixture; they are not conformer-generation benchmark metrics.",
        ],
    }

    for directory in [pipeline_manifest_path.parent, pipeline_results_path.parent, pipeline_log_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    pipeline_manifest_path.write_text(
        json.dumps(pipeline_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pipeline_results_path.write_text(
        json.dumps(pipeline_results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pipeline_log_path.write_text(_pipeline_log_text(pipeline_results), encoding="utf-8")

    return {
        "manifest": pipeline_manifest_path,
        "results": pipeline_results_path,
        "log": pipeline_log_path,
    }


def _default_output_root(config_path: Path) -> Path:
    config = load_config(config_path)
    return Path(config["experiment"]["output_root"])


def _validate_smoke_only_result(result: dict[str, Any], manifest: dict[str, Any]) -> None:
    checks = {
        "result_kind": result.get("result_kind") == "smoke_only_not_benchmark",
        "benchmark_result": result.get("benchmark_result") is False,
        "training_executed": result.get("training_executed") is False,
        "hardware": result.get("hardware") == "cpu_only",
        "slurm_account": result.get("slurm_account") is None,
        "manifest_result_kind": manifest.get("result_kind") == "smoke_only_not_benchmark",
        "manifest_training": manifest.get("training_executed") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Smoke run output failed smoke-only validation: {', '.join(failed)}")


def _summarize_runs(run_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for result in run_results:
        metrics = result["metrics"]
        summaries.append(
            {
                "run_name": result["run_name"],
                "corruption_mode": result["corruption_mode"],
                "backbone": result["backbone"],
                "sample_count": result["sample_count"],
                "dataset": result["dataset"],
                "cov_r_smoke": metrics["cov_mat"]["cov_r_smoke"],
                "mat_r_smoke": metrics["cov_mat"]["mat_r_smoke"],
                "finite_coordinate_rate": metrics["validity"]["finite_coordinate_rate"],
                "mean_abs_bond_length_delta": metrics["validity"]["mean_abs_bond_length_delta"],
                "result_kind": result["result_kind"],
                "benchmark_result": result["benchmark_result"],
            }
        )
    return summaries


def _pipeline_log_text(pipeline_results: dict[str, Any]) -> str:
    lines = [
        f"pipeline_name={pipeline_results['pipeline_name']}",
        f"execution_mode={pipeline_results['execution_mode']}",
        f"run_count={pipeline_results['run_count']}",
        "training_executed=false",
        "hardware=cpu_only",
        "scheduler=none",
        "result_kind=smoke_only_not_benchmark",
    ]
    for run in pipeline_results["runs"]:
        lines.append(
            f"run={run['run_name']} corruption={run['corruption_mode']} "
            f"mat_r_smoke={run['mat_r_smoke']}"
        )
    return "\n".join(lines) + "\n"
