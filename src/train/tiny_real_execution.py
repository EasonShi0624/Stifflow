"""Tiny real-data CPU model execution smoke tests without training."""

from __future__ import annotations

import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

from src.models.smoke_forward import ParameterFreeVelocitySmokeModel, evaluate_velocity_forward_loss
from src.train import tiny_real as tiny_real_data


DEFAULT_PROCESSED_ARTIFACT = tiny_real_data.DEFAULT_PROCESSED_ARTIFACT
DEFAULT_SPLIT_PATH = tiny_real_data.DEFAULT_SPLIT_PATH
DEFAULT_MODES = tiny_real_data.DEFAULT_MODES
RESULT_KIND = "tiny_real_data_model_execution_smoke_not_benchmark"
EXECUTION_MODE = "local_cpu_tiny_real_model_execution_smoke"


def run_tiny_real_execution_smoke_tests(
    processed_artifact_path: str | Path = DEFAULT_PROCESSED_ARTIFACT,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    output_root: str | Path = "experiments",
    seed: int = 0,
    max_molecules: int = 3,
    modes: tuple[str, ...] = DEFAULT_MODES,
) -> dict[str, dict[str, Path]]:
    outputs = {}
    for mode in modes:
        outputs[mode] = run_tiny_real_execution_smoke(
            processed_artifact_path=processed_artifact_path,
            split_path=split_path,
            output_root=output_root,
            seed=seed,
            max_molecules=max_molecules,
            corruption_mode=mode,
        )
    return outputs


def run_tiny_real_execution_smoke(
    processed_artifact_path: str | Path = DEFAULT_PROCESSED_ARTIFACT,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    output_root: str | Path = "experiments",
    seed: int = 0,
    max_molecules: int = 3,
    corruption_mode: str = "isotropic",
) -> dict[str, Path]:
    start = time.time()
    processed_artifact_path = Path(processed_artifact_path)
    split_path = Path(split_path)
    output_root = Path(output_root)
    run_name = _run_name(corruption_mode, seed)
    manifest_path = output_root / "manifests" / f"{run_name}.json"
    results_path = output_root / "results" / f"{run_name}.json"
    log_path = output_root / "logs" / f"{run_name}.log"

    artifact = tiny_real_data._load_tiny_real_artifact(processed_artifact_path)
    records = tiny_real_data._select_records(artifact["molecules"], max_molecules)
    split = tiny_real_data._load_tiny_real_split(split_path)
    split_membership = tiny_real_data._validate_split_coverage(split, records)
    config = tiny_real_data._tiny_real_config(
        corruption_mode=corruption_mode,
        seed=seed,
        output_root=output_root,
        processed_artifact_path=processed_artifact_path,
        split_path=split_path,
        max_molecules=max_molecules,
    )
    config["experiment"]["name"] = run_name
    config["data"]["subset"]["tag"] = "tiny_real_data_model_execution_smoke"
    config["train"] = {
        "enabled": False,
        "dry_run_only": True,
        "max_steps": 0,
        "optimizer_created": False,
        "backward_executed": False,
    }
    config["model"]["execution_smoke_model"] = ParameterFreeVelocitySmokeModel.name

    samples = tiny_real_data._build_flow_matching_samples(
        config,
        records,
        random.Random(seed),
        split_membership,
    )
    model = ParameterFreeVelocitySmokeModel()
    metrics = evaluate_velocity_forward_loss(samples, model=model)
    elapsed = time.time() - start
    split_counts = {key: len(split[key]) for key in ["train", "val", "test"]}

    common = {
        "run_name": run_name,
        "dataset_name": "geom_drugs",
        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
        "processed_artifact_path": str(processed_artifact_path),
        "split_path": str(split_path),
        "random_seed": seed,
        "corruption_mode": corruption_mode,
        "model_name": model.name,
        "backbone": config["model"]["backbone"],
        "execution_mode": EXECUTION_MODE,
        "result_kind": RESULT_KIND,
        "benchmark_result": False,
        "hardware": "cpu_only",
        "gpu_used": False,
        "scheduler": None,
        "slurm_account": None,
        "dry_run": True,
        "training_executed": False,
        "model_training_started": False,
        "training_loop_executed": False,
        "optimizer_created": False,
        "backward_executed": False,
        "checkpoint_path": None,
        "checkpoint_written": False,
        "sampling_executed": False,
        "forward_executed": True,
        "loss_computed": True,
    }
    manifest = {
        **common,
        "git_commit": _git_commit(),
        "config_path": None,
        "config": config,
        "metrics_output_path": str(results_path),
        "bounds": {
            "max_molecules": max_molecules,
            "input_molecule_count": len(artifact["molecules"]),
            "processed_record_count_used": len(records),
            "conformers_per_molecule_used": 1,
            "forward_pass_count": 1,
        },
        "split": {
            "split_type": split["split_type"],
            "degenerate_tiny_split": split.get("degenerate_tiny_split"),
            "counts": split_counts,
        },
        "notes": [
            "Tiny real-data model execution smoke only.",
            "A parameter-free CPU forward pass and velocity loss were computed once on the bounded three-molecule artifact.",
            "No optimizer, backward pass, training loop, checkpoint, model sampling, GPU execution, SLURM job, raw-data mutation, or environment mutation was used.",
        ],
    }
    results = {
        **common,
        "dataset": {
            "name": "geom_drugs",
            "kind": "geom_drugs_tiny_processed_dry_run",
            "molecule_count": len(records),
            "split_type": split["split_type"],
            "split_counts": split_counts,
            "degenerate_tiny_split": split.get("degenerate_tiny_split"),
        },
        "flow_matching": config["flow_matching"],
        "sample_count": len(samples),
        "metrics": {
            "metric_scope": "geom_drugs_tiny_real_model_execution_smoke_forward_loss",
            **metrics,
        },
        "runtime_seconds": elapsed,
        "notes": [
            "This is a tiny real-data CPU model execution smoke on a bounded three-molecule GEOM artifact.",
            "The loss validates forward/loss plumbing only and is not a benchmark result.",
            "No model training, optimizer step, backward pass, checkpoint, model sampling, GPU execution, SLURM job, raw-data modification, or environment mutation was used.",
        ],
    }

    for directory in [manifest_path.parent, results_path.parent, log_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_path.write_text(_log_text(results, manifest_path, results_path), encoding="utf-8")
    return {"manifest": manifest_path, "results": results_path, "log": log_path}


def _run_name(corruption_mode: str, seed: int) -> str:
    if corruption_mode == "isotropic":
        label = "isotropic"
    elif corruption_mode == "stiffness_structured":
        label = "stiffness"
    else:
        raise ValueError(f"Unsupported tiny real execution smoke corruption mode: {corruption_mode}")
    return f"geom_tiny_real_execution_{label}_seed{seed}"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _log_text(results: dict[str, Any], manifest_path: Path, results_path: Path) -> str:
    metrics = results["metrics"]
    return (
        f"run_name={results['run_name']}\n"
        f"corruption_mode={results['corruption_mode']}\n"
        f"execution_mode={results['execution_mode']}\n"
        "training_executed=false\n"
        "model_training_started=false\n"
        "training_loop_executed=false\n"
        "optimizer_created=false\n"
        "backward_executed=false\n"
        "checkpoint_written=false\n"
        "sampling_executed=false\n"
        "forward_executed=true\n"
        "loss_computed=true\n"
        "hardware=cpu_only\n"
        "gpu_used=false\n"
        "scheduler=none\n"
        f"result_kind={RESULT_KIND}\n"
        f"sample_count={results['sample_count']}\n"
        f"mean_velocity_mse={metrics['mean_velocity_mse']}\n"
        f"mean_velocity_mae={metrics['mean_velocity_mae']}\n"
        f"manifest={manifest_path}\n"
        f"results={results_path}\n"
        "note=Tiny real-data model execution smoke only; no scheduler job, GPU use, training loop, checkpoint, or environment mutation.\n"
    )
