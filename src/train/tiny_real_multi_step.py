"""Tiny real-data bounded multi-step training smoke tests."""

from __future__ import annotations

import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

from src.models.single_step_smoke import run_bounded_multistep_velocity_training_smoke
from src.train import tiny_real as tiny_real_data


DEFAULT_PROCESSED_ARTIFACT = tiny_real_data.DEFAULT_PROCESSED_ARTIFACT
DEFAULT_SPLIT_PATH = tiny_real_data.DEFAULT_SPLIT_PATH
DEFAULT_MODES = tiny_real_data.DEFAULT_MODES
RESULT_KIND = "tiny_real_data_bounded_multistep_training_smoke_not_benchmark"
EXECUTION_MODE = "local_cpu_tiny_real_bounded_multistep_training_smoke"
DEFAULT_MAX_STEPS = 3
HARD_MAX_STEPS = 5


def run_tiny_real_multi_step_smoke_tests(
    processed_artifact_path: str | Path = DEFAULT_PROCESSED_ARTIFACT,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    output_root: str | Path = "experiments",
    seed: int = 0,
    max_molecules: int = 3,
    max_steps: int = DEFAULT_MAX_STEPS,
    learning_rate: float = 1.0e-2,
    modes: tuple[str, ...] = DEFAULT_MODES,
) -> dict[str, dict[str, Path]]:
    _validate_bounds(max_molecules=max_molecules, max_steps=max_steps)
    outputs = {}
    for mode in modes:
        outputs[mode] = run_tiny_real_multi_step_smoke(
            processed_artifact_path=processed_artifact_path,
            split_path=split_path,
            output_root=output_root,
            seed=seed,
            max_molecules=max_molecules,
            max_steps=max_steps,
            learning_rate=learning_rate,
            corruption_mode=mode,
        )
    return outputs


def run_tiny_real_multi_step_smoke(
    processed_artifact_path: str | Path = DEFAULT_PROCESSED_ARTIFACT,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    output_root: str | Path = "experiments",
    seed: int = 0,
    max_molecules: int = 3,
    max_steps: int = DEFAULT_MAX_STEPS,
    learning_rate: float = 1.0e-2,
    corruption_mode: str = "isotropic",
) -> dict[str, Path]:
    _validate_bounds(max_molecules=max_molecules, max_steps=max_steps)
    start = time.time()
    processed_artifact_path = Path(processed_artifact_path)
    split_path = Path(split_path)
    output_root = Path(output_root)
    run_name = _run_name(corruption_mode, seed, max_steps)
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
    config["data"]["subset"]["tag"] = "tiny_real_data_bounded_multistep_training_smoke"
    config["model"]["multi_step_smoke_model"] = "single_linear_velocity_smoke"
    config["train"] = {
        "enabled": True,
        "smoke_only": True,
        "max_steps": int(max_steps),
        "hard_max_steps": HARD_MAX_STEPS,
        "learning_rate": float(learning_rate),
        "batch_scope": "all_three_tiny_real_molecules",
        "checkpointing_enabled": False,
    }

    samples = tiny_real_data._build_flow_matching_samples(
        config,
        records,
        random.Random(seed),
        split_membership,
    )
    metrics = run_bounded_multistep_velocity_training_smoke(
        samples,
        seed=seed,
        max_steps=max_steps,
        learning_rate=learning_rate,
    )
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
        "model_name": metrics["model_name"],
        "backbone": config["model"]["backbone"],
        "execution_mode": EXECUTION_MODE,
        "result_kind": RESULT_KIND,
        "benchmark_result": False,
        "hardware": "cpu_only",
        "gpu_used": False,
        "scheduler": None,
        "slurm_account": None,
        "dry_run": True,
        "training_executed": True,
        "model_training_started": True,
        "training_loop_executed": True,
        "bounded_training_smoke_executed": True,
        "larger_real_training_run_started": False,
        "max_steps": int(max_steps),
        "hard_max_steps": HARD_MAX_STEPS,
        "optimizer_created": True,
        "optimizer_step_count": int(max_steps),
        "backward_executed": True,
        "backward_pass_count": int(max_steps),
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
            "max_steps": int(max_steps),
            "hard_max_steps": HARD_MAX_STEPS,
            "optimizer_step_count": int(max_steps),
            "backward_pass_count": int(max_steps),
        },
        "split": {
            "split_type": split["split_type"],
            "degenerate_tiny_split": split.get("degenerate_tiny_split"),
            "counts": split_counts,
        },
        "notes": [
            "Tiny real-data bounded multi-step training smoke only.",
            f"Exactly {int(max_steps)} CPU optimizer steps were run on the bounded three-molecule artifact.",
            "No checkpoint, model sampling, GPU execution, SLURM job, raw-data mutation, environment mutation, benchmark evaluation, or larger real training run was used.",
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
            "metric_scope": "geom_drugs_tiny_real_bounded_multistep_training_smoke",
            **metrics,
        },
        "runtime_seconds": elapsed,
        "notes": [
            "This is a tiny real-data CPU bounded multi-step training smoke on a three-molecule GEOM artifact.",
            "Per-step loss and gradient values validate short training plumbing only and are not benchmark results.",
            "No checkpoint, model sampling, GPU execution, SLURM job, raw-data modification, environment mutation, benchmark evaluation, or larger real training run was used.",
        ],
    }

    for directory in [manifest_path.parent, results_path.parent, log_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_path.write_text(_log_text(results, manifest_path, results_path), encoding="utf-8")
    return {"manifest": manifest_path, "results": results_path, "log": log_path}


def _validate_bounds(max_molecules: int, max_steps: int) -> None:
    if max_molecules != 3:
        raise ValueError("Tiny real multi-step smoke must use exactly the existing three-molecule artifact")
    if max_steps <= 1:
        raise ValueError("Tiny real multi-step smoke requires more than one step")
    if max_steps > HARD_MAX_STEPS:
        raise ValueError(f"Tiny real multi-step smoke is capped at {HARD_MAX_STEPS} steps")


def _run_name(corruption_mode: str, seed: int, max_steps: int) -> str:
    if corruption_mode == "isotropic":
        label = "isotropic"
    elif corruption_mode == "stiffness_structured":
        label = "stiffness"
    else:
        raise ValueError(f"Unsupported tiny real multi-step smoke corruption mode: {corruption_mode}")
    return f"geom_tiny_real_multistep_{label}_steps{int(max_steps)}_seed{seed}"


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
        "training_executed=true\n"
        "model_training_started=true\n"
        "training_loop_executed=true\n"
        "bounded_training_smoke_executed=true\n"
        "larger_real_training_run_started=false\n"
        f"max_steps={results['max_steps']}\n"
        f"hard_max_steps={results['hard_max_steps']}\n"
        "optimizer_created=true\n"
        f"optimizer_step_count={results['optimizer_step_count']}\n"
        "backward_executed=true\n"
        f"backward_pass_count={results['backward_pass_count']}\n"
        "checkpoint_written=false\n"
        "sampling_executed=false\n"
        "forward_executed=true\n"
        "loss_computed=true\n"
        "hardware=cpu_only\n"
        "gpu_used=false\n"
        "scheduler=none\n"
        f"result_kind={RESULT_KIND}\n"
        f"sample_count={results['sample_count']}\n"
        f"initial_loss={metrics['initial_loss']}\n"
        f"final_loss={metrics['final_loss']}\n"
        f"step_record_count={len(metrics['step_records'])}\n"
        f"manifest={manifest_path}\n"
        f"results={results_path}\n"
        "note=Tiny real-data bounded multi-step training smoke only; no checkpoint, GPU use, scheduler job, environment mutation, benchmark, or larger real training run.\n"
    )
