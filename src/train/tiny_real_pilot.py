"""Bounded tiny real-data pilot training runs."""

from __future__ import annotations

import copy
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

from src.models.tiny_pilot import run_tiny_velocity_pilot_training
from src.train import tiny_real as tiny_real_data
from src.utils.config import load_config


DEFAULT_CONFIG_PATHS = (
    Path("configs/train/tiny_real_pilot_isotropic.yaml"),
    Path("configs/train/tiny_real_pilot_stiffness.yaml"),
)
DEFAULT_MODES = tiny_real_data.DEFAULT_MODES
RESULT_KIND = "tiny_real_data_bounded_pilot_not_benchmark"
EXECUTION_MODE = "local_cpu_tiny_real_bounded_pilot"
DEFAULT_MAX_STEPS = 6
HARD_MAX_STEPS = 8


def run_tiny_real_pilot_configs(
    config_paths: tuple[str | Path, ...] = DEFAULT_CONFIG_PATHS,
    output_root: str | Path | None = None,
) -> dict[str, dict[str, Path]]:
    outputs = {}
    for config_path in config_paths:
        config = load_config(config_path)
        if output_root is not None:
            config["experiment"]["output_root"] = str(output_root)
        paths = run_tiny_real_pilot(config=config, config_path=Path(config_path))
        outputs[paths["mode"]] = {
            "manifest": paths["manifest"],
            "results": paths["results"],
            "log": paths["log"],
        }
    return outputs


def run_tiny_real_pilot(
    config: dict[str, Any],
    config_path: str | Path | None = None,
) -> dict[str, Path]:
    config = copy.deepcopy(config)
    _validate_pilot_config(config)
    start = time.time()

    config_path_obj = Path(config_path) if config_path is not None else None
    output_root = Path(config["experiment"]["output_root"])
    seed = int(config["experiment"]["seed"])
    corruption_mode = config["corruption"]["mode"]
    max_molecules = int(config["data"]["subset"]["max_molecules"])
    max_steps = int(config["train"]["max_steps"])
    learning_rate = float(config["train"]["learning_rate"])
    processed_artifact_path = Path(config["data"]["processed_artifact_path"])
    split_path = Path(config["data"]["split_path"])
    run_name = _run_name(corruption_mode, seed, max_steps)
    config["experiment"]["name"] = run_name

    manifest_path = output_root / "manifests" / f"{run_name}.json"
    results_path = output_root / "results" / f"{run_name}.json"
    log_path = output_root / "logs" / f"{run_name}.log"

    artifact = tiny_real_data._load_tiny_real_artifact(processed_artifact_path)
    records = tiny_real_data._select_records(artifact["molecules"], max_molecules)
    split = tiny_real_data._load_tiny_real_split(split_path)
    split_membership = tiny_real_data._validate_split_coverage(split, records)
    samples = tiny_real_data._build_flow_matching_samples(
        config,
        records,
        random.Random(seed),
        split_membership,
    )
    metrics = run_tiny_velocity_pilot_training(
        samples=samples,
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
        "pilot_run": True,
        "bounded_pilot": True,
        "dry_run": False,
        "training_executed": True,
        "model_training_started": True,
        "training_loop_executed": True,
        "bounded_real_training_loop_executed": True,
        "larger_real_training_run_started": False,
        "max_steps": max_steps,
        "hard_max_steps": HARD_MAX_STEPS,
        "optimizer_created": True,
        "optimizer_step_count": max_steps,
        "backward_executed": True,
        "backward_pass_count": max_steps,
        "checkpoint_path": None,
        "checkpoint_written": False,
        "sampling_executed": False,
        "forward_executed": True,
        "loss_computed": True,
    }
    manifest = {
        **common,
        "git_commit": _git_commit(),
        "config_path": str(config_path_obj) if config_path_obj is not None else None,
        "config": config,
        "metrics_output_path": str(results_path),
        "bounds": {
            "max_molecules": max_molecules,
            "input_molecule_count": len(artifact["molecules"]),
            "processed_record_count_used": len(records),
            "conformers_per_molecule_used": 1,
            "train_molecule_count": split_counts["train"],
            "val_molecule_count": split_counts["val"],
            "test_molecule_count": split_counts["test"],
            "max_steps": max_steps,
            "hard_max_steps": HARD_MAX_STEPS,
            "optimizer_step_count": max_steps,
            "backward_pass_count": max_steps,
        },
        "split": {
            "split_type": split["split_type"],
            "degenerate_tiny_split": split.get("degenerate_tiny_split"),
            "counts": split_counts,
        },
        "notes": [
            "Bounded tiny real-data pilot only, not a benchmark result.",
            f"Exactly {max_steps} CPU optimizer steps were run on one train molecule from the bounded three-molecule artifact.",
            "Validation and test losses are interface diagnostics only because each split has one molecule.",
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
            "metric_scope": "geom_drugs_tiny_real_bounded_pilot_training",
            **metrics,
        },
        "runtime_seconds": elapsed,
        "notes": [
            "This is a bounded tiny real-data CPU pilot on a three-molecule GEOM artifact.",
            "The loop trains only the one train-split molecule and records val/test losses as tiny interface diagnostics.",
            "These values are not benchmark results and must not be used for scientific comparison.",
            "No checkpoint, model sampling, GPU execution, SLURM job, raw-data modification, environment mutation, benchmark evaluation, or larger real training run was used.",
        ],
    }

    for directory in [manifest_path.parent, results_path.parent, log_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_path.write_text(_log_text(results, manifest_path, results_path), encoding="utf-8")
    return {"mode": corruption_mode, "manifest": manifest_path, "results": results_path, "log": log_path}


def _validate_pilot_config(config: dict[str, Any]) -> None:
    if config["data"]["dataset_name"] != "geom_drugs":
        raise ValueError("Tiny real pilot config must use dataset_name=geom_drugs")
    if config["data"].get("dataset_kind") != "geom_drugs_tiny_processed_dry_run":
        raise ValueError("Tiny real pilot config must use the bounded tiny processed artifact")
    if int(config["data"]["subset"]["max_molecules"]) != 3:
        raise ValueError("Tiny real pilot must use exactly the existing three-molecule artifact")
    if int(config["data"]["subset"].get("max_conformers_per_mol", 0)) != 1:
        raise ValueError("Tiny real pilot must use one conformer per molecule")
    if config["corruption"]["mode"] not in DEFAULT_MODES:
        raise ValueError(f"Unsupported tiny real pilot corruption mode: {config['corruption']['mode']}")
    if config["model"].get("pilot_model") != "single_linear_velocity_pilot":
        raise ValueError("Tiny real pilot config must name pilot_model=single_linear_velocity_pilot")
    if config["train"].get("enabled") is not True:
        raise ValueError("Tiny real pilot config must enable train.enabled")
    if config["train"].get("pilot_only") is not True:
        raise ValueError("Tiny real pilot config must set train.pilot_only=true")
    if config["sample"].get("enabled") is not False:
        raise ValueError("Tiny real pilot must not enable sampling")
    if config["train"].get("checkpointing_enabled") is not False:
        raise ValueError("Tiny real pilot must not enable checkpointing")
    max_steps = int(config["train"]["max_steps"])
    if max_steps <= 1:
        raise ValueError("Tiny real pilot requires more than one optimizer step")
    if max_steps > HARD_MAX_STEPS:
        raise ValueError(f"Tiny real pilot is capped at {HARD_MAX_STEPS} steps")
    if float(config["train"]["learning_rate"]) <= 0.0:
        raise ValueError("Tiny real pilot learning_rate must be positive")


def _run_name(corruption_mode: str, seed: int, max_steps: int) -> str:
    if corruption_mode == "isotropic":
        label = "isotropic"
    elif corruption_mode == "stiffness_structured":
        label = "stiffness"
    else:
        raise ValueError(f"Unsupported tiny real pilot corruption mode: {corruption_mode}")
    return f"geom_tiny_real_pilot_{label}_steps{int(max_steps)}_seed{seed}"


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
        "pilot_run=true\n"
        "bounded_pilot=true\n"
        "training_executed=true\n"
        "model_training_started=true\n"
        "training_loop_executed=true\n"
        "bounded_real_training_loop_executed=true\n"
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
        f"train_initial_loss={metrics['initial_split_losses']['train']}\n"
        f"train_final_loss={metrics['final_split_losses']['train']}\n"
        f"val_final_loss={metrics['final_split_losses']['val']}\n"
        f"test_final_loss={metrics['final_split_losses']['test']}\n"
        f"step_record_count={len(metrics['step_records'])}\n"
        f"manifest={manifest_path}\n"
        f"results={results_path}\n"
        "note=Bounded tiny real-data pilot only; no checkpoint, GPU use, scheduler job, environment mutation, benchmark, or larger real training run.\n"
    )
