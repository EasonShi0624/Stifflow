"""Tiny real-data checkpoint/reload/sampling smoke tests."""

from __future__ import annotations

import copy
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

from src.models.checkpoint_sampling_smoke import run_checkpoint_sampling_smoke, warm_up_checkpoint_sampling_runtime
from src.train import tiny_real as tiny_real_data
from src.train.tiny_real_multi_step import DEFAULT_MAX_STEPS, HARD_MAX_STEPS
from src.utils.config import load_config


DEFAULT_PROCESSED_ARTIFACT = tiny_real_data.DEFAULT_PROCESSED_ARTIFACT
DEFAULT_SPLIT_PATH = tiny_real_data.DEFAULT_SPLIT_PATH
DEFAULT_MODES = tiny_real_data.DEFAULT_MODES
RESULT_KIND = "tiny_real_data_checkpoint_sampling_smoke_not_benchmark"
EXECUTION_MODE = "local_cpu_tiny_real_checkpoint_sampling_smoke"
DEFAULT_SAMPLE_STEP_SIZE = 0.1
MICRO_PILOT_CONFIG_PATHS = (
    Path("configs/train/micro_pilot_isotropic.yaml"),
    Path("configs/train/micro_pilot_stiffness.yaml"),
)
DATA_SCALE_CHECK_CONFIG_PATHS = (
    Path("configs/train/data_scale_check_isotropic.yaml"),
    Path("configs/train/data_scale_check_stiffness.yaml"),
)
MICRO_PILOT_RESULT_KIND = "tiny_real_data_bounded_micro_pilot_checkpoint_sampling_not_benchmark"
DATA_SCALE_CHECK_RESULT_KIND = "tiny_real_data_bounded_data_scale_check_checkpoint_sampling_not_benchmark"
DATA_SCALE_CHECK_COMPARISON_RESULT_KIND = (
    "tiny_real_data_bounded_data_scale_check_checkpoint_sampling_comparison_not_benchmark"
)
MICRO_PILOT_EXECUTION_MODE = "local_cpu_tiny_real_micro_pilot_checkpoint_sampling"
DATA_SCALE_CHECK_EXECUTION_MODE = "local_cpu_tiny_real_data_scale_check_checkpoint_sampling"
MICRO_PILOT_MAX_MOLECULES = 6
DATA_SCALE_CHECK_MAX_MOLECULES = 30
DEFAULT_MICRO_PILOT_MAX_STEPS = HARD_MAX_STEPS
MICRO_PILOT_CHECKPOINT_SCOPE = "disposable_micro_pilot_checkpoint_sampling_only"
DATA_SCALE_CHECK_CHECKPOINT_SCOPE = "disposable_data_scale_check_checkpoint_sampling_only"
MICRO_PILOT_RUNTIME_SCOPE = "checkpoint_train_save_reload_sample_after_full_cpu_checkpoint_sampling_warmup"
DEFAULT_MICRO_PILOT_CHECKPOINT_SEED_SET = (0, 1, 2)
HARD_MAX_MICRO_PILOT_CHECKPOINT_SEEDS = 3


def run_tiny_real_checkpoint_sampling_smoke_tests(
    processed_artifact_path: str | Path = DEFAULT_PROCESSED_ARTIFACT,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    output_root: str | Path = "experiments",
    seed: int = 0,
    max_molecules: int = 3,
    max_steps: int = DEFAULT_MAX_STEPS,
    learning_rate: float = 1.0e-2,
    sample_step_size: float = DEFAULT_SAMPLE_STEP_SIZE,
    modes: tuple[str, ...] = DEFAULT_MODES,
) -> dict[str, dict[str, Path]]:
    _validate_bounds(max_molecules=max_molecules, max_steps=max_steps, sample_step_size=sample_step_size)
    outputs = {}
    for mode in modes:
        outputs[mode] = run_tiny_real_checkpoint_sampling_smoke(
            processed_artifact_path=processed_artifact_path,
            split_path=split_path,
            output_root=output_root,
            seed=seed,
            max_molecules=max_molecules,
            max_steps=max_steps,
            learning_rate=learning_rate,
            sample_step_size=sample_step_size,
            corruption_mode=mode,
        )
    return outputs


def run_tiny_real_checkpoint_sampling_smoke(
    processed_artifact_path: str | Path = DEFAULT_PROCESSED_ARTIFACT,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    output_root: str | Path = "experiments",
    seed: int = 0,
    max_molecules: int = 3,
    max_steps: int = DEFAULT_MAX_STEPS,
    learning_rate: float = 1.0e-2,
    sample_step_size: float = DEFAULT_SAMPLE_STEP_SIZE,
    corruption_mode: str = "isotropic",
) -> dict[str, Path]:
    _validate_bounds(max_molecules=max_molecules, max_steps=max_steps, sample_step_size=sample_step_size)
    start = time.time()
    processed_artifact_path = Path(processed_artifact_path)
    split_path = Path(split_path)
    output_root = Path(output_root)
    run_name = _run_name(corruption_mode, seed, max_steps)
    manifest_path = output_root / "manifests" / f"{run_name}.json"
    results_path = output_root / "results" / f"{run_name}.json"
    log_path = output_root / "logs" / f"{run_name}.log"
    checkpoint_path = output_root / "checkpoints" / f"{run_name}.pt"

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
    config["data"]["subset"]["tag"] = "tiny_real_data_checkpoint_sampling_smoke"
    config["model"]["checkpoint_sampling_smoke_model"] = "single_linear_velocity_smoke"
    config["train"] = {
        "enabled": True,
        "smoke_only": True,
        "max_steps": int(max_steps),
        "hard_max_steps": HARD_MAX_STEPS,
        "learning_rate": float(learning_rate),
        "batch_scope": "all_three_tiny_real_molecules",
        "checkpointing_enabled": True,
        "checkpoint_scope": "disposable_smoke_only",
    }
    config["sample"] = {
        "enabled": True,
        "smoke_only": True,
        "sample_step_count": 1,
        "sample_step_size": float(sample_step_size),
        "sample_scope": "all_three_tiny_real_molecules",
    }

    samples = tiny_real_data._build_flow_matching_samples(
        config,
        records,
        random.Random(seed),
        split_membership,
    )
    metrics = run_checkpoint_sampling_smoke(
        samples=samples,
        checkpoint_path=checkpoint_path,
        seed=seed,
        max_steps=max_steps,
        learning_rate=learning_rate,
        sample_step_size=sample_step_size,
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
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_written": True,
        "checkpoint_reloaded": True,
        "checkpoint_scope": "disposable_smoke_only",
        "sampling_executed": True,
        "sample_step_count": 1,
        "sample_step_size": float(sample_step_size),
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
            "sample_step_count": 1,
        },
        "split": {
            "split_type": split["split_type"],
            "degenerate_tiny_split": split.get("degenerate_tiny_split"),
            "counts": split_counts,
        },
        "notes": [
            "Tiny real-data checkpoint-and-sampling smoke only.",
            f"A disposable smoke checkpoint was written and reloaded after {int(max_steps)} CPU optimizer steps.",
            "A single bounded sampling step was run for each molecule in the three-molecule artifact.",
            "No raw-data mutation, environment mutation, GPU execution, SLURM job, benchmark evaluation, reusable checkpoint, or larger real training run was used.",
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
            "metric_scope": "geom_drugs_tiny_real_checkpoint_sampling_smoke",
            **metrics,
        },
        "runtime_seconds": elapsed,
        "notes": [
            "This is a tiny real-data CPU checkpoint-and-sampling smoke on a three-molecule GEOM artifact.",
            "Checkpoint reload and sampled-coordinate values validate plumbing only and are not benchmark results.",
            "The checkpoint is disposable smoke output, not a reusable research checkpoint.",
            "No raw-data modification, environment mutation, GPU execution, SLURM job, benchmark evaluation, or larger real training run was used.",
        ],
    }

    for directory in [manifest_path.parent, results_path.parent, log_path.parent, checkpoint_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_path.write_text(_log_text(results, manifest_path, results_path), encoding="utf-8")
    return {"manifest": manifest_path, "results": results_path, "log": log_path, "checkpoint": checkpoint_path}


def run_micro_pilot_checkpoint_sampling_configs(
    config_paths: tuple[str | Path, ...] = MICRO_PILOT_CONFIG_PATHS,
    output_root: str | Path | None = None,
    seed: int = 0,
    max_steps: int = DEFAULT_MICRO_PILOT_MAX_STEPS,
    sample_step_size: float = DEFAULT_SAMPLE_STEP_SIZE,
) -> dict[str, dict[str, Path]]:
    outputs = {}
    for config_path in config_paths:
        config = load_config(config_path)
        if output_root is not None:
            config["experiment"]["output_root"] = str(output_root)
        config["experiment"]["seed"] = int(seed)
        paths = run_micro_pilot_checkpoint_sampling(
            config=config,
            config_path=Path(config_path),
            max_steps=max_steps,
            sample_step_size=sample_step_size,
        )
        outputs[paths["mode"]] = {
            "manifest": paths["manifest"],
            "results": paths["results"],
            "log": paths["log"],
            "checkpoint": paths["checkpoint"],
        }
    return outputs


def run_micro_pilot_checkpoint_sampling_multiseed(
    seed_set: tuple[int, ...] = DEFAULT_MICRO_PILOT_CHECKPOINT_SEED_SET,
    config_paths: tuple[str | Path, ...] = MICRO_PILOT_CONFIG_PATHS,
    output_root: str | Path | None = None,
    max_steps: int = DEFAULT_MICRO_PILOT_MAX_STEPS,
    sample_step_size: float = DEFAULT_SAMPLE_STEP_SIZE,
) -> dict[str, dict[int, dict[str, Path]]]:
    _validate_micro_pilot_checkpoint_seed_set(seed_set)
    outputs: dict[str, dict[int, dict[str, Path]]] = {}
    for seed in seed_set:
        seed_outputs = run_micro_pilot_checkpoint_sampling_configs(
            config_paths=config_paths,
            output_root=output_root,
            seed=int(seed),
            max_steps=max_steps,
            sample_step_size=sample_step_size,
        )
        for mode, paths in seed_outputs.items():
            outputs.setdefault(mode, {})[int(seed)] = paths
    return outputs


def run_data_scale_check_checkpoint_sampling_configs(
    config_paths: tuple[str | Path, ...] = DATA_SCALE_CHECK_CONFIG_PATHS,
    output_root: str | Path | None = None,
    seed: int = 0,
    max_steps: int = DEFAULT_MICRO_PILOT_MAX_STEPS,
    sample_step_size: float = DEFAULT_SAMPLE_STEP_SIZE,
) -> dict[str, dict[str, Path]]:
    return run_micro_pilot_checkpoint_sampling_configs(
        config_paths=config_paths,
        output_root=output_root,
        seed=seed,
        max_steps=max_steps,
        sample_step_size=sample_step_size,
    )


def write_data_scale_check_checkpoint_sampling_comparison(
    result_paths: list[str | Path],
    output_root: str | Path = "experiments",
    seed: int = 0,
    max_molecules: int = DATA_SCALE_CHECK_MAX_MOLECULES,
    max_steps: int = DEFAULT_MICRO_PILOT_MAX_STEPS,
) -> dict[str, Path]:
    output_root = Path(output_root)
    run_name = f"geom_data_scale_check_ckpt_sample_comparison_mols{max_molecules}_steps{max_steps}_seed{seed}"
    manifest_path = output_root / "manifests" / f"{run_name}.json"
    results_path = output_root / "results" / f"{run_name}.json"
    log_path = output_root / "logs" / f"{run_name}.log"
    rows = [_data_scale_check_comparison_row(Path(path)) for path in result_paths]
    by_mode = {row["corruption_mode"]: row for row in rows}
    if set(by_mode) != {"isotropic", "stiffness_structured"}:
        raise ValueError(f"Data-scale-check comparison requires isotropic and stiffness rows, got {sorted(by_mode)}")
    _validate_data_scale_check_rows(rows, seed=seed, max_molecules=max_molecules, max_steps=max_steps)
    metric_comparisons = _data_scale_check_metric_comparisons(by_mode)
    directional_signal_preserved = all(row["stiffness_outcome"] == "better" for row in metric_comparisons)
    common = {
        "run_name": run_name,
        "dataset_name": "geom_drugs",
        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
        "dataset_subset_scope": "data_scale_check_processed_dry_run",
        "result_kind": DATA_SCALE_CHECK_COMPARISON_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_cpu_data_scale_check_checkpoint_sampling_comparison",
        "hardware": "cpu_only",
        "gpu_used": False,
        "scheduler": None,
        "slurm_account": None,
        "pilot_run": True,
        "bounded_pilot": True,
        "micro_pilot_run": False,
        "data_scale_check_run": True,
        "training_executed_by_comparison": False,
        "sampling_executed_by_comparison": False,
        "larger_real_training_run_started": False,
        "raw_data_modified": False,
        "environment_modified": False,
        "input_result_paths": [str(path) for path in result_paths],
        "random_seed": seed,
        "max_molecules": max_molecules,
        "max_steps": max_steps,
    }
    manifest = {
        **common,
        "git_commit": _git_commit(),
        "metrics_output_path": str(results_path),
        "log_path": str(log_path),
        "notes": [
            "Bounded 30-molecule data-scale-check comparison only, not a benchmark result.",
            "The comparison reads two existing bounded pilot outputs and does not run additional training or sampling.",
            "No raw-data mutation, environment mutation, GPU execution, SLURM job, benchmark evaluation, or larger real training run was used.",
        ],
    }
    results = {
        **common,
        "row_count": len(rows),
        "rows": rows,
        "metric_comparisons": metric_comparisons,
        "directional_signal_preserved": directional_signal_preserved,
        "summary": (
            "stiffness_better_on_all_tracked_metrics"
            if directional_signal_preserved
            else "directional_signal_mixed_or_not_preserved"
        ),
        "runtime_note": "Runtime is recorded for accounting only; tiny CPU timing differences are not interpreted as model performance.",
        "notes": [
            "This is a compact bounded data-scale-check comparison, not report-only infrastructure and not a benchmark result.",
            "Lower values are treated as better for reloaded_loss and sampled RMSD diagnostics.",
            "The directional signal is considered preserved only if stiffness is better on reloaded_loss, overall sampled RMSD, and train/val/test sampled RMSD means.",
        ],
    }
    for directory in [manifest_path.parent, results_path.parent, log_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_path.write_text(_data_scale_check_comparison_log(results, manifest_path, results_path), encoding="utf-8")
    return {"manifest": manifest_path, "results": results_path, "log": log_path}


def run_micro_pilot_checkpoint_sampling(
    config: dict[str, Any],
    config_path: str | Path | None = None,
    max_steps: int = DEFAULT_MICRO_PILOT_MAX_STEPS,
    sample_step_size: float = DEFAULT_SAMPLE_STEP_SIZE,
) -> dict[str, Path]:
    config = copy.deepcopy(config)
    config_path_obj = Path(config_path) if config_path is not None else None
    output_root = Path(config["experiment"]["output_root"])
    seed = int(config["experiment"]["seed"])
    corruption_mode = config["corruption"]["mode"]
    max_molecules = int(config["data"]["subset"]["max_molecules"])
    subset_tag = str(config["data"]["subset"].get("tag"))
    scope = _checkpoint_sampling_scope_settings(max_molecules=max_molecules, subset_tag=subset_tag)
    learning_rate = float(config["train"]["learning_rate"])
    processed_artifact_path = Path(config["data"]["processed_artifact_path"])
    split_path = Path(config["data"]["split_path"])
    run_name = _checkpoint_run_name(scope["run_prefix"], corruption_mode, seed, max_molecules, max_steps)
    manifest_path = output_root / "manifests" / f"{run_name}.json"
    results_path = output_root / "results" / f"{run_name}.json"
    log_path = output_root / "logs" / f"{run_name}.log"
    checkpoint_path = output_root / "checkpoints" / f"{run_name}.pt"

    config["experiment"]["name"] = run_name
    config["train"]["max_steps"] = int(max_steps)
    config["train"]["hard_max_steps"] = HARD_MAX_STEPS
    config["train"]["batch_scope"] = scope["batch_scope"]
    config["train"]["checkpointing_enabled"] = True
    config["train"]["checkpoint_scope"] = scope["checkpoint_scope"]
    config["model"]["checkpoint_sampling_smoke_model"] = "single_linear_velocity_smoke"
    config["sample"] = {
        "enabled": True,
        "pilot_only": True,
        "sample_step_count": 1,
        "sample_step_size": float(sample_step_size),
        "sample_scope": scope["sample_scope"],
    }
    _validate_micro_pilot_checkpoint_sampling_config(
        config,
        max_steps=max_steps,
        sample_step_size=sample_step_size,
        scope=scope,
    )

    artifact = tiny_real_data._load_tiny_real_artifact(processed_artifact_path)
    subset_scope = artifact.get("subset_scope") or config["data"].get("subset", {}).get("tag")
    records = tiny_real_data._select_records(artifact["molecules"], max_molecules)
    split = tiny_real_data._load_tiny_real_split(split_path)
    split_membership = tiny_real_data._validate_split_coverage(split, records)
    all_samples = tiny_real_data._build_flow_matching_samples(
        config,
        records,
        random.Random(seed),
        split_membership,
    )
    train_samples = [sample for sample in all_samples if sample["split"] == "train"]
    if not train_samples:
        raise ValueError("Micro-pilot checkpoint/sampling requires a non-empty train split")
    runtime_backend = warm_up_checkpoint_sampling_runtime()
    start = time.time()
    metrics = run_checkpoint_sampling_smoke(
        samples=train_samples,
        sampling_samples=all_samples,
        checkpoint_path=checkpoint_path,
        seed=seed,
        max_steps=max_steps,
        learning_rate=learning_rate,
        sample_step_size=sample_step_size,
    )
    elapsed = time.time() - start
    split_counts = {key: len(split[key]) for key in ["train", "val", "test"]}

    common = {
        "run_name": run_name,
        "dataset_name": "geom_drugs",
        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
        "dataset_subset_scope": subset_scope,
        "processed_artifact_path": str(processed_artifact_path),
        "split_path": str(split_path),
        "random_seed": seed,
        "corruption_mode": corruption_mode,
        "model_name": metrics["model_name"],
        "backbone": config["model"]["backbone"],
        "execution_mode": scope["execution_mode"],
        "result_kind": scope["result_kind"],
        "benchmark_result": False,
        "hardware": "cpu_only",
        "gpu_used": False,
        "scheduler": None,
        "slurm_account": None,
        "pilot_run": True,
        "bounded_pilot": True,
        "micro_pilot_run": scope["micro_pilot_run"],
        "data_scale_check_run": scope["data_scale_check_run"],
        "dry_run": False,
        "training_executed": True,
        "model_training_started": True,
        "training_loop_executed": True,
        "bounded_real_training_loop_executed": True,
        "larger_real_training_run_started": False,
        "max_steps": int(max_steps),
        "hard_max_steps": HARD_MAX_STEPS,
        "optimizer_created": True,
        "optimizer_step_count": int(max_steps),
        "backward_executed": True,
        "backward_pass_count": int(max_steps),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_written": True,
        "checkpoint_reloaded": True,
        "checkpoint_scope": scope["checkpoint_scope"],
        "sampling_executed": True,
        "sample_step_count": 1,
        "sample_step_size": float(sample_step_size),
        "forward_executed": True,
        "loss_computed": True,
        "training_sample_count": len(train_samples),
        "sample_count": len(all_samples),
        "runtime_scope": MICRO_PILOT_RUNTIME_SCOPE,
        "runtime_seconds_reliable": True,
    }
    manifest = {
        **common,
        "git_commit": _git_commit(),
        "config_path": str(config_path_obj) if config_path_obj is not None else None,
        "config": config,
        "metrics_output_path": str(results_path),
        "bounds": {
            "max_molecules": max_molecules,
            "hard_max_molecules": scope["hard_max_molecules"],
            "input_molecule_count": len(artifact["molecules"]),
            "processed_record_count_used": len(records),
            "conformers_per_molecule_used": 1,
            "train_molecule_count": split_counts["train"],
            "val_molecule_count": split_counts["val"],
            "test_molecule_count": split_counts["test"],
            "training_sample_count": len(train_samples),
            "sampling_sample_count": len(all_samples),
            "max_steps": int(max_steps),
            "hard_max_steps": HARD_MAX_STEPS,
            "optimizer_step_count": int(max_steps),
            "backward_pass_count": int(max_steps),
            "sample_step_count": 1,
            "runtime_scope": MICRO_PILOT_RUNTIME_SCOPE,
            "runtime_backend": runtime_backend,
        },
        "split": {
            "split_type": split["split_type"],
            "split_scope": split.get("split_scope"),
            "degenerate_tiny_split": split.get("degenerate_tiny_split"),
            "counts": split_counts,
        },
        "notes": [
            f"{scope['description']} checkpoint-and-sampling run only, not a benchmark result.",
            f"A disposable pilot-only checkpoint was written and reloaded after {int(max_steps)} CPU optimizer steps on the {split_counts['train']}-molecule train split.",
            f"A single bounded sampling step was run across the {max_molecules}-molecule bounded artifact.",
            "No raw-data mutation, environment mutation, GPU execution, SLURM job, benchmark evaluation, reusable checkpoint, or larger real training run was used.",
        ],
    }
    results = {
        **common,
        "dataset": {
            "name": "geom_drugs",
            "kind": "geom_drugs_tiny_processed_dry_run",
            "subset_scope": subset_scope,
            "molecule_count": len(records),
            "split_type": split["split_type"],
            "split_scope": split.get("split_scope"),
            "split_counts": split_counts,
            "degenerate_tiny_split": split.get("degenerate_tiny_split"),
        },
        "flow_matching": config["flow_matching"],
        "metrics": {
            "metric_scope": scope["metric_scope"],
            **metrics,
        },
        "runtime_seconds": elapsed,
        "runtime_scope": MICRO_PILOT_RUNTIME_SCOPE,
        "runtime_seconds_reliable": True,
        "runtime_backend": runtime_backend,
        "runtime_warmup_scope": "full_cpu_checkpoint_sampling_operation",
        "notes": [
            f"This is a {scope['description']} GEOM CPU checkpoint-and-sampling run.",
            "The checkpoint is disposable pilot-only output, not a reusable research checkpoint.",
            "Sampled-coordinate values validate checkpoint reload and sampling plumbing only and are not benchmark results.",
            "runtime_seconds measures only checkpoint train/save/reload/sample after CPU backend warm-up so comparable modes do not include one-time import overhead.",
            "No raw-data modification, environment mutation, GPU execution, SLURM job, benchmark evaluation, or larger real training run was used.",
        ],
    }

    for directory in [manifest_path.parent, results_path.parent, log_path.parent, checkpoint_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_path.write_text(_micro_log_text(results, manifest_path, results_path), encoding="utf-8")
    return {"mode": corruption_mode, "manifest": manifest_path, "results": results_path, "log": log_path, "checkpoint": checkpoint_path}


def _validate_bounds(max_molecules: int, max_steps: int, sample_step_size: float) -> None:
    if max_molecules != 3:
        raise ValueError("Tiny real checkpoint/sampling smoke must use exactly the existing three-molecule artifact")
    if max_steps <= 1:
        raise ValueError("Tiny real checkpoint/sampling smoke requires more than one optimizer step")
    if max_steps > HARD_MAX_STEPS:
        raise ValueError(f"Tiny real checkpoint/sampling smoke is capped at {HARD_MAX_STEPS} steps")
    if sample_step_size <= 0.0 or sample_step_size > 1.0:
        raise ValueError("Tiny real checkpoint/sampling smoke sample_step_size must be in (0, 1]")


def _validate_micro_pilot_checkpoint_sampling_config(
    config: dict[str, Any],
    max_steps: int,
    sample_step_size: float,
    scope: dict[str, Any],
) -> None:
    scope_label = scope["scope_label"]
    if config["data"]["dataset_name"] != "geom_drugs":
        raise ValueError(f"{scope_label} checkpoint/sampling config must use dataset_name=geom_drugs")
    if config["data"].get("dataset_kind") != "geom_drugs_tiny_processed_dry_run":
        raise ValueError(f"{scope_label} checkpoint/sampling config must use the bounded tiny processed artifact")
    if int(config["data"]["subset"]["max_molecules"]) != scope["max_molecules"]:
        raise ValueError(f"{scope_label} checkpoint/sampling must use exactly {scope['max_molecules']} molecules")
    if config["data"]["subset"].get("tag") != scope["subset_tag"]:
        raise ValueError(f"{scope_label} checkpoint/sampling requires subset tag {scope['subset_tag']}")
    if int(config["data"]["subset"].get("max_conformers_per_mol", 0)) != 1:
        raise ValueError(f"{scope_label} checkpoint/sampling must use one conformer per molecule")
    if config["corruption"]["mode"] not in DEFAULT_MODES:
        raise ValueError(f"Unsupported {scope_label} checkpoint/sampling mode: {config['corruption']['mode']}")
    if config["model"].get("pilot_model") != "single_linear_velocity_pilot":
        raise ValueError(f"{scope_label} checkpoint/sampling config must name pilot_model=single_linear_velocity_pilot")
    if config["train"].get("enabled") is not True:
        raise ValueError(f"{scope_label} checkpoint/sampling config must enable train.enabled")
    if config["train"].get("pilot_only") is not True:
        raise ValueError(f"{scope_label} checkpoint/sampling config must set train.pilot_only=true")
    if config["train"].get("checkpointing_enabled") is not True:
        raise ValueError(f"{scope_label} checkpoint/sampling must enable checkpointing")
    if config["sample"].get("enabled") is not True:
        raise ValueError(f"{scope_label} checkpoint/sampling must enable sampling")
    if max_steps <= 1:
        raise ValueError(f"{scope_label} checkpoint/sampling requires more than one optimizer step")
    if max_steps > HARD_MAX_STEPS:
        raise ValueError(f"{scope_label} checkpoint/sampling is capped at {HARD_MAX_STEPS} optimizer steps")
    if float(config["train"]["learning_rate"]) <= 0.0:
        raise ValueError(f"{scope_label} checkpoint/sampling learning_rate must be positive")
    if sample_step_size <= 0.0 or sample_step_size > 1.0:
        raise ValueError(f"{scope_label} checkpoint/sampling sample_step_size must be in (0, 1]")
    if int(config["eval"].get("eval_subset_size", 0)) != scope["max_molecules"]:
        raise ValueError(f"{scope_label} checkpoint/sampling eval_subset_size must match max_molecules")


def _validate_micro_pilot_checkpoint_seed_set(seed_set: tuple[int, ...]) -> None:
    if not seed_set:
        raise ValueError("Micro-pilot checkpoint/sampling multi-seed run requires at least one seed")
    if len(seed_set) > HARD_MAX_MICRO_PILOT_CHECKPOINT_SEEDS:
        raise ValueError(
            f"Micro-pilot checkpoint/sampling multi-seed run is capped at {HARD_MAX_MICRO_PILOT_CHECKPOINT_SEEDS} seeds"
        )
    if len(set(seed_set)) != len(seed_set):
        raise ValueError("Micro-pilot checkpoint/sampling seed set must contain unique seeds")
    if any(int(seed) < 0 for seed in seed_set):
        raise ValueError("Micro-pilot checkpoint/sampling seeds must be non-negative integers")


def _run_name(corruption_mode: str, seed: int, max_steps: int) -> str:
    if corruption_mode == "isotropic":
        label = "isotropic"
    elif corruption_mode == "stiffness_structured":
        label = "stiffness"
    else:
        raise ValueError(f"Unsupported tiny real checkpoint/sampling smoke corruption mode: {corruption_mode}")
    return f"geom_tiny_real_ckpt_sample_{label}_steps{int(max_steps)}_seed{seed}"


def _micro_checkpoint_run_name(corruption_mode: str, seed: int, max_molecules: int, max_steps: int) -> str:
    return _checkpoint_run_name("geom_micro_pilot_ckpt_sample", corruption_mode, seed, max_molecules, max_steps)


def _checkpoint_run_name(
    prefix: str,
    corruption_mode: str,
    seed: int,
    max_molecules: int,
    max_steps: int,
) -> str:
    if corruption_mode == "isotropic":
        label = "isotropic"
    elif corruption_mode == "stiffness_structured":
        label = "stiffness"
    else:
        raise ValueError(f"Unsupported bounded checkpoint/sampling mode: {corruption_mode}")
    return f"{prefix}_{label}_mols{int(max_molecules)}_steps{int(max_steps)}_seed{seed}"


def _checkpoint_sampling_scope_settings(max_molecules: int, subset_tag: str) -> dict[str, Any]:
    if max_molecules == MICRO_PILOT_MAX_MOLECULES and subset_tag == "micro_pilot_processed_dry_run":
        return {
            "scope_label": "micro-pilot",
            "subset_tag": "micro_pilot_processed_dry_run",
            "run_prefix": "geom_micro_pilot_ckpt_sample",
            "description": "Bounded six-molecule micro-pilot",
            "result_kind": MICRO_PILOT_RESULT_KIND,
            "execution_mode": MICRO_PILOT_EXECUTION_MODE,
            "checkpoint_scope": MICRO_PILOT_CHECKPOINT_SCOPE,
            "batch_scope": "micro_pilot_train_split",
            "sample_scope": "all_six_micro_pilot_molecules",
            "metric_scope": "geom_drugs_micro_pilot_checkpoint_sampling",
            "hard_max_molecules": MICRO_PILOT_MAX_MOLECULES,
            "max_molecules": MICRO_PILOT_MAX_MOLECULES,
            "micro_pilot_run": True,
            "data_scale_check_run": False,
        }
    if max_molecules == DATA_SCALE_CHECK_MAX_MOLECULES and subset_tag == "data_scale_check_processed_dry_run":
        return {
            "scope_label": "data-scale-check",
            "subset_tag": "data_scale_check_processed_dry_run",
            "run_prefix": "geom_data_scale_check_ckpt_sample",
            "description": "Bounded 30-molecule data-scale-check",
            "result_kind": DATA_SCALE_CHECK_RESULT_KIND,
            "execution_mode": DATA_SCALE_CHECK_EXECUTION_MODE,
            "checkpoint_scope": DATA_SCALE_CHECK_CHECKPOINT_SCOPE,
            "batch_scope": "data_scale_check_train_split",
            "sample_scope": "all_thirty_data_scale_check_molecules",
            "metric_scope": "geom_drugs_data_scale_check_checkpoint_sampling",
            "hard_max_molecules": DATA_SCALE_CHECK_MAX_MOLECULES,
            "max_molecules": DATA_SCALE_CHECK_MAX_MOLECULES,
            "micro_pilot_run": False,
            "data_scale_check_run": True,
        }
    raise ValueError(
        "Unsupported bounded checkpoint/sampling scope: "
        f"max_molecules={max_molecules} subset_tag={subset_tag!r}"
    )


def _data_scale_check_comparison_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("result_kind") != DATA_SCALE_CHECK_RESULT_KIND:
        raise ValueError(f"Not a data-scale-check checkpoint/sampling result: {path}")
    if payload.get("benchmark_result") is not False:
        raise ValueError(f"Data-scale-check result must be benchmark_result=false: {path}")
    if payload.get("hardware") != "cpu_only" or payload.get("gpu_used") is not False:
        raise ValueError(f"Data-scale-check result must be CPU-only: {path}")
    if payload.get("scheduler") is not None or payload.get("slurm_account") is not None:
        raise ValueError(f"Data-scale-check result unexpectedly records scheduler use: {path}")
    if payload.get("data_scale_check_run") is not True or payload.get("bounded_pilot") is not True:
        raise ValueError(f"Data-scale-check result is not labeled as bounded data-scale check: {path}")
    if payload.get("larger_real_training_run_started") is not False:
        raise ValueError(f"Data-scale-check result unexpectedly started a larger real training run: {path}")
    metrics = payload["metrics"]
    sampled = metrics["sampled_molecules"]
    return {
        "source_path": str(path),
        "run_name": payload["run_name"],
        "corruption_mode": payload["corruption_mode"],
        "random_seed": int(payload["random_seed"]),
        "sample_count": int(payload["sample_count"]),
        "training_sample_count": int(payload["training_sample_count"]),
        "max_steps": int(payload["max_steps"]),
        "runtime_seconds": float(payload["runtime_seconds"]),
        "runtime_scope": payload["runtime_scope"],
        "runtime_seconds_reliable": bool(payload["runtime_seconds_reliable"]),
        "split_counts": payload["dataset"]["split_counts"],
        "reloaded_loss": float(metrics["reloaded_loss"]),
        "sampled_rmsd_overall_mean": _mean_sampled_rmsd(sampled),
        "sampled_rmsd_by_split_mean": {
            split: _mean_sampled_rmsd([item for item in sampled if item["split"] == split])
            for split in ["train", "val", "test"]
        },
        "safety_labels": {
            "benchmark_result": payload["benchmark_result"],
            "hardware": payload["hardware"],
            "gpu_used": payload["gpu_used"],
            "pilot_run": payload["pilot_run"],
            "bounded_pilot": payload["bounded_pilot"],
            "data_scale_check_run": payload["data_scale_check_run"],
            "larger_real_training_run_started": payload["larger_real_training_run_started"],
            "checkpoint_scope": payload["checkpoint_scope"],
            "sampling_executed": payload["sampling_executed"],
            "slurm_account": payload["slurm_account"],
        },
    }


def _validate_data_scale_check_rows(
    rows: list[dict[str, Any]],
    seed: int,
    max_molecules: int,
    max_steps: int,
) -> None:
    for row in rows:
        if row["random_seed"] != seed:
            raise ValueError("Data-scale-check comparison expects a single fixed seed")
        if row["sample_count"] != max_molecules:
            raise ValueError("Data-scale-check comparison sample count mismatch")
        if row["max_steps"] != max_steps:
            raise ValueError("Data-scale-check comparison max_steps mismatch")
        if row["split_counts"] != {"train": max_molecules - 2, "val": 1, "test": 1}:
            raise ValueError("Data-scale-check comparison split counts mismatch")
        if row["runtime_scope"] != MICRO_PILOT_RUNTIME_SCOPE or row["runtime_seconds_reliable"] is not True:
            raise ValueError("Data-scale-check comparison requires reliable fixed-scope runtime metadata")


def _data_scale_check_metric_comparisons(by_mode: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    isotropic = by_mode["isotropic"]
    stiffness = by_mode["stiffness_structured"]
    comparisons = [
        _lower_is_better_comparison("reloaded_loss", isotropic["reloaded_loss"], stiffness["reloaded_loss"]),
        _lower_is_better_comparison(
            "sampled_rmsd_to_target_overall_mean",
            isotropic["sampled_rmsd_overall_mean"],
            stiffness["sampled_rmsd_overall_mean"],
        ),
    ]
    for split in ["train", "val", "test"]:
        comparisons.append(
            _lower_is_better_comparison(
                f"sampled_rmsd_to_target_{split}_mean",
                isotropic["sampled_rmsd_by_split_mean"][split],
                stiffness["sampled_rmsd_by_split_mean"][split],
            )
        )
    return comparisons


def _lower_is_better_comparison(metric: str, isotropic: float, stiffness: float) -> dict[str, Any]:
    delta = stiffness - isotropic
    if abs(delta) <= 1.0e-12:
        outcome = "tied"
    elif delta < 0.0:
        outcome = "better"
    else:
        outcome = "worse"
    return {
        "metric": metric,
        "isotropic": isotropic,
        "stiffness_structured": stiffness,
        "stiffness_minus_isotropic": delta,
        "stiffness_outcome": outcome,
        "direction": "lower_is_better",
    }


def _mean_sampled_rmsd(sampled: list[dict[str, Any]]) -> float:
    if not sampled:
        raise ValueError("Cannot compute sampled RMSD mean over an empty split")
    return sum(float(item["sampled_rmsd_to_target"]) for item in sampled) / len(sampled)


def _data_scale_check_comparison_log(results: dict[str, Any], manifest_path: Path, results_path: Path) -> str:
    return (
        f"run_name={results['run_name']}\n"
        f"result_kind={results['result_kind']}\n"
        "benchmark_result=false\n"
        "data_scale_check_run=true\n"
        "training_executed_by_comparison=false\n"
        "sampling_executed_by_comparison=false\n"
        f"directional_signal_preserved={str(results['directional_signal_preserved']).lower()}\n"
        f"summary={results['summary']}\n"
        f"manifest={manifest_path}\n"
        f"results={results_path}\n"
        "note=Bounded 30-molecule data-scale-check comparison only; no GPU, scheduler job, raw-data mutation, environment mutation, benchmark, or larger real training run.\n"
    )


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
        "bounded_training_smoke_executed=true\n"
        "larger_real_training_run_started=false\n"
        f"max_steps={results['max_steps']}\n"
        "checkpoint_written=true\n"
        "checkpoint_reloaded=true\n"
        "checkpoint_scope=disposable_smoke_only\n"
        "sampling_executed=true\n"
        f"sample_step_count={results['sample_step_count']}\n"
        "hardware=cpu_only\n"
        "gpu_used=false\n"
        "scheduler=none\n"
        f"result_kind={RESULT_KIND}\n"
        f"initial_loss={metrics['initial_loss']}\n"
        f"reloaded_loss={metrics['reloaded_loss']}\n"
        f"sampled_molecule_count={len(metrics['sampled_molecules'])}\n"
        f"manifest={manifest_path}\n"
        f"results={results_path}\n"
        f"checkpoint={results['checkpoint_path']}\n"
        "note=Tiny real-data checkpoint/sampling smoke only; disposable checkpoint, no GPU, scheduler job, environment mutation, benchmark, or larger real training run.\n"
    )


def _micro_log_text(results: dict[str, Any], manifest_path: Path, results_path: Path) -> str:
    metrics = results["metrics"]
    return (
        f"run_name={results['run_name']}\n"
        f"corruption_mode={results['corruption_mode']}\n"
        f"execution_mode={results['execution_mode']}\n"
        "pilot_run=true\n"
        "bounded_pilot=true\n"
        f"micro_pilot_run={str(results['micro_pilot_run']).lower()}\n"
        f"data_scale_check_run={str(results['data_scale_check_run']).lower()}\n"
        "training_executed=true\n"
        "model_training_started=true\n"
        "training_loop_executed=true\n"
        "bounded_real_training_loop_executed=true\n"
        "larger_real_training_run_started=false\n"
        f"max_steps={results['max_steps']}\n"
        f"hard_max_steps={results['hard_max_steps']}\n"
        "checkpoint_written=true\n"
        "checkpoint_reloaded=true\n"
        f"checkpoint_scope={results['checkpoint_scope']}\n"
        "sampling_executed=true\n"
        f"sample_step_count={results['sample_step_count']}\n"
        f"training_sample_count={results['training_sample_count']}\n"
        f"sample_count={results['sample_count']}\n"
        f"runtime_scope={results['runtime_scope']}\n"
        f"runtime_seconds_reliable={str(results['runtime_seconds_reliable']).lower()}\n"
        "hardware=cpu_only\n"
        "gpu_used=false\n"
        "scheduler=none\n"
        f"result_kind={results['result_kind']}\n"
        f"initial_loss={metrics['initial_loss']}\n"
        f"reloaded_loss={metrics['reloaded_loss']}\n"
        f"sampled_molecule_count={len(metrics['sampled_molecules'])}\n"
        f"manifest={manifest_path}\n"
        f"results={results_path}\n"
        f"checkpoint={results['checkpoint_path']}\n"
        "note=Bounded real-data checkpoint/sampling only; disposable pilot checkpoint, no GPU, scheduler job, environment mutation, benchmark, or larger real training run.\n"
    )
