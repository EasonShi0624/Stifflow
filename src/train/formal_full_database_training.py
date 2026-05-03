"""First full-database single-seed formal training path."""

from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

from src.data.geom_full_database import (
    FULL_DATABASE_PROCESSED_ROOT,
    FULL_DATABASE_SCAFFOLD_SPLIT_PATH,
    FULL_DATABASE_SCOPE,
    FULL_DATABASE_SPLIT_ROOT,
)
from src.models.checkpoint_sampling_smoke import (
    run_validation_checkpoint_training,
    warm_up_checkpoint_sampling_runtime,
)
from src.train import tiny_real as tiny_real_data
from src.train.formal_10000_training import (
    FORMAL_10000_HARD_MAX_STEPS,
    FORMAL_10000_LEARNING_RATE,
    FORMAL_10000_MAX_STEPS,
    FORMAL_10000_MIN_DELTA,
    FORMAL_10000_PATIENCE,
    FORMAL_10000_SAMPLE_STEP_SIZE,
)
from src.utils.config import load_config


FORMAL_FULL_DATABASE_CONFIG_PATHS = (
    Path("configs/train/geom_full_database_formal_isotropic.yaml"),
    Path("configs/train/geom_full_database_formal_stiffness.yaml"),
)
FORMAL_FULL_DATABASE_ATTRIBUTION_MODES = (
    "isotropic",
    "harmonic_prior",
    "torsion_only",
    "stiffness_structured",
)
FORMAL_FULL_DATABASE_HYBRID_PILOT_MODES = (
    "hybrid_harmonic_stiffness_v1",
    "harmonic_stiffness_residual_v1",
)
FORMAL_FULL_DATABASE_SUPPORTED_MODES = (
    *FORMAL_FULL_DATABASE_ATTRIBUTION_MODES,
    *FORMAL_FULL_DATABASE_HYBRID_PILOT_MODES,
)
FORMAL_FULL_DATABASE_SEED = 0
FORMAL_FULL_DATABASE_SEED_SET = (0, 1, 2)
FORMAL_FULL_DATABASE_RESULT_KIND = "geom_drugs_full_database_formal_training_not_benchmark"
FORMAL_FULL_DATABASE_EXECUTION_MODE = "cpu_formal_full_database_training_validation_selected_checkpoint"
FORMAL_FULL_DATABASE_GPU_EXECUTION_MODE = "gpu_formal_full_database_training_validation_selected_checkpoint"
FORMAL_FULL_DATABASE_CHECKPOINT_SCOPE = "reusable_formal_full_database_validation_selected_checkpoint"
FORMAL_FULL_DATABASE_CHECKPOINT_RESULT_KIND = (
    "reusable_formal_full_database_validation_selected_checkpoint_not_benchmark"
)
FORMAL_FULL_DATABASE_RUNTIME_SCOPE = "formal_full_database_train_validation_select_save_reload_sample_after_cpu_warmup"


def run_formal_full_database_training_configs(
    config_paths: tuple[str | Path, ...] = FORMAL_FULL_DATABASE_CONFIG_PATHS,
    output_root: str | Path | None = None,
    seed: int = FORMAL_FULL_DATABASE_SEED,
) -> dict[str, dict[str, Path]]:
    if int(seed) not in FORMAL_FULL_DATABASE_SEED_SET:
        raise ValueError(f"full-database formal training seed must be one of {FORMAL_FULL_DATABASE_SEED_SET}")
    outputs = {}
    for config_path in config_paths:
        config = load_config(config_path)
        if output_root is not None:
            config["experiment"]["output_root"] = str(output_root)
        config["experiment"]["seed"] = int(seed)
        paths = run_formal_full_database_training(config=config, config_path=Path(config_path))
        outputs[paths["mode"]] = {
            "manifest": paths["manifest"],
            "results": paths["results"],
            "log": paths["log"],
            "best_checkpoint": paths["best_checkpoint"],
            "latest_checkpoint": paths["latest_checkpoint"],
        }
    return outputs


def run_formal_full_database_training(
    config: dict[str, Any],
    config_path: str | Path | None = None,
) -> dict[str, Path]:
    config = copy.deepcopy(config)
    config_path_obj = Path(config_path) if config_path is not None else None
    validate_formal_full_database_config(config)

    output_root = Path(config["experiment"]["output_root"])
    seed = int(config["experiment"]["seed"])
    corruption_mode = config["corruption"]["mode"]
    max_steps = int(config["train"]["max_steps"])
    hard_max_steps = int(config["train"]["hard_max_steps"])
    learning_rate = float(config["train"]["learning_rate"])
    optimizer_name = str(config["train"].get("optimizer", "SGD"))
    optimizer_weight_decay = float(config["train"].get("optimizer_weight_decay", 0.0))
    lr_scheduler = copy.deepcopy(config["train"].get("lr_scheduler"))
    sample_step_size = float(config["sample"]["sample_step_size"])
    device = str(config["train"].get("device", "cpu"))
    require_cuda = bool(config["train"].get("require_cuda", False))
    processed_artifact_path = Path(config["data"]["processed_artifact_path"])
    split_path = Path(config["data"]["split_path"])
    configured_split_type = str(config["data"]["split_type"])
    run_name = _run_name(
        corruption_mode=corruption_mode,
        seed=seed,
        max_steps=max_steps,
        split_type=configured_split_type,
        corruption_config=config["corruption"],
    )
    run_name_suffix = _run_name_suffix(config)
    if run_name_suffix:
        run_name = f"{run_name}_{run_name_suffix}"
    manifest_path = output_root / "manifests" / f"{run_name}.json"
    results_path = output_root / "results" / f"{run_name}.json"
    log_path = output_root / "logs" / f"{run_name}.log"
    best_checkpoint_path = output_root / "checkpoints" / f"{run_name}_best.pt"
    latest_checkpoint_path = output_root / "checkpoints" / f"{run_name}_latest.pt"

    config["experiment"]["name"] = run_name
    artifact = tiny_real_data._load_tiny_real_artifact(processed_artifact_path)
    if artifact.get("subset_scope") != FULL_DATABASE_SCOPE:
        raise ValueError(f"full-database formal training requires subset_scope={FULL_DATABASE_SCOPE!r}")
    records = artifact["molecules"]
    if len(records) < 20:
        raise ValueError("full-database formal training requires at least 20 extracted molecules")
    split = tiny_real_data._load_tiny_real_split(split_path)
    if split.get("split_scope") != FULL_DATABASE_SCOPE:
        raise ValueError(f"full-database formal training requires split_scope={FULL_DATABASE_SCOPE!r}")
    if split.get("split_type") != configured_split_type:
        raise ValueError("full-database formal training config split_type must match the split artifact")
    split_membership = tiny_real_data._validate_split_coverage(split, records)
    split_counts = {key: len(split[key]) for key in ["train", "val", "test"]}
    if min(split_counts.values()) <= 0:
        raise ValueError("full-database formal training requires non-empty train/val/test splits")
    if configured_split_type == "random":
        expected_val_test = round(len(records) * 0.05)
        if split_counts["val"] != expected_val_test or split_counts["test"] != expected_val_test:
            raise ValueError("full-database formal random split must preserve the 90/5/5 construction")
    elif configured_split_type != "scaffold":
        raise ValueError("full-database formal training supports only random or scaffold splits")

    config["eval"]["eval_subset_size"] = len(records)
    all_samples = tiny_real_data._build_flow_matching_samples(
        config,
        records,
        random.Random(seed),
        split_membership,
    )
    train_samples = [sample for sample in all_samples if sample["split"] == "train"]
    validation_samples = [sample for sample in all_samples if sample["split"] == "val"]
    if len(train_samples) != split_counts["train"]:
        raise ValueError("full-database train sample count does not match the split")
    if len(validation_samples) != split_counts["val"]:
        raise ValueError("full-database validation sample count does not match the split")

    runtime_backend = warm_up_checkpoint_sampling_runtime()
    start = time.time()
    metrics = run_validation_checkpoint_training(
        samples=train_samples,
        validation_samples=validation_samples,
        sampling_samples=all_samples,
        checkpoint_path=best_checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        seed=seed,
        max_steps=max_steps,
        hard_max_steps=hard_max_steps,
        learning_rate=learning_rate,
        sample_step_size=sample_step_size,
        early_stop=copy.deepcopy(config["train"]["early_stop"]),
        checkpoint_result_kind=FORMAL_FULL_DATABASE_CHECKPOINT_RESULT_KIND,
        checkpoint_scope=FORMAL_FULL_DATABASE_CHECKPOINT_SCOPE,
        device=device,
        require_cuda=require_cuda,
        optimizer_name=optimizer_name,
        optimizer_weight_decay=optimizer_weight_decay,
        lr_scheduler=lr_scheduler,
    )
    elapsed = time.time() - start
    scheduler = "slurm" if os.environ.get("SLURM_JOB_ID") else None
    slurm_account = os.environ.get("STIFFLOW_SLURM_ACCOUNT") if scheduler else None

    common = {
        "run_name": run_name,
        "dataset_name": "geom_drugs",
        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
        "dataset_subset_scope": FULL_DATABASE_SCOPE,
        "processed_artifact_path": str(processed_artifact_path),
        "split_path": str(split_path),
        "split_type": configured_split_type,
        "random_seed": seed,
        "corruption_mode": corruption_mode,
        "hybrid_source_strength": _hybrid_source_strength(config["corruption"]),
        "hybrid_source_strength_key": _hybrid_source_strength_key(config["corruption"]),
        "model_name": metrics["model_name"],
        "backbone": config["model"]["backbone"],
        "execution_mode": FORMAL_FULL_DATABASE_GPU_EXECUTION_MODE
        if str(metrics["device"]).startswith("cuda")
        else FORMAL_FULL_DATABASE_EXECUTION_MODE,
        "result_kind": FORMAL_FULL_DATABASE_RESULT_KIND,
        "benchmark_result": False,
        "hardware": "gpu_enabled" if str(metrics["device"]).startswith("cuda") else "cpu_only",
        "gpu_used": bool(str(metrics["device"]).startswith("cuda")),
        "torch_device": metrics["device"],
        "cuda_available": bool(metrics.get("cuda_available")),
        "cuda_required": bool(metrics.get("cuda_required")),
        "cuda_device_name": metrics.get("cuda_device_name"),
        "scheduler": scheduler,
        "slurm_account": slurm_account,
        "formal_training_run": True,
        "full_database_formal_training_run": True,
        "bounded_formal_training_run": False,
        "pilot_run": False,
        "dry_run": False,
        "training_executed": True,
        "model_training_started": True,
        "training_loop_executed": True,
        "formal_training_loop_executed": True,
        "larger_real_training_run_started": True,
        "max_steps": max_steps,
        "hard_max_steps": hard_max_steps,
        "optimizer_created": True,
        "optimizer_name": metrics["optimizer_name"],
        "learning_rate": learning_rate,
        "optimizer_weight_decay": optimizer_weight_decay,
        "final_learning_rate": metrics.get("final_learning_rate"),
        "lr_scheduler": metrics.get("lr_scheduler"),
        "lr_scheduler_name": metrics.get("lr_scheduler_name"),
        "optimizer_step_count": int(metrics["optimizer_step_count"]),
        "backward_executed": True,
        "backward_pass_count": int(metrics["backward_pass_count"]),
        "best_checkpoint_path": str(best_checkpoint_path),
        "latest_checkpoint_path": str(latest_checkpoint_path),
        "checkpoint_path": str(best_checkpoint_path),
        "checkpoint_written": True,
        "checkpoint_reloaded": True,
        "checkpoint_reloaded_path": str(best_checkpoint_path),
        "checkpoint_reusable": True,
        "checkpoint_selected_by_validation": True,
        "checkpoint_selection_metric": "validation_loss_after_step",
        "checkpoint_scope": FORMAL_FULL_DATABASE_CHECKPOINT_SCOPE,
        "sampling_executed": True,
        "sample_step_count": 1,
        "sample_step_size": sample_step_size,
        "forward_executed": True,
        "loss_computed": True,
        "training_sample_count": len(train_samples),
        "validation_sample_count": len(validation_samples),
        "sample_count": len(all_samples),
        "runtime_scope": FORMAL_FULL_DATABASE_RUNTIME_SCOPE,
        "runtime_seconds_reliable": True,
        "configured_max_steps": max_steps,
        "early_stop_enabled": True,
        "early_stop_patience": int(config["train"]["early_stop"]["patience"]),
        "early_stop_min_delta": float(config["train"]["early_stop"]["min_delta"]),
        "early_stop_monitor": config["train"]["early_stop"]["monitor"],
        "stopped_early": bool(metrics["stopped_early"]),
        "early_stop_reason": metrics["stop_reason"],
    }
    manifest = {
        **common,
        "git_commit": _git_commit(),
        "config_path": str(config_path_obj) if config_path_obj is not None else None,
        "config": config,
        "metrics_output_path": str(results_path),
        "bounds": {
            "max_molecules": None,
            "hard_max_molecules": None,
            "input_molecule_count": len(artifact["molecules"]),
            "processed_record_count_used": len(records),
            "conformers_per_molecule_used": 1,
            "train_molecule_count": split_counts["train"],
            "val_molecule_count": split_counts["val"],
            "test_molecule_count": split_counts["test"],
            "training_sample_count": len(train_samples),
            "validation_sample_count": len(validation_samples),
            "sampling_sample_count": len(all_samples),
            "max_steps": max_steps,
            "hard_max_steps": hard_max_steps,
            "executed_train_steps": int(metrics["executed_train_steps"]),
            "optimizer_step_count": int(metrics["optimizer_step_count"]),
            "backward_pass_count": int(metrics["backward_pass_count"]),
            "sample_step_count": 1,
            "runtime_scope": FORMAL_FULL_DATABASE_RUNTIME_SCOPE,
            "runtime_backend": runtime_backend,
            "early_stop": copy.deepcopy(config["train"]["early_stop"]),
            "stopped_early": bool(metrics["stopped_early"]),
            "stop_reason": metrics["stop_reason"],
            "checkpoint_selection_metric": "validation_loss_after_step",
        },
        "split": {
            "split_type": split["split_type"],
            "split_scope": split.get("split_scope"),
            "degenerate_tiny_split": split.get("degenerate_tiny_split"),
            "counts": split_counts,
        },
        "notes": _formal_full_database_notes(
            max_steps=max_steps,
            split_type=configured_split_type,
            optimizer_pilot=bool(config["train"].get("optimizer_pilot", False)),
        ),
    }
    results = {
        **common,
        "dataset": {
            "name": "geom_drugs",
            "kind": "geom_drugs_tiny_processed_dry_run",
            "subset_scope": FULL_DATABASE_SCOPE,
            "molecule_count": len(records),
            "split_type": split["split_type"],
            "split_scope": split.get("split_scope"),
            "split_counts": split_counts,
            "degenerate_tiny_split": split.get("degenerate_tiny_split"),
        },
        "flow_matching": config["flow_matching"],
        "metrics": {
            "metric_scope": "geom_drugs_formal_full_database_training_checkpoint_sampling",
            **metrics,
        },
        "runtime_seconds": elapsed,
        "runtime_backend": runtime_backend,
        "runtime_warmup_scope": "full_cpu_formal_training_checkpoint_sampling_operation",
        "notes": _formal_full_database_notes(
            max_steps=max_steps,
            split_type=configured_split_type,
            optimizer_pilot=bool(config["train"].get("optimizer_pilot", False)),
        ),
    }

    for directory in [manifest_path.parent, results_path.parent, log_path.parent, best_checkpoint_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_path.write_text(_formal_full_database_log_text(results, manifest_path, results_path), encoding="utf-8")
    return {
        "mode": corruption_mode,
        "manifest": manifest_path,
        "results": results_path,
        "log": log_path,
        "best_checkpoint": best_checkpoint_path,
        "latest_checkpoint": latest_checkpoint_path,
    }


def validate_formal_full_database_config(config: dict[str, Any]) -> None:
    if config["data"]["dataset_name"] != "geom_drugs":
        raise ValueError("full-database formal training requires dataset_name=geom_drugs")
    if config["data"].get("dataset_kind") != "geom_drugs_tiny_processed_dry_run":
        raise ValueError("full-database formal training must use the existing processed artifact schema")
    if not _allowed_processed_artifact_path(Path(config["data"]["processed_artifact_path"])):
        raise ValueError("full-database formal training must use the default artifact or a per-run full-database artifact")
    split_type = config["data"].get("split_type")
    if split_type not in {"random", "scaffold"}:
        raise ValueError("full-database formal training split_type must be random or scaffold")
    split_path = Path(config["data"]["split_path"])
    if not _allowed_split_path(split_path):
        raise ValueError("full-database formal training must use the default random/scaffold split or a per-run random_seed0 split artifact")
    if split_type == "scaffold" and split_path != FULL_DATABASE_SCAFFOLD_SPLIT_PATH:
        raise ValueError("full-database scaffold formal training must use the cached scaffold_seed0 split artifact")
    if split_type == "random" and split_path == FULL_DATABASE_SCAFFOLD_SPLIT_PATH:
        raise ValueError("full-database random formal training cannot use the scaffold split artifact")
    if int(config["experiment"]["seed"]) not in FORMAL_FULL_DATABASE_SEED_SET:
        raise ValueError(f"full-database formal training seed must be one of {FORMAL_FULL_DATABASE_SEED_SET}")
    if config["data"]["subset"].get("max_molecules") != "full_database":
        raise ValueError("full-database formal training must label max_molecules as full_database")
    if int(config["data"]["subset"].get("max_conformers_per_mol", 0)) != 1:
        raise ValueError("full-database formal training must preserve one conformer per molecule")
    if config["data"]["subset"].get("tag") != FULL_DATABASE_SCOPE:
        raise ValueError(f"full-database formal training must use tag={FULL_DATABASE_SCOPE!r}")
    if config["corruption"]["mode"] not in set(FORMAL_FULL_DATABASE_SUPPORTED_MODES):
        raise ValueError(f"unsupported full-database formal corruption mode: {config['corruption']['mode']}")
    if config["model"].get("backbone") != "minimal_linear_velocity_pilot":
        raise ValueError("full-database formal training must keep the existing minimal linear backbone")
    if config["model"].get("pilot_model") != "single_linear_velocity_pilot":
        raise ValueError("full-database formal training must keep pilot_model=single_linear_velocity_pilot")
    if config["train"].get("enabled") is not True:
        raise ValueError("full-database formal training requires train.enabled=true")
    if config["train"].get("formal_training") is not True:
        raise ValueError("full-database formal training requires train.formal_training=true")
    if config["train"].get("pilot_only") is not False:
        raise ValueError("full-database formal training must not be labeled pilot_only")
    if config["train"].get("batch_scope") != "full_database_train_split":
        raise ValueError("full-database formal training must use the full-database train split scope")
    if int(config["train"]["max_steps"]) != FORMAL_10000_MAX_STEPS:
        raise ValueError(f"full-database max_steps must remain fixed at {FORMAL_10000_MAX_STEPS}")
    if int(config["train"]["hard_max_steps"]) != FORMAL_10000_HARD_MAX_STEPS:
        raise ValueError(f"full-database hard_max_steps must remain fixed at {FORMAL_10000_HARD_MAX_STEPS}")
    optimizer_pilot = bool(config["train"].get("optimizer_pilot", False))
    optimizer_name = str(config["train"].get("optimizer", "SGD"))
    learning_rate = float(config["train"]["learning_rate"])
    if optimizer_pilot:
        _validate_optimizer_pilot_config(config, optimizer_name=optimizer_name, learning_rate=learning_rate)
    else:
        if learning_rate != FORMAL_10000_LEARNING_RATE:
            raise ValueError("full-database formal training must keep the existing SGD learning rate")
        if optimizer_name != "SGD":
            raise ValueError("full-database formal training must keep optimizer=SGD")
        scheduler = config["train"].get("lr_scheduler")
        if scheduler is not None and scheduler is not False:
            raise ValueError("full-database formal training only allows lr_scheduler in optimizer_pilot configs")
    if config["train"].get("checkpointing_enabled") is not True:
        raise ValueError("full-database formal training requires checkpointing_enabled=true")
    if config["train"].get("checkpoint_scope") != FORMAL_FULL_DATABASE_CHECKPOINT_SCOPE:
        raise ValueError("full-database formal training must use the reusable full-database checkpoint scope")
    device = str(config["train"].get("device", "cpu"))
    if device not in {"cpu", "cuda", "auto"} and not device.startswith("cuda"):
        raise ValueError("full-database formal training device must be cpu, cuda, cuda:<index>, or auto")
    if config["train"].get("require_cuda") not in {None, True, False}:
        raise ValueError("full-database formal training require_cuda must be boolean when provided")
    early_stop = config["train"].get("early_stop")
    if not isinstance(early_stop, dict) or early_stop.get("enabled") is not True:
        raise ValueError("full-database formal training requires validation early stopping")
    if early_stop.get("monitor") != "validation_loss_after_step":
        raise ValueError("full-database early-stop monitor must be validation_loss_after_step")
    if int(early_stop.get("patience", 0)) != FORMAL_10000_PATIENCE:
        raise ValueError(f"full-database patience must remain fixed at {FORMAL_10000_PATIENCE}")
    if float(early_stop.get("min_delta", -1.0)) != FORMAL_10000_MIN_DELTA:
        raise ValueError(f"full-database min_delta must remain fixed at {FORMAL_10000_MIN_DELTA}")
    if config["sample"].get("enabled") is not True:
        raise ValueError("full-database formal training must keep one-step sampling enabled")
    if config["sample"].get("sample_step_count") != 1:
        raise ValueError("full-database formal training must keep sample_step_count=1")
    if float(config["sample"]["sample_step_size"]) != FORMAL_10000_SAMPLE_STEP_SIZE:
        raise ValueError("full-database formal training must keep sample_step_size=0.1")
    if int(config["eval"].get("eval_subset_size", -1)) != 0:
        raise ValueError("full-database eval_subset_size must be 0 in config and set to full artifact size at runtime")


def _allowed_processed_artifact_path(path: Path) -> bool:
    if path == FULL_DATABASE_PROCESSED_ROOT / "molecules.json":
        return True
    root = Path("data/processed/geom_drugs_full_database_runs")
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return len(relative.parts) == 2 and relative.parts[-1] == "molecules.json"


def _allowed_split_path(path: Path) -> bool:
    if path == FULL_DATABASE_SPLIT_ROOT / "random_seed0" / "split.json":
        return True
    if path == FULL_DATABASE_SCAFFOLD_SPLIT_PATH:
        return True
    root = Path("data/splits/geom_drugs_full_database_runs")
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return len(relative.parts) == 3 and relative.parts[-2:] == ("random_seed0", "split.json")


def _validate_optimizer_pilot_config(config: dict[str, Any], optimizer_name: str, learning_rate: float) -> None:
    if config["data"].get("split_type") != "random":
        raise ValueError("optimizer pilot configs are limited to the cached random split")
    if int(config["experiment"]["seed"]) != 0:
        raise ValueError("optimizer pilot configs are limited to seed0")
    if config["corruption"]["mode"] not in {"harmonic_prior", "hybrid_harmonic_stiffness_v1"}:
        raise ValueError("optimizer pilot configs are limited to harmonic_prior and hybrid_harmonic_stiffness_v1")
    allowed_lrs = {
        "SGD": {1.0e-2, 3.0e-3, 1.0e-3},
        "AdamW": {1.0e-3, 3.0e-4, 1.0e-4},
    }
    if optimizer_name not in allowed_lrs:
        raise ValueError("optimizer pilot optimizer must be SGD or AdamW")
    if not any(abs(learning_rate - value) <= 1.0e-12 for value in allowed_lrs[optimizer_name]):
        raise ValueError(f"optimizer pilot learning_rate is not allowed for {optimizer_name}")
    weight_decay = float(config["train"].get("optimizer_weight_decay", 0.0))
    if optimizer_name == "SGD" and weight_decay != 0.0:
        raise ValueError("optimizer pilot SGD configs must keep optimizer_weight_decay=0.0")
    if weight_decay < 0.0:
        raise ValueError("optimizer_weight_decay must be non-negative")
    scheduler = config["train"].get("lr_scheduler")
    if scheduler is None or scheduler is False:
        return
    if not isinstance(scheduler, dict):
        raise ValueError("optimizer pilot lr_scheduler must be a mapping when provided")
    if scheduler.get("name") != "ReduceLROnPlateau":
        raise ValueError("optimizer pilot only supports ReduceLROnPlateau")
    if float(scheduler.get("factor", 0.5)) <= 0.0 or float(scheduler.get("factor", 0.5)) >= 1.0:
        raise ValueError("ReduceLROnPlateau factor must be in (0, 1)")
    if int(scheduler.get("patience", 50)) < 1:
        raise ValueError("ReduceLROnPlateau patience must be positive")


def _run_name(
    corruption_mode: str,
    seed: int,
    max_steps: int,
    split_type: str = "random",
    corruption_config: dict[str, Any] | None = None,
) -> str:
    if corruption_mode == "isotropic":
        label = "isotropic"
    elif corruption_mode == "harmonic_prior":
        label = "harmonic"
    elif corruption_mode == "torsion_only":
        label = "torsion_only"
    elif corruption_mode == "stiffness_structured":
        label = "stiffness"
    elif corruption_mode == "hybrid_harmonic_stiffness_v1":
        label = f"hybrid_harmonic_stiffness_v1_{_lambda_label(_hybrid_strength(corruption_config, 'modulation_strength'))}"
    elif corruption_mode == "harmonic_stiffness_residual_v1":
        label = f"harmonic_stiffness_residual_v1_{_lambda_label(_hybrid_strength(corruption_config, 'residual_strength'))}"
    else:
        raise ValueError(f"Unsupported full-database formal training mode: {corruption_mode}")
    split_label = "" if split_type == "random" else f"_{split_type}"
    return f"geom_formal_full_database{split_label}_train_{label}_steps{int(max_steps)}_seed{seed}"


def _run_name_suffix(config: dict[str, Any]) -> str:
    suffix = str(config.get("experiment", {}).get("run_name_suffix", "")).strip()
    if not suffix:
        return ""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    if any(char not in allowed for char in suffix):
        raise ValueError("experiment.run_name_suffix may only contain letters, numbers, and underscores")
    return suffix


def _hybrid_strength(corruption_config: dict[str, Any] | None, key: str) -> float:
    if corruption_config is None or key not in corruption_config:
        raise ValueError(f"{key} is required to build hybrid full-database run names")
    value = float(corruption_config[key])
    upper_bound = 50.0 if key == "modulation_strength" else 1.0
    if value < 0.0 or value > upper_bound:
        raise ValueError(f"{key} must be in [0, {upper_bound:g}]")
    return value


def _lambda_label(value: float) -> str:
    return f"lambda{int(round(float(value) * 100)):03d}"


def _hybrid_source_strength(corruption_config: dict[str, Any]) -> float | None:
    key = _hybrid_source_strength_key(corruption_config)
    if key is None:
        return None
    return float(corruption_config[key])


def _hybrid_source_strength_key(corruption_config: dict[str, Any]) -> str | None:
    mode = corruption_config.get("mode")
    if mode == "hybrid_harmonic_stiffness_v1":
        return "modulation_strength"
    if mode == "harmonic_stiffness_residual_v1":
        return "residual_strength"
    return None


def _formal_full_database_notes(max_steps: int, split_type: str, optimizer_pilot: bool = False) -> list[str]:
    split_note = (
        "a 90/5/5 random split"
        if split_type == "random"
        else "a cached full-database Bemis-Murcko scaffold split with 90/5/5 target fractions"
    )
    training_note = (
        "The run is an optimizer/LR pilot: model, source construction, checkpoint/reload behavior, one-step sampling diagnostic, validation-driven checkpoint selection, patience, min_delta, and hard cap are frozen; optimizer, learning rate, and optional scheduler are controlled by config."
        if optimizer_pilot
        else f"The run reuses the frozen formal design from the 10000-molecule phase: same model, optimizer, learning rate, checkpoint/reload behavior, one-step sampling diagnostic, validation-driven checkpoint selection, patience, min_delta, and hard cap of {int(max_steps)} steps."
    )
    return [
        "First full-database GEOM-Drugs formal training path, not a benchmark result.",
        training_note,
        f"The data scope is the full available GEOM-Drugs one-conformer processed artifact with {split_note}.",
        "No raw-data modification, environment mutation, benchmark evaluation, or unbounded training is part of this path.",
    ]


def _formal_full_database_log_text(results: dict[str, Any], manifest_path: Path, results_path: Path) -> str:
    return (
        f"run_name={results['run_name']}\n"
        f"result_kind={results['result_kind']}\n"
        "benchmark_result=false\n"
        "formal_training_run=true\n"
        "full_database_formal_training_run=true\n"
        f"corruption_mode={results['corruption_mode']}\n"
        f"seed={results['random_seed']}\n"
        f"max_steps={results['max_steps']}\n"
        f"optimizer_step_count={results['optimizer_step_count']}\n"
        f"stopped_early={results['stopped_early']}\n"
        f"early_stop_reason={results['early_stop_reason']}\n"
        f"checkpoint_selection_metric={results['checkpoint_selection_metric']}\n"
        f"best_checkpoint={results['best_checkpoint_path']}\n"
        f"latest_checkpoint={results['latest_checkpoint_path']}\n"
        f"manifest={manifest_path}\n"
        f"results={results_path}\n"
        "note=Full-database formal training output only; not benchmark results.\n"
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None
