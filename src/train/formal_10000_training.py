"""First formal bounded 10000-molecule training path."""

from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

from src.models.checkpoint_sampling_smoke import (
    run_validation_checkpoint_training,
    warm_up_checkpoint_sampling_runtime,
)
from src.train import tiny_real as tiny_real_data
from src.utils.config import load_config


FORMAL_10000_CONFIG_PATHS = (
    Path("configs/train/data_scale_check_10000_formal_isotropic.yaml"),
    Path("configs/train/data_scale_check_10000_formal_stiffness.yaml"),
)
FORMAL_10000_SEED_SET = (0, 1, 2)
FORMAL_10000_MAX_MOLECULES = 10_000
FORMAL_10000_SPLIT_COUNTS = {"train": 9000, "val": 500, "test": 500}
FORMAL_10000_MAX_STEPS = 2000
FORMAL_10000_HARD_MAX_STEPS = 2000
FORMAL_10000_PATIENCE = 200
FORMAL_10000_MIN_DELTA = 1.0e-6
FORMAL_10000_LEARNING_RATE = 1.0e-2
FORMAL_10000_SAMPLE_STEP_SIZE = 0.1
FORMAL_10000_RESULT_KIND = "tiny_real_data_formal_10000_training_not_benchmark"
FORMAL_10000_SUMMARY_RESULT_KIND = "tiny_real_data_formal_10000_training_multiseed_summary_not_benchmark"
FORMAL_10000_EXECUTION_MODE = "cpu_formal_10000_training_validation_selected_checkpoint"
FORMAL_10000_CHECKPOINT_SCOPE = "reusable_formal_10000_validation_selected_checkpoint"
FORMAL_10000_CHECKPOINT_RESULT_KIND = "reusable_formal_10000_validation_selected_checkpoint_not_benchmark"
FORMAL_10000_RUNTIME_SCOPE = "formal_train_validation_select_save_reload_sample_after_cpu_warmup"
FORMAL_10000_SUMMARY_PATH = Path("experiments/results/geom_formal_10000_training_multiseed_summary.json")
FORMAL_10000_REPORT_PATH = Path("reports/tables/geom_formal_10000_training_multiseed_summary.md")


def run_formal_10000_training_configs(
    config_paths: tuple[str | Path, ...] = FORMAL_10000_CONFIG_PATHS,
    output_root: str | Path | None = None,
    seed: int = 0,
) -> dict[str, dict[str, Path]]:
    outputs = {}
    for config_path in config_paths:
        config = load_config(config_path)
        if output_root is not None:
            config["experiment"]["output_root"] = str(output_root)
        config["experiment"]["seed"] = int(seed)
        paths = run_formal_10000_training(config=config, config_path=Path(config_path))
        outputs[paths["mode"]] = {
            "manifest": paths["manifest"],
            "results": paths["results"],
            "log": paths["log"],
            "best_checkpoint": paths["best_checkpoint"],
            "latest_checkpoint": paths["latest_checkpoint"],
        }
    return outputs


def run_formal_10000_training_multiseed(
    seed_set: tuple[int, ...] = FORMAL_10000_SEED_SET,
    config_paths: tuple[str | Path, ...] = FORMAL_10000_CONFIG_PATHS,
    output_root: str | Path | None = None,
) -> dict[str, dict[int, dict[str, Path]]]:
    _validate_seed_set(seed_set)
    outputs: dict[str, dict[int, dict[str, Path]]] = {}
    for seed in seed_set:
        seed_outputs = run_formal_10000_training_configs(
            config_paths=config_paths,
            output_root=output_root,
            seed=int(seed),
        )
        for mode, paths in seed_outputs.items():
            outputs.setdefault(mode, {})[int(seed)] = paths
    return outputs


def run_formal_10000_training(
    config: dict[str, Any],
    config_path: str | Path | None = None,
) -> dict[str, Path]:
    config = copy.deepcopy(config)
    config_path_obj = Path(config_path) if config_path is not None else None
    _validate_formal_10000_config(config)

    output_root = Path(config["experiment"]["output_root"])
    seed = int(config["experiment"]["seed"])
    corruption_mode = config["corruption"]["mode"]
    max_steps = int(config["train"]["max_steps"])
    hard_max_steps = int(config["train"]["hard_max_steps"])
    learning_rate = float(config["train"]["learning_rate"])
    sample_step_size = float(config["sample"]["sample_step_size"])
    processed_artifact_path = Path(config["data"]["processed_artifact_path"])
    split_path = Path(config["data"]["split_path"])
    run_name = _run_name(corruption_mode=corruption_mode, seed=seed, max_steps=max_steps)
    manifest_path = output_root / "manifests" / f"{run_name}.json"
    results_path = output_root / "results" / f"{run_name}.json"
    log_path = output_root / "logs" / f"{run_name}.log"
    best_checkpoint_path = output_root / "checkpoints" / f"{run_name}_best.pt"
    latest_checkpoint_path = output_root / "checkpoints" / f"{run_name}_latest.pt"

    config["experiment"]["name"] = run_name
    artifact = tiny_real_data._load_tiny_real_artifact(processed_artifact_path)
    records = tiny_real_data._select_records(artifact["molecules"], FORMAL_10000_MAX_MOLECULES)
    split = tiny_real_data._load_tiny_real_split(split_path)
    split_membership = tiny_real_data._validate_split_coverage(split, records)
    split_counts = {key: len(split[key]) for key in ["train", "val", "test"]}
    if split_counts != FORMAL_10000_SPLIT_COUNTS:
        raise ValueError(f"formal 10000 training requires split counts {FORMAL_10000_SPLIT_COUNTS}, got {split_counts}")

    all_samples = tiny_real_data._build_flow_matching_samples(
        config,
        records,
        random.Random(seed),
        split_membership,
    )
    train_samples = [sample for sample in all_samples if sample["split"] == "train"]
    validation_samples = [sample for sample in all_samples if sample["split"] == "val"]
    if len(train_samples) != FORMAL_10000_SPLIT_COUNTS["train"]:
        raise ValueError("formal 10000 training requires the fixed 9000-molecule train split")
    if len(validation_samples) != FORMAL_10000_SPLIT_COUNTS["val"]:
        raise ValueError("formal 10000 training requires the fixed 500-molecule validation split")

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
        checkpoint_result_kind=FORMAL_10000_CHECKPOINT_RESULT_KIND,
        checkpoint_scope=FORMAL_10000_CHECKPOINT_SCOPE,
    )
    elapsed = time.time() - start
    scheduler = "slurm" if os.environ.get("SLURM_JOB_ID") else None
    slurm_account = os.environ.get("STIFFLOW_SLURM_ACCOUNT") if scheduler else None

    common = {
        "run_name": run_name,
        "dataset_name": "geom_drugs",
        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
        "dataset_subset_scope": "data_scale_check_10000_processed_dry_run",
        "processed_artifact_path": str(processed_artifact_path),
        "split_path": str(split_path),
        "random_seed": seed,
        "corruption_mode": corruption_mode,
        "model_name": metrics["model_name"],
        "backbone": config["model"]["backbone"],
        "execution_mode": FORMAL_10000_EXECUTION_MODE,
        "result_kind": FORMAL_10000_RESULT_KIND,
        "benchmark_result": False,
        "hardware": "cpu_only",
        "gpu_used": False,
        "scheduler": scheduler,
        "slurm_account": slurm_account,
        "pilot_run": False,
        "bounded_pilot": False,
        "formal_training_run": True,
        "bounded_formal_training_run": True,
        "data_scale_check_run": True,
        "data_scale_check_10000_run": True,
        "dry_run": False,
        "training_executed": True,
        "model_training_started": True,
        "training_loop_executed": True,
        "formal_training_loop_executed": True,
        "bounded_real_training_loop_executed": True,
        "larger_real_training_run_started": True,
        "max_steps": max_steps,
        "hard_max_steps": hard_max_steps,
        "optimizer_created": True,
        "optimizer_name": metrics["optimizer_name"],
        "learning_rate": learning_rate,
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
        "checkpoint_scope": FORMAL_10000_CHECKPOINT_SCOPE,
        "sampling_executed": True,
        "sample_step_count": 1,
        "sample_step_size": sample_step_size,
        "forward_executed": True,
        "loss_computed": True,
        "training_sample_count": len(train_samples),
        "validation_sample_count": len(validation_samples),
        "sample_count": len(all_samples),
        "runtime_scope": FORMAL_10000_RUNTIME_SCOPE,
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
            "max_molecules": FORMAL_10000_MAX_MOLECULES,
            "hard_max_molecules": FORMAL_10000_MAX_MOLECULES,
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
            "runtime_scope": FORMAL_10000_RUNTIME_SCOPE,
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
        "notes": _formal_notes(max_steps=max_steps),
    }
    results = {
        **common,
        "dataset": {
            "name": "geom_drugs",
            "kind": "geom_drugs_tiny_processed_dry_run",
            "subset_scope": "data_scale_check_10000_processed_dry_run",
            "molecule_count": len(records),
            "split_type": split["split_type"],
            "split_scope": split.get("split_scope"),
            "split_counts": split_counts,
            "degenerate_tiny_split": split.get("degenerate_tiny_split"),
        },
        "flow_matching": config["flow_matching"],
        "metrics": {
            "metric_scope": "geom_drugs_formal_10000_training_checkpoint_sampling",
            **metrics,
        },
        "runtime_seconds": elapsed,
        "runtime_backend": runtime_backend,
        "runtime_warmup_scope": "full_cpu_formal_training_checkpoint_sampling_operation",
        "notes": _formal_notes(max_steps=max_steps),
    }

    for directory in [manifest_path.parent, results_path.parent, log_path.parent, best_checkpoint_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_path.write_text(_formal_log_text(results, manifest_path, results_path), encoding="utf-8")
    return {
        "mode": corruption_mode,
        "manifest": manifest_path,
        "results": results_path,
        "log": log_path,
        "best_checkpoint": best_checkpoint_path,
        "latest_checkpoint": latest_checkpoint_path,
    }


def write_formal_10000_training_multiseed_summary(
    result_paths: list[str | Path],
    summary_path: str | Path = FORMAL_10000_SUMMARY_PATH,
    report_path: str | Path = FORMAL_10000_REPORT_PATH,
    expected_seeds: tuple[int, ...] = FORMAL_10000_SEED_SET,
) -> dict[str, Path]:
    rows = [_summary_row(Path(path)) for path in result_paths]
    _validate_summary_rows(rows, expected_seeds=expected_seeds)
    rows_by_seed = _rows_by_seed(rows)
    per_seed = [
        {
            "seed": seed,
            "metric_comparisons": _metric_comparisons(rows_by_seed[seed]),
        }
        for seed in sorted(rows_by_seed)
    ]
    aggregate = _aggregate_metric_comparisons(per_seed)
    summary = {
        "summary_name": "geom_formal_10000_training_multiseed_summary",
        "result_kind": FORMAL_10000_SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "dataset_name": "geom_drugs",
        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
        "dataset_subset_scope": "data_scale_check_10000_processed_dry_run",
        "hardware": "cpu_only",
        "gpu_used": False,
        "formal_training_summary": True,
        "bounded_formal_training_run": True,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "raw_data_modified": False,
        "environment_modified": False,
        "seed_set": list(expected_seeds),
        "max_molecules": FORMAL_10000_MAX_MOLECULES,
        "split_counts": FORMAL_10000_SPLIT_COUNTS,
        "max_steps": FORMAL_10000_MAX_STEPS,
        "hard_max_steps": FORMAL_10000_HARD_MAX_STEPS,
        "checkpoint_selection_metric": "validation_loss_after_step",
        "input_result_paths": [str(path) for path in result_paths],
        "row_count": len(rows),
        "rows": rows,
        "per_seed_comparisons": per_seed,
        "aggregate_metric_comparisons": aggregate,
        "stiffness_consistency": _consistency(aggregate),
        "notes": [
            "Reporting-only aggregation of first formal bounded 10000-molecule training outputs.",
            "The summary is not a benchmark result and does not run training, sampling, preprocessing, checkpoint loading, SLURM jobs, GPUs, or environment changes.",
            "Metrics are validation/checkpoint-selection and one-step sampling diagnostics for the frozen 10000-molecule setup.",
        ],
    }
    summary_path = Path(summary_path)
    report_path = Path(report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_summary_markdown(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def _validate_formal_10000_config(config: dict[str, Any]) -> None:
    if config["data"]["dataset_name"] != "geom_drugs":
        raise ValueError("formal 10000 training requires dataset_name=geom_drugs")
    if config["data"].get("dataset_kind") != "geom_drugs_tiny_processed_dry_run":
        raise ValueError("formal 10000 training must use the existing bounded processed artifact schema")
    if Path(config["data"]["processed_artifact_path"]) != Path(
        "data/processed/geom_drugs_subset/data_scale_check_10000/molecules.json"
    ):
        raise ValueError("formal 10000 training must use the frozen 10000-molecule processed artifact")
    if Path(config["data"]["split_path"]) != Path("data/splits/geom_drugs_data_scale_check_10000/random_seed0/split.json"):
        raise ValueError("formal 10000 training must use the frozen 9000/500/500 split artifact")
    if int(config["data"]["subset"]["max_molecules"]) != FORMAL_10000_MAX_MOLECULES:
        raise ValueError("formal 10000 training must use exactly 10000 molecules")
    if int(config["data"]["subset"].get("max_conformers_per_mol", 0)) != 1:
        raise ValueError("formal 10000 training must preserve one conformer per molecule")
    if config["data"]["subset"].get("tag") != "data_scale_check_10000_processed_dry_run":
        raise ValueError("formal 10000 training must use the frozen 10000-molecule subset tag")
    if config["corruption"]["mode"] not in {"isotropic", "stiffness_structured"}:
        raise ValueError(f"unsupported formal 10000 corruption mode: {config['corruption']['mode']}")
    if config["model"].get("backbone") != "minimal_linear_velocity_pilot":
        raise ValueError("formal 10000 training must keep the existing minimal linear backbone")
    if config["model"].get("pilot_model") != "single_linear_velocity_pilot":
        raise ValueError("formal 10000 training must keep pilot_model=single_linear_velocity_pilot")
    if config["train"].get("enabled") is not True:
        raise ValueError("formal 10000 training requires train.enabled=true")
    if config["train"].get("formal_training") is not True:
        raise ValueError("formal 10000 training requires train.formal_training=true")
    if config["train"].get("pilot_only") is not False:
        raise ValueError("formal 10000 training must not be labeled pilot_only")
    if config["train"].get("batch_scope") != "data_scale_check_train_split":
        raise ValueError("formal 10000 training must keep the train split batch scope")
    if int(config["train"]["max_steps"]) != FORMAL_10000_MAX_STEPS:
        raise ValueError(f"formal 10000 max_steps must remain fixed at {FORMAL_10000_MAX_STEPS}")
    if int(config["train"]["hard_max_steps"]) != FORMAL_10000_HARD_MAX_STEPS:
        raise ValueError(f"formal 10000 hard_max_steps must remain fixed at {FORMAL_10000_HARD_MAX_STEPS}")
    if float(config["train"]["learning_rate"]) != FORMAL_10000_LEARNING_RATE:
        raise ValueError("formal 10000 training must keep the existing SGD learning rate")
    if config["train"].get("optimizer") != "SGD":
        raise ValueError("formal 10000 training must keep optimizer=SGD")
    if config["train"].get("checkpointing_enabled") is not True:
        raise ValueError("formal 10000 training requires checkpointing_enabled=true")
    if config["train"].get("checkpoint_scope") != FORMAL_10000_CHECKPOINT_SCOPE:
        raise ValueError("formal 10000 training must use the reusable formal checkpoint scope")
    early_stop = config["train"].get("early_stop")
    if not isinstance(early_stop, dict) or early_stop.get("enabled") is not True:
        raise ValueError("formal 10000 training requires validation early stopping")
    if early_stop.get("monitor") != "validation_loss_after_step":
        raise ValueError("formal 10000 early-stop monitor must be validation_loss_after_step")
    if int(early_stop.get("patience", 0)) != FORMAL_10000_PATIENCE:
        raise ValueError(f"formal 10000 patience must remain fixed at {FORMAL_10000_PATIENCE}")
    if float(early_stop.get("min_delta", -1.0)) != FORMAL_10000_MIN_DELTA:
        raise ValueError(f"formal 10000 min_delta must remain fixed at {FORMAL_10000_MIN_DELTA}")
    if config["sample"].get("enabled") is not True:
        raise ValueError("formal 10000 training must keep one-step sampling enabled")
    if config["sample"].get("sample_step_count") != 1:
        raise ValueError("formal 10000 training must keep sample_step_count=1")
    if float(config["sample"]["sample_step_size"]) != FORMAL_10000_SAMPLE_STEP_SIZE:
        raise ValueError("formal 10000 training must keep sample_step_size=0.1")
    if int(config["eval"].get("eval_subset_size", 0)) != FORMAL_10000_MAX_MOLECULES:
        raise ValueError("formal 10000 eval_subset_size must match the frozen 10000-molecule scope")


def _validate_seed_set(seed_set: tuple[int, ...]) -> None:
    if tuple(seed_set) != FORMAL_10000_SEED_SET:
        raise ValueError(f"formal 10000 training seed set must remain fixed at {FORMAL_10000_SEED_SET}")


def _run_name(corruption_mode: str, seed: int, max_steps: int) -> str:
    if corruption_mode == "isotropic":
        label = "isotropic"
    elif corruption_mode == "stiffness_structured":
        label = "stiffness"
    else:
        raise ValueError(f"Unsupported formal 10000 training mode: {corruption_mode}")
    return f"geom_formal_10000_train_{label}_mols10000_steps{int(max_steps)}_seed{seed}"


def _summary_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("result_kind") != FORMAL_10000_RESULT_KIND:
        raise ValueError(f"Not a formal 10000 training result: {path}")
    metrics = payload["metrics"]
    sampled = metrics["sampled_molecules"]
    return {
        "source_path": str(path),
        "run_name": payload["run_name"],
        "corruption_mode": payload["corruption_mode"],
        "random_seed": int(payload["random_seed"]),
        "benchmark_result": payload["benchmark_result"],
        "hardware": payload["hardware"],
        "gpu_used": payload["gpu_used"],
        "scheduler": payload.get("scheduler"),
        "slurm_account": payload.get("slurm_account"),
        "dataset_subset_scope": payload["dataset"]["subset_scope"],
        "split_counts": payload["dataset"]["split_counts"],
        "sample_count": int(payload["sample_count"]),
        "training_sample_count": int(payload["training_sample_count"]),
        "validation_sample_count": int(payload["validation_sample_count"]),
        "max_steps": int(payload["max_steps"]),
        "hard_max_steps": int(payload["hard_max_steps"]),
        "optimizer_name": payload["optimizer_name"],
        "learning_rate": float(payload["learning_rate"]),
        "optimizer_step_count": int(payload["optimizer_step_count"]),
        "stopped_early": bool(payload["stopped_early"]),
        "early_stop_reason": payload["early_stop_reason"],
        "checkpoint_reusable": bool(payload["checkpoint_reusable"]),
        "checkpoint_selected_by_validation": bool(payload["checkpoint_selected_by_validation"]),
        "checkpoint_scope": payload["checkpoint_scope"],
        "sample_step_count": int(payload["sample_step_count"]),
        "sample_step_size": float(payload["sample_step_size"]),
        "best_validation_loss": float(metrics["best_validation_loss"]),
        "best_validation_loss_step": int(metrics["best_validation_loss_step"]),
        "reloaded_validation_loss": float(metrics["reloaded_validation_loss"]),
        "reloaded_train_loss": float(metrics["reloaded_train_loss"]),
        "sampled_rmsd_overall_mean": _mean_sampled_rmsd(sampled),
        "sampled_rmsd_by_split_mean": {
            split: _mean_sampled_rmsd([item for item in sampled if item["split"] == split])
            for split in ["train", "val", "test"]
        },
    }


def _validate_summary_rows(rows: list[dict[str, Any]], expected_seeds: tuple[int, ...]) -> None:
    expected_pairs = {
        ("isotropic", int(seed)) for seed in expected_seeds
    } | {
        ("stiffness_structured", int(seed)) for seed in expected_seeds
    }
    observed_pairs = {(row["corruption_mode"], row["random_seed"]) for row in rows}
    if observed_pairs != expected_pairs:
        raise ValueError(f"formal 10000 summary rows do not match expected mode/seed pairs: {observed_pairs}")
    for row in rows:
        if row["benchmark_result"] is not False or row["hardware"] != "cpu_only" or row["gpu_used"] is not False:
            raise ValueError("formal 10000 summary inputs must be non-benchmark CPU-only results")
        if row["dataset_subset_scope"] != "data_scale_check_10000_processed_dry_run":
            raise ValueError("formal 10000 summary subset scope mismatch")
        if row["split_counts"] != FORMAL_10000_SPLIT_COUNTS:
            raise ValueError("formal 10000 summary split counts mismatch")
        if row["sample_count"] != FORMAL_10000_MAX_MOLECULES:
            raise ValueError("formal 10000 summary sample count mismatch")
        if row["training_sample_count"] != FORMAL_10000_SPLIT_COUNTS["train"]:
            raise ValueError("formal 10000 summary training count mismatch")
        if row["validation_sample_count"] != FORMAL_10000_SPLIT_COUNTS["val"]:
            raise ValueError("formal 10000 summary validation count mismatch")
        if row["max_steps"] != FORMAL_10000_MAX_STEPS or row["hard_max_steps"] != FORMAL_10000_HARD_MAX_STEPS:
            raise ValueError("formal 10000 summary step cap mismatch")
        if row["optimizer_name"] != "SGD" or row["learning_rate"] != FORMAL_10000_LEARNING_RATE:
            raise ValueError("formal 10000 summary optimizer mismatch")
        if row["checkpoint_scope"] != FORMAL_10000_CHECKPOINT_SCOPE:
            raise ValueError("formal 10000 summary checkpoint scope mismatch")
        if row["checkpoint_reusable"] is not True or row["checkpoint_selected_by_validation"] is not True:
            raise ValueError("formal 10000 summary requires reusable validation-selected checkpoints")
        if row["sample_step_count"] != 1 or row["sample_step_size"] != FORMAL_10000_SAMPLE_STEP_SIZE:
            raise ValueError("formal 10000 summary sampling setting mismatch")


def _rows_by_seed(rows: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["random_seed"], {})[row["corruption_mode"]] = row
    for seed, by_mode in grouped.items():
        if set(by_mode) != {"isotropic", "stiffness_structured"}:
            raise ValueError(f"formal 10000 seed {seed} does not have both modes")
    return grouped


def _metric_comparisons(by_mode: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    isotropic = by_mode["isotropic"]
    stiffness = by_mode["stiffness_structured"]
    rows = [
        _lower_is_better("best_validation_loss", isotropic["best_validation_loss"], stiffness["best_validation_loss"]),
        _lower_is_better(
            "reloaded_validation_loss",
            isotropic["reloaded_validation_loss"],
            stiffness["reloaded_validation_loss"],
        ),
        _lower_is_better(
            "sampled_rmsd_to_target_overall_mean",
            isotropic["sampled_rmsd_overall_mean"],
            stiffness["sampled_rmsd_overall_mean"],
        ),
    ]
    for split in ["train", "val", "test"]:
        rows.append(
            _lower_is_better(
                f"sampled_rmsd_to_target_{split}_mean",
                isotropic["sampled_rmsd_by_split_mean"][split],
                stiffness["sampled_rmsd_by_split_mean"][split],
            )
        )
    return rows


def _aggregate_metric_comparisons(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "best_validation_loss",
        "reloaded_validation_loss",
        "sampled_rmsd_to_target_overall_mean",
        "sampled_rmsd_to_target_train_mean",
        "sampled_rmsd_to_target_val_mean",
        "sampled_rmsd_to_target_test_mean",
    ]
    rows = []
    for metric in metrics:
        metric_rows = [
            row
            for seed_row in per_seed
            for row in seed_row["metric_comparisons"]
            if row["metric"] == metric
        ]
        isotropic_values = [row["isotropic"] for row in metric_rows]
        stiffness_values = [row["stiffness_structured"] for row in metric_rows]
        outcomes = [row["stiffness_outcome"] for row in metric_rows]
        rows.append(
            {
                "metric": metric,
                "isotropic_mean": _mean_values(isotropic_values),
                "stiffness_structured_mean": _mean_values(stiffness_values),
                "stiffness_minus_isotropic_mean": _mean_values(stiffness_values)
                - _mean_values(isotropic_values),
                "stiffness_outcome_by_seed": outcomes,
                "stiffness_win_count": outcomes.count("better"),
                "stiffness_tie_count": outcomes.count("tied"),
                "stiffness_loss_count": outcomes.count("worse"),
                "seed_count": len(metric_rows),
                "direction": "lower_is_better",
            }
        )
    return rows


def _consistency(aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = [outcome for row in aggregate for outcome in row["stiffness_outcome_by_seed"]]
    return {
        "stiffness_win_count": outcomes.count("better"),
        "stiffness_tie_count": outcomes.count("tied"),
        "stiffness_loss_count": outcomes.count("worse"),
        "tracked_metric_seed_comparison_count": len(outcomes),
        "verdict": "stiffness_wins_consistently"
        if outcomes and outcomes.count("better") == len(outcomes)
        else "mixed_or_inconclusive",
    }


def _lower_is_better(metric: str, isotropic: float, stiffness: float) -> dict[str, Any]:
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


def _mean_values(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute mean over an empty value list")
    return sum(float(value) for value in values) / len(values)


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Formal 10000-Molecule Training Multi-Seed Summary",
        "",
        "This report aggregates first formal bounded 10000-molecule training outputs. It is not a benchmark result.",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| result_kind | `{summary['result_kind']}` |",
        f"| benchmark_result | `{summary['benchmark_result']}` |",
        f"| max_molecules | `{summary['max_molecules']}` |",
        f"| split_counts | `{summary['split_counts']}` |",
        f"| max_steps | `{summary['max_steps']}` |",
        f"| checkpoint_selection_metric | `{summary['checkpoint_selection_metric']}` |",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Isotropic mean | Stiffness mean | Stiffness - isotropic | Stiffness wins |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["aggregate_metric_comparisons"]:
        lines.append(
            f"| {row['metric']} | {row['isotropic_mean']:.6g} | "
            f"{row['stiffness_structured_mean']:.6g} | {row['stiffness_minus_isotropic_mean']:.6g} | "
            f"{row['stiffness_win_count']}/{row['seed_count']} |"
        )
    lines.extend(
        [
            "",
            f"Stiffness consistency verdict: `{summary['stiffness_consistency']['verdict']}`.",
            "",
            "Runtime and validation diagnostics are accounting fields only; this report does not make benchmark claims.",
            "",
        ]
    )
    return "\n".join(lines)


def _formal_notes(max_steps: int) -> list[str]:
    return [
        "First formal bounded 10000-molecule GEOM training path, not a benchmark result.",
        f"The run keeps the frozen 10000-molecule artifact, 9000/500/500 split, seed set, model, optimizer, checkpoint/reload pattern, and one-step sampling settings; only the training horizon is extended to {int(max_steps)} with validation-driven checkpoint selection.",
        "The best reusable checkpoint is selected by validation_loss_after_step, then reloaded before the bounded one-step sampling diagnostic.",
        "No raw-data modification, environment mutation, GPU execution, benchmark evaluation, or unbounded training is part of this path.",
    ]


def _formal_log_text(results: dict[str, Any], manifest_path: Path, results_path: Path) -> str:
    return (
        f"run_name={results['run_name']}\n"
        f"result_kind={results['result_kind']}\n"
        "benchmark_result=false\n"
        "formal_training_run=true\n"
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
        "note=Bounded formal 10000-molecule training output only; not benchmark results.\n"
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None
