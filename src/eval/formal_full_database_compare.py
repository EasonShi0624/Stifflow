"""Compare full-database formal seed0 isotropic vs stiffness outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.geom_full_database import FULL_DATABASE_SCOPE
from src.train.formal_10000_training import (
    FORMAL_10000_HARD_MAX_STEPS,
    FORMAL_10000_LEARNING_RATE,
    FORMAL_10000_MAX_STEPS,
    FORMAL_10000_MIN_DELTA,
    FORMAL_10000_PATIENCE,
    FORMAL_10000_SAMPLE_STEP_SIZE,
)
from src.train.formal_full_database_training import (
    FORMAL_FULL_DATABASE_CHECKPOINT_SCOPE,
    FORMAL_FULL_DATABASE_RESULT_KIND,
    FORMAL_FULL_DATABASE_RUNTIME_SCOPE,
)


DEFAULT_RESULT_PATHS = [
    Path("experiments/results/geom_formal_full_database_train_isotropic_steps2000_seed0.json"),
    Path("experiments/results/geom_formal_full_database_train_stiffness_steps2000_seed0.json"),
]
DEFAULT_SCAFFOLD_SEED0_RESULT_PATHS = [
    Path("experiments/results/geom_formal_full_database_scaffold_train_isotropic_steps2000_seed0.json"),
    Path("experiments/results/geom_formal_full_database_scaffold_train_stiffness_steps2000_seed0.json"),
]
DEFAULT_MULTI_SEED_RESULT_PATHS = [
    Path(f"experiments/results/geom_formal_full_database_train_isotropic_steps2000_seed{seed}.json")
    for seed in [0, 1, 2]
] + [
    Path(f"experiments/results/geom_formal_full_database_train_stiffness_steps2000_seed{seed}.json")
    for seed in [0, 1, 2]
]
DEFAULT_SCAFFOLD_MULTI_SEED_RESULT_PATHS = [
    Path(f"experiments/results/geom_formal_full_database_scaffold_train_isotropic_steps2000_seed{seed}.json")
    for seed in [0, 1, 2]
] + [
    Path(f"experiments/results/geom_formal_full_database_scaffold_train_stiffness_steps2000_seed{seed}.json")
    for seed in [0, 1, 2]
]
DEFAULT_SUMMARY_PATH = Path("experiments/results/geom_formal_full_database_seed0_summary.json")
DEFAULT_REPORT_PATH = Path("reports/tables/geom_formal_full_database_seed0_summary.md")
DEFAULT_SCAFFOLD_SEED0_SUMMARY_PATH = Path(
    "experiments/results/geom_formal_full_database_scaffold_seed0_summary.json"
)
DEFAULT_SCAFFOLD_SEED0_REPORT_PATH = Path(
    "reports/tables/geom_formal_full_database_scaffold_seed0_summary.md"
)
DEFAULT_MULTI_SEED_SUMMARY_PATH = Path("experiments/results/geom_formal_full_database_multiseed_summary.json")
DEFAULT_MULTI_SEED_REPORT_PATH = Path("reports/tables/geom_formal_full_database_multiseed_summary.md")
DEFAULT_SCAFFOLD_MULTI_SEED_SUMMARY_PATH = Path(
    "experiments/results/geom_formal_full_database_scaffold_multiseed_summary.json"
)
DEFAULT_SCAFFOLD_MULTI_SEED_REPORT_PATH = Path(
    "reports/tables/geom_formal_full_database_scaffold_multiseed_summary.md"
)
SUMMARY_RESULT_KIND = "geom_drugs_full_database_formal_seed0_comparison_not_benchmark"
SCAFFOLD_SEED0_SUMMARY_RESULT_KIND = "geom_drugs_full_database_scaffold_formal_seed0_comparison_not_benchmark"
MULTI_SEED_SUMMARY_RESULT_KIND = "geom_drugs_full_database_formal_multiseed_summary_not_benchmark"
SCAFFOLD_MULTI_SEED_SUMMARY_RESULT_KIND = (
    "geom_drugs_full_database_scaffold_formal_multiseed_summary_not_benchmark"
)
COMPARABLE_METRICS = (
    "best_validation_loss",
    "reloaded_validation_loss",
    "sampled_rmsd_to_target_overall_mean",
    "sampled_rmsd_to_target_train_mean",
    "sampled_rmsd_to_target_val_mean",
    "sampled_rmsd_to_target_test_mean",
)
TIE_TOLERANCE = 1.0e-12


def write_formal_full_database_seed0_comparison(
    result_paths: list[str | Path] | None = None,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Path]:
    paths = [Path(path) for path in (result_paths or DEFAULT_RESULT_PATHS)]
    payloads = [_load_result(path, expected_split_type="random") for path in paths]
    rows = [_row_from_result(path, payload) for path, payload in zip(paths, payloads)]
    by_mode = {row["corruption_mode"]: row for row in rows}
    if set(by_mode) != {"isotropic", "stiffness_structured"}:
        raise ValueError(f"Expected isotropic and stiffness_structured results, got {sorted(by_mode)}")
    _validate_comparable(rows, expected_split_type="random", require_cpu_only_inputs=True)
    metric_comparisons = _metric_comparisons(by_mode)
    verdict = _directional_verdict(metric_comparisons)
    summary = {
        "summary_name": "geom_formal_full_database_seed0_summary",
        "result_kind": SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_cpu_full_database_formal_seed0_report",
        "hardware": "cpu_only",
        "gpu_used": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "input_result_paths": [str(path) for path in paths],
        "row_count": len(rows),
        "dataset_scope": FULL_DATABASE_SCOPE,
        "dataset": {
            "name": "geom_drugs",
            "kind": rows[0]["dataset_kind"],
            "subset_scope": rows[0]["dataset_subset_scope"],
            "molecule_count": rows[0]["molecule_count"],
            "split_type": rows[0]["split_type"],
            "split_scope": rows[0]["split_scope"],
            "split_counts": rows[0]["split_counts"],
        },
        "training_horizon": {
            "max_steps": FORMAL_10000_MAX_STEPS,
            "hard_max_steps": FORMAL_10000_HARD_MAX_STEPS,
            "optimizer": "SGD",
            "learning_rate": FORMAL_10000_LEARNING_RATE,
            "early_stop_monitor": "validation_loss_after_step",
            "early_stop_patience": FORMAL_10000_PATIENCE,
            "early_stop_min_delta": FORMAL_10000_MIN_DELTA,
            "sample_step_count": 1,
            "sample_step_size": FORMAL_10000_SAMPLE_STEP_SIZE,
        },
        "rows": rows,
        "metric_comparisons": metric_comparisons,
        "directional_verdict": verdict,
        "safety_labels": {
            "benchmark_result": False,
            "non_benchmark": True,
            "full_database_formal_training_comparison": True,
            "training_executed_by_summary": False,
            "sampling_executed_by_summary": False,
            "raw_data_modified": False,
            "environment_modified": False,
            "gpu_used": False,
            "result_scope": "full_database_seed0_formal_comparison_only",
        },
        "notes": [
            "Reporting-only comparison of completed full-database GEOM-Drugs formal seed0 outputs.",
            "The summary validates that both inputs use the same full-database one-conformer scope, split, seed, model, optimizer, learning rate, checkpoint/reload behavior, one-step sampling diagnostic, validation-selected checkpoint rule, and hard step cap.",
            "All lower-is-better comparisons are diagnostics from existing result JSONs only.",
            "This is explicitly not a benchmark result and does not run training, sampling, preprocessing, SLURM jobs, GPUs, or environment changes.",
        ],
    }
    summary_path = Path(summary_path)
    report_path = Path(report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_report(summary, title="Full-Database Formal Seed0 Comparison"), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def write_formal_full_database_scaffold_seed0_comparison(
    result_paths: list[str | Path] | None = None,
    summary_path: str | Path = DEFAULT_SCAFFOLD_SEED0_SUMMARY_PATH,
    report_path: str | Path = DEFAULT_SCAFFOLD_SEED0_REPORT_PATH,
) -> dict[str, Path]:
    paths = [Path(path) for path in (result_paths or DEFAULT_SCAFFOLD_SEED0_RESULT_PATHS)]
    payloads = [_load_result(path, expected_split_type="scaffold") for path in paths]
    rows = [_row_from_result(path, payload) for path, payload in zip(paths, payloads)]
    by_mode = {row["corruption_mode"]: row for row in rows}
    if set(by_mode) != {"isotropic", "stiffness_structured"}:
        raise ValueError(f"Expected isotropic and stiffness_structured results, got {sorted(by_mode)}")
    _validate_comparable(rows, expected_split_type="scaffold", require_cpu_only_inputs=False)
    metric_comparisons = _metric_comparisons(by_mode)
    verdict = _directional_verdict(metric_comparisons)
    summary = {
        "summary_name": "geom_formal_full_database_scaffold_seed0_summary",
        "result_kind": SCAFFOLD_SEED0_SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_cpu_full_database_scaffold_formal_seed0_report",
        "hardware": "cpu_only",
        "gpu_used": False,
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "input_result_paths": [str(path) for path in paths],
        "row_count": len(rows),
        "dataset_scope": FULL_DATABASE_SCOPE,
        "dataset": {
            "name": "geom_drugs",
            "kind": rows[0]["dataset_kind"],
            "subset_scope": rows[0]["dataset_subset_scope"],
            "molecule_count": rows[0]["molecule_count"],
            "split_type": rows[0]["split_type"],
            "split_scope": rows[0]["split_scope"],
            "split_counts": rows[0]["split_counts"],
        },
        "training_horizon": {
            "max_steps": FORMAL_10000_MAX_STEPS,
            "hard_max_steps": FORMAL_10000_HARD_MAX_STEPS,
            "optimizer": "SGD",
            "learning_rate": FORMAL_10000_LEARNING_RATE,
            "early_stop_monitor": "validation_loss_after_step",
            "early_stop_patience": FORMAL_10000_PATIENCE,
            "early_stop_min_delta": FORMAL_10000_MIN_DELTA,
            "sample_step_count": 1,
            "sample_step_size": FORMAL_10000_SAMPLE_STEP_SIZE,
        },
        "rows": rows,
        "metric_comparisons": metric_comparisons,
        "directional_verdict": verdict,
        "safety_labels": {
            "benchmark_result": False,
            "non_benchmark": True,
            "full_database_scaffold_formal_training_comparison": True,
            "training_executed_by_summary": False,
            "sampling_executed_by_summary": False,
            "raw_data_modified": False,
            "environment_modified": False,
            "gpu_used_by_summary": False,
            "result_scope": "full_database_scaffold_seed0_formal_comparison_only",
        },
        "notes": [
            "Reporting-only comparison of completed full-database GEOM-Drugs scaffold-split formal seed0 outputs.",
            "The summary validates that both inputs use the same full-database one-conformer scope, scaffold split, seed, model, optimizer, learning rate, checkpoint/reload behavior, one-step sampling diagnostic, validation-selected checkpoint rule, and hard step cap.",
            "The compared inputs were GPU-enabled scheduled training runs; this summary itself does not use GPUs or submit jobs.",
            "All lower-is-better comparisons are diagnostics from existing result JSONs only.",
            "This is explicitly not a benchmark result and does not run training, sampling, preprocessing, SLURM jobs, GPUs, or environment changes.",
        ],
    }
    summary_path = Path(summary_path)
    report_path = Path(report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(
        _markdown_report(summary, title="Full-Database Scaffold Formal Seed0 Comparison"),
        encoding="utf-8",
    )
    return {"summary": summary_path, "report": report_path}


def write_formal_full_database_multiseed_summary(
    result_paths: list[str | Path] | None = None,
    summary_path: str | Path = DEFAULT_MULTI_SEED_SUMMARY_PATH,
    report_path: str | Path = DEFAULT_MULTI_SEED_REPORT_PATH,
    expected_seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, Path]:
    paths = [Path(path) for path in (result_paths or DEFAULT_MULTI_SEED_RESULT_PATHS)]
    payloads = [_load_result(path, expected_split_type="random") for path in paths]
    rows = [_row_from_result(path, payload) for path, payload in zip(paths, payloads)]
    _validate_multiseed_rows(rows, expected_seeds=expected_seeds, expected_split_type="random")
    rows_by_seed = _rows_by_seed(rows)
    per_seed = [
        {
            "seed": seed,
            "rows": rows_by_seed[seed],
            "metric_comparisons": _metric_comparisons(rows_by_seed[seed]),
            "stop_reasons": {
                mode: rows_by_seed[seed][mode]["stop_reason"]
                for mode in ["isotropic", "stiffness_structured"]
            },
        }
        for seed in sorted(rows_by_seed)
    ]
    aggregate = _aggregate_metric_comparisons(per_seed)
    summary = {
        "summary_name": "geom_formal_full_database_multiseed_summary",
        "result_kind": MULTI_SEED_SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_full_database_formal_multiseed_report",
        "hardware": "mixed_completed_results",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "input_result_paths": [str(path) for path in paths],
        "row_count": len(rows),
        "seed_set": list(expected_seeds),
        "dataset_scope": FULL_DATABASE_SCOPE,
        "dataset": {
            "name": "geom_drugs",
            "kind": rows[0]["dataset_kind"],
            "subset_scope": rows[0]["dataset_subset_scope"],
            "molecule_count": rows[0]["molecule_count"],
            "split_type": rows[0]["split_type"],
            "split_scope": rows[0]["split_scope"],
            "split_counts": rows[0]["split_counts"],
        },
        "training_horizon": {
            "max_steps": FORMAL_10000_MAX_STEPS,
            "hard_max_steps": FORMAL_10000_HARD_MAX_STEPS,
            "optimizer": "SGD",
            "learning_rate": FORMAL_10000_LEARNING_RATE,
            "early_stop_monitor": "validation_loss_after_step",
            "early_stop_patience": FORMAL_10000_PATIENCE,
            "early_stop_min_delta": FORMAL_10000_MIN_DELTA,
            "sample_step_count": 1,
            "sample_step_size": FORMAL_10000_SAMPLE_STEP_SIZE,
        },
        "rows": rows,
        "per_seed_comparisons": per_seed,
        "aggregate_metric_comparisons": aggregate,
        "directional_verdict": _multiseed_directional_verdict(per_seed, aggregate),
        "safety_labels": {
            "benchmark_result": False,
            "non_benchmark": True,
            "full_database_formal_training_multiseed_summary": True,
            "training_executed_by_summary": False,
            "sampling_executed_by_summary": False,
            "raw_data_modified": False,
            "environment_modified": False,
            "gpu_used_by_summary": False,
            "result_scope": "full_database_formal_multiseed_reporting_only",
        },
        "notes": [
            "Reporting-only aggregation of completed full-database GEOM-Drugs formal outputs for seeds 0, 1, and 2.",
            "The summary validates that all inputs use the same full-database one-conformer scope, split counts, model, optimizer, learning rate, checkpoint/reload behavior, one-step sampling diagnostic, validation-selected checkpoint rule, and hard step cap.",
            "Seed 0 was CPU-only and seeds 1/2 were GPU-enabled; hardware metadata is retained as run metadata only and is not interpreted as a benchmark timing comparison.",
            "All lower-is-better comparisons are diagnostics from existing result JSONs only.",
            "This is explicitly not a benchmark result and does not run training, sampling, preprocessing, SLURM jobs, GPUs, or environment changes.",
        ],
    }
    summary_path = Path(summary_path)
    report_path = Path(report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown_multiseed_report(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def write_formal_full_database_scaffold_multiseed_summary(
    result_paths: list[str | Path] | None = None,
    summary_path: str | Path = DEFAULT_SCAFFOLD_MULTI_SEED_SUMMARY_PATH,
    report_path: str | Path = DEFAULT_SCAFFOLD_MULTI_SEED_REPORT_PATH,
    expected_seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, Path]:
    paths = [Path(path) for path in (result_paths or DEFAULT_SCAFFOLD_MULTI_SEED_RESULT_PATHS)]
    payloads = [_load_result(path, expected_split_type="scaffold") for path in paths]
    rows = [_row_from_result(path, payload) for path, payload in zip(paths, payloads)]
    _validate_multiseed_rows(
        rows,
        expected_seeds=expected_seeds,
        expected_split_type="scaffold",
        require_shared_data_paths=True,
    )
    rows_by_seed = _rows_by_seed(rows)
    per_seed = [
        {
            "seed": seed,
            "rows": rows_by_seed[seed],
            "metric_comparisons": _metric_comparisons(rows_by_seed[seed]),
            "stop_reasons": {
                mode: rows_by_seed[seed][mode]["stop_reason"]
                for mode in ["isotropic", "stiffness_structured"]
            },
        }
        for seed in sorted(rows_by_seed)
    ]
    aggregate = _aggregate_metric_comparisons(per_seed)
    summary = {
        "summary_name": "geom_formal_full_database_scaffold_multiseed_summary",
        "result_kind": SCAFFOLD_MULTI_SEED_SUMMARY_RESULT_KIND,
        "benchmark_result": False,
        "execution_mode": "local_full_database_scaffold_formal_multiseed_report",
        "hardware": "completed_scaffold_gpu_results",
        "gpu_used_by_summary": False,
        "scheduler": None,
        "slurm_account": None,
        "raw_data_modified": False,
        "environment_modified": False,
        "training_executed_by_summary": False,
        "sampling_executed_by_summary": False,
        "input_result_paths": [str(path) for path in paths],
        "row_count": len(rows),
        "seed_set": list(expected_seeds),
        "dataset_scope": FULL_DATABASE_SCOPE,
        "dataset": {
            "name": "geom_drugs",
            "kind": rows[0]["dataset_kind"],
            "subset_scope": rows[0]["dataset_subset_scope"],
            "molecule_count": rows[0]["molecule_count"],
            "processed_artifact_path": rows[0]["processed_artifact_path"],
            "split_path": rows[0]["split_path"],
            "split_type": rows[0]["split_type"],
            "split_scope": rows[0]["split_scope"],
            "split_counts": rows[0]["split_counts"],
        },
        "training_horizon": {
            "max_steps": FORMAL_10000_MAX_STEPS,
            "hard_max_steps": FORMAL_10000_HARD_MAX_STEPS,
            "optimizer": "SGD",
            "learning_rate": FORMAL_10000_LEARNING_RATE,
            "early_stop_monitor": "validation_loss_after_step",
            "early_stop_patience": FORMAL_10000_PATIENCE,
            "early_stop_min_delta": FORMAL_10000_MIN_DELTA,
            "sample_step_count": 1,
            "sample_step_size": FORMAL_10000_SAMPLE_STEP_SIZE,
        },
        "rows": rows,
        "per_seed_comparisons": per_seed,
        "aggregate_metric_comparisons": aggregate,
        "directional_verdict": _multiseed_directional_verdict(per_seed, aggregate),
        "safety_labels": {
            "benchmark_result": False,
            "non_benchmark": True,
            "full_database_scaffold_formal_training_multiseed_summary": True,
            "training_executed_by_summary": False,
            "sampling_executed_by_summary": False,
            "raw_data_modified": False,
            "environment_modified": False,
            "gpu_used_by_summary": False,
            "result_scope": "full_database_scaffold_formal_multiseed_reporting_only",
        },
        "notes": [
            "Reporting-only aggregation of completed full-database GEOM-Drugs scaffold-split formal outputs for seeds 0, 1, and 2.",
            "The summary validates that all inputs use the same full-database one-conformer scope, cached scaffold split, model, optimizer, learning rate, checkpoint/reload behavior, one-step sampling diagnostic, validation-selected checkpoint rule, and hard step cap.",
            "All input runs are expected to reuse the cached processed artifact and cached scaffold split; only the training seed varies within each mode.",
            "Hardware metadata is retained as run metadata only and is not interpreted as a benchmark timing comparison.",
            "All lower-is-better comparisons are diagnostics from existing result JSONs only.",
            "This is explicitly not a benchmark result and does not run training, sampling, preprocessing, SLURM jobs, GPUs, or environment changes.",
        ],
    }
    summary_path = Path(summary_path)
    report_path = Path(report_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(
        _markdown_multiseed_report(
            summary,
            title="Full-Database Scaffold Formal Multi-Seed Summary",
        ),
        encoding="utf-8",
    )
    return {"summary": summary_path, "report": report_path}


def _load_result(path: Path, expected_split_type: str = "random") -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Full-database formal result does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("result_kind") != FORMAL_FULL_DATABASE_RESULT_KIND:
        raise ValueError(f"Result is not a full-database formal output: {path}")
    if payload.get("benchmark_result") is not False:
        raise ValueError(f"Result is not benchmark_result=false: {path}")
    if payload.get("dataset_subset_scope") != FULL_DATABASE_SCOPE:
        raise ValueError(f"Result has unexpected dataset scope: {path}")
    if payload.get("hardware") not in {"cpu_only", "gpu_enabled"}:
        raise ValueError(f"Result hardware label is not recognized: {path}")
    if not isinstance(payload.get("gpu_used"), bool):
        raise ValueError(f"Result gpu_used must be boolean: {path}")
    if payload.get("formal_training_run") is not True:
        raise ValueError(f"Result is not labeled as a formal training run: {path}")
    if payload.get("full_database_formal_training_run") is not True:
        raise ValueError(f"Result is not labeled as a full-database formal training run: {path}")
    if payload.get("checkpoint_reusable") is not True:
        raise ValueError(f"Result does not contain a reusable checkpoint label: {path}")
    if payload.get("checkpoint_selected_by_validation") is not True:
        raise ValueError(f"Result does not use validation-selected checkpoints: {path}")
    if payload.get("checkpoint_scope") != FORMAL_FULL_DATABASE_CHECKPOINT_SCOPE:
        raise ValueError(f"Result checkpoint scope mismatch: {path}")
    if payload.get("runtime_scope") != FORMAL_FULL_DATABASE_RUNTIME_SCOPE:
        raise ValueError(f"Result runtime scope mismatch: {path}")
    if payload.get("runtime_seconds_reliable") is not True:
        raise ValueError(f"Result runtime_seconds is not marked reliable: {path}")
    dataset = payload.get("dataset") or {}
    if dataset.get("subset_scope") != FULL_DATABASE_SCOPE or dataset.get("split_scope") != FULL_DATABASE_SCOPE:
        raise ValueError(f"Result dataset/split scope mismatch: {path}")
    if dataset.get("split_type") != expected_split_type:
        raise ValueError(f"Result split_type is not {expected_split_type}: {path}")
    split_counts = dataset.get("split_counts")
    if not isinstance(split_counts, dict) or min(int(split_counts[key]) for key in ["train", "val", "test"]) <= 0:
        raise ValueError(f"Result split counts are invalid: {path}")
    metrics = payload.get("metrics") or {}
    for key in ["best_validation_loss", "reloaded_validation_loss", "sampled_molecules"]:
        if key not in metrics:
            raise ValueError(f"Result metrics missing {key}: {path}")
    sampled = metrics["sampled_molecules"]
    expected_sample_count = int(payload["sample_count"])
    if not isinstance(sampled, list) or len(sampled) != expected_sample_count:
        raise ValueError(f"Result sampled_molecules length does not match sample_count: {path}")
    for item in sampled:
        if item.get("split") not in {"train", "val", "test"}:
            raise ValueError(f"Sampled molecule has invalid split in {path}")
        if not isinstance(item.get("sampled_rmsd_to_target"), (int, float)):
            raise ValueError(f"Sampled molecule missing numeric sampled_rmsd_to_target in {path}")
    return payload


def _row_from_result(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    sampled = metrics["sampled_molecules"]
    return {
        "run_name": payload["run_name"],
        "source_path": str(path),
        "processed_artifact_path": payload["processed_artifact_path"],
        "split_path": payload["split_path"],
        "corruption_mode": payload["corruption_mode"],
        "random_seed": int(payload["random_seed"]),
        "result_kind": payload["result_kind"],
        "model_name": payload["model_name"],
        "backbone": payload["backbone"],
        "hardware": payload["hardware"],
        "gpu_used": bool(payload["gpu_used"]),
        "scheduler": payload.get("scheduler"),
        "slurm_account": payload.get("slurm_account"),
        "dataset_kind": payload["dataset"]["kind"],
        "dataset_subset_scope": payload["dataset"]["subset_scope"],
        "molecule_count": int(payload["dataset"]["molecule_count"]),
        "split_type": payload["dataset"]["split_type"],
        "split_scope": payload["dataset"]["split_scope"],
        "split_counts": {key: int(payload["dataset"]["split_counts"][key]) for key in ["train", "val", "test"]},
        "max_steps": int(payload["max_steps"]),
        "hard_max_steps": int(payload["hard_max_steps"]),
        "executed_train_steps": int(metrics["executed_train_steps"]),
        "optimizer_step_count": int(payload["optimizer_step_count"]),
        "training_sample_count": int(payload["training_sample_count"]),
        "validation_sample_count": int(payload["validation_sample_count"]),
        "sample_count": int(payload["sample_count"]),
        "optimizer_name": payload["optimizer_name"],
        "learning_rate": float(payload["learning_rate"]),
        "early_stop_monitor": payload["early_stop_monitor"],
        "early_stop_patience": int(payload["early_stop_patience"]),
        "early_stop_min_delta": float(payload["early_stop_min_delta"]),
        "stopped_early": bool(payload["stopped_early"]),
        "stop_reason": payload["early_stop_reason"],
        "best_validation_loss": float(metrics["best_validation_loss"]),
        "best_validation_loss_step": int(metrics["best_validation_loss_step"]),
        "reloaded_validation_loss": float(metrics["reloaded_validation_loss"]),
        "sample_step_count": int(payload["sample_step_count"]),
        "sample_step_size": float(payload["sample_step_size"]),
        "sampled_rmsd_to_target_overall_mean": _mean(item["sampled_rmsd_to_target"] for item in sampled),
        "sampled_rmsd_to_target_by_split_mean": _average_rmsd_by_split(sampled),
        "runtime_seconds": float(payload["runtime_seconds"]),
        "runtime_scope": payload["runtime_scope"],
        "runtime_seconds_reliable": bool(payload["runtime_seconds_reliable"]),
        "safety_labels": {
            "benchmark_result": payload["benchmark_result"],
            "hardware": payload["hardware"],
            "gpu_used": payload["gpu_used"],
            "scheduler": payload.get("scheduler"),
            "slurm_account": payload.get("slurm_account"),
            "formal_training_run": payload["formal_training_run"],
            "full_database_formal_training_run": payload["full_database_formal_training_run"],
            "checkpoint_reusable": payload["checkpoint_reusable"],
            "checkpoint_selected_by_validation": payload["checkpoint_selected_by_validation"],
            "checkpoint_scope": payload["checkpoint_scope"],
            "sampling_executed": payload["sampling_executed"],
        },
    }


def _validate_comparable(
    rows: list[dict[str, Any]],
    expected_split_type: str,
    require_cpu_only_inputs: bool,
) -> None:
    baseline = rows[0]
    comparable_fields = [
        "random_seed",
        "dataset_kind",
        "dataset_subset_scope",
        "molecule_count",
        "processed_artifact_path",
        "split_path",
        "split_type",
        "split_scope",
        "split_counts",
        "max_steps",
        "hard_max_steps",
        "optimizer_name",
        "learning_rate",
        "early_stop_monitor",
        "early_stop_patience",
        "early_stop_min_delta",
        "sample_step_count",
        "sample_step_size",
        "training_sample_count",
        "validation_sample_count",
        "sample_count",
        "model_name",
        "backbone",
    ]
    for row in rows:
        for field in comparable_fields:
            if row[field] != baseline[field]:
                raise ValueError(f"Full-database formal results are not comparable on {field}")
        if row["random_seed"] != 0:
            raise ValueError("Full-database formal seed0 comparison requires random_seed=0")
        if row["dataset_subset_scope"] != FULL_DATABASE_SCOPE:
            raise ValueError("Full-database formal comparison requires the full-database scope")
        if row["split_type"] != expected_split_type:
            raise ValueError(f"Full-database formal comparison requires split_type={expected_split_type}")
        if row["max_steps"] != FORMAL_10000_MAX_STEPS or row["hard_max_steps"] != FORMAL_10000_HARD_MAX_STEPS:
            raise ValueError("Full-database formal comparison requires the frozen 2000-step horizon")
        if row["optimizer_name"] != "SGD" or row["learning_rate"] != FORMAL_10000_LEARNING_RATE:
            raise ValueError("Full-database formal comparison requires the frozen optimizer settings")
        if row["early_stop_monitor"] != "validation_loss_after_step":
            raise ValueError("Full-database formal comparison requires validation-loss early stopping")
        if row["early_stop_patience"] != FORMAL_10000_PATIENCE:
            raise ValueError("Full-database formal comparison requires the frozen patience")
        if row["early_stop_min_delta"] != FORMAL_10000_MIN_DELTA:
            raise ValueError("Full-database formal comparison requires the frozen min_delta")
        if row["sample_step_count"] != 1 or row["sample_step_size"] != FORMAL_10000_SAMPLE_STEP_SIZE:
            raise ValueError("Full-database formal comparison requires frozen one-step sampling settings")
        if row["safety_labels"]["benchmark_result"] is not False:
            raise ValueError("Full-database formal comparison inputs must be non-benchmark outputs")
        if require_cpu_only_inputs and row["safety_labels"]["gpu_used"] is not False:
            raise ValueError("Full-database formal comparison inputs must be CPU outputs")


def _validate_multiseed_rows(
    rows: list[dict[str, Any]],
    expected_seeds: tuple[int, ...],
    expected_split_type: str,
    require_shared_data_paths: bool = False,
) -> None:
    expected_pairs = {
        ("isotropic", int(seed)) for seed in expected_seeds
    } | {
        ("stiffness_structured", int(seed)) for seed in expected_seeds
    }
    observed_pairs = {(row["corruption_mode"], row["random_seed"]) for row in rows}
    if observed_pairs != expected_pairs:
        raise ValueError(f"Full-database formal multiseed inputs do not match expected mode/seed pairs: {observed_pairs}")
    baseline = rows[0]
    comparable_fields = [
        "dataset_kind",
        "dataset_subset_scope",
        "molecule_count",
        "split_type",
        "split_scope",
        "split_counts",
        "max_steps",
        "hard_max_steps",
        "optimizer_name",
        "learning_rate",
        "early_stop_monitor",
        "early_stop_patience",
        "early_stop_min_delta",
        "sample_step_count",
        "sample_step_size",
        "training_sample_count",
        "validation_sample_count",
        "sample_count",
        "model_name",
        "backbone",
    ]
    if require_shared_data_paths:
        comparable_fields.extend(["processed_artifact_path", "split_path"])
    for row in rows:
        for field in comparable_fields:
            if row[field] != baseline[field]:
                raise ValueError(f"Full-database formal multiseed results are not comparable on {field}")
        if row["dataset_subset_scope"] != FULL_DATABASE_SCOPE:
            raise ValueError("Full-database formal multiseed summary requires the full-database scope")
        if row["split_type"] != expected_split_type:
            raise ValueError(f"Full-database formal multiseed summary requires split_type={expected_split_type}")
        if row["max_steps"] != FORMAL_10000_MAX_STEPS or row["hard_max_steps"] != FORMAL_10000_HARD_MAX_STEPS:
            raise ValueError("Full-database formal multiseed summary requires the frozen 2000-step horizon")
        if row["optimizer_name"] != "SGD" or row["learning_rate"] != FORMAL_10000_LEARNING_RATE:
            raise ValueError("Full-database formal multiseed summary requires frozen optimizer settings")
        if row["early_stop_monitor"] != "validation_loss_after_step":
            raise ValueError("Full-database formal multiseed summary requires validation-loss early stopping")
        if row["early_stop_patience"] != FORMAL_10000_PATIENCE:
            raise ValueError("Full-database formal multiseed summary requires frozen patience")
        if row["early_stop_min_delta"] != FORMAL_10000_MIN_DELTA:
            raise ValueError("Full-database formal multiseed summary requires frozen min_delta")
        if row["sample_step_count"] != 1 or row["sample_step_size"] != FORMAL_10000_SAMPLE_STEP_SIZE:
            raise ValueError("Full-database formal multiseed summary requires frozen one-step sampling settings")
        if row["safety_labels"]["benchmark_result"] is not False:
            raise ValueError("Full-database formal multiseed inputs must be non-benchmark outputs")


def _rows_by_seed(rows: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["random_seed"], {})[row["corruption_mode"]] = row
    for seed, by_mode in grouped.items():
        if set(by_mode) != {"isotropic", "stiffness_structured"}:
            raise ValueError(f"Full-database formal seed {seed} does not have both modes")
    return grouped


def _aggregate_metric_comparisons(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for metric in COMPARABLE_METRICS:
        metric_rows = [
            row
            for seed_row in per_seed
            for row in seed_row["metric_comparisons"]
            if row["metric"] == metric
        ]
        isotropic_values = [row["isotropic"] for row in metric_rows]
        stiffness_values = [row["stiffness_structured"] for row in metric_rows]
        outcomes = [row["stiffness_outcome"] for row in metric_rows]
        iso_mean = _mean(isotropic_values)
        stiff_mean = _mean(stiffness_values)
        rows.append(
            {
                "metric": metric,
                "isotropic_values_by_seed": isotropic_values,
                "stiffness_structured_values_by_seed": stiffness_values,
                "isotropic_mean": iso_mean,
                "stiffness_structured_mean": stiff_mean,
                "stiffness_minus_isotropic_mean": stiff_mean - iso_mean,
                "stiffness_outcome_by_seed": outcomes,
                "stiffness_win_count": outcomes.count("better"),
                "stiffness_tie_count": outcomes.count("tied"),
                "stiffness_loss_count": outcomes.count("worse"),
                "seed_count": len(metric_rows),
                "direction": "lower_is_better",
            }
        )
    return rows


def _multiseed_directional_verdict(
    per_seed: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
) -> dict[str, Any]:
    outcomes = [
        row["stiffness_outcome"]
        for seed_row in per_seed
        for row in seed_row["metric_comparisons"]
    ]
    aggregate_outcomes = [
        "better"
        if row["stiffness_structured_mean"] < row["isotropic_mean"] - TIE_TOLERANCE
        else "tied"
        if abs(row["stiffness_structured_mean"] - row["isotropic_mean"]) <= TIE_TOLERANCE
        else "worse"
        for row in aggregate
    ]
    if outcomes and all(outcome == "better" for outcome in outcomes):
        verdict = "stiffness_better_on_all_tracked_metrics_all_seeds"
    elif outcomes and all(outcome in {"better", "tied"} for outcome in outcomes):
        verdict = "stiffness_better_or_tied_on_all_tracked_metrics_all_seeds"
    else:
        verdict = "mixed_or_inconclusive"
    return {
        "verdict": verdict,
        "stiffness_win_count": outcomes.count("better"),
        "stiffness_tie_count": outcomes.count("tied"),
        "stiffness_loss_count": outcomes.count("worse"),
        "tracked_metric_seed_comparison_count": len(outcomes),
        "aggregate_metric_outcomes": aggregate_outcomes,
        "aggregate_stiffness_win_count": aggregate_outcomes.count("better"),
        "aggregate_stiffness_tie_count": aggregate_outcomes.count("tied"),
        "aggregate_stiffness_loss_count": aggregate_outcomes.count("worse"),
        "tracked_metric_count": len(aggregate_outcomes),
        "direction": "lower_is_better",
    }


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
            isotropic["sampled_rmsd_to_target_overall_mean"],
            stiffness["sampled_rmsd_to_target_overall_mean"],
        ),
    ]
    for split in ["train", "val", "test"]:
        rows.append(
            _lower_is_better(
                f"sampled_rmsd_to_target_{split}_mean",
                isotropic["sampled_rmsd_to_target_by_split_mean"][split],
                stiffness["sampled_rmsd_to_target_by_split_mean"][split],
            )
        )
    return rows


def _directional_verdict(metric_comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = [row["stiffness_outcome"] for row in metric_comparisons]
    if all(outcome == "better" for outcome in outcomes):
        verdict = "stiffness_better_on_all_tracked_metrics"
    elif all(outcome in {"better", "tied"} for outcome in outcomes):
        verdict = "stiffness_better_or_tied_on_all_tracked_metrics"
    else:
        verdict = "mixed_or_inconclusive"
    return {
        "verdict": verdict,
        "stiffness_win_count": outcomes.count("better"),
        "stiffness_tie_count": outcomes.count("tied"),
        "stiffness_loss_count": outcomes.count("worse"),
        "tracked_metric_count": len(outcomes),
        "direction": "lower_is_better",
    }


def _lower_is_better(metric: str, isotropic: float, stiffness: float) -> dict[str, Any]:
    delta = stiffness - isotropic
    if abs(delta) <= TIE_TOLERANCE:
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


def _average_rmsd_by_split(sampled: list[dict[str, Any]]) -> dict[str, float]:
    return {
        split: _mean(item["sampled_rmsd_to_target"] for item in sampled if item["split"] == split)
        for split in ["train", "val", "test"]
    }


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("Cannot compute mean of an empty sequence")
    return sum(items) / len(items)


def _markdown_report(summary: dict[str, Any], title: str) -> str:
    by_mode = {row["corruption_mode"]: row for row in summary["rows"]}
    summary_gpu_label = "gpu_used_by_summary" if "gpu_used_by_summary" in summary else "gpu_used"
    lines = [
        f"# {title}",
        "",
        "This is a reporting-only comparison of completed full-database GEOM-Drugs formal seed0 outputs. It is not a benchmark result.",
        "",
        "## Scope",
        "",
        f"- Dataset scope: `{summary['dataset_scope']}`",
        f"- Molecule count: `{summary['dataset']['molecule_count']}`",
        f"- Split type: `{summary['dataset']['split_type']}`",
        f"- Split counts: `{summary['dataset']['split_counts']}`",
        f"- Seed: `{by_mode['isotropic']['random_seed']}`",
        f"- Max steps / hard cap: `{summary['training_horizon']['max_steps']}` / `{summary['training_horizon']['hard_max_steps']}`",
        f"- Early stop: `{summary['training_horizon']['early_stop_monitor']}`, patience `{summary['training_horizon']['early_stop_patience']}`, min_delta `{summary['training_horizon']['early_stop_min_delta']}`",
        f"- Verdict: `{summary['directional_verdict']['verdict']}`",
        "",
        "## Run Metadata",
        "",
        "| Mode | Input hardware | Input GPU used | Stop reason | Executed steps | Best val step | Runtime seconds |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for mode in ["isotropic", "stiffness_structured"]:
        row = by_mode[mode]
        lines.append(
            f"| {mode} | {row['hardware']} | {row['gpu_used']} | {row['stop_reason']} | {row['executed_train_steps']} | "
            f"{row['best_validation_loss_step']} | {row['runtime_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Metric | Isotropic | Stiffness | Stiffness - isotropic | Outcome |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in summary["metric_comparisons"]:
        lines.append(
            f"| {row['metric']} | {row['isotropic']:.8g} | {row['stiffness_structured']:.8g} | "
            f"{row['stiffness_minus_isotropic']:.8g} | {row['stiffness_outcome']} |"
        )
    lines.extend(
        [
            "",
            "## Safety Labels",
            "",
            f"- `benchmark_result`: `{summary['benchmark_result']}`",
            f"- `training_executed_by_summary`: `{summary['training_executed_by_summary']}`",
            f"- `sampling_executed_by_summary`: `{summary['sampling_executed_by_summary']}`",
            f"- `raw_data_modified`: `{summary['raw_data_modified']}`",
            f"- `environment_modified`: `{summary['environment_modified']}`",
            f"- `{summary_gpu_label}`: `{summary[summary_gpu_label]}`",
            "",
            "These values are diagnostics from existing result JSONs only; no training, sampling, preprocessing, scheduler action, GPU use, or environment change was performed by this comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_multiseed_report(
    summary: dict[str, Any],
    title: str = "Full-Database Formal Multi-Seed Summary",
) -> str:
    lines = [
        f"# {title}",
        "",
        "This is a reporting-only aggregation of completed full-database GEOM-Drugs formal outputs for seeds 0, 1, and 2. It is not a benchmark result.",
        "",
        "## Scope",
        "",
        f"- Dataset scope: `{summary['dataset_scope']}`",
        f"- Molecule count: `{summary['dataset']['molecule_count']}`",
        f"- Split type: `{summary['dataset']['split_type']}`",
        f"- Split counts: `{summary['dataset']['split_counts']}`",
        f"- Seed set: `{summary['seed_set']}`",
        f"- Max steps / hard cap: `{summary['training_horizon']['max_steps']}` / `{summary['training_horizon']['hard_max_steps']}`",
        f"- Early stop: `{summary['training_horizon']['early_stop_monitor']}`, patience `{summary['training_horizon']['early_stop_patience']}`, min_delta `{summary['training_horizon']['early_stop_min_delta']}`",
        f"- Verdict: `{summary['directional_verdict']['verdict']}`",
        "",
        "## Run Metadata",
        "",
        "| Seed | Mode | Hardware | Stop reason | Executed steps | Best val | Reloaded val | RMSD overall |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for seed_row in summary["per_seed_comparisons"]:
        for mode in ["isotropic", "stiffness_structured"]:
            row = seed_row["rows"][mode]
            lines.append(
                f"| {seed_row['seed']} | {mode} | {row['hardware']} | {row['stop_reason']} | "
                f"{row['executed_train_steps']} | {row['best_validation_loss']:.8g} | "
                f"{row['reloaded_validation_loss']:.8g} | {row['sampled_rmsd_to_target_overall_mean']:.8g} |"
            )
    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            "| Metric | Isotropic mean | Stiffness mean | Stiffness - isotropic | Stiffness wins |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["aggregate_metric_comparisons"]:
        lines.append(
            f"| {row['metric']} | {row['isotropic_mean']:.8g} | {row['stiffness_structured_mean']:.8g} | "
            f"{row['stiffness_minus_isotropic_mean']:.8g} | {row['stiffness_win_count']}/{row['seed_count']} |"
        )
    lines.extend(
        [
            "",
            "## Per-Seed Outcomes",
            "",
            "| Seed | Metric | Isotropic | Stiffness | Delta | Outcome |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for seed_row in summary["per_seed_comparisons"]:
        for row in seed_row["metric_comparisons"]:
            lines.append(
                f"| {seed_row['seed']} | {row['metric']} | {row['isotropic']:.8g} | "
                f"{row['stiffness_structured']:.8g} | {row['stiffness_minus_isotropic']:.8g} | "
                f"{row['stiffness_outcome']} |"
            )
    lines.extend(
        [
            "",
            "## Safety Labels",
            "",
            f"- `benchmark_result`: `{summary['benchmark_result']}`",
            f"- `training_executed_by_summary`: `{summary['training_executed_by_summary']}`",
            f"- `sampling_executed_by_summary`: `{summary['sampling_executed_by_summary']}`",
            f"- `raw_data_modified`: `{summary['raw_data_modified']}`",
            f"- `environment_modified`: `{summary['environment_modified']}`",
            f"- `gpu_used_by_summary`: `{summary['gpu_used_by_summary']}`",
            "",
            "These values are diagnostics from existing result JSONs only; no training, sampling, preprocessing, scheduler action, GPU use, or environment change was performed by this summary.",
            "",
        ]
    )
    return "\n".join(lines)
