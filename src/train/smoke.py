"""CPU-only smoke runner for validating config/data/corruption wiring."""

from __future__ import annotations

import copy
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

from src.corruption.base import build_corruptor
from src.data.records import load_records, select_records
from src.data.splits import load_or_create_split, split_file_path
from src.eval.basic_geometry import evaluate_smoke_samples
from src.utils.config import load_config
from src.utils.geometry import assert_finite_coords


def run_smoke(
    config_path: str | Path,
    dry_run: bool,
    output_root_override: str | Path | None = None,
) -> dict[str, Path]:
    start = time.time()
    config_path = Path(config_path)
    config = load_config(config_path)
    if output_root_override is not None:
        config = copy.deepcopy(config)
        config["experiment"]["output_root"] = str(output_root_override)
    if config["train"]["enabled"] and not dry_run:
        raise RuntimeError(
            "Real model training is not implemented in this bootstrap scaffold. "
            "Use --dry-run to validate the smoke path without claiming training ran."
        )

    seed = int(config["experiment"]["seed"])
    rng = random.Random(seed)
    output_root = Path(config["experiment"]["output_root"])
    run_name = str(config["experiment"]["name"])
    manifest_path = output_root / "manifests" / f"{run_name}.json"
    results_path = output_root / "results" / f"{run_name}.json"
    log_path = output_root / "logs" / f"{run_name}.log"
    checkpoint_path = output_root / "checkpoints" / f"{run_name}.pt"

    records = load_records(config["data"]["processed_root"])
    max_molecules = config["data"].get("subset", {}).get("max_molecules")
    records = select_records(records, max_molecules)
    split = load_or_create_split(
        records,
        split_root=config["data"]["split_root"],
        split_type=config["data"]["split_type"],
        dataset_name=config["data"]["dataset_name"],
        seed=seed,
    )

    samples = _build_flow_matching_samples(config, records, rng)
    metrics = evaluate_smoke_samples(samples)
    elapsed = time.time() - start

    manifest = {
        "run_name": run_name,
        "git_commit": _git_commit(),
        "config_path": str(config_path),
        "config": config,
        "dataset_name": config["data"]["dataset_name"],
        "dataset_kind": "synthetic_smoke",
        "dataset_subset": config["data"].get("subset", {}),
        "split_type": config["data"]["split_type"],
        "split_path": str(
            split_file_path(
                config["data"]["split_root"],
                config["data"]["split_type"],
                config["data"]["dataset_name"],
                seed,
            )
        ),
        "model_name": config["model"]["backbone"],
        "corruption_mode": config["corruption"]["mode"],
        "random_seed": seed,
        "checkpoint_path": None if dry_run else str(checkpoint_path),
        "metrics_output_path": str(results_path),
        "execution_mode": "local_smoke",
        "result_kind": "smoke_only_not_benchmark",
        "benchmark_result": False,
        "hardware": "cpu_only",
        "scheduler": None,
        "slurm_account": None,
        "dry_run": dry_run,
        "training_executed": False,
    }

    results = {
        "run_name": run_name,
        "dataset": {
            "name": config["data"]["dataset_name"],
            "kind": "synthetic_smoke",
            "molecule_count": len(records),
            "split_type": config["data"]["split_type"],
            "split_counts": {key: len(split[key]) for key in ["train", "val", "test"]},
        },
        "corruption_mode": config["corruption"]["mode"],
        "backbone": config["model"]["backbone"],
        "flow_matching": config["flow_matching"],
        "sample_count": len(samples),
        "metrics": metrics,
        "runtime_seconds": elapsed,
        "execution_mode": "local_smoke",
        "result_kind": "smoke_only_not_benchmark",
        "benchmark_result": False,
        "hardware": "cpu_only",
        "scheduler": None,
        "slurm_account": None,
        "training_executed": False,
        "notes": [
            "No neural model training was run.",
            "Metrics are smoke-only source-vs-target wiring checks, not benchmark results.",
            "This run used local CPU-only smoke plumbing and did not use SLURM or GPUs.",
        ],
    }

    for path in [manifest_path.parent, results_path.parent, log_path.parent]:
        path.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_path.write_text(_log_text(results, manifest_path, results_path), encoding="utf-8")

    return {"manifest": manifest_path, "results": results_path, "log": log_path}


def _build_flow_matching_samples(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    corruptor = build_corruptor(config, int(config["experiment"]["seed"]))
    max_eval = int(config["eval"]["eval_subset_size"])
    samples = []

    for record in records[:max_eval]:
        conformer = record["conformers"][0]
        x1 = [[float(value) for value in xyz] for xyz in conformer["coords"]]
        assert_finite_coords(x1, f"{record['molecule_id']}.x1")
        x0 = corruptor.corrupt(record, x1)
        assert_finite_coords(x0, f"{record['molecule_id']}.x0")
        t = rng.random()
        xt = _interpolate(x0, x1, t)
        velocity = _velocity(x0, x1)
        samples.append(
            {
                "molecule_id": record["molecule_id"],
                "bonds": record["bonds"],
                "x0": x0,
                "x1": x1,
                "t": t,
                "xt": xt,
                "target_velocity": velocity,
            }
        )
    return samples


def _interpolate(x0: list[list[float]], x1: list[list[float]], t: float) -> list[list[float]]:
    return [
        [(1.0 - t) * float(a) + t * float(b) for a, b in zip(x0_xyz, x1_xyz)]
        for x0_xyz, x1_xyz in zip(x0, x1)
    ]


def _velocity(x0: list[list[float]], x1: list[list[float]]) -> list[list[float]]:
    return [[float(b) - float(a) for a, b in zip(x0_xyz, x1_xyz)] for x0_xyz, x1_xyz in zip(x0, x1)]


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
    return (
        f"run_name={results['run_name']}\n"
        f"execution_mode={results['execution_mode']}\n"
        "training_executed=false\n"
        f"manifest={manifest_path}\n"
        f"results={results_path}\n"
        "note=Smoke-only run; no scheduler job, GPU training, or environment mutation.\n"
    )
