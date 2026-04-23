"""Tiny real-data model-side dry-runs without neural training."""

from __future__ import annotations

import copy
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

from src.corruption.base import build_corruptor
from src.eval.basic_geometry import evaluate_smoke_samples
from src.utils.geometry import assert_finite_coords


DEFAULT_PROCESSED_ARTIFACT = Path("data/processed/geom_drugs_subset/tiny_real_dry_run/molecules.json")
DEFAULT_SPLIT_PATH = Path("data/splits/geom_drugs_tiny_real_dry_run/random_seed0/split.json")
DEFAULT_MODES = ("isotropic", "stiffness_structured")
RESULT_KIND = "tiny_real_data_model_dry_run_not_benchmark"
EXECUTION_MODE = "local_cpu_tiny_real_model_dry_run"


def run_tiny_real_model_dry_runs(
    processed_artifact_path: str | Path = DEFAULT_PROCESSED_ARTIFACT,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    output_root: str | Path = "experiments",
    seed: int = 0,
    max_molecules: int = 3,
    modes: tuple[str, ...] = DEFAULT_MODES,
) -> dict[str, dict[str, Path]]:
    outputs = {}
    for mode in modes:
        outputs[mode] = run_tiny_real_model_dry_run(
            processed_artifact_path=processed_artifact_path,
            split_path=split_path,
            output_root=output_root,
            seed=seed,
            max_molecules=max_molecules,
            corruption_mode=mode,
        )
    return outputs


def run_tiny_real_model_dry_run(
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

    artifact = _load_tiny_real_artifact(processed_artifact_path)
    records = _select_records(artifact["molecules"], max_molecules)
    split = _load_tiny_real_split(split_path)
    split_membership = _validate_split_coverage(split, records)
    config = _tiny_real_config(
        corruption_mode=corruption_mode,
        seed=seed,
        output_root=output_root,
        processed_artifact_path=processed_artifact_path,
        split_path=split_path,
        max_molecules=max_molecules,
    )

    rng = random.Random(seed)
    samples = _build_flow_matching_samples(config, records, rng, split_membership)
    metrics = _tiny_real_metrics(evaluate_smoke_samples(samples))
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
        "model_name": config["model"]["backbone"],
        "execution_mode": EXECUTION_MODE,
        "result_kind": RESULT_KIND,
        "benchmark_result": False,
        "hardware": "cpu_only",
        "scheduler": None,
        "slurm_account": None,
        "dry_run": True,
        "training_executed": False,
        "model_training_started": False,
        "sampling_executed": False,
        "checkpoint_path": None,
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
        },
        "split": {
            "split_type": split["split_type"],
            "degenerate_tiny_split": split.get("degenerate_tiny_split"),
            "counts": split_counts,
        },
        "notes": [
            "Tiny real-data model-side dry-run only.",
            "No neural model training, checkpointing, GPU execution, SLURM job, raw-data mutation, or environment mutation was used.",
            "The run validates source-corruption and flow-matching sample construction on the bounded tiny real artifact.",
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
        "backbone": config["model"]["backbone"],
        "flow_matching": config["flow_matching"],
        "sample_count": len(samples),
        "metrics": metrics,
        "molecules": _sample_summary(samples),
        "runtime_seconds": elapsed,
        "notes": [
            "This is a tiny real-data model-side dry-run on a bounded three-molecule GEOM artifact.",
            "These source-vs-target wiring checks are not benchmark results.",
            "No model training, model sampling, GPU execution, SLURM job, raw-data modification, or environment mutation was used.",
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
        raise ValueError(f"Unsupported tiny real model dry-run corruption mode: {corruption_mode}")
    return f"geom_tiny_real_model_{label}_seed{seed}"


def _tiny_real_config(
    corruption_mode: str,
    seed: int,
    output_root: Path,
    processed_artifact_path: Path,
    split_path: Path,
    max_molecules: int,
) -> dict[str, Any]:
    return {
        "experiment": {
            "name": _run_name(corruption_mode, seed),
            "seed": seed,
            "output_root": str(output_root),
        },
        "data": {
            "dataset_name": "geom_drugs",
            "dataset_kind": "geom_drugs_tiny_processed_dry_run",
            "processed_artifact_path": str(processed_artifact_path),
            "split_path": str(split_path),
            "split_type": "random",
            "subset": {
                "max_molecules": max_molecules,
                "max_conformers_per_mol": 1,
                "tag": "tiny_real_data_model_dry_run",
            },
        },
        "model": {
            "backbone": "egnn_fallback",
            "hidden_dim": 128,
            "num_layers": 4,
            "time_embedding_dim": 64,
        },
        "flow_matching": {
            "objective": "cartesian_linear",
            "interpolation": "linear",
        },
        "corruption": {
            "mode": corruption_mode,
            "stiffness_table": "heuristic_v1",
            "protect_rings": True,
            "protect_amides": True,
            "isotropic_sigma": 0.25,
            "bond_sigma": 0.02,
            "angle_sigma": 0.08,
            "torsion_sigma": 0.30,
        },
        "train": {
            "enabled": False,
            "dry_run_only": True,
            "max_steps": 0,
        },
        "sample": {
            "enabled": False,
            "num_samples_per_mol": 0,
            "num_steps": 0,
        },
        "eval": {
            "enabled": True,
            "compute_cov_mat": True,
            "compute_validity": True,
            "compute_energy_checks": False,
            "eval_subset_size": max_molecules,
        },
        "logging": {
            "save_manifest": True,
            "save_results_json": True,
            "save_log": True,
        },
    }


def _build_flow_matching_samples(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    rng: random.Random,
    split_membership: dict[str, str],
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
        rotatable_bond_count = _rotatable_bond_count(record, config)
        samples.append(
            {
                "molecule_id": record["molecule_id"],
                "split": split_membership[record["molecule_id"]],
                "smiles": record.get("smiles"),
                "atom_count": len(record.get("atom_symbols", [])),
                "bond_count": len(record.get("bonds", [])),
                "rotatable_bond_count": rotatable_bond_count,
                "conformer_id": conformer.get("conformer_id"),
                "bonds": record["bonds"],
                "x0": x0,
                "x1": x1,
                "t": t,
                "xt": _interpolate(x0, x1, t),
                "target_velocity": _velocity(x0, x1),
            }
        )
    return samples


def _rotatable_bond_count(record: dict[str, Any], config: dict[str, Any]) -> int | None:
    value = record.get("rotatable_bond_count")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if not bool(config.get("eval", {}).get("compute_rotatable_bond_count_from_smiles", False)):
        return None
    smiles = record.get("smiles")
    if not smiles:
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import Lipinski
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return int(Lipinski.NumRotatableBonds(mol))


def _load_tiny_real_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Tiny real processed artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_kind") != "geom_drugs_tiny_processed_dry_run":
        raise ValueError("Tiny real artifact has unexpected dataset_kind")
    if payload.get("result_kind") != "processed_output_dry_run_not_benchmark":
        raise ValueError("Tiny real artifact has unexpected result_kind")
    if payload.get("benchmark_result") is not False:
        raise ValueError("Tiny real artifact must be labeled benchmark_result=false")
    records = payload.get("molecules")
    if not isinstance(records, list) or not records:
        raise ValueError("Tiny real artifact must contain a non-empty molecules list")
    for record in records:
        _validate_record(record)
    return payload


def _validate_record(record: dict[str, Any]) -> None:
    for key in ["molecule_id", "smiles", "atom_symbols", "bonds", "conformers"]:
        if key not in record:
            raise ValueError(f"Tiny real record missing field: {key}")
    atom_count = len(record["atom_symbols"])
    if atom_count <= 0:
        raise ValueError(f"{record['molecule_id']} has no atoms")
    if not record["conformers"]:
        raise ValueError(f"{record['molecule_id']} has no conformers")
    coords = record["conformers"][0].get("coords")
    if not isinstance(coords, list) or len(coords) != atom_count:
        raise ValueError(f"{record['molecule_id']} conformer has invalid coordinate count")
    assert_finite_coords(coords, f"{record['molecule_id']}.coords")
    for bond in record["bonds"]:
        if len(bond) < 2:
            raise ValueError(f"{record['molecule_id']} has malformed bond entry")
        i, j = int(bond[0]), int(bond[1])
        if i < 0 or j < 0 or i >= atom_count or j >= atom_count:
            raise ValueError(f"{record['molecule_id']} has out-of-range bond index")


def _select_records(records: list[dict[str, Any]], max_molecules: int) -> list[dict[str, Any]]:
    if max_molecules <= 0:
        raise ValueError("max_molecules must be positive")
    selected = records[:max_molecules]
    if len(selected) < 3:
        raise ValueError(
            "Tiny real model dry-run requires at least three molecules so train/val/test coverage is non-degenerate"
        )
    return selected


def _load_tiny_real_split(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Tiny real split artifact not found: {path}")
    split = json.loads(path.read_text(encoding="utf-8"))
    if split.get("dataset_kind") != "geom_drugs_tiny_processed_dry_run":
        raise ValueError("Tiny real split has unexpected dataset_kind")
    if split.get("result_kind") != "split_dry_run_not_benchmark":
        raise ValueError("Tiny real split has unexpected result_kind")
    if split.get("benchmark_result") is not False:
        raise ValueError("Tiny real split must be labeled benchmark_result=false")
    if split.get("degenerate_tiny_split") is not False:
        raise ValueError("Tiny real model dry-run requires a non-degenerate tiny split")
    for key in ["train", "val", "test"]:
        if not isinstance(split.get(key), list) or not split[key]:
            raise ValueError(f"Tiny real split must contain a non-empty {key} list")
    return split


def _validate_split_coverage(split: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, str]:
    membership = {}
    for split_name in ["train", "val", "test"]:
        for molecule_id in split[split_name]:
            if molecule_id in membership:
                raise ValueError(f"Tiny real split assigns {molecule_id} more than once")
            membership[molecule_id] = split_name

    record_ids = {record["molecule_id"] for record in records}
    split_ids = set(membership)
    if record_ids != split_ids:
        missing = sorted(record_ids - split_ids)
        extra = sorted(split_ids - record_ids)
        raise ValueError(
            "Tiny real split must cover the bounded processed records exactly once; "
            f"missing={missing} extra={extra}"
        )
    return membership


def _tiny_real_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(metrics)
    updated["metric_scope"] = "geom_drugs_tiny_real_model_source_vs_target_dry_run"
    updated["cov_mat"]["note"] = (
        "Tiny real-data dry-run RMSD proxy comparing corrupted source x0 to target x1; not benchmark COV/MAT."
    )
    updated["validity"]["note"] = (
        "Tiny real-data intramolecular wiring checks only; not PoseBusters/MMFF/UFF and not a benchmark."
    )
    return updated


def _sample_summary(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "molecule_id": sample["molecule_id"],
            "split": sample["split"],
            "smiles": sample["smiles"],
            "atom_count": sample["atom_count"],
            "bond_count": sample["bond_count"],
            "rotatable_bond_count": sample["rotatable_bond_count"],
            "conformer_id": sample["conformer_id"],
            "t": sample["t"],
        }
        for sample in samples
    ]


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
    metrics = results["metrics"]
    return (
        f"run_name={results['run_name']}\n"
        f"corruption_mode={results['corruption_mode']}\n"
        f"execution_mode={results['execution_mode']}\n"
        "training_executed=false\n"
        "model_training_started=false\n"
        "sampling_executed=false\n"
        "hardware=cpu_only\n"
        "scheduler=none\n"
        f"result_kind={RESULT_KIND}\n"
        f"sample_count={results['sample_count']}\n"
        f"mat_r_smoke={metrics['cov_mat']['mat_r_smoke']}\n"
        f"mean_abs_bond_length_delta={metrics['validity']['mean_abs_bond_length_delta']}\n"
        f"manifest={manifest_path}\n"
        f"results={results_path}\n"
        "note=Tiny real-data model-side dry-run only; no scheduler job, GPU use, training, or environment mutation.\n"
    )
