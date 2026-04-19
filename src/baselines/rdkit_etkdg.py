"""Tiny local RDKit ETKDGv3 + MMFF smoke baseline.

This module runs only on the synthetic smoke fixture. It is a wiring check for
the future RDKit baseline, not a GEOM-Drugs benchmark result.
"""

from __future__ import annotations

import copy
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from src.data.records import load_records, select_records
from src.data.splits import load_or_create_split, split_file_path
from src.utils.config import load_config
from src.utils.geometry import Coordinates, distance, mean


class RdkitSmokeError(RuntimeError):
    """Raised when the local RDKit smoke baseline cannot run honestly."""

    def __init__(self, message: str, failure_type: str = "runtime") -> None:
        super().__init__(message)
        self.failure_type = failure_type


def run_rdkit_smoke(
    config_path: str | Path = "configs/train/tiny_smoke.yaml",
    output_root_override: str | Path | None = None,
    run_name: str = "smoke_random_rdkit_etkdg_mmff_seed0",
) -> dict[str, Path]:
    start = time.time()
    config_path = Path(config_path)
    config = load_config(config_path)
    if output_root_override is not None:
        config = copy.deepcopy(config)
        config["experiment"]["output_root"] = str(output_root_override)

    seed = int(config["experiment"]["seed"])
    output_root = Path(config["experiment"]["output_root"])
    manifest_path = output_root / "manifests" / f"{run_name}.json"
    results_path = output_root / "results" / f"{run_name}.json"
    log_path = output_root / "logs" / f"{run_name}.log"

    records = load_records(config["data"]["processed_root"])
    records = select_records(records, config["data"].get("subset", {}).get("max_molecules"))
    split = load_or_create_split(
        records,
        split_root=config["data"]["split_root"],
        split_type=config["data"]["split_type"],
        dataset_name=config["data"]["dataset_name"],
        seed=seed,
    )

    molecule_results = []
    failures = []
    rdkit_available = True
    execution_status = "completed"
    environment = _environment_metadata()
    try:
        rdkit = _load_rdkit()
    except RdkitSmokeError as exc:
        rdkit_available = False
        execution_status = "import_failure"
        failures.append(_failure_payload(scope="dependency", exc=exc))
        if _existing_successful_rdkit_result(results_path):
            diagnostic_path = _write_preflight_failure_log(
                log_path,
                run_name=run_name,
                environment=environment,
                error=str(exc),
            )
            raise RdkitSmokeError(
                "RDKit is not importable under the current interpreter, and an existing "
                f"successful RDKit smoke result is present at {results_path}. "
                f"Refusing to overwrite it. Diagnostic log: {diagnostic_path}",
                failure_type="import",
            ) from exc
    else:
        environment["rdkit_version"] = rdkit["rdkit_version"]
        for index, record in enumerate(records):
            try:
                molecule_results.append(_run_one_record(rdkit, record, seed + index))
            except RdkitSmokeError as exc:
                failures.append(
                    _failure_payload(
                        scope="molecule",
                        exc=exc,
                        molecule_id=record.get("molecule_id"),
                    )
                )
            except Exception as exc:  # RDKit can raise several C++-backed exceptions.
                failures.append(
                    {
                        "scope": "molecule",
                        "molecule_id": record.get("molecule_id"),
                        "failure_type": "runtime",
                        "error": str(exc),
                    }
                )

        execution_status = _execution_status(molecule_results, failures)

    elapsed = time.time() - start
    metrics = _summarize_metrics(
        molecule_results,
        failures,
        attempted_molecule_count=len(records) if rdkit_available else 0,
        rdkit_available=rdkit_available,
    )

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
        "baseline_name": "rdkit_etkdg_v3_mmff",
        "random_seed": seed,
        "metrics_output_path": str(results_path),
        "execution_mode": "local_smoke",
        "result_kind": "smoke_only_not_benchmark",
        "benchmark_result": False,
        "hardware": "cpu_only",
        "environment": environment,
        "python_executable": environment["python_executable"],
        "python_version": environment["python_version"],
        "rdkit_version": environment["rdkit_version"],
        "scheduler": None,
        "slurm_account": None,
        "rdkit_available": rdkit_available,
        "execution_status": execution_status,
        "baseline_executed": rdkit_available,
        "training_executed": False,
        "sampling_executed": False,
    }

    results = {
        "run_name": run_name,
        "baseline_name": "rdkit_etkdg_v3_mmff",
        "dataset": {
            "name": config["data"]["dataset_name"],
            "kind": "synthetic_smoke",
            "molecule_count": len(records),
            "split_type": config["data"]["split_type"],
            "split_counts": {key: len(split[key]) for key in ["train", "val", "test"]},
        },
        "metrics": metrics,
        "molecules": molecule_results,
        "failures": failures,
        "runtime_seconds": elapsed,
        "execution_mode": "local_smoke",
        "execution_status": execution_status,
        "result_kind": "smoke_only_not_benchmark",
        "benchmark_result": False,
        "hardware": "cpu_only",
        "environment": environment,
        "python_executable": environment["python_executable"],
        "python_version": environment["python_version"],
        "rdkit_version": environment["rdkit_version"],
        "scheduler": None,
        "slurm_account": None,
        "rdkit_available": rdkit_available,
        "baseline_executed": rdkit_available,
        "training_executed": False,
        "sampling_executed": False,
        "notes": [
            "This wrapper targets RDKit ETKDGv3 + MMFF on the synthetic smoke fixture.",
            "These values are smoke-only plumbing checks, not benchmark results.",
            "No neural model training, GPU execution, SLURM job, or environment mutation was used.",
            "If rdkit_available is false, no RDKit embedding or MMFF optimization was executed.",
            "Failures are categorized as import, fixture, etkdg, mmff, alignment, or runtime.",
        ],
    }

    for directory in [manifest_path.parent, results_path.parent, log_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_path.write_text(_log_text(results, manifest_path, results_path), encoding="utf-8")

    return {"manifest": manifest_path, "results": results_path, "log": log_path}


def run_rdkit_tiny_real_dry_run(
    processed_artifact_path: str | Path = "data/processed/geom_drugs_subset/tiny_real_dry_run/molecules.json",
    split_path: str | Path = "data/splits/geom_drugs_tiny_real_dry_run/random_seed0/split.json",
    output_root: str | Path = "experiments",
    run_name: str = "geom_tiny_real_rdkit_etkdg_mmff_seed0",
    seed: int = 0,
    max_molecules: int = 3,
) -> dict[str, Path]:
    start = time.time()
    processed_artifact_path = Path(processed_artifact_path)
    split_path = Path(split_path)
    output_root = Path(output_root)
    manifest_path = output_root / "manifests" / f"{run_name}.json"
    results_path = output_root / "results" / f"{run_name}.json"
    log_path = output_root / "logs" / f"{run_name}.log"

    artifact = _load_tiny_real_artifact(processed_artifact_path)
    records = artifact["molecules"][:max_molecules]
    split = _load_tiny_real_split(split_path)

    molecule_results = []
    failures = []
    rdkit_available = True
    execution_status = "completed"
    environment = _environment_metadata()
    try:
        rdkit = _load_rdkit()
    except RdkitSmokeError as exc:
        rdkit_available = False
        execution_status = "import_failure"
        failures.append(_failure_payload(scope="dependency", exc=exc))
    else:
        environment["rdkit_version"] = rdkit["rdkit_version"]
        for index, record in enumerate(records):
            try:
                molecule_results.append(_run_one_tiny_real_record(rdkit, record, seed + index))
            except RdkitSmokeError as exc:
                failures.append(
                    _failure_payload(
                        scope="molecule",
                        exc=exc,
                        molecule_id=record.get("molecule_id"),
                    )
                )
            except Exception as exc:  # RDKit can raise C++-backed exceptions.
                failures.append(
                    {
                        "scope": "molecule",
                        "molecule_id": record.get("molecule_id"),
                        "failure_type": "runtime",
                        "error": str(exc),
                    }
                )
        execution_status = _execution_status(molecule_results, failures)

    elapsed = time.time() - start
    metrics = _summarize_tiny_real_metrics(
        molecule_results,
        failures,
        attempted_molecule_count=len(records) if rdkit_available else 0,
        rdkit_available=rdkit_available,
    )
    common = {
        "run_name": run_name,
        "baseline_name": "rdkit_etkdg_v3_mmff",
        "dataset_name": "geom_drugs",
        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
        "processed_artifact_path": str(processed_artifact_path),
        "split_path": str(split_path),
        "random_seed": seed,
        "execution_mode": "local_cpu_tiny_real_data_dry_run",
        "result_kind": "tiny_real_data_dry_run_not_benchmark",
        "benchmark_result": False,
        "hardware": "cpu_only",
        "scheduler": None,
        "slurm_account": None,
        "rdkit_available": rdkit_available,
        "execution_status": execution_status,
        "baseline_executed": rdkit_available,
        "training_executed": False,
        "sampling_executed": False,
        "environment": environment,
        "python_executable": environment["python_executable"],
        "python_version": environment["python_version"],
        "rdkit_version": environment["rdkit_version"],
    }
    manifest = {
        **common,
        "git_commit": _git_commit(),
        "metrics_output_path": str(results_path),
        "bounds": {
            "max_molecules": max_molecules,
            "input_molecule_count": len(artifact["molecules"]),
            "attempted_molecule_count": len(records) if rdkit_available else 0,
        },
        "split": {
            "split_type": split["split_type"],
            "degenerate_tiny_split": split.get("degenerate_tiny_split"),
            "counts": {key: len(split[key]) for key in ["train", "val", "test"]},
        },
    }
    results = {
        **common,
        "dataset": {
            "name": "geom_drugs",
            "kind": "geom_drugs_tiny_processed_dry_run",
            "molecule_count": len(records),
            "split_type": split["split_type"],
            "split_counts": {key: len(split[key]) for key in ["train", "val", "test"]},
        },
        "metrics": metrics,
        "molecules": molecule_results,
        "failures": failures,
        "runtime_seconds": elapsed,
        "notes": [
            "This is a tiny real-data RDKit ETKDGv3 + MMFF dry-run on a bounded GEOM artifact.",
            "These values validate baseline wiring only and are not benchmark results.",
            "No raw data modification, neural model training, GPU execution, SLURM job, or environment mutation was used.",
            "If rdkit_available is false, no RDKit embedding or MMFF optimization was executed.",
        ],
    }

    for directory in [manifest_path.parent, results_path.parent, log_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_path.write_text(_tiny_real_log_text(results, manifest_path, results_path), encoding="utf-8")
    return {"manifest": manifest_path, "results": results_path, "log": log_path}


def _run_one_record(rdkit: dict[str, Any], record: dict[str, Any], seed: int) -> dict[str, Any]:
    Chem = rdkit["Chem"]
    AllChem = rdkit["AllChem"]
    Point3D = rdkit["Point3D"]
    rdMolAlign = rdkit["rdMolAlign"]

    smiles = record["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RdkitSmokeError(f"RDKit could not parse SMILES: {smiles}", failure_type="fixture")
    if mol.GetNumAtoms() != len(record["atom_symbols"]):
        raise RdkitSmokeError(
            f"Atom-count mismatch for {record['molecule_id']}: "
            f"RDKit={mol.GetNumAtoms()} fixture={len(record['atom_symbols'])}",
            failure_type="fixture",
        )

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 1
    embed_status = AllChem.EmbedMolecule(mol_h, params)
    if embed_status != 0:
        raise RdkitSmokeError(
            f"ETKDGv3 embedding failed with status {embed_status}",
            failure_type="etkdg",
        )

    mmff_status = AllChem.MMFFOptimizeMolecule(mol_h, mmffVariant="MMFF94s", maxIters=200)
    props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
    if props is None:
        raise RdkitSmokeError(
            "MMFF properties are unavailable for this molecule",
            failure_type="mmff",
        )
    force_field = AllChem.MMFFGetMoleculeForceField(mol_h, props)
    if force_field is None:
        raise RdkitSmokeError(
            "MMFF force field could not be constructed",
            failure_type="mmff",
        )
    energy = float(force_field.CalcEnergy())

    pred_heavy = Chem.RemoveHs(mol_h)
    predicted_coords = _coords_from_mol(pred_heavy)
    target_coords = [[float(value) for value in xyz] for xyz in record["conformers"][0]["coords"]]

    ref_mol = Chem.Mol(mol)
    ref_conf = Chem.Conformer(ref_mol.GetNumAtoms())
    for atom_index, xyz in enumerate(target_coords):
        ref_conf.SetAtomPosition(atom_index, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    ref_mol.AddConformer(ref_conf, assignId=True)

    try:
        aligned_rmsd = float(rdMolAlign.AlignMol(pred_heavy, ref_mol))
        aligned_coords = _coords_from_mol(pred_heavy)
    except Exception:
        aligned_rmsd = _raw_rmsd(predicted_coords, target_coords)
        aligned_coords = predicted_coords

    bond_deltas = _bond_length_deltas(record["bonds"], aligned_coords, target_coords)
    return {
        "molecule_id": record["molecule_id"],
        "smiles": smiles,
        "embed_status": int(embed_status),
        "mmff_status": int(mmff_status),
        "mmff_converged": mmff_status == 0,
        "mmff_energy": energy,
        "aligned_rmsd_to_fixture": aligned_rmsd,
        "mean_abs_bond_length_delta": mean(abs(value) for value in bond_deltas),
        "heavy_atom_count": len(record["atom_symbols"]),
        "rotatable_bond_count": record.get("rotatable_bond_count"),
    }


def _run_one_tiny_real_record(rdkit: dict[str, Any], record: dict[str, Any], seed: int) -> dict[str, Any]:
    Chem = rdkit["Chem"]
    AllChem = rdkit["AllChem"]
    Point3D = rdkit["Point3D"]
    rdMolAlign = rdkit["rdMolAlign"]

    smiles = record["smiles"]
    mol_h = _mol_from_tiny_real_graph(Chem, record)
    atom_count = len(record["atom_symbols"])
    if mol_h.GetNumAtoms() != atom_count:
        raise RdkitSmokeError(
            f"Atom-count mismatch for {record['molecule_id']}: RDKit={mol_h.GetNumAtoms()} artifact={atom_count}",
            failure_type="fixture",
        )

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 1
    embed_status = AllChem.EmbedMolecule(mol_h, params)
    if embed_status != 0:
        raise RdkitSmokeError(
            f"ETKDGv3 embedding failed with status {embed_status}",
            failure_type="etkdg",
        )

    mmff_status = AllChem.MMFFOptimizeMolecule(mol_h, mmffVariant="MMFF94s", maxIters=200)
    props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
    if props is None:
        raise RdkitSmokeError("MMFF properties are unavailable for this molecule", failure_type="mmff")
    force_field = AllChem.MMFFGetMoleculeForceField(mol_h, props)
    if force_field is None:
        raise RdkitSmokeError("MMFF force field could not be constructed", failure_type="mmff")
    energy = float(force_field.CalcEnergy())

    target_coords = [[float(value) for value in xyz] for xyz in record["conformers"][0]["coords"]]
    ref_mol = Chem.Mol(mol_h)
    ref_mol.RemoveAllConformers()
    ref_conf = Chem.Conformer(ref_mol.GetNumAtoms())
    for atom_index, xyz in enumerate(target_coords):
        ref_conf.SetAtomPosition(atom_index, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    ref_mol.AddConformer(ref_conf, assignId=True)

    predicted_coords = _coords_from_mol(mol_h)
    try:
        aligned_rmsd = float(rdMolAlign.AlignMol(mol_h, ref_mol))
        aligned_coords = _coords_from_mol(mol_h)
    except Exception:
        aligned_rmsd = _raw_rmsd(predicted_coords, target_coords)
        aligned_coords = predicted_coords

    bond_deltas = _bond_length_deltas(record["bonds"], aligned_coords, target_coords)
    return {
        "molecule_id": record["molecule_id"],
        "smiles": smiles,
        "embed_status": int(embed_status),
        "mmff_status": int(mmff_status),
        "mmff_converged": mmff_status == 0,
        "mmff_energy": energy,
        "aligned_rmsd_to_tiny_real_conformer": aligned_rmsd,
        "mean_abs_bond_length_delta": mean(abs(value) for value in bond_deltas),
        "atom_count": atom_count,
        "bond_count": len(record["bonds"]),
        "rotatable_bond_count": record.get("rotatable_bond_count"),
    }


def _mol_from_tiny_real_graph(Chem: Any, record: dict[str, Any]) -> Any:
    rw_mol = Chem.RWMol()
    aromatic_atoms = {
        int(atom_index)
        for bond in record["bonds"]
        if abs(float(bond[2]) - 1.5) < 1.0e-6
        for atom_index in [bond[0], bond[1]]
    }
    formal_charges = record.get("formal_charges") or [0] * len(record["atomic_numbers"])
    for atom_index, atomic_number in enumerate(record["atomic_numbers"]):
        atom = Chem.Atom(int(atomic_number))
        atom.SetFormalCharge(int(formal_charges[atom_index]))
        if atom_index in aromatic_atoms:
            atom.SetIsAromatic(True)
        rw_mol.AddAtom(atom)
    for begin, end, order in record["bonds"]:
        rw_mol.AddBond(int(begin), int(end), _rdkit_bond_type(Chem, float(order)))
    mol = rw_mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise RdkitSmokeError(
            f"RDKit could not sanitize artifact graph for {record['molecule_id']}: {exc}",
            failure_type="fixture",
        ) from exc
    return mol


def _rdkit_bond_type(Chem: Any, order: float) -> Any:
    if abs(order - 1.5) < 1.0e-6:
        return Chem.BondType.AROMATIC
    if abs(order - 2.0) < 1.0e-6:
        return Chem.BondType.DOUBLE
    if abs(order - 3.0) < 1.0e-6:
        return Chem.BondType.TRIPLE
    return Chem.BondType.SINGLE


def _load_rdkit() -> dict[str, Any]:
    try:
        import rdkit  # type: ignore
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem, rdMolAlign  # type: ignore
        from rdkit.Geometry import Point3D  # type: ignore
    except ImportError as exc:
        raise RdkitSmokeError(
            "RDKit is required for the ETKDGv3 + MMFF smoke baseline, but it is not importable. "
            "No fake RDKit results were written.",
            failure_type="import",
        ) from exc
    return {
        "Chem": Chem,
        "AllChem": AllChem,
        "Point3D": Point3D,
        "rdMolAlign": rdMolAlign,
        "rdkit_version": getattr(rdkit, "__version__", "unknown"),
    }


def _coords_from_mol(mol: Any) -> Coordinates:
    conf = mol.GetConformer()
    coords = []
    for atom_index in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_index)
        coords.append([float(pos.x), float(pos.y), float(pos.z)])
    return coords


def _bond_length_deltas(
    bonds: list[list[int]], predicted: Coordinates, target: Coordinates
) -> list[float]:
    deltas = []
    for bond in bonds:
        i, j = int(bond[0]), int(bond[1])
        deltas.append(distance(predicted, i, j) - distance(target, i, j))
    return deltas


def _raw_rmsd(coords_a: Coordinates, coords_b: Coordinates) -> float:
    if len(coords_a) != len(coords_b):
        raise RdkitSmokeError("Cannot compute RMSD for different atom counts", failure_type="alignment")
    squared = 0.0
    for xyz_a, xyz_b in zip(coords_a, coords_b):
        for a, b in zip(xyz_a, xyz_b):
            squared += (float(a) - float(b)) ** 2
    return (squared / len(coords_a)) ** 0.5


def _summarize_metrics(
    molecule_results: list[dict[str, Any]],
    failures: list[dict[str, str]],
    attempted_molecule_count: int,
    rdkit_available: bool,
) -> dict[str, Any]:
    attempted = attempted_molecule_count
    return {
        "metric_scope": "synthetic_smoke_rdkit_etkdg_mmff",
        "rdkit_available": rdkit_available,
        "attempted_molecule_count": attempted,
        "successful_molecule_count": len(molecule_results),
        "failure_count": len(failures),
        "failure_type_counts": _failure_type_counts(failures),
        "success_rate": len(molecule_results) / attempted if attempted else 0.0,
        "mean_aligned_rmsd_to_fixture": _optional_mean(
            result["aligned_rmsd_to_fixture"] for result in molecule_results
        ),
        "mean_abs_bond_length_delta": _optional_mean(
            result["mean_abs_bond_length_delta"] for result in molecule_results
        ),
        "mean_mmff_energy": _optional_mean(result["mmff_energy"] for result in molecule_results),
        "mmff_converged_count": sum(1 for result in molecule_results if result["mmff_converged"]),
        "note": "Smoke-only RDKit baseline metrics on synthetic fixture molecules; not benchmark metrics.",
    }


def _summarize_tiny_real_metrics(
    molecule_results: list[dict[str, Any]],
    failures: list[dict[str, str]],
    attempted_molecule_count: int,
    rdkit_available: bool,
) -> dict[str, Any]:
    attempted = attempted_molecule_count
    return {
        "metric_scope": "geom_drugs_tiny_real_rdkit_etkdg_mmff_dry_run",
        "rdkit_available": rdkit_available,
        "attempted_molecule_count": attempted,
        "successful_molecule_count": len(molecule_results),
        "failure_count": len(failures),
        "failure_type_counts": _failure_type_counts(failures),
        "success_rate": len(molecule_results) / attempted if attempted else 0.0,
        "mean_aligned_rmsd_to_tiny_real_conformer": _optional_mean(
            result["aligned_rmsd_to_tiny_real_conformer"] for result in molecule_results
        ),
        "mean_abs_bond_length_delta": _optional_mean(
            result["mean_abs_bond_length_delta"] for result in molecule_results
        ),
        "mean_mmff_energy": _optional_mean(result["mmff_energy"] for result in molecule_results),
        "mmff_converged_count": sum(1 for result in molecule_results if result["mmff_converged"]),
        "note": "Tiny real-data dry-run metrics for baseline wiring only; not benchmark metrics.",
    }


def _optional_mean(values: Any) -> float | None:
    values_list = list(values)
    if not values_list:
        return None
    return mean(values_list)


def _load_tiny_real_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RdkitSmokeError(f"Tiny real processed artifact not found at {path}", failure_type="fixture")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_kind") != "geom_drugs_tiny_processed_dry_run":
        raise RdkitSmokeError("Tiny real artifact has unexpected dataset_kind", failure_type="fixture")
    if payload.get("benchmark_result") is not False:
        raise RdkitSmokeError("Tiny real artifact must be labeled benchmark_result=false", failure_type="fixture")
    records = payload.get("molecules")
    if not isinstance(records, list) or not records:
        raise RdkitSmokeError("Tiny real artifact must contain a non-empty molecules list", failure_type="fixture")
    for record in records:
        for key in ["molecule_id", "smiles", "atom_symbols", "bonds", "conformers"]:
            if key not in record:
                raise RdkitSmokeError(f"Tiny real record missing field: {key}", failure_type="fixture")
        if not record["conformers"]:
            raise RdkitSmokeError(f"{record['molecule_id']} has no conformers", failure_type="fixture")
    return payload


def _load_tiny_real_split(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RdkitSmokeError(f"Tiny real split artifact not found at {path}", failure_type="fixture")
    split = json.loads(path.read_text(encoding="utf-8"))
    if split.get("result_kind") != "split_dry_run_not_benchmark":
        raise RdkitSmokeError("Tiny real split has unexpected result_kind", failure_type="fixture")
    if split.get("benchmark_result") is not False:
        raise RdkitSmokeError("Tiny real split must be labeled benchmark_result=false", failure_type="fixture")
    for key in ["train", "val", "test"]:
        if key not in split or not isinstance(split[key], list):
            raise RdkitSmokeError(f"Tiny real split is missing list field: {key}", failure_type="fixture")
    molecule_ids = split["train"] + split["val"] + split["test"]
    if len(molecule_ids) != len(set(molecule_ids)):
        raise RdkitSmokeError("Tiny real split assigns a molecule more than once", failure_type="fixture")
    return split


def _environment_metadata() -> dict[str, str | None]:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_full_version": sys.version,
        "rdkit_version": None,
    }


def _failure_payload(
    scope: str,
    exc: RdkitSmokeError,
    molecule_id: str | None = None,
) -> dict[str, str | None]:
    payload = {
        "scope": scope,
        "failure_type": exc.failure_type,
        "error": str(exc),
    }
    if molecule_id is not None:
        payload["molecule_id"] = molecule_id
    return payload


def _execution_status(
    molecule_results: list[dict[str, Any]],
    failures: list[dict[str, str | None]],
) -> str:
    if not failures:
        return "completed"
    if molecule_results:
        return "completed_with_failures"

    failure_types = {failure.get("failure_type") for failure in failures}
    if len(failure_types) == 1:
        failure_type = next(iter(failure_types))
        return f"{failure_type}_failure"
    return "runtime_failure"


def _failure_type_counts(failures: list[dict[str, str | None]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in failures:
        failure_type = str(failure.get("failure_type") or "unknown")
        counts[failure_type] = counts.get(failure_type, 0) + 1
    return counts


def _existing_successful_rdkit_result(results_path: Path) -> bool:
    if not results_path.exists():
        return False
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("baseline_name") == "rdkit_etkdg_v3_mmff"
        and payload.get("result_kind") == "smoke_only_not_benchmark"
        and payload.get("rdkit_available") is True
        and payload.get("baseline_executed") is True
        and payload.get("execution_status") == "completed"
    )


def _write_preflight_failure_log(
    log_path: Path,
    run_name: str,
    environment: dict[str, str | None],
    error: str,
) -> Path:
    diagnostic_path = log_path.with_name(f"{run_name}.preflight_failure.log")
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(
        "\n".join(
            [
                f"run_name={run_name}",
                "execution_status=import_failure",
                "result_kind=smoke_only_not_benchmark",
                "overwrote_manifest=false",
                "overwrote_results=false",
                f"python_executable={environment['python_executable']}",
                f"python_version={environment['python_version']}",
                f"rdkit_version={environment['rdkit_version']}",
                f"error={error}",
                "note=Existing successful RDKit smoke outputs were preserved.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return diagnostic_path


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
        f"baseline_name={results['baseline_name']}\n"
        f"execution_mode={results['execution_mode']}\n"
        "training_executed=false\n"
        "hardware=cpu_only\n"
        "scheduler=none\n"
        "result_kind=smoke_only_not_benchmark\n"
        f"execution_status={results['execution_status']}\n"
        f"rdkit_available={results['rdkit_available']}\n"
        f"python_executable={results['python_executable']}\n"
        f"python_version={results['python_version']}\n"
        f"rdkit_version={results['rdkit_version']}\n"
        f"success_rate={metrics['success_rate']}\n"
        f"manifest={manifest_path}\n"
        f"results={results_path}\n"
        "note=Smoke-only RDKit run; no scheduler job, GPU use, or environment mutation.\n"
    )


def _tiny_real_log_text(results: dict[str, Any], manifest_path: Path, results_path: Path) -> str:
    metrics = results["metrics"]
    return (
        f"run_name={results['run_name']}\n"
        f"baseline_name={results['baseline_name']}\n"
        f"execution_mode={results['execution_mode']}\n"
        "training_executed=false\n"
        "hardware=cpu_only\n"
        "scheduler=none\n"
        "result_kind=tiny_real_data_dry_run_not_benchmark\n"
        f"execution_status={results['execution_status']}\n"
        f"rdkit_available={results['rdkit_available']}\n"
        f"python_executable={results['python_executable']}\n"
        f"python_version={results['python_version']}\n"
        f"rdkit_version={results['rdkit_version']}\n"
        f"success_rate={metrics['success_rate']}\n"
        f"manifest={manifest_path}\n"
        f"results={results_path}\n"
        "note=Tiny real-data RDKit dry-run; no scheduler job, GPU use, training, or environment mutation.\n"
    )
