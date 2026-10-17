"""Repeatable, failure-accounted heavy-atom COV/MAT evaluation of NPZ files."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from stifflow.preflight import fingerprint, provenance

METHODS = {"precision", "alternative_a", "control", "etflow_pretrained"}
DEFAULT_THRESHOLDS = np.arange(0.05, 3.05, 0.05)


def conformer_mol(mol, positions):
    """Copy the graph and set this conformer's coordinates without SMILES caches."""
    result = Chem.Mol(mol)
    result.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i, point in enumerate(positions):
        conformer.SetAtomPosition(i, point.tolist())
    result.AddConformer(conformer, assignId=True)
    return result


def tetrahedral_chirality(mol, generated):
    """Exact atom-index CIP agreement for every specified tetrahedral center.

    Coordinate-derived tags are assigned on graph copies after clearing input
    stereo tags. Unspecified centers and double-bond stereo are not assessed.
    """
    expected_mol = Chem.Mol(mol)
    Chem.AssignStereochemistry(expected_mol, cleanIt=True, force=True)
    expected = {
        atom.GetIdx(): atom.GetProp("_CIPCode")
        for atom in expected_mol.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        and atom.HasProp("_CIPCode")
    }
    if not expected:
        return {"specified_centers": 0, "assessed": 0, "correct": 0, "fraction": None}
    correct = 0
    for positions in generated:
        candidate = conformer_mol(mol, positions)
        Chem.RemoveStereochemistry(candidate)
        Chem.AssignAtomChiralTagsFromStructure(
            candidate, confId=0, replaceExistingTags=True
        )
        Chem.AssignStereochemistry(candidate, cleanIt=True, force=True)
        correct += all(
            candidate.GetAtomWithIdx(i).HasProp("_CIPCode")
            and candidate.GetAtomWithIdx(i).GetProp("_CIPCode") == code
            for i, code in expected.items()
        )
    return {
        "specified_centers": len(expected),
        "assessed": len(generated),
        "correct": correct,
        "fraction": correct / len(generated),
    }


def evaluate_record(record, base_dir, thresholds, ratio=2, allow_disconnected=False):
    """An attempted molecule is always returned, including explicit failures.

    Multi-fragment reference molecules are part of the frozen split.  The
    benchmark CLI keeps rejecting them by default; formal evaluation opts in
    explicitly and records the decision in the report protocol.
    """
    result = {"id": record["id"], "input_status": record["status"]}
    if record["status"] != "ok":
        return result | {
            "status": "failed",
            "reason": record.get("reason", "generation failed"),
        }
    try:
        path = (base_dir / record["npz"]).resolve()
        result["input"] = fingerprint(path)
        mol = Chem.MolFromMolBlock(record["molblock"], removeHs=False, sanitize=True)
        if mol is None:
            raise ValueError("invalid molecule")
        result["multi_fragment"] = len(Chem.GetMolFrags(mol)) != 1
        if result["multi_fragment"] and not allow_disconnected:
            raise ValueError("disconnected molecule")
        n_atoms = mol.GetNumAtoms()
        with np.load(path, allow_pickle=False) as data:
            ref = data["pos_ref"]
            generated = data["pos_gen"]
            numbers = data["atomic_numbers"]
        expected_numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
        if numbers.shape != (n_atoms,) or not np.array_equal(numbers, expected_numbers):
            raise ValueError("atomic_numbers do not match molblock atom order")
        # Atom identity/order comes from the molblock; atomic numbers alone cannot
        # distinguish a permutation of equal elements. Producers must not reorder.
        for name, positions in (("pos_ref", ref), ("pos_gen", generated)):
            if positions.ndim != 3 or positions.shape[1:] != (n_atoms, 3):
                raise ValueError(f"{name} must have shape [conformers, {n_atoms}, 3]")
            if (
                not np.issubdtype(positions.dtype, np.number)
                or not np.isfinite(positions).all()
            ):
                raise ValueError(f"{name} must contain finite numeric coordinates")
        if len(ref) == 0:
            raise ValueError("empty reference ensemble")
        required = ratio * len(ref)
        result.update(
            n_ref=len(ref), n_generated_available=len(generated), n_required=required
        )
        if len(generated) < required:
            raise ValueError(f"insufficient conformers: {len(generated)} < {required}")
        generated = generated[:required]
        references = [Chem.RemoveHs(conformer_mol(mol, p)) for p in ref]
        probes = [Chem.RemoveHs(conformer_mol(mol, p)) for p in generated]
        # Same RemoveHs + GetBestRMS defaults as ETFlow commons/covmat.py.
        # ponytail: dense O(N_ref * N_gen) RMSD; blockwise minima if ensembles grow.
        rmsd = np.array(
            [
                [rdMolAlign.GetBestRMS(Chem.Mol(probe), reference) for probe in probes]
                for reference in references
            ]
        )
        if not np.isfinite(rmsd).all():
            raise ValueError("non-finite pairwise RMSD")
        recall, precision = rmsd.min(axis=1), rmsd.min(axis=0)
        result.update(
            status="ok",
            n_evaluated=required,
            cov_r=(recall[:, None] < thresholds).mean(axis=0).tolist(),
            cov_p=(precision[:, None] < thresholds).mean(axis=0).tolist(),
            mat_r=float(recall.mean()),
            mat_p=float(precision.mean()),
            nearest_reference_rmsd=precision.tolist(),
            nearest_generated_rmsd=recall.tolist(),
            chirality=tetrahedral_chirality(mol, generated),
        )
    except (ValueError, KeyError, OSError, RuntimeError, TypeError) as error:
        result.update(status="failed", reason=f"{type(error).__name__}: {error}")
    return result


def aggregate(records, n_thresholds):
    successful = [r for r in records if r["status"] == "ok"]
    metrics = {}
    for name in ("cov_r", "cov_p", "mat_r", "mat_p"):
        values = np.array([r[name] for r in successful])
        metrics[name + "_mean"] = values.mean(axis=0).tolist() if successful else None
        metrics[name + "_median"] = (
            np.median(values, axis=0).tolist() if successful else None
        )
    total = len(records)
    attempted_cov = {
        name: (
            np.sum([r[name] for r in successful], axis=0) / total
            if successful
            else np.zeros(n_thresholds)
        ).tolist()
        for name in ("cov_r", "cov_p")
    }
    assessed = sum(r["chirality"]["assessed"] for r in successful)
    correct = sum(r["chirality"]["correct"] for r in successful)
    return {
        "attempted": total,
        "succeeded": len(successful),
        "failed": total - len(successful),
        "success_fraction": len(successful) / total,
        "multi_fragment_molecules": sum(
            1 for r in successful if r.get("multi_fragment")
        ),
        "successful_only": metrics,
        "attempted_coverage_failures_zero": attempted_cov,
        "attempted_mat": (
            None
            if len(successful) != total
            else {key: metrics[key + "_mean"] for key in ("mat_r", "mat_p")}
        ),
        "chirality": {
            "assessed": assessed,
            "correct": correct,
            "fraction": correct / assessed if assessed else None,
        },
    }


def evaluate(
    manifest_path,
    thresholds=DEFAULT_THRESHOLDS,
    ratio=2,
    allow_disconnected=False,
):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.ndim != 1 or not len(thresholds) or not np.isfinite(thresholds).all():
        raise ValueError("thresholds must be a nonempty finite vector")
    if np.any(thresholds <= 0) or np.any(np.diff(thresholds) <= 0):
        raise ValueError("thresholds must be positive and strictly increasing")
    if not isinstance(ratio, int) or ratio < 1:
        raise ValueError("ratio must be a positive integer")
    for key in (
        "schema_version",
        "method",
        "baseline_kind",
        "seed",
        "split",
        "config",
        "training_budget",
        "generation",
        "provenance",
        "records",
    ):
        if key not in manifest:
            raise ValueError(f"manifest missing {key}")
    if manifest["schema_version"] != 1 or manifest["method"] not in METHODS:
        raise ValueError("unsupported schema_version or method")
    kind = (
        "external_pretrained"
        if manifest["method"] == "etflow_pretrained"
        else "controlled"
    )
    if manifest["baseline_kind"] != kind:
        raise ValueError(f"{manifest['method']} requires baseline_kind={kind}")
    records = manifest["records"]
    if not records or any(
        not isinstance(r.get("id"), str) or "status" not in r for r in records
    ):
        raise ValueError("records require nonempty string IDs and status")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate attempted molecule ID")
    started = time.perf_counter()
    results = [
        evaluate_record(
            r,
            manifest_path.parent,
            thresholds,
            ratio,
            allow_disconnected=allow_disconnected,
        )
        for r in records
    ]
    return {
        "schema_version": 1,
        "input_manifest": fingerprint(manifest_path),
        "run": {k: v for k, v in manifest.items() if k != "records"},
        "protocol": {
            "coordinates": "angstrom",
            "heavy_atoms": True,
            "rmsd": "RDKit RemoveHs + GetBestRMS defaults; proper alignment",
            "coverage_comparison": "strict <",
            "thresholds": thresholds.tolist(),
            "generated_reference_ratio": ratio,
            "force_field_optimization": False,
            "partial_rmsd_failure": "entire molecule marked failed",
            "connectivity": (
                "multi-fragment reference molecules allowed"
                if allow_disconnected
                else "single-fragment molecules only"
            ),
            "chirality": "all specified tetrahedral atom-index CIP labels; no E/Z metric",
        },
        "environment": {
            "python": sys.executable,
            "python_version": sys.version,
            "numpy": np.__version__,
            "rdkit": rdkit.__version__,
            "module": __file__,
        },
        "evaluation_seconds": time.perf_counter() - started,
        "records": results,
        "aggregate": aggregate(results, len(thresholds)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS
    )
    parser.add_argument("--ratio", type=int, default=2)
    parser.add_argument(
        "--allow-disconnected",
        action="store_true",
        help="evaluate multi-fragment molecules instead of marking them failed",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new report path")
    report = evaluate(
        args.manifest,
        args.thresholds,
        args.ratio,
        allow_disconnected=args.allow_disconnected,
    )
    report["evaluation_provenance"] = provenance()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report["aggregate"], indent=2))
    raise SystemExit(1 if report["aggregate"]["failed"] else 0)


if __name__ == "__main__":
    main()
