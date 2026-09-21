"""Freeze ET-Flow's legacy split order and preserve GEOM conformer identities."""

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem

from .preflight import fingerprint


def build_manifest(raw_dir, split_path, official_test_smiles=None):
    """Freeze the exact nonrecursive Path.glob order used by prepare_data.py.

    The official test list independently checks isolation, but cannot establish
    the historical ordering of every train/validation molecule.
    """
    raw_dir, split_path = Path(raw_dir).resolve(), Path(split_path).resolve()
    files = list(raw_dir.glob("*"))
    if not files or any(p.suffix != ".pickle" for p in files):
        raise ValueError(
            "Expected a nonempty directory containing only GEOM .pickle files"
        )
    splits = np.load(split_path, allow_pickle=True)
    if len(splits) != 3:
        raise ValueError("Expected train/validation/test index arrays")
    assignments = {}
    for name, indices in zip(("train", "val", "test"), splits):
        for index in indices:
            if not isinstance(index, (int, np.integer)) or not 0 <= index < len(files):
                raise ValueError("Invalid split index")
            if int(index) in assignments:
                raise ValueError("Duplicate or overlapping split indices")
            assignments[int(index)] = name
    if len(assignments) != len(files):
        raise ValueError("Splits do not cover the complete raw filename list")
    entries = [
        {"index": i, "file": p.name, "molecule_id": p.stem, "split": assignments[i]}
        for i, p in enumerate(files)
    ]
    if len({e["molecule_id"] for e in entries}) != len(entries):
        raise ValueError("Duplicate molecule IDs")
    official = None
    if official_test_smiles is not None:
        with Path(official_test_smiles).open(newline="") as stream:
            test_ids = [row["smiles"] for row in csv.DictReader(stream)]
        membership = {e["molecule_id"]: e["split"] for e in entries}
        if not test_ids or len(set(test_ids)) != len(test_ids):
            raise ValueError("Official test list is empty or contains duplicate IDs")
        bad = [s for s in test_ids if membership.get(s) != "test"]
        if bad:
            raise ValueError(f"Official test isolation failed for {len(bad)} molecules")
        official = {
            **fingerprint(official_test_smiles),
            "count": len(test_ids),
            "isolated": True,
        }
    names = json.dumps([p.name for p in files], ensure_ascii=False).encode()
    return {
        "schema": 1,
        "raw_dir": str(raw_dir),
        "split_file": fingerprint(split_path),
        "filename_order_sha256": hashlib.sha256(names).hexdigest(),
        "mapping_rule": "ETFlow-reference/scripts/prepare_data.py: list(raw_dir.glob('*')); frozen once",
        "mapping_limit": "Official test membership checks isolation; historical train/val ordering is not independently proven.",
        "official_test": official,
        "counts": {
            name: sum(e["split"] == name for e in entries)
            for name in ("train", "val", "test")
        },
        "entries": entries,
    }


def graph_from_mol(mol):
    """Graph metadata retains atom order, explicit hydrogens and stereochemistry."""
    mol = Chem.Mol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return {
        "atomic_numbers": [a.GetAtomicNum() for a in mol.GetAtoms()],
        "charges": [a.GetFormalCharge() for a in mol.GetAtoms()],
        "atom_mapping": [a.GetAtomMapNum() for a in mol.GetAtoms()],
        "atom_indices": list(range(mol.GetNumAtoms())),
        "chiral_tags": [int(a.GetChiralTag()) for a in mol.GetAtoms()],
        "cip_codes": [
            a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else None
            for a in mol.GetAtoms()
        ],
        "bonds": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()],
        "bond_types": [str(b.GetBondType()) for b in mol.GetBonds()],
        "bond_stereo": [int(b.GetStereo()) for b in mol.GetBonds()],
        "bond_stereo_atoms": [list(b.GetStereoAtoms()) for b in mol.GetBonds()],
    }


def load_molecule(entry, raw_dir, top_n=30):
    """Keep ET-Flow's top-weight 30, with uniform within-molecule sampling.

    Only load trusted local GEOM pickles. No raw file is modified.
    """
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be positive")
    if entry["split"] not in ("train", "val", "test"):
        raise ValueError("Unknown split")
    filename = Path(entry["file"])
    if filename.name != str(filename) or filename.suffix != ".pickle":
        raise ValueError("Manifest filename must be a local .pickle basename")
    path = Path(raw_dir) / filename
    with path.open("rb") as stream:
        raw = pickle.load(stream)
    conformers = list(enumerate(raw["conformers"]))
    if not conformers:
        raise ValueError("No conformers")
    first_mol = conformers[0][1]["rd_mol"]
    if len(Chem.GetMolFrags(first_mol)) != 1:
        raise ValueError("Disconnected components (same exclusion as ET-Flow)")
    graph = graph_from_mol(first_mol)
    if entry["split"] in ("train", "val"):
        conformers = sorted(
            conformers, key=lambda pair: pair[1]["boltzmannweight"], reverse=True
        )[:top_n]
    kept = []
    for raw_index, conf in conformers:
        mol = Chem.Mol(conf["rd_mol"])
        if graph_from_mol(mol) != graph:
            raise ValueError(
                "Atom order, chemical graph or stereochemistry changes across conformers"
            )
        positions = np.array(mol.GetConformer().GetPositions(), dtype=np.float64)
        energy, weight = float(conf["totalenergy"]), float(conf["boltzmannweight"])
        if (
            not np.isfinite(positions).all()
            or not np.isfinite([energy, weight]).all()
            or weight < 0
        ):
            raise ValueError("Invalid conformer coordinates, energy or weight")
        kept.append(
            {
                "conformer_id": f"{entry['molecule_id']}:{raw_index}:{conf.get('geom_id', 'unknown')}",
                "raw_index": raw_index,
                "geom_id": conf.get("geom_id"),
                "positions": positions,
                "rd_mol": mol,
                "energy": energy,
                "boltzmann_weight": weight,
                "sampling_probability": 1 / len(conformers),
            }
        )
    return {
        "molecule_id": entry["molecule_id"],
        "split": entry["split"],
        "graph": graph,
        "conformers": kept,
        "source": fingerprint(path),
        "coordinate_unit": "angstrom",
        "raw_energy_unit": "GEOM original; retained without conversion",
        "sampling": "uniform molecules; uniform retained conformers; top-30 by original Boltzmann weight",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--official-test-smiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.raw_dir, args.split_path, args.official_test_smiles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(manifest, stream, indent=2)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "counts": manifest["counts"],
                "official_test": manifest["official_test"],
            }
        )
    )


if __name__ == "__main__":
    main()
