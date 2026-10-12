"""QM9 v3 scaffold adapter for clean-sde@2f20cad's read-only GEOM memmaps.

Reuse the official GEOM row layout without switching the installed ET-Flow
backbone. Inventory offsets preserve the original pickle/atom correspondence.
"""

import argparse
import hashlib
import json
import os
import pickle
from itertools import pairwise
from pathlib import Path

import numpy as np
from rdkit import Chem

from .data import graph_from_mol
from .preflight import WORKSPACE, fingerprint
from .smoke import external_directory

KIND = "official_memmap_scaffold"
OFFICIAL_COMMIT = "2f20cadeb047b7dc7cd3a363c729373f288c2d08"
SAMPLING = "uniform_conformer_rows"


def verified_path(record):
    path = Path(record["path"])
    if fingerprint(path)["sha256"] != record["sha256"]:
        raise ValueError(f"Metadata hash mismatch: {path}")
    return path


def molecule_identity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    sanitized = mol is not None
    if mol is None:
        # GEOM summary keys include radicals/zwitterions that fail sanitization;
        # fall back to an unsanitized parse and canonical string.
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Unparseable inventory molecule: {smiles}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    if sanitized:
        return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return smiles


class OfficialDataset:
    def __init__(self, specification):
        if (
            specification.get("schema_version") != 2
            or specification.get("dataset_kind") != KIND
            or specification.get("official_commit") != OFFICIAL_COMMIT
            or specification.get("sampling") != SAMPLING
        ):
            raise ValueError("Expected a frozen v3 QM9 scaffold memmap manifest")
        self.specification = specification
        validation = json.loads(verified_path(specification["validation"]).read_text())
        if validation["checks"]["raw_comparison"].get("all_equal") is not True:
            raise ValueError("Official raw/memmap validation did not pass")
        with np.load(
            verified_path(specification["inventory"]), allow_pickle=False
        ) as inv:
            self.keys = inv["keys"].copy()
            self.partitions = inv["partitions"].copy()
            self.atoms = inv["num_atoms"].astype(np.int64)
            self.conformers = inv["num_confs"].astype(np.int64)
        n = len(self.keys)
        if (
            any(a.shape != (n,) for a in (self.partitions, self.atoms, self.conformers))
            or np.any(self.atoms < 1)
            or np.any(self.conformers < 1)
        ):
            raise ValueError("Invalid official inventory counts")
        self.row_offsets = np.r_[0, np.cumsum(self.conformers)]
        self.atom_offsets = np.r_[0, np.cumsum(self.atoms * self.conformers)]
        data_dir = Path(specification["data_dir"])
        shapes = {
            "atomic_inputs.memmap": (np.float32, (int(self.atom_offsets[-1]), 5)),
            "pos_index_range.memmap": (np.int32, (int(self.row_offsets[-1]), 2)),
            "energy.memmap": (np.float32, (int(self.row_offsets[-1]),)),
        }
        arrays = {}
        for name, (dtype, shape) in shapes.items():
            path = data_dir / name
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            if (
                path.stat().st_size != size
                or specification["memmaps"][name]["bytes"] != size
                or validation["checks"]["file_sizes"][name] != size
                or validation["hashes"][name]
                != specification["memmaps"][name]["sha256"]
            ):
                raise ValueError(f"Memmap size differs from verified inventory: {name}")
            # Large hashes come from SHHPC's full validation, not another full scan.
            digest = specification["memmaps"][name]["sha256"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Missing full-validation hash: {name}")
            arrays[name] = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        self.atomic = arrays["atomic_inputs.memmap"]
        self.ranges = arrays["pos_index_range.memmap"]
        self.energy = arrays["energy.memmap"]
        self.raw_dir = Path(specification["raw_dir"]).resolve()
        self.summary_path = verified_path(specification["raw_summary"])
        self.summary = None

    @classmethod
    def from_manifest(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def groups(self, split):
        rows = np.load(
            verified_path(self.specification["splits"][split]), allow_pickle=False
        )
        if (
            rows.ndim != 1
            or not np.issubdtype(rows.dtype, np.integer)
            or not len(rows)
            or np.any(rows < 0)
            or np.any(rows >= self.row_offsets[-1])
        ):
            raise ValueError(f"Invalid official {split} row indices")
        rows = np.sort(rows.astype(np.int64))
        if np.any(np.diff(rows) == 0):
            raise ValueError(f"Duplicate official {split} rows")
        inventory_indices = np.searchsorted(self.row_offsets[1:], rows, side="right")
        boundaries = np.r_[0, np.flatnonzero(np.diff(inventory_indices)) + 1, len(rows)]
        groups, identities = [], set()
        for start, end in pairwise(boundaries):
            index = int(inventory_indices[start])
            if str(self.partitions[index]) != "qm9":
                raise ValueError("QM9 scaffold indices include another partition")
            identity = "qm9:" + molecule_identity(str(self.keys[index]))
            if identity in identities:
                raise ValueError(f"Ambiguous duplicate chemical identity: {identity}")
            identities.add(identity)
            groups.append(
                {
                    "id": identity,
                    "inventory_index": index,
                    "row_ids": rows[start:end].tolist(),
                }
            )
        return groups

    def validate_splits(self):
        groups = {split: self.groups(split) for split in ("train", "val", "test")}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if {g["id"] for g in groups[left]} & {g["id"] for g in groups[right]}:
                raise ValueError(f"Official molecule leakage: {left}/{right}")
        return {
            split: {
                "molecules": len(items),
                "rows": sum(len(g["row_ids"]) for g in items),
            }
            for split, items in groups.items()
        }

    def candidates(self, split, seed=42):
        return sorted(
            self.groups(split),
            key=lambda g: hashlib.sha256(f"{seed}:{g['id']}".encode()).hexdigest(),
        )

    def load_molecule(self, entry, split, limit=None):
        if self.summary is None:
            self.summary = json.loads(self.summary_path.read_text())
        index = entry["inventory_index"]
        key = str(self.keys[index])
        path = (self.raw_dir / self.summary[key]["pickle_path"]).resolve()
        if not path.is_relative_to(self.raw_dir):
            raise ValueError("Raw pickle path escapes the read-only dataset")
        with path.open("rb") as stream:
            raw = pickle.load(stream)  # Trusted local GEOM artifacts only.
        conformers = raw["conformers"]
        if len(conformers) != int(self.conformers[index]):
            raise ValueError("Raw conformer count differs from official inventory")
        rows = entry["row_ids"] if limit is None else entry["row_ids"][:limit]
        selected, graph = [], None
        for row in rows:
            raw_index = row - int(self.row_offsets[index])
            if not 0 <= raw_index < len(conformers):
                raise ValueError("Official row does not belong to its molecule")
            start = int(self.atom_offsets[index] + raw_index * self.atoms[index])
            end = start + int(self.atoms[index])
            if not np.array_equal(self.ranges[row], [start, end]):
                raise ValueError("Official row/atom offsets disagree")
            block = np.asarray(self.atomic[start:end])
            conf = conformers[raw_index]
            mol = Chem.Mol(conf["rd_mol"])
            metadata = graph_from_mol(mol)
            if graph is None:
                graph = metadata
            if graph != metadata or len(Chem.GetMolFrags(mol)) != 1:
                raise ValueError(
                    "Atom order, topology or stereo varies across official rows"
                )
            if (
                not np.isfinite(block).all()
                or not np.array_equal(block[:, 0], metadata["atomic_numbers"])
                or not np.array_equal(block[:, 1], metadata["charges"])
                or not np.array_equal(
                    block[:, 2:], mol.GetConformer().GetPositions().astype(np.float32)
                )
                or not np.isfinite(self.energy[row])
                or np.float32(conf["totalenergy"]) != self.energy[row]
            ):
                raise ValueError(
                    "Official memmap differs from raw atom order/coordinates/energy"
                )
            positions = np.array(block[:, 2:], dtype=np.float64)
            weight = float(conf["boltzmannweight"])
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("Invalid original conformer weight")
            for atom, xyz in enumerate(positions):
                mol.GetConformer().SetAtomPosition(atom, xyz)
            selected.append(
                {
                    "row_id": row,
                    "raw_index": raw_index,
                    "conformer_id": f"{entry['id']}:row:{row}",
                    "positions": positions,
                    "rd_mol": mol,
                    "energy": float(self.energy[row]),
                    "boltzmann_weight": weight,
                }
            )
        return {
            "molecule_id": entry["id"],
            "split": split,
            "graph": graph,
            "conformers": selected,
            "source": fingerprint(path),
        }


def build_manifest(workspace):
    workspace = Path(workspace).resolve()
    report_path = workspace / "runs/etflow-official/phase3-validation.json"
    report = json.loads(report_path.read_text())
    if report["checks"]["raw_comparison"].get("all_equal") is not True:
        raise ValueError("Official raw/memmap validation did not pass")
    specification = {
        "schema_version": 2,
        "dataset_kind": KIND,
        "sampling": SAMPLING,
        "official_commit": OFFICIAL_COMMIT,
        "partition": "qm9",
        "split_scheme": "v3_scaffold",
        "data_dir": str(workspace / "cache/etflow-official/geom-preprocessed"),
        "raw_dir": str(workspace / "dataset/rdkit_folder"),
        "raw_summary": fingerprint(workspace / "dataset/rdkit_folder/summary_qm9.json"),
        "inventory": fingerprint(
            workspace / "runs/etflow-official/phase1-inventory.npz"
        ),
        "validation": fingerprint(report_path),
        "splits": {
            split: fingerprint(
                workspace
                / f"cache/etflow-official/splits/expanded/scaffold_qm9_{split}_indices.npy"
            )
            for split in ("train", "val", "test")
        },
        "memmaps": {
            name: {"bytes": size, "sha256": report["hashes"][name]}
            for name, size in report["checks"]["file_sizes"].items()
        },
        "unmapped": report["checks"]["scaffold_matching"],
    }
    specification["checks"] = OfficialDataset(specification).validate_splits()
    return specification


def validate_label_manifest(manifest):
    if (
        manifest.get("schema_version") != 2
        or manifest.get("dataset_kind") != KIND
        or manifest.get("sampling") != SAMPLING
        or manifest.get("status") != "completed"
        or manifest.get("audit_partial") is not False
        or manifest.get("requested_attempts_completed") is not True
    ):
        raise ValueError("Training requires complete official-conformer labels")
    specification = json.loads(verified_path(manifest["data_manifest"]).read_text())
    if specification.get("dataset_kind") != KIND or specification.get("checks") is None:
        raise ValueError("Unverified official data manifest")
    expected = {
        g["id"]: g["row_ids"]
        for g in OfficialDataset(specification).groups(manifest["split"])
    }
    ids, rows = set(), set()
    for record in manifest["records"]:
        selected = record["row_ids"]
        if (
            record["id"] in ids
            or record["split"] != manifest["split"]
            or selected != expected.get(record["id"])
            or record["audit_partial"]
            or not selected
            or len(selected) != record["retained_conformers"]
            or len(selected) != len(record["conformer_ids"])
            or any(not isinstance(row, int) or row < 0 for row in selected)
            or len(set(selected)) != len(selected)
            or rows.intersection(selected)
        ):
            raise ValueError("Incomplete, duplicate or unpaired official rows")
        ids.add(record["id"])
        rows.update(selected)
    if not ids or len(ids) != manifest["requested_molecules"]:
        raise ValueError("Official shared molecule list is incomplete")
    return ids, rows


def validate_label_pair(train, val):
    train_ids, train_rows = validate_label_manifest(train)
    val_ids, val_rows = validate_label_manifest(val)
    if (
        train["split"] != "train"
        or val["split"] != "val"
        or train["data_manifest"]["sha256"] != val["data_manifest"]["sha256"]
        or train["selection_seed"] != val["selection_seed"]
        or train_ids & val_ids
        or train_rows & val_rows
    ):
        raise ValueError(
            "Official train/validation dataset or molecule/row isolation differs"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("STIFFLOW_WORKSPACE", WORKSPACE)),
    )
    parser.add_argument("--output", type=external_directory, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Full split metadata checks belong in an SHHPC allocation")
    if args.output.exists():
        parser.error("Existing official manifests are never overwritten")
    specification = build_manifest(args.workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(specification, stream, indent=2, allow_nan=False)
    print(json.dumps(specification["checks"]))


if __name__ == "__main__":
    main()
