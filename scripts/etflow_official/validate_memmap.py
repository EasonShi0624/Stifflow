"""Phase 3a: validate the official clean-sde memmap preprocessed dataset.

Exact totals, per-molecule boundaries vs Phase 1 inventory, finite energies,
raw-pickle sampling, and official split expansion (scaffold + random).
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path

import numpy as np

WS = Path(os.environ.get("STIFFLOW_WORKSPACE", Path(__file__).resolve().parents[3]))
RAW = WS / "dataset/rdkit_folder"
DEST = WS / "cache/etflow-official/geom-preprocessed"
ZEN = WS / "cache/etflow-official/zenodo"
SPLITS = WS / "cache/etflow-official/splits"
EXPANDED = SPLITS / "expanded"
INVENTORY = WS / "runs/etflow-official/phase1-inventory.npz"
REPORT = WS / "runs/etflow-official/phase3-validation.json"

TOTAL_NUM_MOLS = 437580
TOTAL_NUM_CONFORMERS = 33078483
TOTAL_NUM_ATOMS = 1656320500
TOTAL_ATOMS_UNIQUE_MOLS = 15904498

_RAW = None
_PARENT_STATE = None


def sha256_file(path, block=1 << 24):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def compare_molecule(index):
    """Compare one sampled molecule against its raw pickle."""
    import datamol as dm

    key = str(_INVENTORY[0][index])
    atoms = int(_INVENTORY[2][index])
    confs = int(_INVENTORY[3][index])
    pickle_path = str(_INVENTORY[4][index])
    mol_rows = _INVENTORY_ROWS[index]
    row0, row1 = int(mol_rows[0]), int(mol_rows[1])
    atomic = _ATOMIC
    pidx = _PIDX
    energy = _ENERGY
    block = atomic[pidx[row0, 0] : pidx[row1 - 1, 1]]
    positions = block[:, 2:].reshape(confs, atoms, 3)
    numbers = block[:, 0].reshape(confs, atoms)
    charges = block[:, 1].reshape(confs, atoms)
    with (RAW / pickle_path).open("rb") as stream:
        raw = pickle.load(stream)
    conf_list = raw["conformers"]
    check(
        len(conf_list) == confs, f"{key}: conformer count {len(conf_list)} != {confs}"
    )
    raw_pos = np.concatenate(
        [conf["rd_mol"].GetConformer().GetPositions() for conf in conf_list], axis=0
    ).astype(np.float32)
    expected_numbers = np.array(
        [atom.GetAtomicNum() for atom in conf_list[0]["rd_mol"].GetAtoms()],
        dtype=np.float32,
    )
    expected_charges = np.array(
        [atom.GetFormalCharge() for atom in conf_list[0]["rd_mol"].GetAtoms()],
        dtype=np.float32,
    )
    raw_energy = np.array([conf["totalenergy"] for conf in conf_list], dtype=np.float32)
    np.testing.assert_array_equal(positions.reshape(-1, 3), raw_pos)
    check(np.array_equal(numbers[0], expected_numbers), f"{key}: atomic numbers")
    check(np.array_equal(charges[0], expected_charges), f"{key}: charges")
    np.testing.assert_allclose(energy[row0:row1], raw_energy, rtol=0, atol=0)
    expected_smiles = [
        dm.to_smiles(
            conf["rd_mol"],
            canonical=False,
            explicit_hs=True,
            with_atom_indices=True,
            isomeric=True,
        )
        for conf in conf_list
    ]
    for row, expected in zip(range(row0, row1), expected_smiles):
        check(
            str(_SMILES_UNIQUE[_SMILES_INV[row]]) == expected,
            f"{key}: smiles row {row}",
        )
    return key


def init_worker(inventory):
    global _INVENTORY, _INVENTORY_ROWS, _ATOMIC, _PIDX, _ENERGY, _SMILES_UNIQUE, _SMILES_INV
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _INVENTORY = inventory
    rows = []
    offset = 0
    for confs in inventory[3]:
        rows.append((offset, offset + int(confs)))
        offset += int(confs)
    _INVENTORY_ROWS = np.asarray(rows, dtype=np.int64)
    state = _PARENT_STATE
    _ATOMIC = state["atomic"]
    _PIDX = state["pidx"]
    _ENERGY = state["energy"]
    _SMILES_UNIQUE = state["smiles_uniques"]
    _SMILES_INV = state["smiles_inv"]


def expand_rows(count_indices):
    rows = np.concatenate(
        [
            np.arange(int(start), int(start) + int(count), dtype=np.int64)
            for start, count in zip(count_indices[0], count_indices[1])
        ]
    )
    return np.unique(rows)


def canonical(smiles):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        try:
            import datamol as dm

            mol = dm.to_mol(smiles, remove_hs=False, ordered=True)
        except (ValueError, RuntimeError, TypeError):
            mol = None
    if mol is None:
        return None
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)


def ordered_signature(smiles):
    """Atom-map-ordered (atomic number, formal charge) sequence."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if sorted(maps) != list(range(mol.GetNumAtoms())):
        return None
    atoms = sorted(mol.GetAtoms(), key=lambda atom: atom.GetAtomMapNum())
    return tuple((atom.GetAtomicNum(), atom.GetFormalCharge()) for atom in atoms)


def identity_of_key(smiles):
    """Molecule identity consistent with stifflow.official_data.molecule_identity."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    sanitized = mol is not None
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    if sanitized:
        return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=400)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--hash", action="store_true")
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Full memmap validation requires an SHHPC allocation")
    if REPORT.exists() or EXPANDED.exists():
        parser.error("Existing validation reports/splits are protected; reuse them")
    if not 1 <= args.sample <= TOTAL_NUM_MOLS or args.workers < 1:
        parser.error("Invalid sample or worker count")
    started = time.perf_counter()
    report = {"dest": str(DEST), "checks": {}}

    # A. declared shapes and exact file sizes
    shapes = {
        "atomic_inputs.memmap": ("float32", (TOTAL_NUM_ATOMS, 5)),
        "pos_index_range.memmap": ("int32", (TOTAL_NUM_CONFORMERS, 2)),
        "energy.memmap": ("float32", (TOTAL_NUM_CONFORMERS,)),
    }
    sizes = {}
    for name, (dtype, shape) in shapes.items():
        path = DEST / name
        itemsize = np.dtype(dtype).itemsize
        expected_bytes = int(np.prod(shape)) * itemsize
        actual = path.stat().st_size
        check(actual == expected_bytes, f"{name}: {actual} != {expected_bytes}")
        sizes[name] = actual
    report["checks"]["file_sizes"] = sizes

    # B. inventory and per-molecule boundaries
    inventory = np.load(INVENTORY, allow_pickle=True)
    keys, partitions, num_atoms, num_confs = (
        inventory["keys"],
        inventory["partitions"],
        inventory["num_atoms"].astype(np.int64),
        inventory["num_confs"].astype(np.int64),
    )
    check(len(keys) == TOTAL_NUM_MOLS, "inventory molecule count")
    check(int(num_confs.sum()) == TOTAL_NUM_CONFORMERS, "inventory conformer count")
    atom_total = int((num_atoms * num_confs).sum())
    check(atom_total == TOTAL_NUM_ATOMS, "inventory atom count")
    check(
        int(num_atoms.sum()) == TOTAL_ATOMS_UNIQUE_MOLS, "inventory unique atom count"
    )
    pidx = np.memmap(
        DEST / "pos_index_range.memmap",
        dtype=np.int32,
        mode="r",
        shape=(TOTAL_NUM_CONFORMERS, 2),
    )
    starts = np.asarray(pidx[:, 0], dtype=np.int64)
    ends = np.asarray(pidx[:, 1], dtype=np.int64)
    check(int(starts[0]) == 0, "first conformer does not start at 0")
    check(
        int(ends[-1]) == TOTAL_NUM_ATOMS,
        "last conformer does not end at TOTAL_NUM_ATOMS",
    )
    check(bool(np.all(ends > starts)), "nonpositive conformer span")
    check(bool(np.all(starts[1:] == ends[:-1])), "conformer atom ranges not contiguous")
    widths = ends - starts
    conf_offset = np.concatenate([[0], np.cumsum(num_confs)])
    expected_width = np.repeat(num_atoms, num_confs)
    check(
        bool(np.array_equal(widths, expected_width)),
        "per-conformer atom counts differ from inventory",
    )
    atom_offsets = np.concatenate([[0], np.cumsum(num_atoms * num_confs)])
    check(
        bool(np.array_equal(starts[conf_offset[:-1]], atom_offsets[:-1])),
        "molecule starts differ from inventory",
    )
    report["checks"]["molecules_and_atoms"] = {
        "molecules": len(keys),
        "conformers": len(widths),
        "atoms": int(atom_total),
    }

    # C. smiles/subset inversion arrays and per-molecule consistency
    npz = np.load(DEST / "smiles.npz", allow_pickle=False)
    inv_smiles = npz["inv_indices"]
    uniques_list = [str(s) for s in npz["uniques"]]
    del npz
    subset_npz = np.load(DEST / "subset.npz", allow_pickle=False)
    subset = subset_npz["inv_indices"]
    check(
        len(inv_smiles) == TOTAL_NUM_CONFORMERS and len(subset) == TOTAL_NUM_CONFORMERS,
        "npz lengths",
    )
    check(bool(np.all(inv_smiles < len(uniques_list))), "smiles inv index out of range")
    first_rows, last_rows = conf_offset[:-1], conf_offset[1:] - 1
    check(
        bool(np.array_equal(subset, np.repeat(subset[first_rows], num_confs))),
        "molecule spans mixed subset ids",
    )
    mixed_smiles = int(np.sum(inv_smiles[first_rows] != inv_smiles[last_rows]))
    report["checks"]["npz"] = {
        "smiles_uniques": len(uniques_list),
        "molecules_with_mixed_conformer_smiles": mixed_smiles,
        "subset_uniques": [str(x) for x in subset_npz["uniques"]],
    }

    # D. energies finite (full scan) and no zero-fill tail
    energy = np.memmap(
        DEST / "energy.memmap",
        dtype=np.float32,
        mode="r",
        shape=(TOTAL_NUM_CONFORMERS,),
    )
    check(bool(np.isfinite(energy).all()), "nonfinite energy entries")
    atomic = np.memmap(
        DEST / "atomic_inputs.memmap",
        dtype=np.float32,
        mode="r",
        shape=(TOTAL_NUM_ATOMS, 5),
    )
    tail = atomic[TOTAL_NUM_ATOMS - 50 :]
    check(
        bool(np.isfinite(tail).all()) and bool(np.abs(tail).sum() > 0), "zero-fill tail"
    )
    sample_rows = np.linspace(0, TOTAL_NUM_ATOMS - 1, 20000, dtype=np.int64)
    check(
        bool(np.isfinite(atomic[sample_rows]).all()), "nonfinite atomic_inputs sample"
    )
    report["checks"]["energy"] = {"finite": True, "sampled_atoms": 20000}

    # E. raw-pickle sampling
    rng = np.random.default_rng(42)
    sample_idx = np.sort(rng.choice(TOTAL_NUM_MOLS, size=args.sample, replace=False))
    summary_paths = {}
    for partition in ("qm9", "drugs"):
        with (RAW / f"summary_{partition}.json").open() as stream:
            summary = json.load(stream)
        for key, value in summary.items():
            if value.get("pickle_path"):
                summary_paths[(partition, key)] = value["pickle_path"]
    pickle_for_index = [
        summary_paths[(str(partitions[i]), str(keys[i]))] for i in range(len(keys))
    ]
    inventory_tuple = (keys, partitions, num_atoms, num_confs, pickle_for_index)
    global _PARENT_STATE
    _PARENT_STATE = {
        "smiles_uniques": uniques_list,
        "smiles_inv": inv_smiles,
        "atomic": atomic,
        "pidx": pidx,
        "energy": energy,
    }
    with mp.Pool(
        args.workers, initializer=init_worker, initargs=(inventory_tuple,)
    ) as pool:
        compared = pool.map(compare_molecule, sample_idx.tolist(), chunksize=4)
    report["checks"]["raw_comparison"] = {"sampled": len(compared), "all_equal": True}

    # F. official splits: expand and cross-validate
    v2q, v2d = ZEN / "v2/QM9", ZEN / "v2/DRUGS"
    expanded = {}
    for name, path in {
        "random_qm9_train": v2q / "train_indices.npy",
        "random_qm9_val": v2q / "val_indices.npy",
        "random_drugs_train": v2d / "train_indices.npy",
        "random_drugs_val": v2d / "val_indices.npy",
        "scaffold_qm9_train": v2q / "scaffold_train_indices.npy",
        "scaffold_qm9_val": v2q / "scaffold_val_indices.npy",
    }.items():
        arr = np.load(path)
        check(
            arr.size > 0 and arr.min() >= 0 and arr.max() < TOTAL_NUM_CONFORMERS,
            f"{name} out of range",
        )
        expanded[name] = arr

    random_qm9_test = expand_rows(np.load(v2q / "count_indices.npy"))
    random_drugs_test = expand_rows(np.load(v2d / "count_indices.npy"))
    scaffold_qm9_test = expand_rows(np.load(v2q / "scaffold_qm9_count_indices.npy"))
    for name, arr in (
        ("random_qm9_test", random_qm9_test),
        ("random_drugs_test", random_drugs_test),
        ("scaffold_qm9_test", scaffold_qm9_test),
    ):
        check(arr.max() < TOTAL_NUM_CONFORMERS, f"{name} out of range")
        expanded[name] = arr
    checks = {
        "random_qm9_test_in_train": len(
            np.intersect1d(random_qm9_test, expanded["random_qm9_train"])
        ),
        "random_qm9_test_in_val": len(
            np.intersect1d(random_qm9_test, expanded["random_qm9_val"])
        ),
        "random_drugs_test_in_train": len(
            np.intersect1d(random_drugs_test, expanded["random_drugs_train"])
        ),
        "random_drugs_test_in_val": len(
            np.intersect1d(random_drugs_test, expanded["random_drugs_val"])
        ),
        "scaffold_qm9_test_in_train": len(
            np.intersect1d(scaffold_qm9_test, expanded["scaffold_qm9_train"])
        ),
        "scaffold_qm9_test_in_val": len(
            np.intersect1d(scaffold_qm9_test, expanded["scaffold_qm9_val"])
        ),
    }

    # map v3 scaffold SMILES to conformer rows via smiles.npz (order-sensitive)
    unique_index = {s: i for i, s in enumerate(uniques_list)}
    order = np.argsort(inv_smiles, kind="stable")
    sorted_inv = inv_smiles[order]
    boundaries = np.flatnonzero(np.diff(sorted_inv)) + 1
    bounds = np.concatenate([[0], boundaries, [order.size]])
    row_by_unique = {
        int(sorted_inv[bounds[i]]): order[bounds[i] : bounds[i + 1]]
        for i in range(len(bounds) - 1)
    }
    scaffold_files = {
        "qm9": ZEN / "v3/data/scaffold/qm9",
        "drugs": ZEN / "v3/data/scaffold/drugs",
    }
    canonical_map = None
    signature_cache = {}

    def unique_signature(uid):
        if uid not in signature_cache:
            signature_cache[uid] = ordered_signature(uniques_list[uid])
        return signature_cache[uid]

    def resolve(smiles):
        """Return (unique_id, method); unique_id None means unmapped/ambiguous."""
        if smiles in unique_index:
            return unique_index[smiles], "exact"
        c = canonical(smiles)
        ids = canonical_map.get(c, []) if c is not None else []
        if not ids:
            return None, "unmapped"
        signature = ordered_signature(smiles)
        if signature is None:
            return (ids[0], "canonical_unique") if len(ids) == 1 else (None, "ambiguous")
        matched = [uid for uid in ids if unique_signature(uid) == signature]
        if len(matched) == 1:
            return matched[0], "fingerprint"
        if not matched:
            target = sorted(signature)
            matched = [
                uid
                for uid in ids
                if unique_signature(uid) is not None
                and sorted(unique_signature(uid)) == target
            ]
            if len(matched) == 1:
                return matched[0], "multiset"
            if not matched:
                return None, "fingerprint_miss"
        return matched[0], "fingerprint_tied"

    scaffold_rows = {}
    scaffold_match = {}
    partition_identities = {}
    for partition, base in scaffold_files.items():
        for split in ("train", "val", "test"):
            with (base / f"scaffold_{split}_smiles.pkl").open("rb") as stream:
                smiles_list = pickle.load(stream)
            if canonical_map is None and any(s not in unique_index for s in smiles_list):
                print("building canonical fallback map ...", flush=True)
                canonical_map = {}
                for i, s in enumerate(uniques_list):
                    c = canonical(s)
                    if c is not None:
                        canonical_map.setdefault(c, []).append(i)
            train_identities = partition_identities.get((partition, "train"))
            val_identities = partition_identities.get((partition, "val"))
            seen = set()
            rows = []
            stats = {
                "molecules": len(smiles_list),
                "exact": 0,
                "fingerprint": 0,
                "multiset": 0,
                "canonical_unique": 0,
                "fingerprint_tied": 0,
                "duplicates_skipped": 0,
                "overlap_removed": 0,
                "unmapped": 0,
                "unmapped_examples": [],
            }
            for s in smiles_list:
                uid, method = resolve(s)
                if uid is None:
                    stats["unmapped"] += 1
                    if len(stats["unmapped_examples"]) < 5:
                        stats["unmapped_examples"].append([method, s[:80]])
                    continue
                row0 = int(row_by_unique[uid][0])
                index = int(np.searchsorted(conf_offset[1:], row0, side="right"))
                identity = identity_of_key(str(keys[index]))
                if identity is None:
                    stats["unmapped"] += 1
                    if len(stats["unmapped_examples"]) < 5:
                        stats["unmapped_examples"].append(["identity", s[:80]])
                    continue
                if identity in seen:
                    stats["duplicates_skipped"] += 1
                    continue
                seen.add(identity)
                if split == "val" and train_identities and identity in train_identities:
                    stats["overlap_removed"] += 1
                    continue
                if split == "test" and (
                    (train_identities and identity in train_identities)
                    or (val_identities and identity in val_identities)
                ):
                    stats["overlap_removed"] += 1
                    continue
                rows.append(np.arange(conf_offset[index], conf_offset[index + 1]))
                stats[method] += 1
            merged = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
            scaffold_rows[f"scaffold_{partition}_{split}"] = np.unique(merged)
            scaffold_match[f"scaffold_{partition}_{split}"] = stats
            partition_identities[(partition, split)] = seen
    report["checks"]["scaffold_matching"] = scaffold_match
    v2_sources = {
        "train": expanded["scaffold_qm9_train"],
        "val": expanded["scaffold_qm9_val"],
        "test": scaffold_qm9_test,
    }
    for split in ("train", "val", "test"):
        v2_arr = np.unique(v2_sources[split])
        v3_arr = scaffold_rows[f"scaffold_qm9_{split}"]
        overlap = np.intersect1d(v2_arr, v3_arr, assume_unique=True)
        checks[f"scaffold_qm9_v2_vs_v3_{split}_equal"] = bool(
            np.array_equal(v2_arr, v3_arr)
        )
        checks[f"scaffold_qm9_v2_vs_v3_{split}_v2_only"] = int(
            v2_arr.size - overlap.size
        )
        checks[f"scaffold_qm9_v2_vs_v3_{split}_v3_only"] = int(
            v3_arr.size - overlap.size
        )
    for partition in ("qm9", "drugs"):
        tr = scaffold_rows[f"scaffold_{partition}_train"]
        va = scaffold_rows[f"scaffold_{partition}_val"]
        te = scaffold_rows[f"scaffold_{partition}_test"]
        checks[f"scaffold_{partition}_train_val_overlap"] = len(np.intersect1d(tr, va))
        checks[f"scaffold_{partition}_test_in_train"] = len(np.intersect1d(te, tr))
        checks[f"scaffold_{partition}_test_in_val"] = len(np.intersect1d(te, va))
        molecule_sets = {}
        unparseable = {}
        for split, rows in (("train", tr), ("val", va), ("test", te)):
            indices = np.unique(np.searchsorted(conf_offset[1:], rows, side="right"))
            identities = {identity_of_key(str(keys[i])) for i in indices}
            unparseable[split] = int(None in identities)
            molecule_sets[split] = {x for x in identities if x is not None}
        checks[f"scaffold_{partition}_unparseable_identities"] = unparseable
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = len(molecule_sets[left] & molecule_sets[right])
            checks[f"scaffold_{partition}_{left}_{right}_molecule_overlap"] = overlap
        if partition == "qm9":
            diagnostics = {
                "sizes": {split: len(values) for split, values in molecule_sets.items()},
                "scaffold_match": scaffold_match,
                "row_overlaps": {
                    "train/val": checks[f"scaffold_{partition}_train_val_overlap"],
                    "train/test": checks[f"scaffold_{partition}_test_in_train"],
                    "val/test": checks[f"scaffold_{partition}_test_in_val"],
                },
                "identity_overlaps": {},
            }
            for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                inter = molecule_sets[left] & molecule_sets[right]
                diagnostics["identity_overlaps"][f"{left}/{right}"] = {
                    "count": len(inter),
                    "examples": sorted(inter)[:8],
                }
            (
                WS / "runs/etflow-official/phase4-clean-diagnostics.json"
            ).write_text(json.dumps(diagnostics, indent=2))
            check(
                all(
                    checks[f"scaffold_{partition}_{left}_{right}_molecule_overlap"] == 0
                    for left, right in (
                        ("train", "val"),
                        ("train", "test"),
                        ("val", "test"),
                    )
                ),
                "scaffold molecule leakage: qm9",
            )
            check(
                all(
                    checks[name] == 0
                    for name in (
                        f"scaffold_{partition}_train_val_overlap",
                        f"scaffold_{partition}_test_in_train",
                        f"scaffold_{partition}_test_in_val",
                    )
                ),
                f"scaffold row leakage: {partition}",
            )
            check(
                scaffold_match[f"scaffold_{partition}_test"]["unmapped"] == 0,
                f"unmapped scaffold {partition} test molecules",
            )
            check(
                not any(unparseable.values()),
                f"unparseable scaffold {partition} identities",
            )
    report["checks"]["splits"] = checks

    EXPANDED.mkdir(parents=True, exist_ok=False)
    to_save = {
        "random_qm9_train": expanded["random_qm9_train"],
        "random_qm9_val": expanded["random_qm9_val"],
        "random_qm9_test": random_qm9_test,
        "random_drugs_train": expanded["random_drugs_train"],
        "random_drugs_val": expanded["random_drugs_val"],
        "random_drugs_test": random_drugs_test,
        "scaffold_qm9_train": scaffold_rows["scaffold_qm9_train"],
        "scaffold_qm9_val": scaffold_rows["scaffold_qm9_val"],
        "scaffold_qm9_test": scaffold_rows["scaffold_qm9_test"],
        "scaffold_drugs_train": scaffold_rows["scaffold_drugs_train"],
        "scaffold_drugs_val": scaffold_rows["scaffold_drugs_val"],
        "scaffold_drugs_test": scaffold_rows["scaffold_drugs_test"],
    }
    saved = {}
    for name, arr in to_save.items():
        path = EXPANDED / f"{name}_indices.npy"
        with path.open("xb") as stream:
            np.save(stream, arr)
        saved[name] = {"rows": int(arr.size), "path": str(path.relative_to(WS))}
    report["expanded_splits"] = saved

    if args.hash:
        report["hashes"] = {
            name: sha256_file(DEST / name)
            for name in (
                "atomic_inputs.memmap",
                "pos_index_range.memmap",
                "energy.memmap",
            )
        }
    report["seconds"] = time.perf_counter() - started
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report["checks"]["splits"], indent=2))
    print(json.dumps(report["checks"]["molecules_and_atoms"], indent=2))
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
