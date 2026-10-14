"""Small synthetic files check row identity, split leakage and label completeness."""

import copy
import hashlib
import json
import pickle
from unittest.mock import patch

import numpy as np
import pytest
from rdkit import Chem

from stifflow.geometry import centered, centered_basis
from stifflow.official_data import (
    OfficialDataset,
    build_manifest,
    graph_signatures,
    validate_label_manifest,
    validate_label_pair,
)
from stifflow.official_labels import build_cache
from stifflow.preflight import fingerprint
from stifflow.training import draw_record, load_records, make_loader


@pytest.fixture
def official(tmp_path):
    raw = tmp_path / "dataset/rdkit_folder"
    raw.mkdir(parents=True)
    data = tmp_path / "cache/etflow-official/geom-preprocessed"
    data.mkdir(parents=True)
    runs = tmp_path / "runs/etflow-official"
    runs.mkdir(parents=True)
    splits = tmp_path / "cache/etflow-official/splits/expanded"
    splits.mkdir(parents=True)
    keys = ["CC", "CO", "CN", "CF"]
    counts = [2, 3, 2, 1]
    summary, atoms, blocks, ranges, energies = {}, [], [], [], []
    offset = 0
    for molecule, (key, count) in enumerate(zip(keys, counts)):
        mol = Chem.AddHs(Chem.MolFromSmiles(key))
        n = mol.GetNumAtoms()
        atoms.append(n)
        conformers = []
        for conf_index in range(count):
            copy_mol = Chem.Mol(mol)
            xyz = np.arange(n * 3).reshape(n, 3) * 0.11 + molecule + conf_index * 0.03
            conf = Chem.Conformer(n)
            for index, pos in enumerate(xyz):
                conf.SetAtomPosition(index, pos)
            copy_mol.AddConformer(conf)
            numbers = np.array([a.GetAtomicNum() for a in copy_mol.GetAtoms()])
            charges = np.zeros(n)
            blocks.append(np.column_stack([numbers, charges, xyz]).astype(np.float32))
            ranges.append([offset, offset + n])
            offset += n
            energy = float(molecule + conf_index)
            energies.append(energy)
            conformers.append(
                {
                    "rd_mol": copy_mol,
                    "totalenergy": energy,
                    "boltzmannweight": 1.0 / count,
                }
            )
        filename = f"molecule-{molecule}.pickle"
        with (raw / filename).open("wb") as stream:
            pickle.dump({"conformers": conformers}, stream)
        summary[key] = {"pickle_path": filename}
    (raw / "summary_qm9.json").write_text(json.dumps(summary))
    np.concatenate(blocks).tofile(data / "atomic_inputs.memmap")
    np.asarray(ranges, dtype=np.int32).tofile(data / "pos_index_range.memmap")
    np.asarray(energies, dtype=np.float32).tofile(data / "energy.memmap")
    np.savez_compressed(
        runs / "phase1-inventory.npz",
        keys=keys,
        partitions=["qm9"] * 4,
        num_atoms=atoms,
        num_confs=counts,
    )
    for split, rows in {"train": [0, 1, 2, 3, 4], "val": [5, 6], "test": [7]}.items():
        np.save(
            splits / f"scaffold_qm9_{split}_indices.npy",
            np.asarray(rows, dtype=np.int64),
        )
    names = ("atomic_inputs.memmap", "pos_index_range.memmap", "energy.memmap")
    report = {
        "checks": {
            "raw_comparison": {"all_equal": True},
            "file_sizes": {name: (data / name).stat().st_size for name in names},
            "scaffold_matching": {},
        },
        "hashes": {
            name: hashlib.sha256((data / name).read_bytes()).hexdigest()
            for name in names
        },
    }
    (runs / "phase3-validation.json").write_text(json.dumps(report))
    manifest = tmp_path / "official.json"
    manifest.write_text(json.dumps(build_manifest(tmp_path)))
    return OfficialDataset.from_manifest(manifest), manifest, tmp_path


def synthetic_stiffness(mol):
    # Synthetic SPD labels exercise I/O; they do not validate physical Hessians.
    target = mol.GetConformer().GetPositions().copy()
    Q = centered_basis(len(target))
    dimension = Q.shape[1]
    K = np.eye(dimension) * (1 + target[0, 0])
    return {
        "target": target,
        "reference": centered(target),
        "K": K,
        "Q": Q,
        "H_raw": K,
        "epsilon_E": 0.5924849497,
        "spectrum": np.diag(K),
        "floored_mask": np.zeros(dimension, dtype=bool),
        "audit": {"synthetic": True},
    }


def cache(official, split="train", count=2, audit=False, output_name="labels"):
    dataset, manifest, root = official
    with patch(
        "stifflow.stiffness.estimate_stiffness", side_effect=synthetic_stiffness
    ), patch("stifflow.official_labels.provenance", return_value={"synthetic": True}):
        return build_cache(
            dataset,
            manifest,
            root / output_name,
            split,
            count,
            {"sigma0": 1.0, "rho": 1.0},
            audit=audit,
        )


def test_all_official_rows_keep_atom_order_and_original_target(official):
    dataset, _, root = official
    for entry in dataset.groups("train"):
        molecule = dataset.load_molecule(entry, "train")
        assert [c["row_id"] for c in molecule["conformers"]] == entry["row_ids"]
        for conf in molecule["conformers"]:
            begin, end = dataset.ranges[conf["row_id"]]
            np.testing.assert_array_equal(
                conf["positions"], dataset.atomic[begin:end, 2:]
            )
            assert conf["positions"].dtype == np.float64
    assert dataset.validate_splits()["train"]["rows"] == 5
    original = fingerprint(root / "dataset/rdkit_folder/molecule-0.pickle")
    dataset.load_molecule(dataset.groups("train")[0], "train")
    assert fingerprint(original["path"])["sha256"] == original["sha256"]


@pytest.mark.parametrize("change_type", [False, True])
def test_bond_order_is_ignored_but_bond_type_is_checked(official, change_type):
    dataset, _, root = official
    path = root / "dataset/rdkit_folder/molecule-0.pickle"
    with path.open("rb") as stream:
        raw = pickle.load(stream)
    mol = raw["conformers"][1]["rd_mol"]
    changed = Chem.RWMol(mol)
    bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType())
        for b in mol.GetBonds()
    ]
    for i, j, _ in bonds:
        changed.RemoveBond(i, j)
    for i, j, kind in reversed(bonds):
        changed.AddBond(j, i, kind)
    if change_type:
        changed.GetBondWithIdx(0).SetBondType(Chem.BondType.DOUBLE)
    raw["conformers"][1]["rd_mol"] = changed.GetMol()
    with path.open("wb") as stream:
        pickle.dump(raw, stream)
    entry = dataset.groups("train")[0]
    if change_type:
        with pytest.raises(ValueError, match="topology"):
            dataset.load_molecule(entry, "train")
    else:
        assert len(dataset.load_molecule(entry, "train")["conformers"]) == 2


def test_stereo_variation_requires_explicit_reference_mode(official):
    from stifflow.data import graph_from_mol

    dataset, _, root = official
    with (root / "dataset/rdkit_folder/molecule-0.pickle").open("rb") as stream:
        mol = pickle.load(stream)["conformers"][0]["rd_mol"]
    first = graph_from_mol(mol)
    changed = copy.deepcopy(first)
    changed["cip_codes"][
        0
    ] = "R"  # Exercise the metadata policy independently of 3D perception.
    entry = dataset.groups("train")[0]
    with patch(
        "stifflow.official_data.graph_from_mol", side_effect=[first, changed]
    ), pytest.raises(ValueError, match="Stereochemistry"):
        dataset.load_molecule(entry, "train")
    with patch("stifflow.official_data.graph_from_mol", side_effect=[first, changed]):
        loaded = dataset.load_molecule(entry, "train", allow_stereo_variation=True)
    assert loaded["stereo_variation_rows"] == [entry["row_ids"][1]]
    left, right = [
        graph_from_mol(Chem.MolFromSmiles(s))
        for s in ("F[C@](Cl)(Br)I", "F[C@@](Cl)(Br)I")
    ]
    assert graph_signatures(left)[0] == graph_signatures(right)[0]
    assert graph_signatures(left)[1] != graph_signatures(right)[1]


def test_topology_variation_requires_explicit_reference_mode(official):
    from stifflow.data import graph_from_mol

    dataset, _, root = official
    with (root / "dataset/rdkit_folder/molecule-0.pickle").open("rb") as stream:
        mol = pickle.load(stream)["conformers"][0]["rd_mol"]
    first = graph_from_mol(mol)
    changed = copy.deepcopy(first)
    changed["bond_types"][0] = (
        "DOUBLE" if first["bond_types"][0] != "DOUBLE" else "SINGLE"
    )  # Indexed topology differs across rows (labels stay strict, references opt in).
    entry = dataset.groups("train")[0]
    with patch(
        "stifflow.official_data.graph_from_mol", side_effect=[first, changed]
    ), pytest.raises(ValueError, match="topology"):
        dataset.load_molecule(entry, "train")
    with patch("stifflow.official_data.graph_from_mol", side_effect=[first, changed]):
        loaded = dataset.load_molecule(entry, "train", allow_topology_variation=True)
    assert loaded["topology_variation_rows"] == [entry["row_ids"][1]]
    assert loaded["stereo_variation_rows"] == []


def test_molecule_leakage_is_rejected_even_when_rows_do_not_overlap(official):
    dataset, _, _ = official
    dataset.keys = dataset.keys.astype(object)
    dataset.keys[2] = "C(C)"  # Same chemical molecule as CC, different spelling/rows.
    with pytest.raises(ValueError, match="molecule leakage"):
        dataset.validate_splits()


def test_duplicate_or_wrong_atom_spans_fail(official):
    dataset, _, root = official
    split = (
        root / "cache/etflow-official/splits/expanded/scaffold_qm9_train_indices.npy"
    )
    np.save(split, np.asarray([0, 0], dtype=np.int64))
    dataset.specification["splits"]["train"] = fingerprint(split)
    with pytest.raises(ValueError, match="Duplicate"):
        dataset.groups("train")
    ranges = root / "cache/etflow-official/geom-preprocessed/pos_index_range.memmap"
    changed = np.memmap(ranges, dtype=np.int32, mode="r+", shape=dataset.ranges.shape)
    changed[0, 0] = 1
    changed.flush()
    with pytest.raises(ValueError, match="offsets"):
        dataset.load_molecule(
            {"id": "qm9:CC", "inventory_index": 0, "row_ids": [0]}, "train"
        )


def test_complete_cache_pairs_rows_for_all_three_methods_and_protects_output(official):
    report = cache(official)
    validate_label_manifest(report)
    path = official[2] / "labels/manifest.json"
    for method in ("precision", "alternative_a", "control"):
        records = load_records(path, "train", 2, method, report["source_config"])
        assert sum(len(r["positions"]) for r in records) == 5
        for record, accepted in zip(records, report["records"]):
            assert record["row_ids"] == accepted["row_ids"]
            assert len(record["labels"]) == len(accepted["row_ids"])
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        cache(official)
    assert path.read_bytes() == before


def test_partial_cache_cannot_be_disguised_as_complete(official):
    report = cache(official)
    modified = copy.deepcopy(report)
    item = modified["records"][0]
    item["row_ids"] = item["row_ids"][:1]
    item["conformer_ids"] = item["conformer_ids"][:1]
    item["retained_conformers"] = 1
    with pytest.raises(ValueError, match="unpaired"):
        validate_label_manifest(modified)
    audit = cache(official, audit=True, output_name="audit")
    with pytest.raises(ValueError, match="complete official"):
        validate_label_manifest(audit)


def test_cached_row_reordering_fails_even_with_a_new_file_hash(official):
    report = cache(official)
    directory = official[2] / "labels"
    record = report["records"][0]
    path = directory / record["npz"]
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    arrays["row_ids"] = arrays["row_ids"][::-1]
    np.savez_compressed(path, **arrays)
    record["cache"] = fingerprint(path)
    (directory / "manifest.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="row/label pairing"):
        load_records(
            directory / "manifest.json",
            "train",
            2,
            "precision",
            report["source_config"],
        )


def test_checkpoint_comparison_detects_a_changed_optimizer_or_rng():
    import torch

    from stifflow.verify_resume import compare_states

    state = {"optimizer": [torch.tensor([0.2, 0.3])], "rng": {"state": 123}}
    compare_states(state, copy.deepcopy(state))
    changed = copy.deepcopy(state)
    changed["rng"]["state"] = 124
    with pytest.raises(ValueError, match="rng.state"):
        compare_states(state, changed)
    changed = copy.deepcopy(state)
    changed["optimizer"][0][0] = 0.21
    with pytest.raises(AssertionError):
        compare_states(state, changed)


def test_one_failed_row_rejects_whole_molecule_and_uses_next_candidate(official):
    from stifflow.stiffness import build_label_arrays

    dataset, manifest, root = official
    candidates = dataset.candidates("train")
    failed_row = candidates[0]["row_ids"][-1]

    def reject_row(conformers, source):
        if failed_row in [c["row_id"] for c in conformers]:
            raise ValueError(f"failed row {failed_row}")
        return build_label_arrays(conformers, source)

    with patch(
        "stifflow.official_labels.build_label_arrays", side_effect=reject_row
    ), patch(
        "stifflow.stiffness.estimate_stiffness", side_effect=synthetic_stiffness
    ), patch(
        "stifflow.official_labels.provenance", return_value={}
    ):
        report = build_cache(
            dataset,
            manifest,
            root / "replacements",
            "train",
            1,
            {"sigma0": 1.0, "rho": 1.0},
        )
    assert report["records"][0]["id"] == candidates[1]["id"]
    assert report["rejected_molecules"] == 1
    assert len(list((root / "replacements").glob("*.npz"))) == 1
    assert report["attempts"][0]["status"] == "rejected"
    validate_label_manifest(report)


def test_insufficient_labels_are_not_training_ready(official):
    report = cache(official, count=3)
    assert report["status"] == "insufficient_labels"
    with pytest.raises(ValueError, match="complete official"):
        validate_label_manifest(report)


def test_validation_references_have_no_hessian_and_pair_is_isolated(official):
    train = cache(official)
    val = cache(official, split="val", count=1, output_name="validation")
    validate_label_pair(train, val)
    refs = official[2] / "validation/references"
    reference = json.loads((refs / "manifest.json").read_text())
    for item in reference["records"]:
        with np.load(refs / item["reference_npz"], allow_pickle=False) as arrays:
            assert set(arrays.files) == {"pos_ref", "atomic_numbers"}
            assert len(arrays["pos_ref"]) == item["reference_count"] == 2
    modified = copy.deepcopy(val)
    modified["selection_seed"] = 7
    with pytest.raises(ValueError, match="isolation differs"):
        validate_label_pair(train, modified)


def test_row_uniform_sampling_and_actual_loader_resume(official):
    draws = [draw_record([1, 4], 42, i, "uniform_conformer_rows") for i in range(3000)]
    assert abs(np.mean([record == 0 for record, _ in draws]) - 0.2) < 0.04
    report = cache(official)
    records = load_records(
        official[2] / "labels/manifest.json",
        "train",
        2,
        "control",
        report["source_config"],
    )
    full = list(make_loader(records, 2, 4, 42, sampling="uniform_conformer_rows"))
    resumed = list(
        make_loader(records, 2, 2, 42, start_step=2, sampling="uniform_conformer_rows")
    )
    for (_, expected), (_, actual) in zip(full[2:], resumed):
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left["x1"], right["x1"])
