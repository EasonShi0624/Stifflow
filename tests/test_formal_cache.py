import copy
import json
import pickle
from itertools import islice
from unittest.mock import patch

import numpy as np
import pytest
from rdkit import Chem
from test_official_data import (
    cache,
    synthetic_stiffness,
)
from test_official_data import (
    official as official,  # noqa: PLC0414 -- explicit pytest fixture re-export
)

from stifflow.cache_io import atomic_json
from stifflow.formal_cache import build_shard, merge_shards
from stifflow.freeze_data import VARIANT, freeze
from stifflow.official_data import (
    OfficialDataset,
    validate_label_manifest,
    validate_label_pair,
)
from stifflow.packed_data import PackedRecords
from stifflow.preflight import fingerprint
from stifflow.training import advance_cursor, load_records, make_loader

SOURCE = {"sigma0": 1.0, "rho": 1.0}


@pytest.fixture
def frozen(official):
    dataset, _manifest, root = official
    receipt = root / "synthetic-mapping.jsonl"
    receipt.write_text("synthetic unit fixture, not a real mapping audit\n")
    spec = copy.deepcopy(dataset.specification)
    spec["variant"] = {
        "name": VARIANT,
        "mapping_matches_existing": True,
        "mapping_receipt": fingerprint(receipt),
        "counts": dataset.validate_splits(),
    }
    new = root / "frozen.json"
    atomic_json(new, spec)
    return OfficialDataset.from_manifest(new), new, root


def build(frozen, split="train", reject_first=False, reuse=()):
    from stifflow.stiffness import build_label_arrays

    dataset, manifest, root = frozen
    first = dataset.candidates(split)[0]["row_ids"][0]

    def labels(conformers, source):
        if reject_first and first in [c["row_id"] for c in conformers]:
            raise ValueError("deliberate physical-label rejection")
        return build_label_arrays(conformers, source)

    with patch(
        "stifflow.stiffness.estimate_stiffness", side_effect=synthetic_stiffness
    ), patch("stifflow.formal_cache.provenance", return_value={}), patch(
        "stifflow.official_labels.build_label_arrays", side_effect=labels
    ):
        paths = []
        for i in range(len(dataset.candidates(split))):
            output = root / f"{split}-shard-{i}"
            report = build_shard(
                dataset,
                manifest,
                output,
                split,
                SOURCE,
                shard_index=i,
                shard_size=1,
                reuse=reuse,
            )
            assert report["status"] == "completed"
            paths.append(output / "shard.json")
    return paths


def pack(frozen, paths, split="train"):
    dataset, manifest, root = frozen
    with patch("stifflow.formal_cache.provenance", return_value={}):
        return merge_shards(
            dataset,
            manifest,
            paths,
            root / f"{split}-packed",
            split,
            SOURCE,
            rows_per_shard=2,
        )


def test_all_candidates_complete_with_whole_molecule_rejection(frozen):
    report = pack(frozen, build(frozen, reject_first=True))
    assert report["candidate_molecules"] == report["attempted_molecules"] == 2
    assert report["requested_molecules"] == report["rejected_molecules"] == 1
    validate_label_manifest(report)
    missing = copy.deepcopy(report)
    missing["attempted_molecules"] = 1
    with pytest.raises(ValueError, match="every frozen candidate"):
        validate_label_manifest(missing)


def test_pack_reads_paired_rows_and_bounds_memory_for_all_methods(frozen):
    report = pack(frozen, build(frozen))
    validate_label_manifest(report)
    for method in ("precision", "alternative_a", "control"):
        store = PackedRecords(report, method, max_open_parts=1, graph_cache_size=1)
        snapshots = []
        for i, record in enumerate(report["records"]):
            for j, row in enumerate(record["row_ids"]):
                graph, sample = store.sample(i, j)
                begin, end = frozen[0].ranges[row]
                np.testing.assert_array_equal(
                    sample["x1"], frozen[0].atomic[begin:end, 2:]
                )
                assert graph.num_nodes == len(sample["x1"])
                assert sample["source"].S.dtype == np.float64
                snapshots.append(sample)
                assert len(store.parts) <= 1 and len(store.graphs) <= 1
        store.close()
        # Previously returned rows must stay valid after mmap eviction/close.
        assert all(np.isfinite(s["x1"]).all() for s in snapshots)
        assert all(s["label"] is None for s in snapshots) == (method == "control")


def test_worker_and_packing_resume_protects_completed_results(frozen):
    paths = build(frozen)
    before = [p.read_bytes() for p in paths]
    dataset, manifest, root = frozen
    with patch(
        "stifflow.formal_cache.label_record",
        side_effect=AssertionError("must not recompute"),
    ), patch("stifflow.formal_cache.provenance", return_value={}):
        for i, path in enumerate(paths):
            build_shard(
                dataset,
                manifest,
                path.parent,
                "train",
                SOURCE,
                shard_index=i,
                shard_size=1,
                resume=True,
            )
        first = merge_shards(
            dataset,
            manifest,
            paths,
            root / "train-packed",
            "train",
            SOURCE,
            rows_per_shard=2,
        )
        hashes = [
            f["sha256"] for p in first["factor_shards"] for f in p["files"].values()
        ]
        second = merge_shards(
            dataset,
            manifest,
            paths,
            root / "train-packed",
            "train",
            SOURCE,
            rows_per_shard=2,
            resume=True,
        )
        assert hashes == [
            f["sha256"] for p in second["factor_shards"] for f in p["files"].values()
        ]
    assert before == [p.read_bytes() for p in paths]
    with pytest.raises(FileExistsError):
        build_shard(dataset, manifest, paths[0].parent, "train", SOURCE, shard_size=1)
    with pytest.raises(ValueError, match="Missing"):
        pack(frozen, paths[:1])


def test_v2_pilot_reuse_does_not_repeat_hessian_work(frozen, official):
    cache(official)
    old = official[2] / "labels/manifest.json"
    with patch(
        "stifflow.formal_cache.label_record",
        side_effect=AssertionError("must reuse"),
    ):
        paths = build(frozen, reuse=[old])
    report = pack(frozen, paths)
    for record in report["records"]:
        with open(record["audit_receipt"]["path"]) as stream:
            receipt = json.load(stream)
        assert receipt["attempt"]["reused"]


def test_seed_and_accumulation_cursor_preserve_draws(frozen):
    pack(frozen, build(frozen))
    path = frozen[2] / "train-packed/manifest.json"
    records = load_records(path, "train", "all", "control", SOURCE)

    def loader(seed, start=0):
        return make_loader(
            records,
            2,
            4,
            seed,
            sampling="uniform_conformer_rows",
            accumulate_grad_batches=2,
            consumed_samples=start,
        )

    full = list(loader(43))
    resumed = list(islice(loader(43, 8), 4))
    for (_, left), (_, right) in zip(full[4:], resumed):
        assert [r["draw_id"] for r in left] == [r["draw_id"] for r in right]
        for a, b in zip(left, right):
            np.testing.assert_array_equal(a["x1"], b["x1"])
    assert advance_cursor(8, resumed[0][1]) == 10
    with pytest.raises(RuntimeError, match="repeated/skipped"):
        advance_cursor(0, resumed[0][1])
    assert records.manifest["selection_seed"] == 42
    records.close()


def test_train_val_pair_and_modified_packed_row_fail(frozen):
    train = pack(frozen, build(frozen))
    val = pack(frozen, build(frozen, split="val"), split="val")
    validate_label_pair(train, val)
    path = train["row_index"]["path"]
    index = np.load(path)
    index[0, 2] = -1
    np.save(path, index)
    train["row_index"] = fingerprint(path)
    with pytest.raises(ValueError, match="pairing"):
        PackedRecords(train, "precision")


@pytest.mark.parametrize(
    "extra_split,row_mismatch,passes",
    [
        (None, False, True),
        ("train", False, True),
        ("test", False, False),
        ("train", True, False),
    ],
)
def test_freeze_preserves_existing_rows_and_records_exclusions(
    official, extra_split, row_mismatch, passes
):
    from stifflow.identity import SplitMapper

    dataset, manifest, root = official
    base = root / "scaffold"
    base.mkdir()
    for split, keys in {
        "train": ["CC", "CC", "CO"],
        "val": ["CN", "CC"],
        "test": ["CF"],
    }.items():
        if split == extra_split:
            keys.append("unit:ambiguous")
        with (base / f"scaffold_{split}_smiles.pkl").open("wb") as file:
            pickle.dump(keys, file)
    uniques, inverse = [], []
    for i, count in enumerate(dataset.conformers):
        mol = Chem.AddHs(Chem.MolFromSmiles(str(dataset.keys[i])))
        for j, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(j)
        uniques.append(Chem.MolToSmiles(mol, canonical=False))
        inverse.extend([i] * count)
    smiles = root / "smiles.npz"
    np.savez(smiles, uniques=uniques, inv_indices=inverse)
    original_resolve = SplitMapper.resolve

    def resolve(mapper, text):
        return (
            (None, "ambiguous_group")
            if text == "unit:ambiguous"
            else original_resolve(mapper, text)
        )

    if row_mismatch:
        changed = root / "different-rows.npy"
        np.save(changed, np.asarray([1, 2, 3, 4], dtype=np.int64))
        dataset.specification["splits"]["train"] = fingerprint(changed)
    with patch("stifflow.freeze_data.provenance", return_value={}), patch.object(
        SplitMapper, "resolve", resolve
    ):
        if not passes:
            with pytest.raises(ValueError, match="Strict mapping"):
                freeze(dataset, manifest, base, smiles, root / "strict")
            assert not (root / "strict/manifest.json").exists()
            assert not json.loads((root / "strict/audit.json").read_text())["passed"]
            return
        result = freeze(dataset, manifest, base, smiles, root / "strict")
    assert result["passed"]
    expected_exclusions = {"duplicate": 1}
    if extra_split == "train":
        expected_exclusions["ambiguous_group"] = 1
    assert result["checks"]["train"]["exclusions"] == expected_exclusions
    assert result["checks"]["val"]["exclusions"] == {"prior_split_overlap": 1}
    frozen_spec = json.loads((root / "strict/manifest.json").read_text())
    assert frozen_spec["variant"]["name"] == VARIANT
