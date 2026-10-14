"""Small CPU tests of speed-cache contracts and metrics; no GPU or real Hessians."""

import csv
import pickle
from unittest.mock import patch

import numpy as np
import pytest
from rdkit import Chem

from stifflow.cache_io import atomic_json, read_json
from stifflow.packed_data import PackedRecords
from stifflow.precision import prepare_label
from stifflow.preflight import fingerprint
from stifflow.speed_benchmark import case, timing_summary
from stifflow.speed_data import prepare_speed_cache
from stifflow.training import load_records, make_loader


def test_real_row_contract_and_packed_speed_cache_cannot_enter_training(tmp_path):
    from stifflow_alt_a.path import prepare_label as alt_label

    raw = tmp_path / "raw"
    raw.mkdir()
    entries = []
    for text in ("C", "CC", "CCC", "CCO", "CCCC"):
        mol = Chem.AddHs(Chem.MolFromSmiles(text))
        xyz = np.arange(mol.GetNumAtoms() * 3).reshape(-1, 3) * 0.03
        conf = Chem.Conformer(mol.GetNumAtoms())
        for atom, pos in enumerate(xyz):
            conf.SetAtomPosition(atom, pos)
        mol.AddConformer(conf)
        with (raw / (text + ".pickle")).open("wb") as stream:
            pickle.dump({"conformers": [{"rd_mol": mol}]}, stream)
        entries.append(
            {"file": text + ".pickle", "molecule_id": text, "split": "train"}
        )
    test_file = tmp_path / "test.csv"
    with test_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["smiles"])
        writer.writeheader()
        writer.writerow({"smiles": "N"})
    split = tmp_path / "split.json"
    atomic_json(
        split,
        {
            "raw_dir": str(raw),
            "entries": entries,
            "official_test": {**fingerprint(test_file), "isolated": True},
        },
    )

    def labels(conformers, source):
        row = conformers[0]
        assert row["conformer_id"]
        np.testing.assert_array_equal(
            row["positions"], row["rd_mol"].GetConformer().GetPositions()
        )
        dimension = source.S.shape[0]
        K = np.eye(dimension) * 2
        primary, alternative = prepare_label(source, K, 0.592485), alt_label(
            source, K, 0.592485
        )
        return (
            {
                "positions": row["positions"][None],
                "precision_values": primary["values"][None],
                "precision_vectors": primary["vectors"][None],
                "alt_L_K": alternative["L_K"][None],
            },
            [],
            [],
        )

    source_config = {"sigma0": 1.0, "rho": 1.0}
    output = tmp_path / "speed"
    with patch("stifflow.speed_data.build_label_arrays", side_effect=labels):
        manifest = prepare_speed_cache(split, output, source_config, count=2)
    assert manifest["speed_only"] and manifest["audit_partial"]
    assert manifest["stiffness_preprocessing_seconds"] > 0
    assert manifest["selection_filters"] == [
        {"id": "C", "reason": "outside 8..32 atoms", "atoms": 5}
    ]
    for method in ("precision", "alternative_a", "control"):
        records = PackedRecords(manifest, method)
        payload = next(
            iter(
                make_loader(
                    records,
                    2,
                    1,
                    42,
                    sampling="uniform_conformer_rows",
                    num_workers=0,
                    prefetch_factor=4,
                    pin_memory=False,
                )
            )
        )
        assert len(payload[1]) == 2
        assert [r["draw_id"] for r in payload[1]] == [0, 1]
        assert all(r["x1"].dtype == np.float64 for r in payload[1])
        records.close()
    assert all(
        item["status"] == "accepted" for item in read_json(output / "attempts.json")
    )
    with pytest.raises(ValueError, match="Speed-only"):
        load_records(
            output / "manifest.json", "train", "all", "precision", source_config
        )


def test_case_and_timing_metrics_use_optimizer_updates_and_samples():
    assert case("precision", 32, 128)["accumulation"] == 4
    assert case("precision", 64, 512)["accumulation"] == 8
    with pytest.raises(ValueError):
        case("precision", 12, 128)
    stats = timing_summary(
        [
            {"seconds": 1.0, "samples": 128, "atoms": 2560},
            {"seconds": 3.0, "samples": 128, "atoms": 2560},
        ]
    )
    assert stats["measured_updates"] == 2
    assert stats["seconds_per_update"] == 2
    assert stats["samples_per_second"] == 64
    assert stats["atoms_per_second"] == 1280
    assert timing_summary([]) == {"measured_updates": 0}
