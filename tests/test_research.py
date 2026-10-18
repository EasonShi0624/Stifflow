"""Synthetic stereochemical and statistical checks; no formal results are used."""

import copy
import json

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from stifflow_benchmark.research_compare import label_identity, paired
from stifflow_benchmark.research_diagnostics import (
    audit_ensemble,
    audit_mapping,
    clipping_log,
    select_records,
    stereo_audit,
)


def test_mirror_audit_distinguishes_geometry_from_stereo():
    mol = Chem.AddHs(Chem.MolFromSmiles("F[C@](Cl)(Br)I"))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    pos = mol.GetConformer().GetPositions()
    mirror = pos * [-1, 1, 1]
    audit = audit_ensemble(mol, pos[None], np.stack([pos, mirror]))
    assert audit["generated_stereo"]["counts"] == {"correct": 1, "all_inverted": 1}
    assert audit["reference_stereo"]["fraction"] == 1
    (pair,) = audit["wrong_stereo_mirror_pairs"]
    assert pair["mirror_rmsd"] < 1e-6 and pair["original_rmsd"] > 0.1
    assert audit["generated_stereo"]["fraction"] == 0.5
    planar = pos.copy()
    planar[:, 2] = 0
    assert stereo_audit(mol, [planar])["patterns"] == ["missing_or_non_RS"]


def test_multiple_centers_are_not_called_a_global_mirror():
    mol = Chem.AddHs(Chem.MolFromSmiles("F[C@H](Cl)[C@H](Br)I"))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    pos = mol.GetConformer().GetPositions()
    expected = stereo_audit(mol, [pos])
    assert len(expected["expected"]) == 2
    # Exchange positions of the two terminal neighbors at one center only.
    perturbed = pos.copy()
    center = next(iter(expected["expected"]))
    terminals = [
        a.GetIdx()
        for a in mol.GetAtomWithIdx(center).GetNeighbors()
        if a.GetDegree() == 1
    ]
    perturbed[terminals[:2]] = perturbed[terminals[:2][::-1]]
    assert stereo_audit(mol, [perturbed])["patterns"] == ["partially_inverted"]


def test_selection_and_log_denominator(tmp_path):
    records = [{"id": str(i), "status": "ok"} for i in range(100)]
    assert select_records(records, 64) == select_records(records[::-1], 64)
    with pytest.raises(ValueError):
        select_records(records + records[:1], 64)
    path = tmp_path / "log.txt"
    path.write_text(
        "INFO Clipped gradient with value 200.0 while allowed 100.0\nnoise\nINFO Clipped gradient with value 11.0 while allowed 10.0\n"
    )
    assert clipping_log(path)["logged_event_fraction"] is None
    assert clipping_log(path, 20)["logged_event_fraction"] == 0.1
    with pytest.raises(ValueError):
        clipping_log(path, 1)


def reports():
    training = {
        "train": "train",
        "val": "val",
        "official_data": "official",
        "sampling": "uniform_conformer_rows",
        "dataset_seed": 42,
        "training_seed": 42,
        "row_rng_domain": 2,
        "effective_batch_size": 128,
        "profile": "formal",
        "config": {
            "source": {"rho": 1.0},
            "model_args": {"first_cycle_steps": 250000},
            "batch_size": 32,
            "profiles": {"formal": {"accumulate_grad_batches": 4}},
        },
    }
    base = {
        "run": {
            "split": "val",
            "method": "control",
            "baseline_kind": "controlled",
            "training_budget": {"steps": 25000},
            "config": {"model": {"method": "control", "seed": 42}, "steps": 50},
            "training_data": training,
            "seed": 42,
            "input_manifest": {"sha256": "refs"},
        },
        "protocol": {"thresholds": [0.5]},
        "records": [
            {
                "id": str(i),
                "status": "ok",
                "mat_p": 0.2,
                "mat_r": 0.1,
                "cov_r": [0.9],
                "chirality": {"assessed": 2, "fraction": 1.0},
            }
            for i in range(10)
        ],
    }
    candidate = copy.deepcopy(base)
    candidate["run"]["method"] = "linear"
    candidate["run"]["config"]["model"]["method"] = "linear"
    candidate["run"]["training_data"]["profile"] = "research"
    candidate["run"]["training_data"]["config"]["profiles"] = {
        "research": {"accumulate_grad_batches": 4}
    }
    for r in candidate["records"]:
        r["mat_p"] = 0.18
    return candidate, base


def test_paired_gate_and_mismatch_rejection():
    candidate, base = reports()
    result = paired(candidate, base, bootstrap=100)
    assert result["advance_to_second_seed"]
    assert result["relative_mat_p_change"] == pytest.approx(-0.1)
    candidate["records"].reverse()
    assert paired(candidate, base, bootstrap=100) == result
    for key, value in (
        ("split", "test"),
        ("seed", 43),
        ("training_budget", {"steps": 250000}),
    ):
        changed = copy.deepcopy(candidate)
        changed["run"][key] = value
        with pytest.raises(ValueError):
            paired(changed, base, bootstrap=100)
    changed = copy.deepcopy(candidate)
    changed["run"]["training_data"]["config"]["model_args"]["first_cycle_steps"] = 25000
    with pytest.raises(ValueError):
        paired(changed, base, bootstrap=100)
    candidate["records"][0]["status"] = "failed"
    with pytest.raises(ValueError):
        paired(candidate, base, bootstrap=100)


def test_missing_stereochemical_evidence_fails_gate():
    candidate, base = reports()
    for report in (candidate, base):
        for row in report["records"]:
            row["chirality"] = {"assessed": 0, "fraction": None}
    assert not paired(candidate, base, bootstrap=100)["advance_to_second_seed"]


def test_label_equivalence_keeps_cohort_and_matrix_hashes(tmp_path):
    from stifflow.preflight import fingerprint

    manifest = {
        "schema_version": 3,
        "split": "train",
        "status": "completed",
        "all_candidates_processed": True,
        "dataset_kind": "official_memmap_scaffold",
        "sampling": "uniform_conformer_rows",
        "source_config": {},
        "stiffness_config": {},
        "label_implementation": {"code": "original"},
        "data_manifest": {"sha256": "frozen", "path": "/canonical/data.json"},
        "records": [{"id": "molecule", "row_ids": [1, 2]}],
        "graph_shards": [],
        "factor_shards": [
            {
                "files": {
                    "positions": {
                        "path": "/canonical/x.npy",
                        "sha256": "positions",
                        "bytes": 12,
                    }
                }
            }
        ],
        "row_index": {"sha256": "rows", "path": "/canonical/rows.npy"},
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(manifest))
    identity = label_identity(first, fingerprint(first)["sha256"], "train")
    manifest["factor_shards"][0]["files"]["positions"]["path"] = "/staged/x.npy"
    second.write_text(json.dumps(manifest))
    assert label_identity(second, fingerprint(second)["sha256"], "train") == identity
    manifest["records"][0]["row_ids"] = [1, 3]
    second.write_text(json.dumps(manifest))
    assert label_identity(second, fingerprint(second)["sha256"], "train") != identity
    with pytest.raises(ValueError, match="hash"):
        label_identity(first, "wrong", "train")
    manifest["records"][0]["row_ids"] = [1, 2]
    manifest["factor_shards"][0]["files"]["positions"]["sha256"] = "changed"
    second.write_text(json.dumps(manifest))
    assert label_identity(second, fingerprint(second)["sha256"], "train") != identity


def test_disconnected_protocol_is_explicit(tmp_path):
    from stifflow_benchmark.evaluate import conformer_mol, evaluate_record

    mol = Chem.MolFromSmiles("C.C")
    pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    np.savez(
        tmp_path / "samples.npz",
        pos_ref=pos[None],
        pos_gen=np.stack([pos, pos]),
        atomic_numbers=[6, 6],
    )
    record = {
        "id": "multi",
        "status": "ok",
        "molblock": Chem.MolToMolBlock(conformer_mol(mol, pos)),
        "npz": "samples.npz",
    }
    assert evaluate_record(record, tmp_path, np.array([0.5]))["status"] == "failed"
    result = evaluate_record(record, tmp_path, np.array([0.5]), allow_disconnected=True)
    assert result["status"] == "ok" and result["multi_fragment"]


def test_mapping_audit_flags_difference_without_repair():
    from stifflow.data import graph_from_mol

    mol = Chem.MolFromSmiles("CC")
    original = np.array([[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]])
    canonical = original[:, ::-1].copy()

    class Dataset:
        def load_molecule(self, *args, **kwargs):
            return {
                "graph": graph_from_mol(mol),
                "conformers": [{"positions": canonical[0]}],
                "topology_variation_rows": [7],
                "stereo_variation_rows": [],
                "reference_atom_mappings": [{"row_id": 7, "canonical_to_raw": [1, 0]}],
                "source": {"sha256": "raw"},
            }

    result = audit_mapping(
        {"molblock": Chem.MolToMolBlock(mol)},
        original,
        Dataset(),
        {"row_ids": [7]},
        "val",
    )
    assert result["different_rows"] == [7]
    assert not result["matches_current_canonical_coordinates"]
    assert result["conditioning_topology_matches"]
    changed = audit_mapping(
        {"molblock": Chem.MolToMolBlock(Chem.MolFromSmiles("C.C"))},
        original,
        Dataset(),
        {"row_ids": [7]},
        "val",
    )
    assert not changed["conditioning_topology_matches"]
    np.testing.assert_array_equal(original, [[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]])
