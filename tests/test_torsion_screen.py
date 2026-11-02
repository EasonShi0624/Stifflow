"""Small synthetic panel checks; no GEOM analysis or model execution."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation
from stifflow.geometry import centered_basis, rigid_subspaces
from stifflow.preflight import fingerprint
from stifflow_benchmark import torsion_screen as ts


def molecule(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol


def record(mol, identity="molecule", count=3):
    return {
        "id": identity,
        "molblock": Chem.MolToMolBlock(mol),
        "reference_count": count,
    }


@pytest.mark.parametrize(
    "smiles,group,strict,ntorsions",
    [
        ("CCCC", "one", 1, 1),
        ("CCCOCC", "multiple", 3, 3),
        ("CC(=O)NCC", "conjugated", 1, 2),
        ("COC(=O)CC", "conjugated", 1, 2),
        ("C#CC#C", "zero", 0, 0),
    ],
)
def test_heavy_atom_rotor_definition_and_restricted_bonds(
    smiles, group, strict, ntorsions
):
    mol = molecule(smiles)
    actual = ts.describe(record(mol))
    assert (actual["group"], actual["strict_rotors"], len(actual["torsions"])) == (
        group,
        strict,
        ntorsions,
    )
    assert ts.describe(record(Chem.RemoveHs(mol)))["strict_rotors"] == strict
    for equivalent in actual["torsions"]:
        for quartet in equivalent:
            assert all(mol.GetAtomWithIdx(i).GetAtomicNum() > 1 for i in quartet)


def test_permutation_and_proper_rotation_preserve_torsions():
    mol = molecule("CCCOCC")
    point = mol.GetConformer().GetPositions()
    perturbed = ts.displaced(mol, point, ts.describe(record(mol))["torsions"][0][0], 40)
    before = ts.torsion_metrics(
        ts.torsion_angles(mol, [point]), ts.torsion_angles(mol, [perturbed])
    )
    order = list(reversed(range(mol.GetNumAtoms())))
    permuted = Chem.RenumberAtoms(mol, order)
    rotation = Rotation.from_rotvec([0.3, -0.4, 0.2]).as_matrix()
    transformed = [p[order] @ rotation + [2, 3, -1] for p in (point, perturbed)]
    after = ts.torsion_metrics(
        ts.torsion_angles(permuted, transformed[:1]),
        ts.torsion_angles(permuted, transformed[1:]),
    )
    assert_allclose(
        [before["mat_r"], before["mat_p"]], [after["mat_r"], after["mat_p"]], atol=1e-12
    )
    assert ts.describe(record(permuted))["group"] == ts.describe(record(mol))["group"]


def test_periodic_tfd_and_no_torsion_is_not_zero_error():
    assert ts.torsion_metrics([[([179.0], 180.0)]], [[([-179.0], 180.0)]])[
        "mat_p"
    ] == pytest.approx(2 / 180)
    assert ts.torsion_metrics([[]], [[]])["status"] == "not_applicable"


def test_selection_is_outcome_independent_and_does_not_fill_shortfalls():
    rows = [
        {
            "id": str(i),
            "status": "ok",
            "group": "one",
            "connected": True,
            "heavy_atoms": 8,
            "reference_count": 4,
        }
        for i in range(8)
    ]
    ids, counts = ts.choose_panel(rows)
    assert ts.choose_panel(rows[::-1]) == (ids, counts)
    assert len(ids) == 3 and counts["conjugated"]["shortfall"] == 3
    rows[0].update(heavy_atoms=6)
    assert "0" not in ts.choose_panel(rows)[0]


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "provenance", lambda: {"synthetic": True})
    monkeypatch.setattr(ts, "plots", lambda *args: None)
    # Integration fixtures check pairing/denominators; the shared helper tests bootstrap itself.
    original_bootstrap = ts.bootstrap_delta
    monkeypatch.setattr(
        ts, "bootstrap_delta", lambda v: original_bootstrap(v, replicates=100)
    )
    mapping = tmp_path / "mapping.jsonl"
    mapping.write_text("synthetic mapping\n")
    data_path = tmp_path / "data.json"
    data_path.write_text(
        json.dumps(
            {
                "variant": {
                    "name": ts.VARIANT,
                    "mapping_matches_existing": True,
                    "mapping_receipt": fingerprint(mapping),
                    "counts": {"val": {"molecules": 4}},
                }
            }
        )
    )
    ref_dir = tmp_path / "references"
    ref_dir.mkdir()
    records = []
    for i, smiles in enumerate(("N#Cc1ccncc1", "CCc1ccncc1", "CCCCOCC", "CC(=O)NCCC")):
        mol = molecule(smiles)
        r = record(mol, str(i))
        pos = mol.GetConformer().GetPositions()
        path = ref_dir / f"{i}.npz"
        np.savez(
            path,
            pos_ref=np.repeat(pos[None], 3, axis=0),
            atomic_numbers=[a.GetAtomicNum() for a in mol.GetAtoms()],
        )
        r.update(reference_npz=path.name, cache=fingerprint(path))
        records.append(r)
    ref_path = ref_dir / "manifest.json"
    ref_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split": "val",
                "selection_frozen": True,
                "variant": ts.VARIANT,
                "data_manifest": fingerprint(data_path),
                "records": records,
            }
        )
    )
    panel_dir = tmp_path / "panel"
    panel = ts.select(ref_path, panel_dir)
    pairs = []
    for method in ts.METHODS[:2]:
        for seed in ts.SEEDS:
            directory = tmp_path / f"{method}-{seed}"
            directory.mkdir()
            generated, evaluated = [], []
            for r in records:
                with np.load(ref_dir / r["reference_npz"]) as saved:
                    ref = saved["pos_ref"]
                    path = directory / (r["id"] + ".npz")
                    np.savez(
                        path,
                        pos_ref=ref,
                        pos_gen=np.repeat(ref[:1], 6, axis=0),
                        atomic_numbers=saved["atomic_numbers"],
                    )
                generated.append(
                    {
                        "id": r["id"],
                        "molblock": r["molblock"],
                        "status": "ok",
                        "npz": path.name,
                        "npz_fingerprint": fingerprint(path),
                    }
                )
                evaluated.append(
                    {
                        "id": r["id"],
                        "status": "ok",
                        "input": fingerprint(path),
                        "mat_p": 0.2 if method == "control" else 0.18,
                        "mat_r": 0.1,
                        "cov_r": [0.9],
                        "cov_p": [0.9],
                        "chirality": {"assessed": 0, "fraction": None},
                    }
                )
            manifest = {
                "schema_version": 1,
                "method": method,
                "baseline_kind": "controlled",
                "split": "val",
                "seed": 42,
                "input_manifest": fingerprint(ref_path),
                "training_budget": {"steps": 250000},
                "generation": {"expected": 4, "postprocessing": "none"},
                "config": {"steps": 50, "model": {"method": method, "seed": seed}},
                "training_data": {
                    "official_data": fingerprint(data_path)["sha256"],
                    "training_seed": seed,
                    "train": "train-hash",
                    "val": "val-hash",
                    "config": {"source": {"rho": 1.0}},
                },
                "records": generated,
            }
            manifest_path, evaluation_path = (
                directory / "manifest.json",
                directory / "evaluation.json",
            )
            manifest_path.write_text(json.dumps(manifest))
            report = {
                "input_manifest": fingerprint(manifest_path),
                "run": {k: v for k, v in manifest.items() if k != "records"},
                "protocol": {
                    "generated_reference_ratio": 2,
                    "force_field_optimization": False,
                    "heavy_atoms": True,
                    "thresholds": [0.5],
                    "coordinates": "angstrom",
                    "coverage_comparison": "strict <",
                },
                "records": evaluated,
            }
            evaluation_path.write_text(json.dumps(report))
            pairs.append((manifest_path, evaluation_path))
    return panel_dir / "panel.json", panel, records, pairs


def test_saved_only_campaign_roundtrip_and_seed_unit(campaign, tmp_path):
    panel_path, panel, _, pairs = campaign
    assert len(panel["ids"]) == 4
    result = ts.analyze(panel_path, pairs, tmp_path / "analysis")
    assert result["all_saved_runs_valid"] and result["energy_gradient_queries"] == 0
    assert not result["panel_complete"]  # one/group, not the requested three
    group = result["geometry"]["alternative_a-control/one/all"]["metrics"]["mat_p"]
    assert group["delta"]["molecules"] == 1  # three seeds are not three molecules
    assert group["delta"]["mean"] == pytest.approx(-0.02)
    assert len(group["per_seed_delta"]) == 3
    assert all(
        (panel_path.parent / (hashlib.sha256(r.encode()).hexdigest() + ".svg")).exists()
        for r in panel["ids"]
    )
    paired = result["paired_panel"]["alternative_a-control/one/tfd_mat_p"]
    assert paired["delta"]["molecules"] == 1
    assert paired["molecules"][0]["delta"] == pytest.approx(0)


def test_panel_tampering_and_test_split_are_rejected(campaign):
    path, panel, _, _ = campaign
    changed = copy.deepcopy(panel)
    changed["ids"] = panel["ids"][::-1]
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="selection was changed"):
        ts.load_panel(path)
    reference_path = Path(panel["references"]["path"])
    reference = json.loads(reference_path.read_text())
    reference["split"] = "test"
    reference_path.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="validation references"):
        ts.references(reference_path)


@pytest.mark.parametrize(
    "change", ["budget", "reference", "seed", "missing", "staged", "evaluation"]
)
def test_mismatched_runs_rejected(campaign, change):
    panel_path, _, _, pairs = campaign
    panel, records, _ = ts.load_panel(panel_path)
    manifest_path, evaluation_path = pairs[-1]
    manifest, report = ts.read_json(manifest_path), ts.read_json(evaluation_path)
    if change == "budget":
        manifest["training_budget"]["steps"] = 25000
    elif change == "reference":
        manifest["input_manifest"]["sha256"] = "wrong"
    elif change == "seed":
        manifest["config"]["model"]["seed"] = 123
    elif change == "missing":
        manifest["records"].pop()
    elif change == "staged":
        manifest["training_data"]["train"] = "unproven-staging"
    else:
        report["run"]["seed"] = 123
    if change != "evaluation":
        manifest_path.write_text(json.dumps(manifest))
        report["run"] = {k: v for k, v in manifest.items() if k != "records"}
        report["input_manifest"] = fingerprint(manifest_path)
    evaluation_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        ts.load_runs(pairs, panel, records)


def test_changed_coordinate_file_fails_without_changing_the_panel(campaign, tmp_path):
    path, panel, _, pairs = campaign
    manifest = ts.read_json(pairs[0][0])
    identity = panel["ids"][0]
    changed = next(r for r in manifest["records"] if r["id"] == identity)
    changed_path = pairs[0][0].parent / changed["npz"]
    with np.load(changed_path) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    arrays["pos_gen"][0, 0, 0] += 0.01
    np.savez(changed_path, **arrays)
    result = ts.analyze(path, pairs, tmp_path / "changed")
    assert result["selected_ids"] == panel["ids"]
    assert not result["all_saved_runs_valid"]
    item = next(r for r in result["records"] if r["id"] == identity)
    assert item["runs"]["alternative_a/42"]["status"] == "failed"


def test_failed_molecule_cannot_produce_favorable_geometry_subset(campaign):
    path, _, _, pairs = campaign
    panel, records, inventory = ts.load_panel(path)
    runs = ts.load_runs(pairs, panel, records)
    group = "multiple"
    identity = next(r["id"] for r in inventory if r["group"] == group)
    runs["control", 43]["metrics"][identity].update(
        status="failed", reason="synthetic failure"
    )
    result = ts.geometry_summary(runs, inventory)[f"alternative_a-control/{group}/all"]
    assert result["metrics"]["mat_p"]["delta"] is None
    assert result["metrics"]["mat_p"]["failures"][0]["id"] == identity


def test_force_failure_retained_and_original_positions_unchanged(monkeypatch):
    mol = molecule("CCCC")
    point = mol.GetConformer().GetPositions()
    before = point.copy()

    class Failing:
        def __init__(self, mol):
            self.n = 0

        def __call__(self, p):
            self.n += 1
            if self.n == 2:
                raise ValueError("synthetic failure")
            return 0.0, np.ones_like(p)

    monkeypatch.setattr(ts, "MMFF94", Failing)
    queries = []
    result = ts.force_audit(mol, [point], [point] * 3, queries.append, {"id": "butane"})
    assert result["status"] == "failed" and result["force_median"] is None
    assert len(queries) == 4 and sum(q["status"] == "failed" for q in queries) == 1
    assert_allclose(point, before, atol=0, rtol=0)


def test_local_taylor_uses_raw_signed_hessian_and_marks_missing_archive(monkeypatch):
    mol = molecule("CCCC")
    point = ts.centered(mol.GetConformer().GetPositions())
    quartet = ts.describe(record(mol))["torsions"][0][0]
    original = point.copy()
    displaced = ts.displaced(mol, point, quartet, 2)
    assert np.linalg.norm(displaced - point) > 0.01
    assert_allclose(point, original, atol=0, rtol=0)
    # An arbitrary quadratic, including negative curvature, exercises the shared Taylor kernel.
    Q = np.eye(point.size)
    H = np.diag(np.linspace(-1, 3, point.size))
    value = lambda p: (
        float(0.5 * p.ravel() @ H @ p.ravel()),
        (H @ p.ravel()).reshape(p.shape),
    )
    result = ts.taylor_metrics(displaced, point, H, Q, value(displaced), value(point))
    assert result["energy_relative_error"] < 1e-10
    assert result["force_relative_error"] < 1e-10
    with pytest.raises(KeyError):
        ts.local_checks(mol, point[None], {}, {}, lambda r: None)


def test_full_local_scan_calibration_floor_and_rotation(tmp_path, monkeypatch):
    mol = molecule("CCCC")
    origin = ts.centered(mol.GetConformer().GetPositions())
    Q = centered_basis(mol.GetNumAtoms())
    _, Qvib = rigid_subspaces(origin, Q)
    spectrum = np.linspace(0.1, 2, Qvib.shape[1])
    spectrum[7] = 0.5  # A retained mode exactly on the floor must survive roundoff.
    H = (Qvib * spectrum) @ Qvib.T
    arrays = {
        "positions": origin[None].copy(),
        "reference": origin[None].copy(),
        "H_raw": H[None].copy(),
        "spectrum": spectrum[None],
        "floored_mask": (spectrum < 0.5)[None],
        "row_ids": np.array([17]),
    }
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "record": {
                    "labels": [
                        {"settings": {"kappa_min": 0.5, "negative_tolerance": 1e-3}}
                    ]
                }
            }
        )
    )
    label = {"id": "butane", "audit_receipt": fingerprint(receipt)}
    monkeypatch.setattr(
        ts, "load_archive", lambda *args: (mol, SimpleNamespace(Q=Q), arrays, {})
    )

    class Quadratic:
        def __init__(self, mol):
            pass

        def __call__(self, point):
            delta = Q.T @ (point - origin).ravel()
            return float(0.5 * delta @ H @ delta), (Q @ H @ delta).reshape(-1, 3)

    monkeypatch.setattr(ts, "MMFF94", Quadratic)
    queries = []
    before = arrays["positions"].copy()
    result = ts.local_checks(mol, origin[None], label, {}, queries.append)
    assert result["status"] == "ok" and len(result["checks"]) == 1
    check = result["checks"][0]
    assert len(queries) == 9 and len(check["scan"]) == 7
    assert (
        check["regularized_curvature_kcal_mol_rad2"]
        >= check["raw_curvature_kcal_mol_rad2"]
    )
    assert 0 < check["floored_tangent_fraction"] < 1
    assert max(r["force_relative_error"] for r in check["scan"]) < 1e-10
    assert max(r["energy_relative_error"] for r in check["scan"]) < 1e-10
    assert_allclose(arrays["positions"], before, rtol=0, atol=0)
    rotation = Rotation.from_rotvec([0.3, -0.1, 0.5]).as_matrix()
    W = Q.T @ np.kron(np.eye(len(origin)), rotation.T) @ Q
    origin = origin @ rotation
    H = W @ H @ W.T
    arrays.update(positions=origin[None], reference=origin[None], H_raw=H[None])
    rotated = ts.local_checks(mol, origin[None], label, {}, lambda r: None)["checks"][0]
    for key in (
        "raw_curvature_kcal_mol_rad2",
        "regularized_curvature_kcal_mol_rad2",
        "floored_tangent_fraction",
    ):
        assert rotated[key] == pytest.approx(check[key], rel=1e-8, abs=1e-8)
    arrays["floored_mask"] = ~arrays["floored_mask"]
    with pytest.raises(ValueError, match="floored modes"):
        ts.local_checks(mol, origin[None], label, {}, lambda r: None)


def test_missing_labels_preserve_tfd_and_acceptance_is_explicit(
    campaign, tmp_path, monkeypatch
):
    path, _, _, pairs = campaign
    monkeypatch.setattr(ts, "validate_label_manifest", lambda spec: None)
    panel, _records, inventory = ts.load_panel(path)
    water = molecule("O")
    dummy = {"id": "not-on-panel", "molblock": Chem.MolToMolBlock(water)}
    label_path = tmp_path / "labels.json"
    label_path.write_text(
        json.dumps(
            {
                "split": "val",
                "data_manifest": {"sha256": panel["official_data"]},
                "source_config": {"rho": 1.0},
                "records": [dummy],
            }
        )
    )
    # Analytic force fixture avoids a real-data MMFF campaign on Torch.
    monkeypatch.setattr(
        ts, "force_audit", lambda *args: {"status": "ok", "force_median": 1.0}
    )
    result = ts.analyze(
        path,
        pairs,
        tmp_path / "missing-labels",
        val_labels=label_path,
        physics=True,
        limit=2,
    )
    assert result["acceptance_only"] and result["all_saved_runs_valid"]
    assert not result["physics_complete"] and not result["panel_complete"]
    assert len(result["selected_ids"]) == 2
    assert any(ts.indexed(inventory)[i]["torsions"] for i in result["selected_ids"])
    assert all(r["local"]["status"] == "unavailable" for r in result["records"])
    assert all(
        r["runs"]["alternative_a/42"]["torsion"]["status"] in ("ok", "not_applicable")
        for r in result["records"]
    )


def test_exported_angle_and_joint_plots(tmp_path):
    mol = molecule("CCCOCC")
    meta = ts.describe(record(mol))
    angles = ts.torsion_angles(mol, [mol.GetConformer().GetPositions()])
    ts.plots(
        tmp_path,
        "example",
        meta,
        {"reference": [angles], "alternative_a": [angles], "control": [angles]},
    )
    assert len(list(tmp_path.glob("*-angles.png"))) == 1
    assert len(list(tmp_path.glob("*-joint-*.png"))) == len(meta["adjacent_torsions"])
