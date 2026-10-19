"""Small analytic/one-molecule physics checks; no saved-data audit or model run."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit import Chem
from rdkit.Chem import AllChem
from stifflow.features import graph_from_mol
from stifflow.geometry import centered
from stifflow.locality import choose_start
from stifflow.preflight import fingerprint
from stifflow.source import prepare_source
from stifflow_alt_a.path import prepare_label
from stifflow_benchmark import stiffness_diagnostics as sd
from stifflow_benchmark.research_compare import comparison_signature, paired
from stifflow_benchmark.research_diagnostics import select_records
from stifflow_benchmark.stiffness_compare import METHODS, screen


@pytest.fixture
def water():
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol


def test_mmff_gradient_agrees_with_energy_at_each_queried_geometry(water):
    potential = sd.MMFF94(water)
    x = water.GetConformer().GetPositions()
    other = x.copy()
    other[1] += [0.1, -0.07, 0.03]
    for point in [x, other, x]:
        _, grad = potential(point)
        for atom, axis in [(0, 0), (1, 1), (2, 2)]:
            plus, minus = point.copy(), point.copy()
            plus[atom, axis] += 1e-6
            minus[atom, axis] -= 1e-6
            assert_allclose(
                grad[atom, axis],
                (potential(plus)[0] - potential(minus)[0]) / 2e-6,
                atol=2e-6,
            )
    assert sd.rms_force(np.ones((3, 3))) == pytest.approx(np.sqrt(3))


def test_taylor_keeps_first_derivative_and_signed_raw_curvature():
    Q = np.eye(6)
    H = np.diag([-2.0, 1.0, 3.0, 4.0, 2.0, 1.0])
    origin = np.arange(6).reshape(2, 3) * 0.1
    point = origin.copy()
    point[0, 0] += 0.3

    def value(x):
        return 0.5 * x.ravel() @ H @ x.ravel(), (H @ x.ravel()).reshape(2, 3)

    result = sd.taylor_metrics(point, origin, H, Q, value(point), value(origin))
    assert result["quadratic_kcal_mol"] < 0
    assert result["energy_relative_error"] < 1e-12
    assert result["force_relative_error"] < 1e-12


def test_physics_chirality_check_does_not_accept_a_reflection():
    mol = Chem.AddHs(Chem.MolFromSmiles("F[C@](Cl)(Br)I"))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    point = mol.GetConformer().GetPositions()
    assert sd.stereo_matches(mol, point)
    assert not sd.stereo_matches(mol, point * [-1, 1, 1])


def test_full_archive_verifies_receipts_source_and_row_pairing(tmp_path, water):
    source = prepare_source(graph_from_mol(water))
    pos = water.GetConformer().GetPositions()[None]
    archive = tmp_path / "full.npz"
    np.savez(
        archive,
        positions=pos,
        reference=pos,
        H_raw=np.eye(6)[None],
        alt_L_K=source.S[None],
        source_S=source.S,
        source_C0=source.C0,
        floored_mask=np.zeros((1, 3), dtype=bool),
        spectrum=np.ones((1, 3)),
        row_ids=[7],
    )
    full = {
        "id": "water",
        "split": "train",
        "row_ids": [7],
        "audit_partial": False,
        "molblock": Chem.MolToMolBlock(water),
        "cache": fingerprint(archive),
        "labels": [
            {
                "potential": "MMFF94",
                "units": {
                    "length": "angstrom",
                    "energy": "kcal/mol",
                    "stiffness": "kcal/mol/angstrom^2",
                },
            }
        ],
    }
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"attempt": {"status": "success"}, "record": full}))
    record = {
        "id": "water",
        "split": "train",
        "row_ids": [7],
        "audit_receipt": fingerprint(receipt),
    }
    mol, actual, arrays, inputs = sd.load_archive(record, {"sigma0": 1.0, "rho": 1.0})
    assert mol.GetNumAtoms() == 3
    assert_allclose(actual.S, source.S)
    assert arrays["row_ids"].tolist() == [7]
    assert inputs["archive"]["sha256"] == fingerprint(archive)["sha256"]
    archive.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        sd.load_archive(record, {"sigma0": 1.0, "rho": 1.0})


def test_locality_quadratic_limit_is_exact_and_keeps_all_queries(water, monkeypatch):
    source = prepare_source(graph_from_mol(water))
    origin = centered(water.GetConformer().GetPositions())
    H = np.diag(np.arange(1.0, 7.0))
    label = prepare_label(source, H, 0.592485)

    class Quadratic:
        queries = 0

        def __init__(self, mol):
            pass

        def __call__(self, point):
            self.queries += 1
            d = source.Q.T @ (point - origin).ravel()
            return float(0.5 * d @ H @ d), (source.Q @ H @ d).reshape(-1, 3)

    monkeypatch.setattr(sd, "MMFF94", Quadratic)
    arrays = {
        "row_ids": [7],
        "positions": [origin],
        "reference": [origin],
        "H_raw": [H],
        "alt_L_K": [label["L_K"]],
        "spectrum": [np.arange(1.0, 4.0)],
        "floored_mask": [[False] * 3],
    }
    points = []
    result = sd.locality_molecule("water", water, source, arrays, 0, points.append)
    assert len(points) == 4 * 4 * 16
    assert result["energy_gradient_queries"] == len(points) + 2
    assert all(
        c["valid_signal_fraction"] == pytest.approx(1) for c in result["candidates"]
    )
    candidates = [
        c
        | {"force_prediction_delta": {"mean": -1, "ci95": [-2, -0.5], "molecules": 128}}
        for c in result["candidates"]
    ]
    assert choose_start(candidates) is None
    assert max(p["target"]["energy_relative_error"] for p in points) < 1e-10


def test_physics_raw_force_is_not_filtered_by_stereo_relaxation_or_missing_labels(
    water, monkeypatch
):
    pos = water.GetConformer().GetPositions()
    original = pos.copy()

    class FixedForce:
        queries = 0

        def __init__(self, mol):
            pass

        def __call__(self, x):
            self.queries += 1
            return 10.0, np.ones_like(x)

    monkeypatch.setattr(sd, "MMFF94", FixedForce)
    monkeypatch.setattr(
        sd,
        "relaxation",
        lambda *args: {
            "status": "ok",
            "stereo_changed": True,
            "valid_strain": False,
            "strain_kcal_mol": 3.0,
        },
    )
    points = []
    report = sd.audit_molecule(
        "water", water, pos[None], np.stack([pos, pos]), points.append
    )
    assert report["status"] == "ok" and report["force_median"] == pytest.approx(
        np.sqrt(3)
    )
    assert report["strain_median_stereo_preserving"] is None
    assert report["relaxation_stereo_changes"] == 2
    assert report["taylor_covered_conformers"] == 0
    assert_allclose(water.GetConformer().GetPositions(), original, atol=0)

    class BrokenForce(FixedForce):
        def __call__(self, x):
            raise ValueError("bad geometry")

    monkeypatch.setattr(sd, "MMFF94", BrokenForce)
    report = sd.audit_molecule(
        "water", water, pos[None], np.stack([pos, pos]), lambda row: None
    )
    assert report["status"] == "failed" and report["force_median"] is None
    assert report["expected_conformers"] == report["failed_conformers"] == 2


def test_xtb_gradient_parser_units_atom_count_and_open_shell_rejection(water, tmp_path):
    text = "$grad\n cycle = 1 SCF energy = -2.0D+00 |dE/dxyz| = 0.1\n0 0 0 o\n0 0 1 h\n0 1 0 h\n1D-3 0 0\n0 2D-3 0\n0 0 3D-3\n$end\n"
    energy, gradient = sd.parse_xtb_gradient(text, 3)
    assert energy == pytest.approx(-2 * sd.HARTREE_KCAL_MOL)
    assert gradient[1, 1] == pytest.approx(
        0.002 * sd.HARTREE_KCAL_MOL / sd.BOHR_ANGSTROM
    )
    with pytest.raises(ValueError):
        sd.parse_xtb_gradient(text, 4)
    radical = Chem.AddHs(Chem.MolFromSmiles("[CH3]"))
    with pytest.raises(ValueError, match="closed-shell"):
        sd.xtb_value(
            radical, np.zeros((radical.GetNumAtoms(), 3)), "not-executed", tmp_path
        )


def test_physics_binds_coordinates_to_geometry_and_keeps_missing_label_rows(
    tmp_path, water, monkeypatch
):
    records = [{"id": str(i), "status": "ok"} for i in range(512)]
    manifest = {
        "method": "control",
        "split": "val",
        "baseline_kind": "controlled",
        "seed": 42,
        "training_budget": {"steps": 25000},
        "config": {"steps": 50},
        "records": records,
    }
    path = tmp_path / "ensembles.json"
    path.write_text(json.dumps(manifest))
    evaluation = {
        "input_manifest": fingerprint(path),
        "run": {k: v for k, v in manifest.items() if k != "records"},
        "protocol": {"generated_reference_ratio": 2, "force_field_optimization": False},
        "records": [r | {"input": {"sha256": "old-coordinates"}} for r in records],
    }
    ep = tmp_path / "evaluation.json"
    ep.write_text(json.dumps(evaluation))
    lp = tmp_path / "val.json"
    lp.write_text(json.dumps({"split": "val", "source_config": {}, "records": []}))
    monkeypatch.setattr(sd, "validate_label_manifest", lambda manifest: None)
    positions = water.GetConformer().GetPositions()
    monkeypatch.setattr(
        sd,
        "read_ensemble",
        lambda *args: (
            water,
            positions[None],
            np.stack([positions, positions]),
            {"sha256": "current-coordinates"},
        ),
    )
    called = []

    def fake_audit(identity, *args):
        called.append(identity)
        return {
            "id": identity,
            "status": "ok",
            "force_median": 1.0,
            "oracle_force_median": None,
        }

    monkeypatch.setattr(sd, "audit_molecule", fake_audit)
    args = SimpleNamespace(
        manifest=path,
        evaluation=ep,
        val_labels=lp,
        xtb=None,
        output=tmp_path / "output",
    )
    failed = sd.run_physics(args, lambda row: None)
    assert not called and failed["aggregate"]["failed_molecules"] == 64
    assert failed["aggregate"]["force_mean_molecule_median"] is None
    for row in evaluation["records"]:
        row["input"]["sha256"] = "current-coordinates"
    ep.write_text(json.dumps(evaluation))
    complete = sd.run_physics(args, lambda row: None)
    assert len(called) == 64 and complete["aggregate"]["failed_molecules"] == 0
    assert complete["aggregate"]["force_mean_molecule_median"] == 1.0
    assert all(row["label_error"] for row in complete["records"])


def matched_reports():
    training = {
        "train": "train",
        "val": "val",
        "official_data": "official",
        "sampling": "uniform_conformer_rows",
        "dataset_seed": 42,
        "training_seed": 42,
        "row_rng_domain": 2,
        "effective_batch_size": 128,
        "profile": "research",
        "config": {
            "source": {"rho": 1.0},
            "model_args": {"first_cycle_steps": 250000},
            "batch_size": 32,
            "profiles": {"research": {"accumulate_grad_batches": 4}},
        },
    }
    geometry, physics = {}, {}
    forces = {
        "alternative_a": 10.0,
        "control": 11.0,
        "a_rot_null": 12.0,
        "a_delayed": 8.0,
        "a_delayed_rot_null": 11.0,
    }
    for method in METHODS:
        train = copy.deepcopy(training)
        model = {"method": method, "seed": 42}
        if method.startswith("a_delayed"):
            model.update(stiffness_start=0.5, locality_report_sha256="a" * 64)
            train.update(stiffness_start=0.5, locality_report_sha256="a" * 64)
        run = {
            "method": method,
            "split": "val",
            "baseline_kind": "controlled",
            "seed": 42,
            "training_budget": {"steps": 25000},
            "config": {"model": model, "steps": 50},
            "training_data": train,
            "input_manifest": {"sha256": "refs"},
        }
        records = [
            {
                "id": str(i),
                "status": "ok",
                "mat_r": 0.1,
                "mat_p": 0.2,
                "cov_r": [0.9],
                "cov_p": [0.8],
                "chirality": {"assessed": 2, "fraction": 0.9},
            }
            for i in range(512)
        ]
        geometry[method] = {
            "run": run,
            "protocol": {
                "thresholds": [0.5],
                "generated_reference_ratio": 2,
                "force_field_optimization": False,
            },
            "records": records,
        }
        selected = select_records(records, 64)
        physics[method] = {
            "schema_version": 1,
            "kind": "stifflow-physics",
            "status": "completed",
            "split": "val",
            "method": method,
            "protocol": sd.PHYSICS_PROTOCOL,
            "run": copy.deepcopy(run),
            "environment": {
                "numpy": "unit-test",
                "scipy": "unit-test",
                "rdkit": "unit-test",
            },
            "records": [
                {
                    "id": r["id"],
                    "status": "ok",
                    "force_median": forces[method],
                    "oracle_force_median": None,
                }
                for r in selected
            ],
            "oracle": {"status": "not_requested"},
            "oracle_ids": sorted(r["id"] for r in selected[:16]),
        }
    return geometry, physics


def test_factorial_screen_has_physics_gate_and_shared_molecule_bootstrap():
    geometry, physics = matched_reports()
    result = screen(geometry, physics)
    assert result["advance_candidate"] and not result["formal_method_upgrade"]
    assert result["force_candidate_minus_baseline"]["alternative_a"][
        "relative_change"
    ] == pytest.approx(-0.2)
    assert result["direction_difference_in_differences"]["mean"] == pytest.approx(-1.0)
    assert result["timing_specific_direction_evidence"]
    for method in METHODS:
        geometry[method]["records"].reverse()
        physics[method]["records"].reverse()
    assert screen(geometry, physics) == result
    physics["a_delayed"]["records"][0].update(status="failed", force_median=None)
    assert not screen(geometry, physics)["advance_candidate"]


def test_schedule_improvement_without_directional_benefit_fails_gate():
    geometry, physics = matched_reports()
    for r in physics["a_delayed_rot_null"]["records"]:
        r["force_median"] = 8.0
    result = screen(geometry, physics)
    assert not result["criteria"]["force_ci_below_a_delayed_rot_null"]
    assert not result["advance_candidate"]
    geometry, physics = matched_reports()
    for r in geometry["a_delayed"]["records"]:
        r["cov_p"] = [0.7]
    assert not screen(geometry, physics)["criteria"]["geometry_cov_p_vs_alternative_a"]


def test_comparison_rejects_changed_locality_model_protocol_or_cohort():
    geometry, physics = matched_reports()
    comparison_signature(geometry["a_delayed"])
    with pytest.raises(ValueError, match="legacy geometry gate"):
        paired(geometry["a_delayed"], geometry["alternative_a"])
    geometry["a_delayed_rot_null"]["run"]["config"]["model"]["stiffness_start"] = 0.75
    with pytest.raises(ValueError, match="provenance differ"):
        screen(geometry, physics)
    for mutate in [
        lambda g, p: g["a_delayed"]["run"].update(split="test"),
        lambda g, p: p["a_delayed"]["records"].pop(),
        lambda g, p: p["a_delayed"]["protocol"].update(molecules=63),
    ]:
        g, p = copy.deepcopy(matched_reports())
        mutate(g, p)
        with pytest.raises(ValueError):
            screen(g, p)


def test_independent_oracle_reversal_blocks_advancement():
    geometry, physics = matched_reports()
    for method in METHODS:
        physics[method]["oracle"] = {
            "status": "requested",
            "binary": {"sha256": "b" * 64},
            "version": "unit-test-xTB",
        }
        for row in physics[method]["records"]:
            row["oracle_force_median"] = 20.0 if method == "a_delayed" else 10.0
    result = screen(geometry, physics)
    assert result["oracle"]["status"] == "reversed"
    assert not result["advance_candidate"]
    assert result["claim_scope"] == "MMFF94 only"


@pytest.mark.parametrize(
    "task,arguments",
    [
        ("locality", ["--train-labels", "train.json", "--config", "cfg.yaml"]),
        (
            "physics",
            ["ensemble.json", "--evaluation", "eval.json", "--val-labels", "val.json"],
        ),
    ],
)
def test_diagnostics_require_allocation_before_reading_or_creating_output(
    task, arguments, tmp_path, monkeypatch
):
    import sys

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["stiffness", task, *arguments, "--output", str(tmp_path / "result")],
    )
    with pytest.raises(SystemExit) as exc:
        sd.main()
    assert exc.value.code == 2 and not (tmp_path / "result").exists()
