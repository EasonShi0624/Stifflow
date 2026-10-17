"""Small synthetic CPU checks, without GEOM data or weights."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from stifflow_benchmark.evaluate import conformer_mol, evaluate, tetrahedral_chirality
from stifflow_benchmark.generate import validate_resume
from stifflow_benchmark.summarize import summarize


class EvaluationTest(unittest.TestCase):
    def test_roundtrip_failures_and_stereo(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("F[C@](Cl)(Br)I"))
        self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=42), 0)
        pos = mol.GetConformer().GetPositions()
        reflection = pos.copy()
        reflection[:, 0] *= -1
        chiral = tetrahedral_chirality(mol, np.stack([pos, reflection]))
        self.assertEqual(chiral["assessed"], 2)
        self.assertEqual(chiral["correct"], 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(
                root / "samples.npz",
                pos_ref=pos[None],
                pos_gen=np.stack([pos + 3, pos + 7]),
                atomic_numbers=[a.GetAtomicNum() for a in mol.GetAtoms()],
            )
            record = {
                "id": "synthetic",
                "status": "ok",
                "npz": "samples.npz",
                "molblock": Chem.MolToMolBlock(conformer_mol(mol, pos)),
            }
            manifest = {
                "schema_version": 1,
                "method": "precision",
                "baseline_kind": "controlled",
                "seed": 42,
                "split": "synthetic_unit_test",
                "training_budget": {"steps": 0},
                "config": {},
                "generation": {"nfe": None},
                "provenance": {},
                "records": [
                    record,
                    {
                        "id": "generation_failure",
                        "status": "failed",
                        "reason": "synthetic failure",
                    },
                    record | {"id": "missing_file", "npz": "missing.npz"},
                ],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            report = evaluate(path, [0.05, 0.5])
            self.assertEqual(report["aggregate"]["attempted"], 3)
            self.assertEqual(report["aggregate"]["succeeded"], 1)
            self.assertEqual(report["aggregate"]["failed"], 2)
            self.assertIsNone(report["aggregate"]["attempted_mat"])
            np.testing.assert_allclose(
                report["aggregate"]["successful_only"]["cov_r_mean"], 1
            )
            np.testing.assert_allclose(
                report["aggregate"]["attempted_coverage_failures_zero"]["cov_r"], 1 / 3
            )
            self.assertLess(report["records"][0]["mat_r"], 1e-6)
            self.assertEqual(report["records"][0]["chirality"]["fraction"], 1)
            again = evaluate(path, [0.05, 0.5])
            self.assertEqual(report["records"], again["records"])
            self.assertEqual(report["aggregate"], again["aggregate"])
            json.dumps(report, allow_nan=False)
            # Matched controls can be summarized; a budget change must be rejected.
            report["run"]["training_data"] = {
                "train_sha256": "synthetic-train",
                "val_sha256": "synthetic-val",
            }
            report["run"]["input_manifest"] = {"sha256": "synthetic-references"}
            report_path = root / "evaluation.json"
            report_path.write_text(json.dumps(report))
            self.assertEqual(len(summarize([report_path])["runs"]), 1)
            report["run"]["training_budget"]["steps"] = 1
            other = root / "other.json"
            other.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "Controlled runs differ"):
                summarize([report_path, other])
            manifest["records"] = [record | {"id": "bad_order"}]
            path.write_text(json.dumps(manifest))
            np.savez(
                root / "samples.npz",
                pos_ref=pos[None],
                pos_gen=np.stack([pos, pos]),
                atomic_numbers=np.arange(mol.GetNumAtoms()),
            )
            self.assertIn("atom order", evaluate(path)["records"][0]["reason"])
            manifest["records"] *= 2
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                evaluate(path)

    def test_disconnected_molecules_and_staged_label_normalization(self):
        mol = Chem.MolFromSmiles("C.C")
        positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(
                root / "disconnected.npz",
                pos_ref=positions[None],
                pos_gen=np.stack([positions, positions]),
                atomic_numbers=[6, 6],
            )
            manifest = {
                "schema_version": 1,
                "method": "precision",
                "baseline_kind": "controlled",
                "seed": 42,
                "split": "synthetic_unit_test",
                "training_budget": {"steps": 0},
                "config": {},
                "generation": {"nfe": None},
                "provenance": {},
                "records": [
                    {
                        "id": "disconnected",
                        "status": "ok",
                        "npz": "disconnected.npz",
                        "molblock": Chem.MolToMolBlock(conformer_mol(mol, positions)),
                    }
                ],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            strict = evaluate(path, [0.05, 0.5])
            self.assertEqual(strict["aggregate"]["failed"], 1)
            self.assertIn("disconnected", strict["records"][0]["reason"])
            allowed = evaluate(path, [0.05, 0.5], allow_disconnected=True)
            self.assertEqual(allowed["aggregate"]["failed"], 0)
            self.assertEqual(allowed["aggregate"]["multi_fragment_molecules"], 1)
            self.assertIn("multi-fragment", allowed["protocol"]["connectivity"])

        def report(train, val, official="data"):
            return {
                "run": {
                    "method": "precision",
                    "baseline_kind": "controlled",
                    "seed": 42,
                    "split": "synthetic",
                    "training_budget": {"steps": 0},
                    "training_data": {
                        "train": train,
                        "val": val,
                        "official_data": official,
                        "training_seed": 42,
                    },
                    "input_manifest": {"sha256": "references"},
                    "config": {"model": {"method": "precision", "seed": 42, "x": 1}},
                    "generation": {},
                },
                "protocol": {"thresholds": [0.05, 0.5]},
                "records": [{"id": "molecule"}],
                "aggregate": {"attempted": 1, "succeeded": 1, "failed": 0},
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            other_data = root / "other-data.json"
            first.write_text(json.dumps(report("staged-a", "staged-b")))
            second.write_text(json.dumps(report("staged-c", "staged-d")))
            other_data.write_text(
                json.dumps(report("staged-a", "staged-b", official="other"))
            )
            with self.assertRaisesRegex(ValueError, "Controlled runs differ"):
                summarize([first, second])
            normalized = summarize([first, second], normalize_staged_labels=True)
            self.assertEqual(len(normalized["runs"]), 2)
            self.assertIn("data_identity", normalized)
            self.assertEqual(
                normalized["runs"][0]["staged_label_manifests"],
                {"train": "staged-a", "val": "staged-b"},
            )
            with self.assertRaisesRegex(ValueError, "Controlled runs differ"):
                summarize([first, other_data], normalize_staged_labels=True)

    def test_resume_validation_skips_completed_and_rejects_changes(self):
        checkpoint = {
            "hyper_parameters": {"method": "precision", "seed": 42},
            "global_step": 250000,
            "training_signature": {"official_data": "data"},
        }
        checkpoint_fingerprint = {"sha256": "checkpoint"}
        input_fingerprint = {"sha256": "references"}
        current_provenance = {"repos": {"stifflow-v1": {"head": "head", "diff": ""}}}
        specification = {
            "split": "val",
            "records": [{"id": "a"}, {"id": "b"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            np.savez(
                output / "a.npz",
                pos_ref=np.zeros((1, 2, 3)),
                pos_gen=np.zeros((2, 2, 3)),
                atomic_numbers=[6, 6],
            )
            report = {
                "schema_version": 1,
                "method": "precision",
                "baseline_kind": "controlled",
                "seed": 42,
                "split": "val",
                "config": {"model": checkpoint["hyper_parameters"], "steps": 50},
                "training_budget": {"steps": 250000},
                "training_data": checkpoint["training_signature"],
                "checkpoint": checkpoint_fingerprint,
                "input_manifest": input_fingerprint,
                "provenance": current_provenance,
                "records": [
                    {"id": "a", "status": "ok", "npz": "a.npz"},
                    {"id": "b", "status": "failed", "reason": "interrupted"},
                ],
            }
            done = validate_resume(
                report,
                specification,
                output,
                checkpoint_fingerprint,
                input_fingerprint,
                checkpoint,
                "precision",
                42,
                50,
                current_provenance,
            )
            self.assertEqual(list(done), ["a"])
            changed = dict(report) | {"checkpoint": {"sha256": "other"}}
            with self.assertRaisesRegex(ValueError, "checkpoint differs"):
                validate_resume(
                    changed,
                    specification,
                    output,
                    checkpoint_fingerprint,
                    input_fingerprint,
                    checkpoint,
                    "precision",
                    42,
                    50,
                    current_provenance,
                )
            changed = dict(report) | {"provenance": {"repos": {}}}
            with self.assertRaisesRegex(ValueError, "different code revisions"):
                validate_resume(
                    changed,
                    specification,
                    output,
                    checkpoint_fingerprint,
                    input_fingerprint,
                    checkpoint,
                    "precision",
                    42,
                    50,
                    current_provenance,
                )
            changed = dict(report) | {"config": {"model": {}, "steps": 50}}
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                validate_resume(
                    changed,
                    specification,
                    output,
                    checkpoint_fingerprint,
                    input_fingerprint,
                    checkpoint,
                    "precision",
                    42,
                    50,
                    current_provenance,
                )
            changed = dict(report) | {
                "records": [
                    {"id": "a", "status": "ok", "npz": "a.npz"},
                    {"id": "a", "status": "failed"},
                ]
            }
            with self.assertRaisesRegex(ValueError, "duplicate or unexpected"):
                validate_resume(
                    changed,
                    specification,
                    output,
                    checkpoint_fingerprint,
                    input_fingerprint,
                    checkpoint,
                    "precision",
                    42,
                    50,
                    current_provenance,
                )


if __name__ == "__main__":
    unittest.main()
