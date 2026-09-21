"""Small synthetic CPU checks, without GEOM data or weights."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from stifflow_benchmark.evaluate import conformer_mol, evaluate, tetrahedral_chirality
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


if __name__ == "__main__":
    unittest.main()
