from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.corruption.isotropic import IsotropicCorruptor
from src.corruption.stiffness_structured import StiffnessStructuredCorruptor
from src.baselines import rdkit_etkdg
from src.baselines.rdkit_etkdg import RdkitSmokeError, run_rdkit_smoke
from src.data.records import load_records
from src.data.splits import make_random_split
from src.train.pipeline import run_smoke_pipeline
from src.train.smoke import run_smoke
from src.utils.config import load_config


class SmokeScaffoldTests(unittest.TestCase):
    def test_config_loads(self) -> None:
        config = load_config("configs/train/tiny_smoke.yaml")
        self.assertEqual(config["data"]["dataset_name"], "synthetic_smoke")
        self.assertEqual(config["corruption"]["mode"], "isotropic")

    def test_random_split_keeps_molecules_unique(self) -> None:
        records = load_records("data/processed/smoke_fixture")
        split = make_random_split(records, dataset_name="synthetic_smoke", seed=0)
        molecule_ids = split["train"] + split["val"] + split["test"]
        self.assertEqual(len(molecule_ids), len(set(molecule_ids)))
        self.assertEqual(len(molecule_ids), len(records))

    def test_isotropic_corruption_is_seed_deterministic(self) -> None:
        config = load_config("configs/train/tiny_smoke.yaml")
        record = load_records("data/processed/smoke_fixture")[0]
        coords = record["conformers"][0]["coords"]
        first = IsotropicCorruptor(config, seed=7).corrupt(record, coords)
        second = IsotropicCorruptor(config, seed=7).corrupt(record, coords)
        self.assertEqual(first, second)

    def test_stiffness_ring_atoms_are_more_protected(self) -> None:
        config = load_config("configs/train/tiny_smoke.yaml")
        config["corruption"]["mode"] = "stiffness_structured"
        record = next(
            item for item in load_records("data/processed/smoke_fixture")
            if item["molecule_id"] == "smoke_cyclopropane"
        )
        sigmas = StiffnessStructuredCorruptor(config, seed=0).atom_sigmas(record, atom_count=3)
        self.assertTrue(all(sigma == config["corruption"]["bond_sigma"] for sigma in sigmas))
        self.assertLess(config["corruption"]["bond_sigma"], config["corruption"]["torsion_sigma"])

    def test_dry_run_writes_manifest_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "smoke.yaml"
            config_path.write_text(
                """
experiment:
  name: smoke_random_isotropic_seed0
  seed: 0
  output_root: {output_root}
data:
  dataset_name: synthetic_smoke
  raw_root: data/raw/geom_drugs
  processed_root: data/processed/smoke_fixture
  split_root: {split_root}
  split_type: random
  subset:
    max_molecules: 6
    max_conformers_per_mol: 2
    tag: synthetic_smoke
model:
  backbone: egnn_fallback
  hidden_dim: 128
  num_layers: 4
  time_embedding_dim: 64
flow_matching:
  objective: cartesian_linear
  interpolation: linear
corruption:
  mode: isotropic
  stiffness_table: heuristic_v1
  protect_rings: true
  protect_amides: true
  isotropic_sigma: 0.25
  bond_sigma: 0.02
  angle_sigma: 0.08
  torsion_sigma: 0.30
train:
  enabled: true
  batch_size: 8
  max_steps: 100
  learning_rate: 1.0e-4
  num_workers: 0
sample:
  enabled: true
  num_samples_per_mol: 2
  num_steps: 20
eval:
  enabled: true
  compute_cov_mat: true
  compute_validity: true
  compute_energy_checks: false
  eval_subset_size: 64
logging:
  save_manifest: true
  save_results_json: true
  save_log: true
""".format(output_root=tmp_path / "experiments", split_root=tmp_path / "splits"),
                encoding="utf-8",
            )
            paths = run_smoke(config_path, dry_run=True)
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            results = json.loads(paths["results"].read_text(encoding="utf-8"))
            self.assertFalse(manifest["training_executed"])
            self.assertEqual(manifest["result_kind"], "smoke_only_not_benchmark")
            self.assertEqual(results["dataset"]["kind"], "synthetic_smoke")
            self.assertFalse(results["benchmark_result"])
            self.assertEqual(results["hardware"], "cpu_only")
            self.assertEqual(results["sample_count"], 6)

    def test_pipeline_runs_both_smoke_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = run_smoke_pipeline(output_root=tmp, pipeline_name="test_smoke_pair")
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            results = json.loads(paths["results"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["result_kind"], "smoke_only_not_benchmark")
            self.assertFalse(manifest["benchmark_result"])
            self.assertEqual(results["run_count"], 2)
            self.assertFalse(results["training_executed"])
            self.assertEqual(results["hardware"], "cpu_only")
            self.assertEqual(
                {run["corruption_mode"] for run in results["runs"]},
                {"isotropic", "stiffness_structured"},
            )

    def test_rdkit_smoke_baseline_writes_smoke_only_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = run_rdkit_smoke(output_root_override=tmp, run_name="test_rdkit_smoke")
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            results = json.loads(paths["results"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["baseline_name"], "rdkit_etkdg_v3_mmff")
            self.assertEqual(manifest["result_kind"], "smoke_only_not_benchmark")
            self.assertFalse(results["benchmark_result"])
            self.assertEqual(results["hardware"], "cpu_only")
            self.assertIn("python_executable", manifest)
            self.assertIn("python_version", manifest)
            self.assertIn("rdkit_version", manifest)
            self.assertIn("python_executable", results)
            self.assertIn("python_version", results)
            self.assertIn("rdkit_version", results)
            self.assertEqual(manifest["python_executable"], results["python_executable"])
            if results["rdkit_available"]:
                self.assertEqual(results["execution_status"], "completed")
                self.assertIsInstance(results["rdkit_version"], str)
                self.assertEqual(results["metrics"]["attempted_molecule_count"], 6)
                self.assertGreater(results["metrics"]["successful_molecule_count"], 0)
            else:
                self.assertEqual(results["execution_status"], "import_failure")
                self.assertFalse(results["baseline_executed"])
                self.assertEqual(results["metrics"]["attempted_molecule_count"], 0)
                self.assertEqual(results["metrics"]["failure_type_counts"].get("import"), 1)

    def test_rdkit_import_failure_preserves_existing_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            results_dir = output_root / "results"
            results_dir.mkdir(parents=True)
            result_path = results_dir / "test_rdkit_smoke.json"
            original = {
                "baseline_name": "rdkit_etkdg_v3_mmff",
                "result_kind": "smoke_only_not_benchmark",
                "rdkit_available": True,
                "baseline_executed": True,
                "execution_status": "completed",
            }
            result_path.write_text(json.dumps(original), encoding="utf-8")

            with patch.object(
                rdkit_etkdg,
                "_load_rdkit",
                side_effect=RdkitSmokeError("forced import failure", failure_type="import"),
            ):
                with self.assertRaises(RdkitSmokeError):
                    run_rdkit_smoke(output_root_override=output_root, run_name="test_rdkit_smoke")

            self.assertEqual(json.loads(result_path.read_text(encoding="utf-8")), original)
            self.assertTrue((output_root / "logs" / "test_rdkit_smoke.preflight_failure.log").exists())


if __name__ == "__main__":
    unittest.main()
