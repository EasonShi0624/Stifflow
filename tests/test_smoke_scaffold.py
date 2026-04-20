from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.corruption.isotropic import IsotropicCorruptor
from src.corruption.stiffness_structured import StiffnessStructuredCorruptor
from src.baselines import rdkit_etkdg
from src.baselines.rdkit_etkdg import RdkitSmokeError, run_rdkit_smoke, run_rdkit_tiny_real_dry_run
from src.data.records import load_records
from src.data.splits import make_random_split
from src.data.geom_preflight import run_geom_preflight
from src.data.geom_split_dry_run import run_geom_tiny_split_dry_run
from src.eval.smoke_compare import write_smoke_comparison
from src.eval.tiny_real_compare import write_tiny_real_comparison
from src.eval.tiny_real_lifecycle import write_tiny_real_lifecycle_summary
from src.eval.tiny_real_pilot_compare import write_tiny_real_pilot_comparison
from src.train.pipeline import run_smoke_pipeline
from src.train.smoke import run_smoke
from src.train.tiny_real import run_tiny_real_model_dry_runs
from src.train.tiny_real_checkpoint_sampling import run_tiny_real_checkpoint_sampling_smoke_tests
from src.train.tiny_real_execution import run_tiny_real_execution_smoke_tests
from src.train.tiny_real_multi_step import run_tiny_real_multi_step_smoke_tests
from src.train.tiny_real_pilot import run_tiny_real_pilot_configs
from src.train.tiny_real_single_step import run_tiny_real_single_step_smoke_tests
from src.utils.config import load_config


class _FakePoint:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class _FakeAtom:
    def __init__(self, index: int, symbol: str, atomic_num: int, formal_charge: int = 0) -> None:
        self._index = index
        self._symbol = symbol
        self._atomic_num = atomic_num
        self._formal_charge = formal_charge

    def GetIdx(self) -> int:
        return self._index

    def GetSymbol(self) -> str:
        return self._symbol

    def GetAtomicNum(self) -> int:
        return self._atomic_num

    def GetFormalCharge(self) -> int:
        return self._formal_charge

    def IsInRing(self) -> bool:
        return False


class _FakeBond:
    def __init__(self, begin: int, end: int, order: float) -> None:
        self._begin = begin
        self._end = end
        self._order = order

    def GetBeginAtomIdx(self) -> int:
        return self._begin

    def GetEndAtomIdx(self) -> int:
        return self._end

    def GetBondTypeAsDouble(self) -> float:
        return self._order


class _FakeConformer:
    def __init__(self, coords: list[list[float]]) -> None:
        self._coords = coords

    def GetAtomPosition(self, atom_index: int) -> _FakePoint:
        x, y, z = self._coords[atom_index]
        return _FakePoint(x, y, z)


class _FakeMol:
    def __init__(self, coords: list[list[float]]) -> None:
        self._atoms = [_FakeAtom(0, "C", 6), _FakeAtom(1, "O", 8)]
        self._bonds = [_FakeBond(0, 1, 1.0)]
        self._conformer = _FakeConformer(coords)

    def GetNumAtoms(self) -> int:
        return len(self._atoms)

    def GetNumConformers(self) -> int:
        return 1

    def GetAtoms(self) -> list[_FakeAtom]:
        return self._atoms

    def GetBonds(self) -> list[_FakeBond]:
        return self._bonds

    def GetConformer(self) -> _FakeConformer:
        return self._conformer


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

    def test_rdkit_tiny_real_import_failure_writes_nonbenchmark_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed" / "molecules.json"
            split_path = root / "splits" / "split.json"
            processed.parent.mkdir(parents=True)
            split_path.parent.mkdir(parents=True)
            molecules = [
                {
                    "molecule_id": f"geom_tiny_{index}",
                    "smiles": "CO",
                    "atom_symbols": ["C", "O", "H", "H", "H", "H"],
                    "bonds": [[0, 1, 1], [0, 2, 1], [0, 3, 1], [0, 4, 1], [1, 5, 1]],
                    "conformers": [{"conformer_id": str(index), "coords": [[0, 0, 0]] * 6}],
                }
                for index in range(3)
            ]
            processed.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "result_kind": "processed_output_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "molecules": molecules,
                    }
                ),
                encoding="utf-8",
            )
            split_path.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "result_kind": "split_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "split_type": "random",
                        "degenerate_tiny_split": False,
                        "train": ["geom_tiny_0"],
                        "val": ["geom_tiny_1"],
                        "test": ["geom_tiny_2"],
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(
                rdkit_etkdg,
                "_load_rdkit",
                side_effect=RdkitSmokeError("forced import failure", failure_type="import"),
            ):
                paths = run_rdkit_tiny_real_dry_run(
                    processed_artifact_path=processed,
                    split_path=split_path,
                    output_root=root / "experiments",
                    run_name="test_geom_tiny_rdkit",
                )

            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            results = json.loads(paths["results"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["result_kind"], "tiny_real_data_dry_run_not_benchmark")
            self.assertEqual(results["result_kind"], "tiny_real_data_dry_run_not_benchmark")
            self.assertFalse(results["benchmark_result"])
            self.assertEqual(results["hardware"], "cpu_only")
            self.assertFalse(results["training_executed"])
            self.assertFalse(results["baseline_executed"])
            self.assertEqual(results["execution_status"], "import_failure")
            self.assertEqual(results["metrics"]["attempted_molecule_count"], 0)
            self.assertEqual(results["metrics"]["failure_type_counts"].get("import"), 1)
            self.assertEqual(results["dataset"]["split_counts"], {"train": 1, "val": 1, "test": 1})

    def test_tiny_real_model_dry_runs_write_nonbenchmark_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed" / "molecules.json"
            split_path = root / "splits" / "split.json"
            processed.parent.mkdir(parents=True)
            split_path.parent.mkdir(parents=True)
            molecules = [
                {
                    "molecule_id": f"geom_tiny_{index}",
                    "smiles": "CO",
                    "atom_symbols": ["C", "O", "H"],
                    "bonds": [[0, 1, 1], [0, 2, 1]],
                    "conformers": [
                        {
                            "conformer_id": str(index),
                            "coords": [
                                [0.0, 0.0, 0.0],
                                [1.2 + 0.1 * index, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                            ],
                        }
                    ],
                    "ring_atom_indices": [],
                    "rotatable_bond_count": 0,
                }
                for index in range(3)
            ]
            processed.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "result_kind": "processed_output_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "molecules": molecules,
                    }
                ),
                encoding="utf-8",
            )
            split_path.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "result_kind": "split_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "split_type": "random",
                        "degenerate_tiny_split": False,
                        "train": ["geom_tiny_0"],
                        "val": ["geom_tiny_1"],
                        "test": ["geom_tiny_2"],
                    }
                ),
                encoding="utf-8",
            )

            outputs = run_tiny_real_model_dry_runs(
                processed_artifact_path=processed,
                split_path=split_path,
                output_root=root / "experiments",
                seed=0,
                max_molecules=3,
            )

            self.assertEqual(set(outputs), {"isotropic", "stiffness_structured"})
            for mode, paths in outputs.items():
                manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                results = json.loads(paths["results"].read_text(encoding="utf-8"))
                self.assertEqual(manifest["result_kind"], "tiny_real_data_model_dry_run_not_benchmark")
                self.assertEqual(results["result_kind"], "tiny_real_data_model_dry_run_not_benchmark")
                self.assertFalse(manifest["benchmark_result"])
                self.assertFalse(results["benchmark_result"])
                self.assertEqual(manifest["hardware"], "cpu_only")
                self.assertFalse(manifest["training_executed"])
                self.assertFalse(manifest["model_training_started"])
                self.assertFalse(results["sampling_executed"])
                self.assertEqual(results["corruption_mode"], mode)
                self.assertEqual(results["sample_count"], 3)
                self.assertEqual(results["dataset"]["split_counts"], {"train": 1, "val": 1, "test": 1})
                self.assertEqual(
                    results["metrics"]["metric_scope"],
                    "geom_drugs_tiny_real_model_source_vs_target_dry_run",
                )
                self.assertIn("not benchmark", " ".join(results["notes"]))

    def test_tiny_real_execution_smoke_writes_forward_loss_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed" / "molecules.json"
            split_path = root / "splits" / "split.json"
            processed.parent.mkdir(parents=True)
            split_path.parent.mkdir(parents=True)
            molecules = [
                {
                    "molecule_id": f"geom_tiny_{index}",
                    "smiles": "CO",
                    "atom_symbols": ["C", "O", "H"],
                    "bonds": [[0, 1, 1], [0, 2, 1]],
                    "conformers": [
                        {
                            "conformer_id": str(index),
                            "coords": [
                                [0.0, 0.0, 0.0],
                                [1.2 + 0.1 * index, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                            ],
                        }
                    ],
                    "ring_atom_indices": [],
                    "rotatable_bond_count": 0,
                }
                for index in range(3)
            ]
            processed.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "result_kind": "processed_output_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "molecules": molecules,
                    }
                ),
                encoding="utf-8",
            )
            split_path.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "result_kind": "split_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "split_type": "random",
                        "degenerate_tiny_split": False,
                        "train": ["geom_tiny_0"],
                        "val": ["geom_tiny_1"],
                        "test": ["geom_tiny_2"],
                    }
                ),
                encoding="utf-8",
            )

            outputs = run_tiny_real_execution_smoke_tests(
                processed_artifact_path=processed,
                split_path=split_path,
                output_root=root / "experiments",
                seed=0,
                max_molecules=3,
            )

            self.assertEqual(set(outputs), {"isotropic", "stiffness_structured"})
            for mode, paths in outputs.items():
                manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                results = json.loads(paths["results"].read_text(encoding="utf-8"))
                self.assertEqual(manifest["result_kind"], "tiny_real_data_model_execution_smoke_not_benchmark")
                self.assertEqual(results["result_kind"], "tiny_real_data_model_execution_smoke_not_benchmark")
                self.assertFalse(results["benchmark_result"])
                self.assertEqual(results["hardware"], "cpu_only")
                self.assertFalse(results["gpu_used"])
                self.assertFalse(results["training_executed"])
                self.assertFalse(results["training_loop_executed"])
                self.assertFalse(results["optimizer_created"])
                self.assertFalse(results["backward_executed"])
                self.assertFalse(results["checkpoint_written"])
                self.assertFalse(results["sampling_executed"])
                self.assertTrue(results["forward_executed"])
                self.assertTrue(results["loss_computed"])
                self.assertEqual(results["corruption_mode"], mode)
                self.assertEqual(results["sample_count"], 3)
                self.assertEqual(results["metrics"]["forward_pass_count"], 1)
                self.assertEqual(results["metrics"]["trainable_parameter_count"], 0)
                self.assertGreaterEqual(results["metrics"]["mean_velocity_mse"], 0.0)
                self.assertEqual(results["dataset"]["split_counts"], {"train": 1, "val": 1, "test": 1})

    def test_tiny_real_single_step_smoke_writes_training_step_outputs(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not importable in this interpreter")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed" / "molecules.json"
            split_path = root / "splits" / "split.json"
            processed.parent.mkdir(parents=True)
            split_path.parent.mkdir(parents=True)
            molecules = [
                {
                    "molecule_id": f"geom_tiny_{index}",
                    "smiles": "CO",
                    "atom_symbols": ["C", "O", "H"],
                    "bonds": [[0, 1, 1], [0, 2, 1]],
                    "conformers": [
                        {
                            "conformer_id": str(index),
                            "coords": [
                                [0.0, 0.0, 0.0],
                                [1.2 + 0.1 * index, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                            ],
                        }
                    ],
                    "ring_atom_indices": [],
                    "rotatable_bond_count": 0,
                }
                for index in range(3)
            ]
            processed.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "result_kind": "processed_output_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "molecules": molecules,
                    }
                ),
                encoding="utf-8",
            )
            split_path.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "result_kind": "split_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "split_type": "random",
                        "degenerate_tiny_split": False,
                        "train": ["geom_tiny_0"],
                        "val": ["geom_tiny_1"],
                        "test": ["geom_tiny_2"],
                    }
                ),
                encoding="utf-8",
            )

            outputs = run_tiny_real_single_step_smoke_tests(
                processed_artifact_path=processed,
                split_path=split_path,
                output_root=root / "experiments",
                seed=0,
                max_molecules=3,
            )

            self.assertEqual(set(outputs), {"isotropic", "stiffness_structured"})
            for mode, paths in outputs.items():
                manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                results = json.loads(paths["results"].read_text(encoding="utf-8"))
                self.assertEqual(
                    manifest["result_kind"],
                    "tiny_real_data_single_step_training_smoke_not_benchmark",
                )
                self.assertEqual(
                    results["result_kind"],
                    "tiny_real_data_single_step_training_smoke_not_benchmark",
                )
                self.assertFalse(results["benchmark_result"])
                self.assertEqual(results["hardware"], "cpu_only")
                self.assertFalse(results["gpu_used"])
                self.assertTrue(results["training_executed"])
                self.assertTrue(results["model_training_started"])
                self.assertFalse(results["training_loop_executed"])
                self.assertTrue(results["single_step_training_smoke_executed"])
                self.assertEqual(results["max_steps"], 1)
                self.assertTrue(results["optimizer_created"])
                self.assertEqual(results["optimizer_step_count"], 1)
                self.assertTrue(results["backward_executed"])
                self.assertEqual(results["backward_pass_count"], 1)
                self.assertFalse(results["checkpoint_written"])
                self.assertFalse(results["sampling_executed"])
                self.assertTrue(results["forward_executed"])
                self.assertTrue(results["loss_computed"])
                self.assertEqual(results["corruption_mode"], mode)
                self.assertEqual(results["sample_count"], 3)
                self.assertEqual(results["metrics"]["optimizer_step_count"], 1)
                self.assertEqual(results["metrics"]["backward_pass_count"], 1)
                self.assertGreater(results["metrics"]["trainable_parameter_count"], 0)
                self.assertGreaterEqual(results["metrics"]["loss_before_step"], 0.0)
                self.assertGreaterEqual(results["metrics"]["loss_after_step"], 0.0)
                self.assertEqual(results["dataset"]["split_counts"], {"train": 1, "val": 1, "test": 1})

    def test_tiny_real_multi_step_smoke_writes_bounded_step_records(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not importable in this interpreter")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed" / "molecules.json"
            split_path = root / "splits" / "split.json"
            processed.parent.mkdir(parents=True)
            split_path.parent.mkdir(parents=True)
            molecules = [
                {
                    "molecule_id": f"geom_tiny_{index}",
                    "smiles": "CO",
                    "atom_symbols": ["C", "O", "H"],
                    "bonds": [[0, 1, 1], [0, 2, 1]],
                    "conformers": [
                        {
                            "conformer_id": str(index),
                            "coords": [
                                [0.0, 0.0, 0.0],
                                [1.2 + 0.1 * index, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                            ],
                        }
                    ],
                    "ring_atom_indices": [],
                    "rotatable_bond_count": 0,
                }
                for index in range(3)
            ]
            processed.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "result_kind": "processed_output_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "molecules": molecules,
                    }
                ),
                encoding="utf-8",
            )
            split_path.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "result_kind": "split_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "split_type": "random",
                        "degenerate_tiny_split": False,
                        "train": ["geom_tiny_0"],
                        "val": ["geom_tiny_1"],
                        "test": ["geom_tiny_2"],
                    }
                ),
                encoding="utf-8",
            )

            outputs = run_tiny_real_multi_step_smoke_tests(
                processed_artifact_path=processed,
                split_path=split_path,
                output_root=root / "experiments",
                seed=0,
                max_molecules=3,
                max_steps=3,
            )

            self.assertEqual(set(outputs), {"isotropic", "stiffness_structured"})
            for mode, paths in outputs.items():
                manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                results = json.loads(paths["results"].read_text(encoding="utf-8"))
                self.assertEqual(
                    manifest["result_kind"],
                    "tiny_real_data_bounded_multistep_training_smoke_not_benchmark",
                )
                self.assertEqual(
                    results["result_kind"],
                    "tiny_real_data_bounded_multistep_training_smoke_not_benchmark",
                )
                self.assertFalse(results["benchmark_result"])
                self.assertEqual(results["hardware"], "cpu_only")
                self.assertFalse(results["gpu_used"])
                self.assertTrue(results["training_executed"])
                self.assertTrue(results["training_loop_executed"])
                self.assertTrue(results["bounded_training_smoke_executed"])
                self.assertFalse(results["larger_real_training_run_started"])
                self.assertEqual(results["max_steps"], 3)
                self.assertEqual(results["hard_max_steps"], 5)
                self.assertEqual(results["optimizer_step_count"], 3)
                self.assertEqual(results["backward_pass_count"], 3)
                self.assertFalse(results["checkpoint_written"])
                self.assertFalse(results["sampling_executed"])
                self.assertEqual(results["corruption_mode"], mode)
                self.assertEqual(results["sample_count"], 3)
                self.assertEqual(results["metrics"]["optimizer_step_count"], 3)
                self.assertEqual(len(results["metrics"]["step_records"]), 3)
                self.assertEqual([row["optimizer_step"] for row in results["metrics"]["step_records"]], [1, 2, 3])
                self.assertGreater(results["metrics"]["trainable_parameter_count"], 0)
                self.assertGreaterEqual(results["metrics"]["initial_loss"], 0.0)
                self.assertGreaterEqual(results["metrics"]["final_loss"], 0.0)
                self.assertEqual(results["dataset"]["split_counts"], {"train": 1, "val": 1, "test": 1})

    def test_tiny_real_checkpoint_sampling_smoke_writes_reload_and_sampling_outputs(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not importable in this interpreter")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed" / "molecules.json"
            split_path = root / "splits" / "split.json"
            processed.parent.mkdir(parents=True)
            split_path.parent.mkdir(parents=True)
            molecules = [
                {
                    "molecule_id": f"geom_tiny_{index}",
                    "smiles": "CO",
                    "atom_symbols": ["C", "O", "H"],
                    "bonds": [[0, 1, 1], [0, 2, 1]],
                    "conformers": [
                        {
                            "conformer_id": str(index),
                            "coords": [
                                [0.0, 0.0, 0.0],
                                [1.2 + 0.1 * index, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                            ],
                        }
                    ],
                    "ring_atom_indices": [],
                    "rotatable_bond_count": 0,
                }
                for index in range(3)
            ]
            processed.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "result_kind": "processed_output_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "molecules": molecules,
                    }
                ),
                encoding="utf-8",
            )
            split_path.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "result_kind": "split_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "split_type": "random",
                        "degenerate_tiny_split": False,
                        "train": ["geom_tiny_0"],
                        "val": ["geom_tiny_1"],
                        "test": ["geom_tiny_2"],
                    }
                ),
                encoding="utf-8",
            )

            outputs = run_tiny_real_checkpoint_sampling_smoke_tests(
                processed_artifact_path=processed,
                split_path=split_path,
                output_root=root / "experiments",
                seed=0,
                max_molecules=3,
                max_steps=3,
            )

            self.assertEqual(set(outputs), {"isotropic", "stiffness_structured"})
            for mode, paths in outputs.items():
                manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                results = json.loads(paths["results"].read_text(encoding="utf-8"))
                self.assertTrue(paths["checkpoint"].exists())
                self.assertEqual(
                    manifest["result_kind"],
                    "tiny_real_data_checkpoint_sampling_smoke_not_benchmark",
                )
                self.assertEqual(
                    results["result_kind"],
                    "tiny_real_data_checkpoint_sampling_smoke_not_benchmark",
                )
                self.assertFalse(results["benchmark_result"])
                self.assertEqual(results["hardware"], "cpu_only")
                self.assertFalse(results["gpu_used"])
                self.assertTrue(results["training_executed"])
                self.assertTrue(results["bounded_training_smoke_executed"])
                self.assertFalse(results["larger_real_training_run_started"])
                self.assertEqual(results["max_steps"], 3)
                self.assertTrue(results["checkpoint_written"])
                self.assertTrue(results["checkpoint_reloaded"])
                self.assertEqual(results["checkpoint_scope"], "disposable_smoke_only")
                self.assertTrue(results["sampling_executed"])
                self.assertEqual(results["sample_step_count"], 1)
                self.assertEqual(results["corruption_mode"], mode)
                self.assertEqual(results["sample_count"], 3)
                self.assertEqual(results["metrics"]["checkpoint_path"], str(paths["checkpoint"]))
                self.assertTrue(results["metrics"]["checkpoint_written"])
                self.assertTrue(results["metrics"]["checkpoint_reloaded"])
                self.assertFalse(results["metrics"]["checkpoint_benchmark_result"])
                self.assertEqual(len(results["metrics"]["train_steps"]), 3)
                self.assertEqual(len(results["metrics"]["sampled_molecules"]), 3)
                self.assertEqual(results["dataset"]["split_counts"], {"train": 1, "val": 1, "test": 1})

    def test_tiny_real_pilot_writes_bounded_training_outputs(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not importable in this interpreter")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed" / "molecules.json"
            split_path = root / "splits" / "split.json"
            processed.parent.mkdir(parents=True)
            split_path.parent.mkdir(parents=True)
            molecules = [
                {
                    "molecule_id": f"geom_tiny_{index}",
                    "smiles": "CO",
                    "atom_symbols": ["C", "O", "H"],
                    "bonds": [[0, 1, 1], [0, 2, 1]],
                    "conformers": [
                        {
                            "conformer_id": str(index),
                            "coords": [
                                [0.0, 0.0, 0.0],
                                [1.2 + 0.1 * index, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                            ],
                        }
                    ],
                    "ring_atom_indices": [],
                    "rotatable_bond_count": 0,
                }
                for index in range(3)
            ]
            processed.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "result_kind": "processed_output_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "molecules": molecules,
                    }
                ),
                encoding="utf-8",
            )
            split_path.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "result_kind": "split_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "split_type": "random",
                        "degenerate_tiny_split": False,
                        "train": ["geom_tiny_0"],
                        "val": ["geom_tiny_1"],
                        "test": ["geom_tiny_2"],
                    }
                ),
                encoding="utf-8",
            )
            config_paths = []
            for mode, label in [("isotropic", "isotropic"), ("stiffness_structured", "stiffness")]:
                config_path = root / f"{label}.yaml"
                config_path.write_text(
                    f"""
experiment:
  name: geom_tiny_real_pilot_{label}_steps4_seed0
  seed: 0
  output_root: {root / "experiments"}

data:
  dataset_name: geom_drugs
  dataset_kind: geom_drugs_tiny_processed_dry_run
  raw_root: data/raw/geom_drugs
  processed_root: {processed.parent}
  split_root: {split_path.parent}
  processed_artifact_path: {processed}
  split_path: {split_path}
  split_type: random
  subset:
    max_molecules: 3
    max_conformers_per_mol: 1
    tag: tiny_real_data_bounded_pilot

model:
  backbone: minimal_linear_velocity_pilot
  pilot_model: single_linear_velocity_pilot
  hidden_dim: 0
  num_layers: 1
  time_embedding_dim: 1

flow_matching:
  objective: cartesian_linear
  interpolation: linear

corruption:
  mode: {mode}
  stiffness_table: heuristic_v1
  protect_rings: true
  protect_amides: true
  isotropic_sigma: 0.25
  bond_sigma: 0.02
  angle_sigma: 0.08
  torsion_sigma: 0.30

train:
  enabled: true
  pilot_only: true
  batch_scope: train_split_only_one_molecule
  max_steps: 4
  hard_max_steps: 8
  learning_rate: 1.0e-2
  num_workers: 0
  checkpointing_enabled: false

sample:
  enabled: false
  num_samples_per_mol: 0
  num_steps: 0

eval:
  enabled: true
  eval_subset_size: 3
  compute_train_val_test_losses: true
  compute_cov_mat: false
  compute_validity: false
  compute_energy_checks: false

logging:
  save_manifest: true
  save_results_json: true
  save_log: true
""",
                    encoding="utf-8",
                )
                config_paths.append(config_path)

            outputs = run_tiny_real_pilot_configs(config_paths=tuple(config_paths))

            self.assertEqual(set(outputs), {"isotropic", "stiffness_structured"})
            for mode, paths in outputs.items():
                manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                results = json.loads(paths["results"].read_text(encoding="utf-8"))
                self.assertEqual(manifest["result_kind"], "tiny_real_data_bounded_pilot_not_benchmark")
                self.assertEqual(results["result_kind"], "tiny_real_data_bounded_pilot_not_benchmark")
                self.assertFalse(results["benchmark_result"])
                self.assertEqual(results["hardware"], "cpu_only")
                self.assertFalse(results["gpu_used"])
                self.assertTrue(results["pilot_run"])
                self.assertTrue(results["bounded_pilot"])
                self.assertFalse(results["dry_run"])
                self.assertTrue(results["training_executed"])
                self.assertTrue(results["training_loop_executed"])
                self.assertTrue(results["bounded_real_training_loop_executed"])
                self.assertFalse(results["larger_real_training_run_started"])
                self.assertEqual(results["max_steps"], 4)
                self.assertEqual(results["hard_max_steps"], 8)
                self.assertEqual(results["optimizer_step_count"], 4)
                self.assertEqual(results["backward_pass_count"], 4)
                self.assertFalse(results["checkpoint_written"])
                self.assertFalse(results["sampling_executed"])
                self.assertEqual(results["corruption_mode"], mode)
                self.assertEqual(results["sample_count"], 3)
                self.assertEqual(results["metrics"]["optimizer_step_count"], 4)
                self.assertEqual(len(results["metrics"]["step_records"]), 4)
                self.assertEqual(results["metrics"]["split_sample_counts"], {"train": 1, "val": 1, "test": 1})
                self.assertEqual(set(results["metrics"]["final_split_losses"]), {"train", "val", "test"})
                self.assertGreater(results["metrics"]["trainable_parameter_count"], 0)
                self.assertEqual(results["dataset"]["split_counts"], {"train": 1, "val": 1, "test": 1})
                self.assertIn("not benchmark", " ".join(results["notes"]))

    def test_tiny_real_micro_pilot_accepts_six_molecule_subset(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not importable in this interpreter")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed" / "micro_pilot" / "molecules.json"
            split_path = root / "splits" / "random_seed0" / "split.json"
            processed.parent.mkdir(parents=True)
            split_path.parent.mkdir(parents=True)
            molecules = [
                {
                    "molecule_id": f"geom_micro_{index}",
                    "smiles": "CO",
                    "atom_symbols": ["C", "O", "H"],
                    "bonds": [[0, 1, 1], [0, 2, 1]],
                    "conformers": [
                        {
                            "conformer_id": str(index),
                            "coords": [
                                [0.0, 0.0, 0.0],
                                [1.2 + 0.1 * index, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                            ],
                        }
                    ],
                    "ring_atom_indices": [],
                    "rotatable_bond_count": 0,
                }
                for index in range(6)
            ]
            processed.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "subset_scope": "micro_pilot_processed_dry_run",
                        "result_kind": "processed_output_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "molecules": molecules,
                    }
                ),
                encoding="utf-8",
            )
            split_path.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "result_kind": "split_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "split_type": "random",
                        "split_scope": "micro_pilot_processed_dry_run",
                        "degenerate_tiny_split": False,
                        "train": [f"geom_micro_{index}" for index in range(4)],
                        "val": ["geom_micro_4"],
                        "test": ["geom_micro_5"],
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "micro.yaml"
            config_path.write_text(
                f"""
experiment:
  name: geom_micro_pilot_isotropic_mols6_steps3_seed0
  seed: 0
  output_root: {root / "experiments"}

data:
  dataset_name: geom_drugs
  dataset_kind: geom_drugs_tiny_processed_dry_run
  raw_root: data/raw/geom_drugs
  processed_root: {processed.parent}
  split_root: {split_path.parent}
  processed_artifact_path: {processed}
  split_path: {split_path}
  split_type: random
  subset:
    max_molecules: 6
    max_conformers_per_mol: 1
    tag: micro_pilot_processed_dry_run

model:
  backbone: minimal_linear_velocity_pilot
  pilot_model: single_linear_velocity_pilot
  hidden_dim: 0
  num_layers: 1
  time_embedding_dim: 1

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
  pilot_only: true
  batch_scope: micro_pilot_train_split
  max_steps: 3
  hard_max_steps: 8
  learning_rate: 1.0e-2
  num_workers: 0
  checkpointing_enabled: false

sample:
  enabled: false
  num_samples_per_mol: 0
  num_steps: 0

eval:
  enabled: true
  eval_subset_size: 6
  compute_train_val_test_losses: true
  compute_cov_mat: false
  compute_validity: false
  compute_energy_checks: false

logging:
  save_manifest: true
  save_results_json: true
  save_log: true
""",
                encoding="utf-8",
            )

            outputs = run_tiny_real_pilot_configs(config_paths=(config_path,))
            results = json.loads(outputs["isotropic"]["results"].read_text(encoding="utf-8"))
            self.assertTrue(results["micro_pilot_run"])
            self.assertEqual(results["sample_count"], 6)
            self.assertEqual(results["dataset"]["split_counts"], {"train": 4, "val": 1, "test": 1})
            self.assertEqual(results["metrics"]["split_sample_counts"], {"train": 4, "val": 1, "test": 1})
            self.assertEqual(len(results["metrics"]["step_records"]), 3)
            self.assertFalse(results["checkpoint_written"])
            self.assertFalse(results["sampling_executed"])

    def test_smoke_comparison_writes_summary_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            iso = output_root / "iso.json"
            stiff = output_root / "stiff.json"
            rdkit = output_root / "rdkit.json"
            common = {
                "result_kind": "smoke_only_not_benchmark",
                "benchmark_result": False,
                "hardware": "cpu_only",
                "slurm_account": None,
                "training_executed": False,
                "dataset": {"kind": "synthetic_smoke"},
            }
            iso.write_text(json.dumps({
                **common,
                "run_name": "iso",
                "corruption_mode": "isotropic",
                "sample_count": 2,
                "metrics": {
                    "cov_mat": {"cov_r_smoke": 0.5, "mat_r_smoke": 0.4},
                    "validity": {"finite_coordinate_rate": 1.0, "mean_abs_bond_length_delta": 0.2},
                },
            }), encoding="utf-8")
            stiff.write_text(json.dumps({
                **common,
                "run_name": "stiff",
                "corruption_mode": "stiffness_structured",
                "sample_count": 2,
                "metrics": {
                    "cov_mat": {"cov_r_smoke": 1.0, "mat_r_smoke": 0.3},
                    "validity": {"finite_coordinate_rate": 1.0, "mean_abs_bond_length_delta": 0.1},
                },
            }), encoding="utf-8")
            rdkit.write_text(json.dumps({
                **common,
                "run_name": "rdkit",
                "baseline_name": "rdkit_etkdg_v3_mmff",
                "execution_status": "completed",
                "rdkit_version": "test",
                "metrics": {
                    "attempted_molecule_count": 2,
                    "success_rate": 1.0,
                    "mean_aligned_rmsd_to_fixture": 0.05,
                    "mean_abs_bond_length_delta": 0.02,
                    "mean_mmff_energy": -1.0,
                },
            }), encoding="utf-8")

            paths = write_smoke_comparison(
                result_paths=[iso, stiff, rdkit],
                summary_path=output_root / "summary.json",
                report_path=output_root / "summary.md",
            )
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            report = paths["report"].read_text(encoding="utf-8")
            self.assertEqual(summary["result_kind"], "smoke_only_not_benchmark")
            self.assertFalse(summary["benchmark_result"])
            self.assertEqual(summary["row_count"], 3)
            self.assertIn("Smoke Comparison Summary", report)
            self.assertIn("not a benchmark result", report)

    def test_tiny_real_comparison_writes_summary_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            rdkit = output_root / "rdkit.json"
            iso = output_root / "iso.json"
            stiff = output_root / "stiff.json"
            common = {
                "benchmark_result": False,
                "hardware": "cpu_only",
                "scheduler": None,
                "slurm_account": None,
                "training_executed": False,
                "sampling_executed": False,
                "dataset": {
                    "kind": "geom_drugs_tiny_processed_dry_run",
                    "split_counts": {"train": 1, "val": 1, "test": 1},
                },
            }
            rdkit.write_text(json.dumps({
                **common,
                "run_name": "rdkit",
                "result_kind": "tiny_real_data_dry_run_not_benchmark",
                "baseline_name": "rdkit_etkdg_v3_mmff",
                "execution_status": "completed",
                "rdkit_version": "test",
                "metrics": {
                    "attempted_molecule_count": 3,
                    "success_rate": 1.0,
                    "mean_aligned_rmsd_to_tiny_real_conformer": 0.5,
                    "mean_abs_bond_length_delta": 0.02,
                    "mean_mmff_energy": -1.0,
                },
            }), encoding="utf-8")
            for path, mode, mat, bond_delta in [
                (iso, "isotropic", 0.4, 0.2),
                (stiff, "stiffness_structured", 0.3, 0.1),
            ]:
                path.write_text(json.dumps({
                    **common,
                    "run_name": mode,
                    "result_kind": "tiny_real_data_model_dry_run_not_benchmark",
                    "corruption_mode": mode,
                    "sample_count": 3,
                    "metrics": {
                        "cov_mat": {"cov_r_smoke": 1.0, "mat_r_smoke": mat},
                        "validity": {
                            "finite_coordinate_rate": 1.0,
                            "mean_abs_bond_length_delta": bond_delta,
                        },
                    },
                }), encoding="utf-8")

            paths = write_tiny_real_comparison(
                result_paths=[rdkit, iso, stiff],
                summary_path=output_root / "summary.json",
                report_path=output_root / "summary.md",
            )
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            report = paths["report"].read_text(encoding="utf-8")
            self.assertEqual(summary["result_kind"], "tiny_real_data_comparison_dry_run_not_benchmark")
            self.assertFalse(summary["benchmark_result"])
            self.assertFalse(summary["training_executed"])
            self.assertFalse(summary["sampling_executed"])
            self.assertEqual(summary["row_count"], 3)
            self.assertEqual(
                {row["method"] for row in summary["rows"]},
                {"rdkit_etkdg_v3_mmff", "isotropic", "stiffness_structured"},
            )
            self.assertIn("Tiny Real-Data Dry-Run Summary", report)
            self.assertIn("not a benchmark result", report)

    def test_tiny_real_lifecycle_summary_writes_stage_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            common = {
                "benchmark_result": False,
                "hardware": "cpu_only",
                "scheduler": None,
                "slurm_account": None,
                "dataset": {
                    "kind": "geom_drugs_tiny_processed_dry_run",
                    "molecule_count": 3,
                    "split_counts": {"train": 1, "val": 1, "test": 1},
                },
            }
            payloads = [
                {
                    **common,
                    "run_name": "rdkit",
                    "result_kind": "tiny_real_data_dry_run_not_benchmark",
                    "baseline_name": "rdkit_etkdg_v3_mmff",
                    "training_executed": False,
                    "sampling_executed": False,
                    "metrics": {
                        "attempted_molecule_count": 3,
                        "success_rate": 1.0,
                        "mean_aligned_rmsd_to_tiny_real_conformer": 0.5,
                        "mean_abs_bond_length_delta": 0.02,
                        "mean_mmff_energy": -1.0,
                    },
                },
                {
                    **common,
                    "run_name": "model_iso",
                    "result_kind": "tiny_real_data_model_dry_run_not_benchmark",
                    "corruption_mode": "isotropic",
                    "training_executed": False,
                    "sampling_executed": False,
                    "sample_count": 3,
                    "metrics": {
                        "cov_mat": {"mat_r_smoke": 0.4},
                        "validity": {"mean_abs_bond_length_delta": 0.1},
                    },
                },
                {
                    **common,
                    "run_name": "model_stiff",
                    "result_kind": "tiny_real_data_model_dry_run_not_benchmark",
                    "corruption_mode": "stiffness_structured",
                    "training_executed": False,
                    "sampling_executed": False,
                    "sample_count": 3,
                    "metrics": {
                        "cov_mat": {"mat_r_smoke": 0.3},
                        "validity": {"mean_abs_bond_length_delta": 0.05},
                    },
                },
                {
                    **common,
                    "run_name": "execution_iso",
                    "result_kind": "tiny_real_data_model_execution_smoke_not_benchmark",
                    "corruption_mode": "isotropic",
                    "gpu_used": False,
                    "training_executed": False,
                    "sampling_executed": False,
                    "sample_count": 3,
                    "metrics": {"sample_count": 3, "mean_velocity_mse": 0.2},
                },
                {
                    **common,
                    "run_name": "single_iso",
                    "result_kind": "tiny_real_data_single_step_training_smoke_not_benchmark",
                    "corruption_mode": "isotropic",
                    "gpu_used": False,
                    "training_executed": True,
                    "sampling_executed": False,
                    "checkpoint_written": False,
                    "sample_count": 3,
                    "optimizer_step_count": 1,
                    "backward_pass_count": 1,
                    "metrics": {
                        "sample_count": 3,
                        "loss_before_step": 0.5,
                        "loss_after_step": 0.4,
                        "optimizer_step_count": 1,
                        "backward_pass_count": 1,
                    },
                },
                {
                    **common,
                    "run_name": "multi_iso",
                    "result_kind": "tiny_real_data_bounded_multistep_training_smoke_not_benchmark",
                    "corruption_mode": "isotropic",
                    "gpu_used": False,
                    "training_executed": True,
                    "sampling_executed": False,
                    "checkpoint_written": False,
                    "larger_real_training_run_started": False,
                    "sample_count": 3,
                    "optimizer_step_count": 3,
                    "backward_pass_count": 3,
                    "metrics": {
                        "sample_count": 3,
                        "initial_loss": 0.5,
                        "final_loss": 0.3,
                        "optimizer_step_count": 3,
                        "backward_pass_count": 3,
                        "step_records": [{"optimizer_step": 1}, {"optimizer_step": 2}, {"optimizer_step": 3}],
                    },
                },
                {
                    **common,
                    "run_name": "ckpt_iso",
                    "result_kind": "tiny_real_data_checkpoint_sampling_smoke_not_benchmark",
                    "corruption_mode": "isotropic",
                    "gpu_used": False,
                    "training_executed": True,
                    "sampling_executed": True,
                    "checkpoint_written": True,
                    "checkpoint_reloaded": True,
                    "checkpoint_scope": "disposable_smoke_only",
                    "larger_real_training_run_started": False,
                    "sample_count": 3,
                    "optimizer_step_count": 3,
                    "backward_pass_count": 3,
                    "metrics": {
                        "sample_count": 3,
                        "initial_loss": 0.5,
                        "reloaded_loss": 0.35,
                        "optimizer_step_count": 3,
                        "backward_pass_count": 3,
                        "train_steps": [{"optimizer_step": 1}, {"optimizer_step": 2}, {"optimizer_step": 3}],
                        "sampled_molecules": [{"sampled_rmsd_to_target": 0.2}],
                    },
                },
            ]
            result_paths = []
            for index, payload in enumerate(payloads):
                path = output_root / f"result_{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                result_paths.append(path)

            paths = write_tiny_real_lifecycle_summary(
                result_paths=result_paths,
                summary_path=output_root / "summary.json",
                report_path=output_root / "summary.md",
            )
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            report = paths["report"].read_text(encoding="utf-8")
            self.assertEqual(summary["result_kind"], "tiny_real_data_lifecycle_smoke_summary_not_benchmark")
            self.assertFalse(summary["benchmark_result"])
            self.assertFalse(summary["larger_real_training_run_started"])
            self.assertEqual(summary["row_count"], len(payloads))
            self.assertEqual(summary["stage_counts"]["model_dry_run"], 2)
            self.assertEqual(summary["stage_counts"]["checkpoint_sampling_smoke"], 1)
            self.assertIn("Tiny Real-Data End-to-End Smoke Lifecycle Summary", report)
            self.assertIn("not a benchmark result", report)
            self.assertIn("checkpoint_sampling_smoke", report)

    def test_tiny_real_pilot_comparison_writes_summary_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            result_paths = []
            for mode, final_train in [("isotropic", 0.35), ("stiffness_structured", 0.30)]:
                path = output_root / f"{mode}.json"
                payload = {
                    "run_name": f"pilot_{mode}",
                    "result_kind": "tiny_real_data_bounded_pilot_not_benchmark",
                    "benchmark_result": False,
                    "hardware": "cpu_only",
                    "gpu_used": False,
                    "scheduler": None,
                    "slurm_account": None,
                    "pilot_run": True,
                    "bounded_pilot": True,
                    "training_executed": True,
                    "training_loop_executed": True,
                    "bounded_real_training_loop_executed": True,
                    "larger_real_training_run_started": False,
                    "checkpoint_written": False,
                    "sampling_executed": False,
                    "max_steps": 2,
                    "hard_max_steps": 8,
                    "optimizer_step_count": 2,
                    "backward_pass_count": 2,
                    "sample_count": 3,
                    "corruption_mode": mode,
                    "dataset": {
                        "kind": "geom_drugs_tiny_processed_dry_run",
                        "molecule_count": 3,
                        "split_type": "random",
                        "split_counts": {"train": 1, "val": 1, "test": 1},
                    },
                    "metrics": {
                        "metric_scope": "geom_drugs_tiny_real_bounded_pilot_training",
                        "sample_count": 3,
                        "split_sample_counts": {"train": 1, "val": 1, "test": 1},
                        "trainable_parameter_count": 15,
                        "optimizer_name": "SGD",
                        "learning_rate": 0.01,
                        "max_steps": 2,
                        "optimizer_step_count": 2,
                        "backward_pass_count": 2,
                        "initial_split_losses": {"train": 0.5, "val": 0.6, "test": 0.7},
                        "final_split_losses": {"train": final_train, "val": 0.45, "test": 0.55},
                        "step_records": [
                            {
                                "optimizer_step": 1,
                                "train_loss_before_step": 0.5,
                                "gradient_l2_norm": 1.0,
                                "split_losses_after_step": {"train": 0.4, "val": 0.5, "test": 0.6},
                            },
                            {
                                "optimizer_step": 2,
                                "train_loss_before_step": 0.4,
                                "gradient_l2_norm": 0.8,
                                "split_losses_after_step": {
                                    "train": final_train,
                                    "val": 0.45,
                                    "test": 0.55,
                                },
                            },
                        ],
                    },
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                result_paths.append(path)

            paths = write_tiny_real_pilot_comparison(
                result_paths=result_paths,
                summary_path=output_root / "summary.json",
                report_path=output_root / "summary.md",
            )
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            report = paths["report"].read_text(encoding="utf-8")
            self.assertEqual(summary["result_kind"], "tiny_real_data_bounded_pilot_comparison_not_benchmark")
            self.assertFalse(summary["benchmark_result"])
            self.assertFalse(summary["gpu_used"])
            self.assertFalse(summary["training_executed_by_summary"])
            self.assertFalse(summary["sampling_executed_by_summary"])
            self.assertFalse(summary["larger_real_training_run_started"])
            self.assertEqual(summary["row_count"], 2)
            self.assertEqual(
                {row["corruption_mode"] for row in summary["rows"]},
                {"isotropic", "stiffness_structured"},
            )
            for row in summary["rows"]:
                self.assertEqual(row["split_counts"], {"train": 1, "val": 1, "test": 1})
                self.assertEqual(len(row["step_loss_trace"]), 2)
                self.assertFalse(row["safety_labels"]["benchmark_result"])
                self.assertTrue(row["safety_labels"]["bounded_pilot"])
                self.assertFalse(row["safety_labels"]["checkpoint_written"])
                self.assertFalse(row["safety_labels"]["sampling_executed"])
            self.assertIn("Tiny Real-Data Bounded Pilot Summary", report)
            self.assertIn("not a benchmark result", report)
            self.assertIn("Per-step train loss trace", report)

    def test_micro_pilot_comparison_writes_summary_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            result_paths = []
            for mode, final_train in [("isotropic", 0.43), ("stiffness_structured", 0.40)]:
                path = output_root / f"{mode}.json"
                payload = {
                    "run_name": f"micro_{mode}",
                    "result_kind": "tiny_real_data_bounded_pilot_not_benchmark",
                    "benchmark_result": False,
                    "hardware": "cpu_only",
                    "gpu_used": False,
                    "scheduler": None,
                    "slurm_account": None,
                    "pilot_run": True,
                    "bounded_pilot": True,
                    "micro_pilot_run": True,
                    "training_executed": True,
                    "training_loop_executed": True,
                    "bounded_real_training_loop_executed": True,
                    "larger_real_training_run_started": False,
                    "checkpoint_written": False,
                    "sampling_executed": False,
                    "max_steps": 2,
                    "hard_max_steps": 8,
                    "optimizer_step_count": 2,
                    "backward_pass_count": 2,
                    "sample_count": 6,
                    "corruption_mode": mode,
                    "dataset": {
                        "kind": "geom_drugs_tiny_processed_dry_run",
                        "subset_scope": "micro_pilot_processed_dry_run",
                        "molecule_count": 6,
                        "split_type": "random",
                        "split_counts": {"train": 4, "val": 1, "test": 1},
                    },
                    "metrics": {
                        "metric_scope": "geom_drugs_tiny_real_bounded_pilot_training",
                        "sample_count": 6,
                        "split_sample_counts": {"train": 4, "val": 1, "test": 1},
                        "trainable_parameter_count": 15,
                        "optimizer_name": "SGD",
                        "learning_rate": 0.01,
                        "max_steps": 2,
                        "optimizer_step_count": 2,
                        "backward_pass_count": 2,
                        "initial_split_losses": {"train": 0.7, "val": 0.6, "test": 0.5},
                        "final_split_losses": {"train": final_train, "val": 0.42, "test": 0.33},
                        "step_records": [
                            {
                                "optimizer_step": 1,
                                "train_loss_before_step": 0.7,
                                "gradient_l2_norm": 1.1,
                                "split_losses_after_step": {"train": 0.55, "val": 0.5, "test": 0.4},
                            },
                            {
                                "optimizer_step": 2,
                                "train_loss_before_step": 0.55,
                                "gradient_l2_norm": 0.9,
                                "split_losses_after_step": {
                                    "train": final_train,
                                    "val": 0.42,
                                    "test": 0.33,
                                },
                            },
                        ],
                    },
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                result_paths.append(path)

            paths = write_tiny_real_pilot_comparison(
                result_paths=result_paths,
                summary_path=output_root / "summary.json",
                report_path=output_root / "summary.md",
                comparison_scope="micro_pilot",
            )
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            report = paths["report"].read_text(encoding="utf-8")
            self.assertEqual(
                summary["result_kind"],
                "tiny_real_data_bounded_micro_pilot_comparison_not_benchmark",
            )
            self.assertEqual(summary["comparison_scope"], "micro_pilot")
            self.assertFalse(summary["benchmark_result"])
            self.assertFalse(summary["gpu_used"])
            self.assertFalse(summary["training_executed_by_summary"])
            self.assertFalse(summary["sampling_executed_by_summary"])
            self.assertFalse(summary["larger_real_training_run_started"])
            self.assertEqual(summary["row_count"], 2)
            for row in summary["rows"]:
                self.assertTrue(row["micro_pilot_run"])
                self.assertEqual(row["dataset_subset_scope"], "micro_pilot_processed_dry_run")
                self.assertEqual(row["split_counts"], {"train": 4, "val": 1, "test": 1})
                self.assertEqual(len(row["step_loss_trace"]), 2)
                self.assertFalse(row["safety_labels"]["benchmark_result"])
                self.assertTrue(row["safety_labels"]["bounded_pilot"])
                self.assertTrue(row["safety_labels"]["micro_pilot_run"])
                self.assertFalse(row["safety_labels"]["checkpoint_written"])
                self.assertFalse(row["safety_labels"]["sampling_executed"])
            self.assertIn("Tiny Real-Data Bounded Micro-Pilot Summary", report)
            self.assertIn("micro_pilot", report)
            self.assertIn("not a benchmark result", report)

    def test_geom_preflight_reports_missing_raw_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = run_geom_preflight(
                raw_root=root / "missing_geom",
                processed_root=root / "processed",
                manifest_path=root / "manifest.json",
                report_path=root / "report.md",
            )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            report = paths["report"].read_text(encoding="utf-8")
            self.assertEqual(manifest["result_kind"], "preflight_not_benchmark")
            self.assertFalse(manifest["benchmark_result"])
            self.assertFalse(manifest["raw_data_modified"])
            self.assertFalse(manifest["processed_data_written"])
            self.assertEqual(manifest["raw_summary"]["status"], "missing_root")
            self.assertIn("GEOM-Drugs Preflight", report)

    def test_geom_preflight_detects_candidate_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "geom_drugs"
            (raw / "rdkit_folder" / "drugs").mkdir(parents=True)
            (raw / "summary_drugs.json").write_text("{}", encoding="utf-8")
            paths = run_geom_preflight(
                raw_root=raw,
                processed_root=root / "processed",
                manifest_path=root / "manifest.json",
                report_path=root / "report.md",
            )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["raw_summary"]["status"], "present_candidate_layout")
            self.assertIn("summary_drugs.json", manifest["raw_summary"]["present_layout_hints"])
            self.assertIn("rdkit_folder", manifest["raw_summary"]["present_layout_hints"])
            self.assertFalse((root / "processed").exists())

    def test_geom_preflight_reads_bounded_pickle_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "geom_drugs"
            drugs = raw / "rdkit_folder" / "drugs"
            drugs.mkdir(parents=True)
            for index in range(2):
                record = {
                    "smiles": f"C{index}",
                    "totalconfs": 2,
                    "uniqueconfs": 2,
                    "charge": 0,
                    "datasets": ["unit"],
                    "conformers": [
                        {
                            "geom_id": index * 10 + conformer_index,
                            "set": 1,
                            "relativeenergy": float(conformer_index),
                            "rd_mol": {"not": "serialized to manifest"},
                        }
                        for conformer_index in range(2)
                    ],
                }
                with (drugs / f"mol_{index}.pickle").open("wb") as handle:
                    pickle.dump(record, handle)

            paths = run_geom_preflight(
                raw_root=raw,
                processed_root=root / "processed",
                manifest_path=root / "manifest.json",
                report_path=root / "report.md",
                max_dry_run_molecules=1,
                max_dry_run_conformers_per_mol=1,
            )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            dry_run = manifest["preprocessing_dry_run"]
            self.assertEqual(dry_run["status"], "completed_metadata_only")
            self.assertEqual(dry_run["molecule_metadata_records_read"], 1)
            self.assertEqual(dry_run["conformer_metadata_records_read"], 1)
            self.assertFalse(dry_run["processed_data_written"])
            sample = dry_run["sample_records"][0]
            self.assertEqual(sample["conformer_count_in_file"], 2)
            self.assertEqual(sample["sampled_conformer_count"], 1)
            self.assertIn("smiles", sample["molecule_metadata"])
            self.assertNotIn("conformers", sample["molecule_metadata"])
            self.assertNotIn("rd_mol", sample["sampled_conformer_metadata"][0])
            self.assertFalse((root / "processed").exists())

    def test_geom_tiny_processed_output_writes_bounded_nonbenchmark_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "geom_drugs"
            drugs = raw / "rdkit_folder" / "drugs"
            processed = root / "processed"
            drugs.mkdir(parents=True)
            for index in range(3):
                record = {
                    "smiles": "CO",
                    "totalconfs": 2,
                    "uniqueconfs": 2,
                    "charge": 0,
                    "conformers": [
                        {
                            "geom_id": 100 + index,
                            "relativeenergy": 0.0,
                            "rd_mol": _FakeMol([[0.0, 0.0, 0.0], [1.2 + index, 0.0, 0.0]]),
                        },
                        {
                            "geom_id": 200 + index,
                            "relativeenergy": 0.1,
                            "rd_mol": _FakeMol([[0.0, 0.1, 0.0], [1.1 + index, 0.0, 0.0]]),
                        },
                    ],
                }
                with (drugs / f"tiny_mol_{index}.pickle").open("wb") as handle:
                    pickle.dump(record, handle)

            paths = run_geom_preflight(
                raw_root=raw,
                processed_root=processed,
                manifest_path=root / "manifest.json",
                report_path=root / "report.md",
                write_tiny_processed_output=True,
                max_processed_conformers_per_mol=1,
            )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            artifact = json.loads(paths["processed_artifact"].read_text(encoding="utf-8"))
            processed_manifest = json.loads(paths["processed_manifest"].read_text(encoding="utf-8"))
            self.assertTrue(manifest["processed_output_dry_run_written"])
            self.assertEqual(artifact["dataset_kind"], "geom_drugs_tiny_processed_dry_run")
            self.assertEqual(artifact["result_kind"], "processed_output_dry_run_not_benchmark")
            self.assertFalse(artifact["benchmark_result"])
            self.assertEqual(artifact["bounds"]["max_molecules"], 3)
            self.assertEqual(artifact["molecule_count"], 3)
            self.assertEqual(artifact["conformer_count"], 3)
            molecule = artifact["molecules"][0]
            self.assertEqual(molecule["atom_symbols"], ["C", "O"])
            self.assertEqual(molecule["bonds"], [[0, 1, 1]])
            self.assertEqual(len(molecule["conformers"]), 1)
            self.assertNotIn("rd_mol", json.dumps(artifact))
            self.assertEqual(processed_manifest["processed_data_written"], True)
            self.assertEqual(processed_manifest["full_processed_data_written"], False)

    def test_geom_tiny_split_dry_run_writes_nonbenchmark_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed_artifact = root / "processed" / "tiny_real_dry_run" / "molecules.json"
            processed_artifact.parent.mkdir(parents=True)
            processed_artifact.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "result_kind": "processed_output_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "molecules": [
                            {
                                "molecule_id": f"geom_tiny_{index}",
                                "smiles": "CO",
                                "conformers": [{"conformer_id": "100", "coords": [[0, 0, 0], [1, 0, 0]]}],
                            }
                            for index in range(3)
                        ],
                    }
                ),
                encoding="utf-8",
            )

            paths = run_geom_tiny_split_dry_run(
                processed_artifact_path=processed_artifact,
                split_root=root / "splits",
                split_type="random",
                seed=0,
                max_molecules=3,
            )
            split = json.loads(paths["split"].read_text(encoding="utf-8"))
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(split["result_kind"], "split_dry_run_not_benchmark")
            self.assertFalse(split["benchmark_result"])
            self.assertFalse(split["degenerate_tiny_split"])
            self.assertEqual(len(split["train"]), 1)
            self.assertEqual(len(split["val"]), 1)
            self.assertEqual(len(split["test"]), 1)
            self.assertEqual(
                sorted(split["train"] + split["val"] + split["test"]),
                ["geom_tiny_0", "geom_tiny_1", "geom_tiny_2"],
            )
            self.assertEqual(manifest["result_kind"], "split_dry_run_not_benchmark")
            self.assertFalse(manifest["benchmark_result"])
            self.assertFalse(manifest["training_executed"])
            self.assertFalse(manifest["raw_data_modified"])
            self.assertFalse(manifest["processed_data_modified"])
            self.assertEqual(manifest["counts"], {"train": 1, "val": 1, "test": 1})

    def test_geom_micro_pilot_processed_output_and_split_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "geom_drugs"
            drugs = raw / "rdkit_folder" / "drugs"
            processed = root / "processed"
            drugs.mkdir(parents=True)
            for index in range(6):
                record = {
                    "smiles": "CO",
                    "totalconfs": 1,
                    "uniqueconfs": 1,
                    "charge": 0,
                    "conformers": [
                        {
                            "geom_id": 100 + index,
                            "relativeenergy": 0.0,
                            "rd_mol": _FakeMol([[0.0, 0.0, 0.0], [1.2 + index, 0.0, 0.0]]),
                        }
                    ],
                }
                with (drugs / f"micro_mol_{index}.pickle").open("wb") as handle:
                    pickle.dump(record, handle)

            paths = run_geom_preflight(
                raw_root=raw,
                processed_root=processed,
                manifest_path=root / "manifest.json",
                report_path=root / "report.md",
                write_micro_pilot_output=True,
                max_processed_conformers_per_mol=1,
            )
            artifact = json.loads(paths["processed_artifact"].read_text(encoding="utf-8"))
            processed_manifest = json.loads(paths["processed_manifest"].read_text(encoding="utf-8"))
            self.assertEqual(artifact["subset_scope"], "micro_pilot_processed_dry_run")
            self.assertEqual(artifact["bounds"]["max_molecules"], 6)
            self.assertEqual(artifact["bounds"]["hard_max_molecules"], 6)
            self.assertEqual(artifact["molecule_count"], 6)
            self.assertEqual(artifact["conformer_count"], 6)
            self.assertFalse(artifact["benchmark_result"])
            self.assertFalse(processed_manifest["full_processed_data_written"])

            split_paths = run_geom_tiny_split_dry_run(
                processed_artifact_path=paths["processed_artifact"],
                split_root=root / "splits",
                split_type="random",
                seed=0,
                max_molecules=6,
                split_scope="micro_pilot_processed_dry_run",
            )
            split = json.loads(split_paths["split"].read_text(encoding="utf-8"))
            manifest = json.loads(split_paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(split["split_scope"], "micro_pilot_processed_dry_run")
            self.assertFalse(split["benchmark_result"])
            self.assertFalse(split["degenerate_tiny_split"])
            self.assertEqual(len(split["train"]), 4)
            self.assertEqual(len(split["val"]), 1)
            self.assertEqual(len(split["test"]), 1)
            self.assertEqual(manifest["bounds"]["hard_max_molecules"], 6)
            self.assertEqual(manifest["counts"], {"train": 4, "val": 1, "test": 1})


if __name__ == "__main__":
    unittest.main()
