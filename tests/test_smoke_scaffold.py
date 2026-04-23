from __future__ import annotations

import json
import copy
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.corruption.harmonic import HarmonicPriorCorruptor
from src.corruption.isotropic import IsotropicCorruptor
from src.corruption.stiffness_structured import StiffnessStructuredCorruptor
from src.corruption.torsion_only import TorsionOnlyCorruptor
from src.baselines import rdkit_etkdg
from src.baselines.rdkit_etkdg import RdkitSmokeError, run_rdkit_smoke, run_rdkit_tiny_real_dry_run
from src.data.records import load_records
from src.data.splits import make_random_split
from src.data.geom_full_database import (
    FULL_DATABASE_SCAFFOLD_SPLIT_PATH,
    FULL_DATABASE_SCOPE,
    prepare_geom_full_database_artifact,
    prepare_geom_full_database_scaffold_split,
    prepare_geom_full_database_split,
)
from src.data.geom_preflight import run_geom_preflight
from src.data.geom_split_dry_run import make_tiny_split, run_geom_tiny_split_dry_run
from src.eval.smoke_compare import write_smoke_comparison
from src.eval.micro_pilot_checkpoint_sampling_compare import (
    write_micro_pilot_checkpoint_sampling_comparison,
    write_micro_pilot_checkpoint_sampling_multiseed_summary,
)
from src.eval.data_scale_convergence_diagnostics import write_data_scale_check_convergence_diagnostics
from src.eval.formal_full_database_compare import (
    MULTI_SEED_SUMMARY_RESULT_KIND as FORMAL_FULL_DATABASE_MULTI_SEED_SUMMARY_RESULT_KIND,
    SCAFFOLD_MULTI_SEED_SUMMARY_RESULT_KIND as FORMAL_FULL_DATABASE_SCAFFOLD_MULTI_SEED_SUMMARY_RESULT_KIND,
    SCAFFOLD_SEED0_SUMMARY_RESULT_KIND as FORMAL_FULL_DATABASE_SCAFFOLD_SEED0_SUMMARY_RESULT_KIND,
    SUMMARY_RESULT_KIND as FORMAL_FULL_DATABASE_SUMMARY_RESULT_KIND,
    write_formal_full_database_multiseed_summary,
    write_formal_full_database_scaffold_multiseed_summary,
    write_formal_full_database_scaffold_seed0_comparison,
    write_formal_full_database_seed0_comparison,
)
from src.eval.formal_attribution import (
    ATTRIBUTION_SUMMARY_RESULT_KIND,
    write_formal_attribution_phase_summary,
)
from src.eval.tiny_real_compare import write_tiny_real_comparison
from src.eval.tiny_real_lifecycle import write_tiny_real_lifecycle_summary
from src.eval.tiny_real_pilot_compare import write_micro_pilot_multiseed_summary, write_tiny_real_pilot_comparison
from src.models.checkpoint_sampling_smoke import (
    run_bounded_checkpoint_sampling_pilot,
    run_validation_checkpoint_training,
)
from src.train.formal_10000_training import (
    FORMAL_10000_CHECKPOINT_SCOPE,
    FORMAL_10000_MAX_STEPS,
    FORMAL_10000_RESULT_KIND,
    FORMAL_10000_SEED_SET,
    FORMAL_10000_SPLIT_COUNTS,
    _validate_formal_10000_config,
    write_formal_10000_training_multiseed_summary,
)
from src.train.formal_full_database_training import (
    FORMAL_FULL_DATABASE_CHECKPOINT_SCOPE,
    FORMAL_FULL_DATABASE_SEED,
    FORMAL_FULL_DATABASE_SEED_SET,
    validate_formal_full_database_config,
)
from src.train.pipeline import run_smoke_pipeline
from src.train.smoke import run_smoke
from src.train.tiny_real import run_tiny_real_model_dry_runs
from src.train.tiny_real_checkpoint_sampling import (
    run_data_scale_check_checkpoint_sampling_configs,
    run_data_scale_check_checkpoint_sampling_multiseed,
    run_micro_pilot_checkpoint_sampling_configs,
    run_micro_pilot_checkpoint_sampling_multiseed,
    run_tiny_real_checkpoint_sampling_smoke_tests,
    write_data_scale_check_checkpoint_sampling_comparison,
    write_data_scale_check_checkpoint_sampling_multiseed_summary,
)
from src.train.tiny_real_execution import run_tiny_real_execution_smoke_tests
from src.train.tiny_real_multi_step import run_tiny_real_multi_step_smoke_tests
from src.train.tiny_real_pilot import run_micro_pilot_multiseed, run_tiny_real_pilot_configs
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

    def test_harmonic_corruption_is_seed_deterministic(self) -> None:
        config = load_config("configs/train/tiny_smoke.yaml")
        config["corruption"]["mode"] = "harmonic_prior"
        config["corruption"]["harmonic_smoothing_steps"] = 2
        record = load_records("data/processed/smoke_fixture")[0]
        coords = record["conformers"][0]["coords"]
        first = HarmonicPriorCorruptor(config, seed=11).corrupt(record, coords)
        second = HarmonicPriorCorruptor(config, seed=11).corrupt(record, coords)
        self.assertEqual(first, second)
        self.assertNotEqual(first, coords)
        self.assertEqual(len(first), len(coords))

    def test_torsion_only_corruption_preserves_bond_lengths(self) -> None:
        config = load_config("configs/train/tiny_smoke.yaml")
        config["corruption"]["mode"] = "torsion_only"
        config["corruption"]["torsion_only_max_bonds"] = 1
        record = {
            "molecule_id": "torsion_fixture",
            "bonds": [[0, 1, 1.0], [1, 2, 1.0], [2, 3, 1.0]],
            "ring_atom_indices": [],
        }
        coords = [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        corrupted = TorsionOnlyCorruptor(config, seed=3).corrupt(record, coords)
        self.assertNotEqual(corrupted, coords)
        for i, j, _ in record["bonds"]:
            before = sum((coords[i][axis] - coords[j][axis]) ** 2 for axis in range(3)) ** 0.5
            after = sum((corrupted[i][axis] - corrupted[j][axis]) ** 2 for axis in range(3)) ** 0.5
            self.assertAlmostEqual(before, after, places=7)

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

    def test_micro_pilot_multiseed_runs_bounded_seed_set(self) -> None:
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
            config_paths = []
            for mode, label in [("isotropic", "isotropic"), ("stiffness_structured", "stiffness")]:
                config_path = root / f"{label}.yaml"
                config_path.write_text(
                    f"""
experiment:
  name: geom_micro_pilot_{label}_mols6_steps2_seed0
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
  batch_scope: micro_pilot_train_split
  max_steps: 2
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
                config_paths.append(config_path)

            outputs = run_micro_pilot_multiseed(seed_set=(0, 1), config_paths=tuple(config_paths))
            self.assertEqual(set(outputs), {"isotropic", "stiffness_structured"})
            result_paths = []
            for mode in outputs:
                self.assertEqual(set(outputs[mode]), {0, 1})
                for seed, paths in outputs[mode].items():
                    results = json.loads(paths["results"].read_text(encoding="utf-8"))
                    result_paths.append(paths["results"])
                    self.assertEqual(results["random_seed"], seed)
                    self.assertTrue(results["micro_pilot_run"])
                    self.assertEqual(results["sample_count"], 6)
                    self.assertEqual(results["dataset"]["split_counts"], {"train": 4, "val": 1, "test": 1})
                    self.assertFalse(results["checkpoint_written"])
                    self.assertFalse(results["sampling_executed"])

            summary_paths = write_micro_pilot_multiseed_summary(
                result_paths=result_paths,
                summary_path=root / "summary.json",
                report_path=root / "summary.md",
                expected_seeds=(0, 1),
            )
            summary = json.loads(summary_paths["summary"].read_text(encoding="utf-8"))
            self.assertEqual(
                summary["result_kind"],
                "tiny_real_data_bounded_micro_pilot_multiseed_summary_not_benchmark",
            )
            self.assertEqual(summary["seed_set"], [0, 1])
            self.assertEqual(summary["row_count"], 4)
            self.assertEqual(summary["mode_statistics"]["isotropic"]["seed_count"], 2)

    def test_micro_pilot_checkpoint_sampling_runs_single_seed_six_molecule_subset(self) -> None:
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
            config_paths = []
            for mode, label in [("isotropic", "isotropic"), ("stiffness_structured", "stiffness")]:
                config_path = root / f"{label}.yaml"
                config_path.write_text(
                    f"""
experiment:
  name: geom_micro_pilot_ckpt_sample_{label}_mols6_steps2_seed0
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
  batch_scope: micro_pilot_train_split
  max_steps: 2
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
                config_paths.append(config_path)

            outputs = run_micro_pilot_checkpoint_sampling_configs(config_paths=tuple(config_paths), max_steps=2)
            self.assertEqual(set(outputs), {"isotropic", "stiffness_structured"})
            for mode, paths in outputs.items():
                manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                results = json.loads(paths["results"].read_text(encoding="utf-8"))
                self.assertTrue(paths["checkpoint"].exists())
                self.assertEqual(
                    manifest["result_kind"],
                    "tiny_real_data_bounded_micro_pilot_checkpoint_sampling_not_benchmark",
                )
                self.assertEqual(
                    results["result_kind"],
                    "tiny_real_data_bounded_micro_pilot_checkpoint_sampling_not_benchmark",
                )
                self.assertFalse(results["benchmark_result"])
                self.assertEqual(results["hardware"], "cpu_only")
                self.assertFalse(results["gpu_used"])
                self.assertTrue(results["pilot_run"])
                self.assertTrue(results["bounded_pilot"])
                self.assertTrue(results["micro_pilot_run"])
                self.assertTrue(results["training_executed"])
                self.assertTrue(results["bounded_real_training_loop_executed"])
                self.assertFalse(results["larger_real_training_run_started"])
                self.assertEqual(results["max_steps"], 2)
                self.assertEqual(results["hard_max_steps"], 5)
                self.assertTrue(results["checkpoint_written"])
                self.assertTrue(results["checkpoint_reloaded"])
                self.assertEqual(results["checkpoint_scope"], "disposable_micro_pilot_checkpoint_sampling_only")
                self.assertTrue(results["sampling_executed"])
                self.assertEqual(results["sample_step_count"], 1)
                self.assertEqual(results["corruption_mode"], mode)
                self.assertEqual(results["training_sample_count"], 4)
                self.assertEqual(results["sample_count"], 6)
                self.assertEqual(results["dataset"]["split_counts"], {"train": 4, "val": 1, "test": 1})
                self.assertEqual(results["metrics"]["training_sample_count"], 4)
                self.assertEqual(results["metrics"]["sample_count"], 6)
                self.assertEqual(len(results["metrics"]["train_steps"]), 2)
                self.assertEqual(len(results["metrics"]["sampled_molecules"]), 6)
                self.assertEqual(results["metrics"]["checkpoint_path"], str(paths["checkpoint"]))
                self.assertFalse(results["metrics"]["checkpoint_benchmark_result"])
                self.assertEqual(
                    results["runtime_scope"],
                    "checkpoint_train_save_reload_sample_after_full_cpu_checkpoint_sampling_warmup",
                )
                self.assertTrue(results["runtime_seconds_reliable"])

    def test_data_scale_check_checkpoint_sampling_accepts_thirty_molecule_subset(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not importable in this interpreter")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed" / "data_scale_check" / "molecules.json"
            split_path = root / "splits" / "random_seed0" / "split.json"
            processed.parent.mkdir(parents=True)
            split_path.parent.mkdir(parents=True)
            molecules = [
                {
                    "molecule_id": f"geom_scale_{index:02d}",
                    "smiles": "CO",
                    "atom_symbols": ["C", "O", "H"],
                    "bonds": [[0, 1, 1], [0, 2, 1]],
                    "conformers": [
                        {
                            "conformer_id": str(index),
                            "coords": [
                                [0.0, 0.0, 0.0],
                                [1.2 + 0.02 * index, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                            ],
                        }
                    ],
                    "ring_atom_indices": [],
                    "rotatable_bond_count": 0,
                }
                for index in range(30)
            ]
            processed.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "subset_scope": "data_scale_check_processed_dry_run",
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
                        "split_scope": "data_scale_check_processed_dry_run",
                        "degenerate_tiny_split": False,
                        "train": [f"geom_scale_{index:02d}" for index in range(28)],
                        "val": ["geom_scale_28"],
                        "test": ["geom_scale_29"],
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
  name: geom_data_scale_check_ckpt_sample_{label}_mols30_steps2_seed0
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
    max_molecules: 30
    max_conformers_per_mol: 1
    tag: data_scale_check_processed_dry_run

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
  batch_scope: data_scale_check_train_split
  max_steps: 2
  hard_max_steps: 5
  learning_rate: 1.0e-2
  num_workers: 0
  checkpointing_enabled: false

sample:
  enabled: false
  num_samples_per_mol: 0
  num_steps: 0

eval:
  enabled: true
  eval_subset_size: 30
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

            outputs = run_data_scale_check_checkpoint_sampling_configs(config_paths=tuple(config_paths), max_steps=2)
            result_paths = []
            for mode, paths in outputs.items():
                results = json.loads(paths["results"].read_text(encoding="utf-8"))
                result_paths.append(paths["results"])
                self.assertEqual(
                    results["result_kind"],
                    "tiny_real_data_bounded_data_scale_check_checkpoint_sampling_not_benchmark",
                )
                self.assertTrue(results["data_scale_check_run"])
                self.assertFalse(results["micro_pilot_run"])
                self.assertFalse(results["benchmark_result"])
                self.assertEqual(results["hardware"], "cpu_only")
                self.assertFalse(results["gpu_used"])
                self.assertEqual(results["max_steps"], 2)
                self.assertEqual(results["training_sample_count"], 28)
                self.assertEqual(results["sample_count"], 30)
                self.assertEqual(results["dataset"]["split_counts"], {"train": 28, "val": 1, "test": 1})
                self.assertEqual(results["checkpoint_scope"], "disposable_data_scale_check_checkpoint_sampling_only")
                self.assertEqual(len(results["metrics"]["sampled_molecules"]), 30)

            comparison = write_data_scale_check_checkpoint_sampling_comparison(
                result_paths=result_paths,
                output_root=root / "experiments",
                seed=0,
                max_molecules=30,
                max_steps=2,
            )
            comparison_results = json.loads(comparison["results"].read_text(encoding="utf-8"))
            self.assertEqual(
                comparison_results["result_kind"],
                "tiny_real_data_bounded_data_scale_check_checkpoint_sampling_comparison_not_benchmark",
            )
            self.assertTrue(comparison_results["data_scale_check_run"])
            self.assertFalse(comparison_results["benchmark_result"])
            self.assertEqual(len(comparison_results["metric_comparisons"]), 5)

    def test_data_scale_check_checkpoint_sampling_multiseed_summary_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_paths = []
            molecule_ids = [f"geom_scale_{index:02d}" for index in range(30)]
            split_by_index = ["train"] * 28 + ["val", "test"]
            for seed in [0, 1, 2]:
                for mode, label, offset in [
                    ("isotropic", "isotropic", 0.30),
                    ("stiffness_structured", "stiffness", 0.20),
                ]:
                    path = root / f"{label}_seed{seed}.json"
                    sampled = [
                        offset + 0.001 * seed + 0.001 * index
                        for index in range(30)
                    ]
                    payload = {
                        "run_name": f"geom_data_scale_check_ckpt_sample_{label}_mols30_steps5_seed{seed}",
                        "dataset_name": "geom_drugs",
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "dataset_subset_scope": "data_scale_check_processed_dry_run",
                        "processed_artifact_path": "data/processed/geom_drugs_subset/data_scale_check/molecules.json",
                        "split_path": "data/splits/geom_drugs_data_scale_check/random_seed0/split.json",
                        "random_seed": seed,
                        "corruption_mode": mode,
                        "model_name": "single_linear_velocity_smoke",
                        "backbone": "minimal_linear_velocity_pilot",
                        "execution_mode": "local_cpu_tiny_real_data_scale_check_checkpoint_sampling",
                        "result_kind": "tiny_real_data_bounded_data_scale_check_checkpoint_sampling_not_benchmark",
                        "benchmark_result": False,
                        "hardware": "cpu_only",
                        "gpu_used": False,
                        "scheduler": None,
                        "slurm_account": None,
                        "pilot_run": True,
                        "bounded_pilot": True,
                        "micro_pilot_run": False,
                        "data_scale_check_run": True,
                        "dry_run": False,
                        "training_executed": True,
                        "model_training_started": True,
                        "training_loop_executed": True,
                        "bounded_real_training_loop_executed": True,
                        "larger_real_training_run_started": False,
                        "max_steps": 5,
                        "hard_max_steps": 5,
                        "optimizer_created": True,
                        "optimizer_step_count": 5,
                        "backward_executed": True,
                        "backward_pass_count": 5,
                        "checkpoint_path": f"experiments/checkpoints/{label}_seed{seed}.pt",
                        "checkpoint_written": True,
                        "checkpoint_reloaded": True,
                        "checkpoint_scope": "disposable_data_scale_check_checkpoint_sampling_only",
                        "sampling_executed": True,
                        "sample_step_count": 1,
                        "sample_step_size": 0.1,
                        "forward_executed": True,
                        "loss_computed": True,
                        "training_sample_count": 28,
                        "sample_count": 30,
                        "runtime_scope": "checkpoint_train_save_reload_sample_after_full_cpu_checkpoint_sampling_warmup",
                        "runtime_seconds_reliable": True,
                        "dataset": {
                            "name": "geom_drugs",
                            "kind": "geom_drugs_tiny_processed_dry_run",
                            "subset_scope": "data_scale_check_processed_dry_run",
                            "molecule_count": 30,
                            "split_type": "random",
                            "split_scope": "data_scale_check_processed_dry_run",
                            "split_counts": {"train": 28, "val": 1, "test": 1},
                            "degenerate_tiny_split": False,
                        },
                        "flow_matching": {"objective": "cartesian_linear", "interpolation": "linear"},
                        "metrics": {
                            "metric_scope": "geom_drugs_data_scale_check_checkpoint_sampling",
                            "model_name": "single_linear_velocity_smoke",
                            "reloaded_loss": 0.60 + 0.01 * seed if mode == "isotropic" else 0.50 + 0.01 * seed,
                            "train_steps": [{"optimizer_step": index + 1} for index in range(5)],
                            "sampled_molecules": [
                                {
                                    "molecule_id": molecule_id,
                                    "split": split,
                                    "atom_count": 3,
                                    "sample_step_count": 1,
                                    "sample_step_size": 0.1,
                                    "sampled_rmsd_to_target": rmsd,
                                    "predicted_velocity_norm": 1.0,
                                }
                                for molecule_id, split, rmsd in zip(molecule_ids, split_by_index, sampled)
                            ],
                        },
                        "runtime_seconds": 0.12 + 0.01 * seed,
                        "notes": ["Bounded data-scale-check pilot only, not benchmark results."],
                    }
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    result_paths.append(path)

            first = write_data_scale_check_checkpoint_sampling_multiseed_summary(
                result_paths=result_paths,
                summary_path=root / "summary.json",
                report_path=root / "summary.md",
                expected_seeds=(0, 1, 2),
            )
            first_summary_text = first["summary"].read_text(encoding="utf-8")
            second = write_data_scale_check_checkpoint_sampling_multiseed_summary(
                result_paths=result_paths,
                summary_path=root / "summary2.json",
                report_path=root / "summary2.md",
                expected_seeds=(0, 1, 2),
            )
            self.assertEqual(first_summary_text, second["summary"].read_text(encoding="utf-8"))

            summary = json.loads(first_summary_text)
            report = first["report"].read_text(encoding="utf-8")
            self.assertEqual(
                summary["result_kind"],
                "tiny_real_data_bounded_data_scale_check_checkpoint_sampling_multiseed_summary_not_benchmark",
            )
            self.assertEqual(summary["seed_set"], [0, 1, 2])
            self.assertEqual(summary["row_count"], 6)
            self.assertTrue(summary["data_scale_check_run"])
            self.assertEqual(summary["stiffness_consistency"]["verdict"], "stiffness_wins_consistently")
            self.assertTrue(summary["runtime_summary"]["all_runtime_seconds_reliable"])
            aggregate = {row["metric"]: row for row in summary["aggregate_metric_comparisons"]}
            self.assertEqual(aggregate["reloaded_loss"]["stiffness_win_count"], 3)
            self.assertEqual(aggregate["sampled_rmsd_to_target_overall_mean"]["stiffness_win_count"], 3)
            self.assertIn("Bounded 30-Molecule Checkpoint/Sampling Multi-Seed Summary", report)
            self.assertIn("stiffness_wins_consistently", report)

    def test_data_scale_check_500_comparison_accepts_bounded_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_paths = []
            molecule_ids = [f"geom_scale500_{index:03d}" for index in range(500)]
            split_by_index = ["train"] * 450 + ["val"] * 25 + ["test"] * 25
            for mode, label, offset in [
                ("isotropic", "isotropic", 0.40),
                ("stiffness_structured", "stiffness", 0.30),
            ]:
                path = root / f"{label}.json"
                sampled = [offset + 0.0001 * index for index in range(500)]
                payload = {
                    "run_name": f"geom_data_scale_check_ckpt_sample_{label}_mols500_steps5_seed0",
                    "dataset_name": "geom_drugs",
                    "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                    "dataset_subset_scope": "data_scale_check_500_processed_dry_run",
                    "processed_artifact_path": "data/processed/geom_drugs_subset/data_scale_check_500/molecules.json",
                    "split_path": "data/splits/geom_drugs_data_scale_check_500/random_seed0/split.json",
                    "random_seed": 0,
                    "corruption_mode": mode,
                    "model_name": "single_linear_velocity_smoke",
                    "backbone": "minimal_linear_velocity_pilot",
                    "execution_mode": "local_cpu_tiny_real_data_scale_check_checkpoint_sampling",
                    "result_kind": "tiny_real_data_bounded_data_scale_check_checkpoint_sampling_not_benchmark",
                    "benchmark_result": False,
                    "hardware": "cpu_only",
                    "gpu_used": False,
                    "scheduler": None,
                    "slurm_account": None,
                    "pilot_run": True,
                    "bounded_pilot": True,
                    "micro_pilot_run": False,
                    "data_scale_check_run": True,
                    "data_scale_check_500_run": True,
                    "dry_run": False,
                    "training_executed": True,
                    "model_training_started": True,
                    "training_loop_executed": True,
                    "bounded_real_training_loop_executed": True,
                    "larger_real_training_run_started": False,
                    "max_steps": 5,
                    "hard_max_steps": 5,
                    "optimizer_created": True,
                    "optimizer_step_count": 5,
                    "backward_executed": True,
                    "backward_pass_count": 5,
                    "checkpoint_path": f"experiments/checkpoints/{label}_seed0.pt",
                    "checkpoint_written": True,
                    "checkpoint_reloaded": True,
                    "checkpoint_scope": "disposable_data_scale_check_500_checkpoint_sampling_only",
                    "sampling_executed": True,
                    "sample_step_count": 1,
                    "sample_step_size": 0.1,
                    "forward_executed": True,
                    "loss_computed": True,
                    "training_sample_count": 450,
                    "sample_count": 500,
                    "runtime_scope": "checkpoint_train_save_reload_sample_after_full_cpu_checkpoint_sampling_warmup",
                    "runtime_seconds_reliable": True,
                    "dataset": {
                        "name": "geom_drugs",
                        "kind": "geom_drugs_tiny_processed_dry_run",
                        "subset_scope": "data_scale_check_500_processed_dry_run",
                        "molecule_count": 500,
                        "split_type": "random",
                        "split_scope": "data_scale_check_500_processed_dry_run",
                        "split_counts": {"train": 450, "val": 25, "test": 25},
                        "degenerate_tiny_split": False,
                    },
                    "flow_matching": {"objective": "cartesian_linear", "interpolation": "linear"},
                    "metrics": {
                        "metric_scope": "geom_drugs_data_scale_check_checkpoint_sampling",
                        "model_name": "single_linear_velocity_smoke",
                        "reloaded_loss": 0.70 if mode == "isotropic" else 0.60,
                        "train_steps": [{"optimizer_step": index + 1} for index in range(5)],
                        "sampled_molecules": [
                            {
                                "molecule_id": molecule_id,
                                "split": split,
                                "atom_count": 3,
                                "sample_step_count": 1,
                                "sample_step_size": 0.1,
                                "sampled_rmsd_to_target": rmsd,
                                "predicted_velocity_norm": 1.0,
                            }
                            for molecule_id, split, rmsd in zip(molecule_ids, split_by_index, sampled)
                        ],
                    },
                    "runtime_seconds": 0.5,
                    "notes": ["Bounded 500-molecule data-scale-check pilot only, not benchmark results."],
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                result_paths.append(path)

            comparison = write_data_scale_check_checkpoint_sampling_comparison(
                result_paths=result_paths,
                output_root=root / "experiments",
                seed=0,
                max_molecules=500,
                max_steps=5,
            )
            results = json.loads(comparison["results"].read_text(encoding="utf-8"))
            self.assertEqual(results["max_molecules"], 500)
            self.assertEqual(results["dataset_subset_scope"], "data_scale_check_500_processed_dry_run")
            self.assertTrue(results["data_scale_check_500_run"])
            self.assertEqual(results["hard_max_molecules"], 500)
            self.assertTrue(results["directional_signal_preserved"])
            self.assertEqual(len(results["metric_comparisons"]), 5)
            for row in results["rows"]:
                self.assertEqual(row["sample_count"], 500)
                self.assertEqual(row["training_sample_count"], 450)
                self.assertTrue(row["runtime_seconds_reliable"])

    def test_data_scale_check_10000_split_uses_bounded_heldout_partitions(self) -> None:
        records = [{"molecule_id": f"geom_scale10000_{index:05d}"} for index in range(10_000)]
        split = make_tiny_split(
            records,
            split_type="random",
            seed=0,
            split_scope="data_scale_check_10000_processed_dry_run",
        )
        self.assertEqual(len(split["train"]), 9000)
        self.assertEqual(len(split["val"]), 500)
        self.assertEqual(len(split["test"]), 500)
        self.assertFalse(split["benchmark_result"])
        self.assertFalse(split["degenerate_tiny_split"])
        self.assertEqual(
            len(set(split["train"] + split["val"] + split["test"])),
            10_000,
        )

    def test_data_scale_check_10000_comparison_accepts_bounded_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_paths = []
            molecule_ids = [f"geom_scale10000_{index:05d}" for index in range(10_000)]
            split_by_index = ["train"] * 9000 + ["val"] * 500 + ["test"] * 500
            for mode, label, offset in [
                ("isotropic", "isotropic", 0.40),
                ("stiffness_structured", "stiffness", 0.30),
            ]:
                path = root / f"{label}.json"
                sampled = [offset + 0.000001 * index for index in range(10_000)]
                payload = {
                    "run_name": f"geom_data_scale_check_ckpt_sample_{label}_mols10000_steps5_seed0",
                    "dataset_name": "geom_drugs",
                    "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                    "dataset_subset_scope": "data_scale_check_10000_processed_dry_run",
                    "processed_artifact_path": "data/processed/geom_drugs_subset/data_scale_check_10000/molecules.json",
                    "split_path": "data/splits/geom_drugs_data_scale_check_10000/random_seed0/split.json",
                    "random_seed": 0,
                    "corruption_mode": mode,
                    "model_name": "single_linear_velocity_smoke",
                    "backbone": "minimal_linear_velocity_pilot",
                    "execution_mode": "local_cpu_tiny_real_data_scale_check_checkpoint_sampling",
                    "result_kind": "tiny_real_data_bounded_data_scale_check_checkpoint_sampling_not_benchmark",
                    "benchmark_result": False,
                    "hardware": "cpu_only",
                    "gpu_used": False,
                    "scheduler": None,
                    "slurm_account": None,
                    "pilot_run": True,
                    "bounded_pilot": True,
                    "micro_pilot_run": False,
                    "data_scale_check_run": True,
                    "data_scale_check_10000_run": True,
                    "dry_run": False,
                    "training_executed": True,
                    "model_training_started": True,
                    "training_loop_executed": True,
                    "bounded_real_training_loop_executed": True,
                    "larger_real_training_run_started": False,
                    "max_steps": 5,
                    "hard_max_steps": 5,
                    "optimizer_created": True,
                    "optimizer_step_count": 5,
                    "backward_executed": True,
                    "backward_pass_count": 5,
                    "checkpoint_path": f"experiments/checkpoints/{label}_seed0.pt",
                    "checkpoint_written": True,
                    "checkpoint_reloaded": True,
                    "checkpoint_scope": "disposable_data_scale_check_10000_checkpoint_sampling_only",
                    "sampling_executed": True,
                    "sample_step_count": 1,
                    "sample_step_size": 0.1,
                    "forward_executed": True,
                    "loss_computed": True,
                    "training_sample_count": 9000,
                    "sample_count": 10_000,
                    "runtime_scope": "checkpoint_train_save_reload_sample_after_full_cpu_checkpoint_sampling_warmup",
                    "runtime_seconds_reliable": True,
                    "dataset": {
                        "name": "geom_drugs",
                        "kind": "geom_drugs_tiny_processed_dry_run",
                        "subset_scope": "data_scale_check_10000_processed_dry_run",
                        "molecule_count": 10_000,
                        "split_type": "random",
                        "split_scope": "data_scale_check_10000_processed_dry_run",
                        "split_counts": {"train": 9000, "val": 500, "test": 500},
                        "degenerate_tiny_split": False,
                    },
                    "flow_matching": {"objective": "cartesian_linear", "interpolation": "linear"},
                    "metrics": {
                        "metric_scope": "geom_drugs_data_scale_check_checkpoint_sampling",
                        "model_name": "single_linear_velocity_smoke",
                        "reloaded_loss": 0.70 if mode == "isotropic" else 0.60,
                        "train_steps": [{"optimizer_step": index + 1} for index in range(5)],
                        "sampled_molecules": [
                            {
                                "molecule_id": molecule_id,
                                "split": split,
                                "atom_count": 3,
                                "sample_step_count": 1,
                                "sample_step_size": 0.1,
                                "sampled_rmsd_to_target": rmsd,
                                "predicted_velocity_norm": 1.0,
                            }
                            for molecule_id, split, rmsd in zip(molecule_ids, split_by_index, sampled)
                        ],
                    },
                    "runtime_seconds": 5.0,
                    "notes": ["Bounded 10000-molecule data-scale-check pilot only, not benchmark results."],
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                result_paths.append(path)

            comparison = write_data_scale_check_checkpoint_sampling_comparison(
                result_paths=result_paths,
                output_root=root / "experiments",
                seed=0,
                max_molecules=10_000,
                max_steps=5,
            )
            results = json.loads(comparison["results"].read_text(encoding="utf-8"))
            self.assertEqual(results["max_molecules"], 10_000)
            self.assertEqual(results["dataset_subset_scope"], "data_scale_check_10000_processed_dry_run")
            self.assertTrue(results["data_scale_check_10000_run"])
            self.assertEqual(results["hard_max_molecules"], 10_000)
            self.assertTrue(results["directional_signal_preserved"])
            self.assertEqual(len(results["metric_comparisons"]), 5)
            for row in results["rows"]:
                self.assertEqual(row["sample_count"], 10_000)
                self.assertEqual(row["training_sample_count"], 9000)
                self.assertEqual(row["split_counts"], {"train": 9000, "val": 500, "test": 500})
                self.assertTrue(row["runtime_seconds_reliable"])

    def test_data_scale_check_500_multiseed_summary_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_paths = []
            molecule_ids = [f"geom_scale500_{index:03d}" for index in range(500)]
            split_by_index = ["train"] * 450 + ["val"] * 25 + ["test"] * 25
            for seed in [0, 1, 2]:
                for mode, label, offset in [
                    ("isotropic", "isotropic", 0.40),
                    ("stiffness_structured", "stiffness", 0.30),
                ]:
                    path = root / f"{label}_seed{seed}.json"
                    sampled = [offset + 0.0001 * index + 0.00001 * seed for index in range(500)]
                    payload = {
                        "run_name": f"geom_data_scale_check_ckpt_sample_{label}_mols500_steps5_seed{seed}",
                        "dataset_name": "geom_drugs",
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "dataset_subset_scope": "data_scale_check_500_processed_dry_run",
                        "processed_artifact_path": "data/processed/geom_drugs_subset/data_scale_check_500/molecules.json",
                        "split_path": "data/splits/geom_drugs_data_scale_check_500/random_seed0/split.json",
                        "random_seed": seed,
                        "corruption_mode": mode,
                        "model_name": "single_linear_velocity_smoke",
                        "backbone": "minimal_linear_velocity_pilot",
                        "execution_mode": "local_cpu_tiny_real_data_scale_check_checkpoint_sampling",
                        "result_kind": "tiny_real_data_bounded_data_scale_check_checkpoint_sampling_not_benchmark",
                        "benchmark_result": False,
                        "hardware": "cpu_only",
                        "gpu_used": False,
                        "scheduler": None,
                        "slurm_account": None,
                        "pilot_run": True,
                        "bounded_pilot": True,
                        "micro_pilot_run": False,
                        "data_scale_check_run": True,
                        "data_scale_check_500_run": True,
                        "dry_run": False,
                        "training_executed": True,
                        "model_training_started": True,
                        "training_loop_executed": True,
                        "bounded_real_training_loop_executed": True,
                        "larger_real_training_run_started": False,
                        "max_steps": 5,
                        "hard_max_steps": 5,
                        "optimizer_created": True,
                        "optimizer_step_count": 5,
                        "backward_executed": True,
                        "backward_pass_count": 5,
                        "checkpoint_path": f"experiments/checkpoints/{label}_seed{seed}.pt",
                        "checkpoint_written": True,
                        "checkpoint_reloaded": True,
                        "checkpoint_scope": "disposable_data_scale_check_500_checkpoint_sampling_only",
                        "sampling_executed": True,
                        "sample_step_count": 1,
                        "sample_step_size": 0.1,
                        "forward_executed": True,
                        "loss_computed": True,
                        "training_sample_count": 450,
                        "sample_count": 500,
                        "runtime_scope": "checkpoint_train_save_reload_sample_after_full_cpu_checkpoint_sampling_warmup",
                        "runtime_seconds_reliable": True,
                        "dataset": {
                            "name": "geom_drugs",
                            "kind": "geom_drugs_tiny_processed_dry_run",
                            "subset_scope": "data_scale_check_500_processed_dry_run",
                            "molecule_count": 500,
                            "split_type": "random",
                            "split_scope": "data_scale_check_500_processed_dry_run",
                            "split_counts": {"train": 450, "val": 25, "test": 25},
                            "degenerate_tiny_split": False,
                        },
                        "flow_matching": {"objective": "cartesian_linear", "interpolation": "linear"},
                        "metrics": {
                            "metric_scope": "geom_drugs_data_scale_check_checkpoint_sampling",
                            "model_name": "single_linear_velocity_smoke",
                            "reloaded_loss": 0.70 + 0.01 * seed if mode == "isotropic" else 0.60 + 0.01 * seed,
                            "train_steps": [{"optimizer_step": index + 1} for index in range(5)],
                            "sampled_molecules": [
                                {
                                    "molecule_id": molecule_id,
                                    "split": split,
                                    "atom_count": 3,
                                    "sample_step_count": 1,
                                    "sample_step_size": 0.1,
                                    "sampled_rmsd_to_target": rmsd,
                                    "predicted_velocity_norm": 1.0,
                                }
                                for molecule_id, split, rmsd in zip(molecule_ids, split_by_index, sampled)
                            ],
                        },
                        "runtime_seconds": 0.50 + 0.01 * seed,
                        "notes": ["Bounded 500-molecule data-scale-check pilot only, not benchmark results."],
                    }
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    result_paths.append(path)

            first = write_data_scale_check_checkpoint_sampling_multiseed_summary(
                result_paths=result_paths,
                summary_path=root / "summary.json",
                report_path=root / "summary.md",
                expected_seeds=(0, 1, 2),
                max_molecules=500,
            )
            first_summary_text = first["summary"].read_text(encoding="utf-8")
            second = write_data_scale_check_checkpoint_sampling_multiseed_summary(
                result_paths=result_paths,
                summary_path=root / "summary2.json",
                report_path=root / "summary2.md",
                expected_seeds=(0, 1, 2),
                max_molecules=500,
            )
            self.assertEqual(first_summary_text, second["summary"].read_text(encoding="utf-8"))

            summary = json.loads(first_summary_text)
            report = first["report"].read_text(encoding="utf-8")
            self.assertEqual(summary["max_molecules"], 500)
            self.assertEqual(summary["split_counts"], {"train": 450, "val": 25, "test": 25})
            self.assertTrue(summary["data_scale_check_500_run"])
            self.assertEqual(summary["row_count"], 6)
            self.assertEqual(summary["stiffness_consistency"]["verdict"], "stiffness_wins_consistently")
            self.assertTrue(summary["runtime_summary"]["all_runtime_seconds_reliable"])
            aggregate = {row["metric"]: row for row in summary["aggregate_metric_comparisons"]}
            self.assertEqual(aggregate["reloaded_loss"]["stiffness_win_count"], 3)
            self.assertEqual(aggregate["sampled_rmsd_to_target_test_mean"]["stiffness_win_count"], 3)
            self.assertIn("Bounded 500-Molecule Checkpoint/Sampling Multi-Seed Summary", report)
            self.assertIn("stiffness_wins_consistently", report)

    def test_data_scale_check_10000_multiseed_summary_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_paths = []
            molecule_ids = [f"geom_scale10000_{index:05d}" for index in range(10_000)]
            split_by_index = ["train"] * 9000 + ["val"] * 500 + ["test"] * 500
            for seed in [0, 1, 2]:
                for mode, label, offset in [
                    ("isotropic", "isotropic", 0.40),
                    ("stiffness_structured", "stiffness", 0.30),
                ]:
                    path = root / f"{label}_seed{seed}.json"
                    sampled = [offset + 0.000001 * index + 0.00001 * seed for index in range(10_000)]
                    payload = {
                        "run_name": f"geom_data_scale_check_ckpt_sample_{label}_mols10000_steps5_seed{seed}",
                        "dataset_name": "geom_drugs",
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "dataset_subset_scope": "data_scale_check_10000_processed_dry_run",
                        "processed_artifact_path": "data/processed/geom_drugs_subset/data_scale_check_10000/molecules.json",
                        "split_path": "data/splits/geom_drugs_data_scale_check_10000/random_seed0/split.json",
                        "random_seed": seed,
                        "corruption_mode": mode,
                        "model_name": "single_linear_velocity_smoke",
                        "backbone": "minimal_linear_velocity_pilot",
                        "execution_mode": "local_cpu_tiny_real_data_scale_check_checkpoint_sampling",
                        "result_kind": "tiny_real_data_bounded_data_scale_check_checkpoint_sampling_not_benchmark",
                        "benchmark_result": False,
                        "hardware": "cpu_only",
                        "gpu_used": False,
                        "scheduler": None,
                        "slurm_account": None,
                        "pilot_run": True,
                        "bounded_pilot": True,
                        "micro_pilot_run": False,
                        "data_scale_check_run": True,
                        "data_scale_check_10000_run": True,
                        "dry_run": False,
                        "training_executed": True,
                        "model_training_started": True,
                        "training_loop_executed": True,
                        "bounded_real_training_loop_executed": True,
                        "larger_real_training_run_started": False,
                        "max_steps": 5,
                        "hard_max_steps": 5,
                        "optimizer_created": True,
                        "optimizer_step_count": 5,
                        "backward_executed": True,
                        "backward_pass_count": 5,
                        "checkpoint_path": f"experiments/checkpoints/{label}_seed{seed}.pt",
                        "checkpoint_written": True,
                        "checkpoint_reloaded": True,
                        "checkpoint_scope": "disposable_data_scale_check_10000_checkpoint_sampling_only",
                        "sampling_executed": True,
                        "sample_step_count": 1,
                        "sample_step_size": 0.1,
                        "forward_executed": True,
                        "loss_computed": True,
                        "training_sample_count": 9000,
                        "sample_count": 10_000,
                        "runtime_scope": "checkpoint_train_save_reload_sample_after_full_cpu_checkpoint_sampling_warmup",
                        "runtime_seconds_reliable": True,
                        "dataset": {
                            "name": "geom_drugs",
                            "kind": "geom_drugs_tiny_processed_dry_run",
                            "subset_scope": "data_scale_check_10000_processed_dry_run",
                            "molecule_count": 10_000,
                            "split_type": "random",
                            "split_scope": "data_scale_check_10000_processed_dry_run",
                            "split_counts": {"train": 9000, "val": 500, "test": 500},
                            "degenerate_tiny_split": False,
                        },
                        "flow_matching": {"objective": "cartesian_linear", "interpolation": "linear"},
                        "metrics": {
                            "metric_scope": "geom_drugs_data_scale_check_checkpoint_sampling",
                            "model_name": "single_linear_velocity_smoke",
                            "reloaded_loss": 0.70 + 0.01 * seed if mode == "isotropic" else 0.60 + 0.01 * seed,
                            "train_steps": [{"optimizer_step": index + 1} for index in range(5)],
                            "sampled_molecules": [
                                {
                                    "molecule_id": molecule_id,
                                    "split": split,
                                    "atom_count": 3,
                                    "sample_step_count": 1,
                                    "sample_step_size": 0.1,
                                    "sampled_rmsd_to_target": rmsd,
                                    "predicted_velocity_norm": 1.0,
                                }
                                for molecule_id, split, rmsd in zip(molecule_ids, split_by_index, sampled)
                            ],
                        },
                        "runtime_seconds": 5.00 + 0.01 * seed,
                        "notes": ["Bounded 10000-molecule data-scale-check pilot only, not benchmark results."],
                    }
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    result_paths.append(path)

            first = write_data_scale_check_checkpoint_sampling_multiseed_summary(
                result_paths=result_paths,
                summary_path=root / "summary.json",
                report_path=root / "summary.md",
                expected_seeds=(0, 1, 2),
                max_molecules=10_000,
            )
            first_summary_text = first["summary"].read_text(encoding="utf-8")
            second = write_data_scale_check_checkpoint_sampling_multiseed_summary(
                result_paths=result_paths,
                summary_path=root / "summary2.json",
                report_path=root / "summary2.md",
                expected_seeds=(0, 1, 2),
                max_molecules=10_000,
            )
            self.assertEqual(first_summary_text, second["summary"].read_text(encoding="utf-8"))

            summary = json.loads(first_summary_text)
            report = first["report"].read_text(encoding="utf-8")
            self.assertEqual(summary["max_molecules"], 10_000)
            self.assertEqual(summary["split_counts"], {"train": 9000, "val": 500, "test": 500})
            self.assertTrue(summary["data_scale_check_10000_run"])
            self.assertEqual(summary["row_count"], 6)
            self.assertEqual(summary["stiffness_consistency"]["verdict"], "stiffness_wins_consistently")
            self.assertTrue(summary["runtime_summary"]["all_runtime_seconds_reliable"])
            aggregate = {row["metric"]: row for row in summary["aggregate_metric_comparisons"]}
            self.assertEqual(aggregate["reloaded_loss"]["stiffness_win_count"], 3)
            self.assertEqual(aggregate["sampled_rmsd_to_target_val_mean"]["stiffness_win_count"], 3)
            self.assertIn("Bounded 10000-Molecule Checkpoint/Sampling Multi-Seed Summary", report)
            self.assertIn("stiffness_wins_consistently", report)

    def test_bounded_checkpoint_sampling_pilot_records_early_stop(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not importable in this interpreter")

        with tempfile.TemporaryDirectory() as tmp:
            samples = [
                {
                    "molecule_id": "geom_early_stop_0",
                    "split": "train",
                    "t": 0.5,
                    "xt": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    "x1": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    "target_velocity": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                }
            ]
            metrics = run_bounded_checkpoint_sampling_pilot(
                samples=samples,
                sampling_samples=samples,
                checkpoint_path=Path(tmp) / "early_stop.pt",
                seed=0,
                max_steps=20,
                hard_max_steps=20,
                learning_rate=1.0e-2,
                sample_step_size=0.1,
                early_stop={
                    "enabled": True,
                    "monitor": "train_loss_before_step",
                    "patience": 2,
                    "min_delta": 100.0,
                },
            )
            self.assertTrue(metrics["early_stop_enabled"])
            self.assertTrue(metrics["stopped_early"])
            self.assertEqual(metrics["stop_reason"], "patience_exhausted")
            self.assertEqual(metrics["configured_max_steps"], 20)
            self.assertLess(metrics["executed_train_steps"], 20)
            self.assertEqual(metrics["optimizer_step_count"], metrics["executed_train_steps"])
            self.assertEqual(metrics["backward_pass_count"], metrics["executed_train_steps"])
            self.assertEqual(metrics["early_stop_patience"], 2)
            self.assertEqual(metrics["early_stop_monitor"], "train_loss_before_step")
            self.assertEqual(len(metrics["sampled_molecules"]), 1)

    def test_validation_checkpoint_training_writes_reusable_best_and_latest(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not importable in this interpreter")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_samples = [
                {
                    "molecule_id": "formal_train_0",
                    "split": "train",
                    "t": 0.5,
                    "xt": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    "x1": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
                    "target_velocity": [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
                }
            ]
            validation_samples = [
                {
                    "molecule_id": "formal_val_0",
                    "split": "val",
                    "t": 0.5,
                    "xt": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    "x1": [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0]],
                    "target_velocity": [[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]],
                }
            ]
            metrics = run_validation_checkpoint_training(
                samples=train_samples,
                validation_samples=validation_samples,
                sampling_samples=train_samples + validation_samples,
                checkpoint_path=root / "best.pt",
                latest_checkpoint_path=root / "latest.pt",
                seed=0,
                max_steps=205,
                hard_max_steps=205,
                learning_rate=1.0e-2,
                sample_step_size=0.1,
                early_stop={
                    "enabled": True,
                    "monitor": "validation_loss_after_step",
                    "patience": 2,
                    "min_delta": 100.0,
                },
                checkpoint_scope="test_reusable_validation_checkpoint",
            )

            self.assertTrue((root / "best.pt").exists())
            self.assertTrue((root / "latest.pt").exists())
            self.assertTrue(metrics["checkpoint_reusable"])
            self.assertTrue(metrics["checkpoint_selected_by_validation"])
            self.assertEqual(metrics["checkpoint_selection_metric"], "validation_loss_after_step")
            self.assertEqual(metrics["checkpoint_reloaded_path"], str(root / "best.pt"))
            self.assertTrue(metrics["stopped_early"])
            self.assertEqual(metrics["stop_reason"], "validation_patience_exhausted")
            self.assertLess(metrics["executed_train_steps"], 205)
            self.assertEqual(metrics["validation_sample_count"], 1)
            self.assertEqual(len(metrics["sampled_molecules"]), 2)

    def test_formal_10000_configs_freeze_scope_and_selection_settings(self) -> None:
        for path, mode in [
            ("configs/train/data_scale_check_10000_formal_isotropic.yaml", "isotropic"),
            ("configs/train/data_scale_check_10000_formal_stiffness.yaml", "stiffness_structured"),
        ]:
            config = load_config(path)
            _validate_formal_10000_config(config)
            self.assertEqual(config["corruption"]["mode"], mode)
            self.assertEqual(config["data"]["subset"]["max_molecules"], 10_000)
            self.assertEqual(config["data"]["subset"]["max_conformers_per_mol"], 1)
            self.assertEqual(config["model"]["backbone"], "minimal_linear_velocity_pilot")
            self.assertEqual(config["model"]["pilot_model"], "single_linear_velocity_pilot")
            self.assertEqual(config["train"]["optimizer"], "SGD")
            self.assertEqual(config["train"]["max_steps"], FORMAL_10000_MAX_STEPS)
            self.assertEqual(config["train"]["learning_rate"], 1.0e-2)
            self.assertEqual(config["train"]["checkpoint_scope"], FORMAL_10000_CHECKPOINT_SCOPE)
            self.assertEqual(config["train"]["early_stop"]["monitor"], "validation_loss_after_step")
            self.assertEqual(config["sample"]["sample_step_count"], 1)
            self.assertEqual(config["sample"]["sample_step_size"], 0.1)
            self.assertEqual(FORMAL_10000_SEED_SET, (0, 1, 2))
            self.assertEqual(FORMAL_10000_SPLIT_COUNTS, {"train": 9000, "val": 500, "test": 500})

    def test_geom_full_database_artifact_and_split_preserve_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_drugs = root / "raw" / "rdkit_folder" / "drugs"
            raw_drugs.mkdir(parents=True)
            for index in range(20):
                raw_record = {
                    "smiles": f"C{index}",
                    "conformers": [
                        {
                            "geom_id": f"conf_{index}",
                            "rd_mol": _FakeMol(
                                [
                                    [float(index), 0.0, 0.0],
                                    [float(index), 1.0, 0.0],
                                ]
                            ),
                        }
                    ],
                }
                with (raw_drugs / f"mol_{index:03d}.pickle").open("wb") as handle:
                    pickle.dump(raw_record, handle)

            processed = prepare_geom_full_database_artifact(
                raw_root=root / "raw",
                processed_root=root / "processed",
                max_atoms=8,
            )
            artifact = json.loads(Path(processed["artifact"]).read_text(encoding="utf-8"))
            self.assertEqual(artifact["dataset_kind"], "geom_drugs_tiny_processed_dry_run")
            self.assertEqual(artifact["result_kind"], "processed_output_dry_run_not_benchmark")
            self.assertEqual(artifact["subset_scope"], FULL_DATABASE_SCOPE)
            self.assertFalse(artifact["benchmark_result"])
            self.assertEqual(artifact["molecule_count"], 20)
            self.assertEqual(processed["source_pickle_count"], 20)
            self.assertEqual(processed["skip_reasons"], {})
            self.assertEqual(artifact["bounds"]["max_molecules"], None)
            self.assertTrue(artifact["bounds"]["one_conformer_per_molecule"])

            split_paths = prepare_geom_full_database_split(
                processed_artifact_path=processed["artifact"],
                split_root=root / "splits",
                seed=0,
            )
            split = json.loads(Path(split_paths["split"]).read_text(encoding="utf-8"))
            self.assertEqual(split["dataset_kind"], "geom_drugs_tiny_processed_dry_run")
            self.assertEqual(split["result_kind"], "split_dry_run_not_benchmark")
            self.assertEqual(split["split_scope"], FULL_DATABASE_SCOPE)
            self.assertFalse(split["benchmark_result"])
            self.assertFalse(split["degenerate_tiny_split"])
            self.assertEqual(split_paths["counts"], {"train": 18, "val": 1, "test": 1})
            self.assertEqual(len(set(split["train"] + split["val"] + split["test"])), 20)

    def test_geom_full_database_scaffold_split_reuses_processed_artifact(self) -> None:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            self.skipTest("RDKit is not importable")

        smiles_values = [
            "CCO",
            "CCN",
            "CCC",
            "CCCl",
            "CCBr",
            "CC(=O)N",
            "CCOC",
            "CCS",
            "CC(C)O",
            "CC(C)N",
            "c1ccccc1",
            "Cc1ccccc1",
            "Oc1ccccc1",
            "Nc1ccccc1",
            "c1ccncc1",
            "Cc1ccncc1",
            "C1CCCCC1",
            "CC1CCCCC1",
            "C1CCNCC1",
            "CC1CCNCC1",
            "c1ccoc1",
            "Cc1ccoc1",
            "c1ccsc1",
            "Cc1ccsc1",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "processed" / "molecules.json"
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_text(
                json.dumps(
                    {
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "result_kind": "processed_output_dry_run_not_benchmark",
                        "benchmark_result": False,
                        "subset_scope": FULL_DATABASE_SCOPE,
                        "molecule_count": len(smiles_values),
                        "conformer_count": len(smiles_values),
                        "molecules": [
                            {"molecule_id": f"mol_{index:03d}", "smiles": smiles}
                            for index, smiles in enumerate(smiles_values)
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            split_paths = prepare_geom_full_database_scaffold_split(
                processed_artifact_path=artifact_path,
                split_root=root / "splits",
                seed=0,
            )
            split = json.loads(Path(split_paths["split"]).read_text(encoding="utf-8"))
            manifest = json.loads(Path(split_paths["manifest"]).read_text(encoding="utf-8"))
            split_ids = split["train"] + split["val"] + split["test"]

            self.assertEqual(split["split_type"], "scaffold")
            self.assertEqual(split["split_scope"], FULL_DATABASE_SCOPE)
            self.assertFalse(split["benchmark_result"])
            self.assertFalse(split["degenerate_tiny_split"])
            self.assertEqual(len(split_ids), len(smiles_values))
            self.assertEqual(len(set(split_ids)), len(smiles_values))
            self.assertGreater(split_paths["counts"]["train"], 0)
            self.assertGreater(split_paths["counts"]["val"], 0)
            self.assertGreater(split_paths["counts"]["test"], 0)
            self.assertEqual(manifest["processed_artifact_reused"], True)
            self.assertEqual(manifest["split_type"], "scaffold")
            self.assertEqual(manifest["stereo_fallback_molecule_count"], 0)
            self.assertEqual(manifest["skipped_molecule_count"], 0)
            self.assertEqual(split["stereo_fallback_molecule_count"], 0)
            self.assertEqual(split["skipped_molecule_count"], 0)
            self.assertFalse(manifest["raw_data_modified"])
            self.assertFalse(manifest["training_executed"])

    def test_geom_full_database_scaffold_split_retries_without_stereo(self) -> None:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            self.skipTest("RDKit is not importable")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = self._write_scaffold_test_artifact(root)
            call_count = {"count": 0}

            def fake_murcko_scaffold(mol):
                call_count["count"] += 1
                if call_count["count"] == 1:
                    raise RuntimeError("Pre-condition Violation bad bond stereo")
                return f"scaffold_{call_count['count']:03d}"

            with patch(
                "rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles",
                side_effect=fake_murcko_scaffold,
            ):
                split_paths = prepare_geom_full_database_scaffold_split(
                    processed_artifact_path=artifact_path,
                    split_root=root / "splits",
                    seed=0,
                )

            split = json.loads(Path(split_paths["split"]).read_text(encoding="utf-8"))
            manifest = json.loads(Path(split_paths["manifest"]).read_text(encoding="utf-8"))
            split_ids = split["train"] + split["val"] + split["test"]

            self.assertEqual(len(split_ids), 24)
            self.assertEqual(split_paths["stereo_fallback_molecule_count"], 1)
            self.assertEqual(split_paths["skipped_molecule_count"], 0)
            self.assertEqual(split["stereo_fallback_molecule_count"], 1)
            self.assertEqual(split["skipped_molecule_count"], 0)
            self.assertEqual(manifest["stereo_fallback_molecule_count"], 1)
            self.assertEqual(manifest["skipped_molecule_count"], 0)
            self.assertEqual(manifest["stereo_fallback_examples"][0]["molecule_id"], "mol_000")
            self.assertIn("bad bond stereo", manifest["stereo_fallback_examples"][0]["first_error"])

    def test_geom_full_database_scaffold_split_skips_after_failed_stereo_retry(self) -> None:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            self.skipTest("RDKit is not importable")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = self._write_scaffold_test_artifact(root)
            call_count = {"count": 0}

            def fake_murcko_scaffold(mol):
                call_count["count"] += 1
                if call_count["count"] in {1, 2}:
                    raise RuntimeError("Pre-condition Violation bad bond stereo")
                return f"scaffold_{call_count['count']:03d}"

            with patch(
                "rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles",
                side_effect=fake_murcko_scaffold,
            ):
                split_paths = prepare_geom_full_database_scaffold_split(
                    processed_artifact_path=artifact_path,
                    split_root=root / "splits",
                    seed=0,
                )

            split = json.loads(Path(split_paths["split"]).read_text(encoding="utf-8"))
            manifest = json.loads(Path(split_paths["manifest"]).read_text(encoding="utf-8"))
            split_ids = split["train"] + split["val"] + split["test"]

            self.assertEqual(len(split_ids), 23)
            self.assertNotIn("mol_000", set(split_ids))
            self.assertEqual(split_paths["stereo_fallback_molecule_count"], 0)
            self.assertEqual(split_paths["skipped_molecule_count"], 1)
            self.assertEqual(split["skipped_molecule_count"], 1)
            self.assertEqual(split["skipped_molecule_ids"], ["mol_000"])
            self.assertEqual(manifest["skipped_molecule_count"], 1)
            self.assertEqual(
                manifest["skipped_molecule_examples"][0]["reason"],
                "scaffold_generation_failed_after_stereo_removal",
            )
            self.assertIn("bad bond stereo", manifest["skipped_molecule_examples"][0]["first_error"])
            self.assertIn("bad bond stereo", manifest["skipped_molecule_examples"][0]["retry_error"])
            self.assertFalse(manifest["benchmark_result"])
            self.assertFalse(manifest["training_executed"])

    def test_formal_full_database_configs_freeze_formal_design(self) -> None:
        for path, mode in [
            ("configs/train/geom_full_database_formal_isotropic.yaml", "isotropic"),
            ("configs/train/geom_full_database_formal_harmonic.yaml", "harmonic_prior"),
            ("configs/train/geom_full_database_formal_torsion_only.yaml", "torsion_only"),
            ("configs/train/geom_full_database_formal_stiffness.yaml", "stiffness_structured"),
            ("configs/train/geom_full_database_scaffold_formal_isotropic.yaml", "isotropic"),
            ("configs/train/geom_full_database_scaffold_formal_harmonic.yaml", "harmonic_prior"),
            ("configs/train/geom_full_database_scaffold_formal_torsion_only.yaml", "torsion_only"),
            ("configs/train/geom_full_database_scaffold_formal_stiffness.yaml", "stiffness_structured"),
        ]:
            config = load_config(path)
            validate_formal_full_database_config(config)
            self.assertEqual(config["corruption"]["mode"], mode)
            self.assertEqual(config["data"]["subset"]["max_molecules"], "full_database")
            self.assertEqual(config["data"]["subset"]["max_conformers_per_mol"], 1)
            self.assertEqual(config["data"]["subset"]["tag"], FULL_DATABASE_SCOPE)
            self.assertEqual(config["model"]["backbone"], "minimal_linear_velocity_pilot")
            self.assertEqual(config["model"]["pilot_model"], "single_linear_velocity_pilot")
            self.assertEqual(config["train"]["optimizer"], "SGD")
            self.assertEqual(config["train"]["max_steps"], FORMAL_10000_MAX_STEPS)
            self.assertEqual(config["train"]["learning_rate"], 1.0e-2)
            self.assertEqual(config["train"]["checkpoint_scope"], FORMAL_FULL_DATABASE_CHECKPOINT_SCOPE)
            self.assertEqual(config["train"]["early_stop"]["monitor"], "validation_loss_after_step")
            self.assertEqual(config["sample"]["sample_step_count"], 1)
            self.assertEqual(config["sample"]["sample_step_size"], 0.1)
            self.assertEqual(config["eval"]["eval_subset_size"], 0)
            self.assertEqual(FORMAL_FULL_DATABASE_SEED, 0)
            self.assertEqual(FORMAL_FULL_DATABASE_SEED_SET, (0, 1, 2))

    def test_formal_full_database_scaffold_configs_reuse_cached_split(self) -> None:
        for path in [
            "configs/train/geom_full_database_scaffold_formal_isotropic.yaml",
            "configs/train/geom_full_database_scaffold_formal_harmonic.yaml",
            "configs/train/geom_full_database_scaffold_formal_torsion_only.yaml",
            "configs/train/geom_full_database_scaffold_formal_stiffness.yaml",
        ]:
            config = load_config(path)
            validate_formal_full_database_config(config)
            self.assertEqual(config["data"]["split_type"], "scaffold")
            self.assertEqual(config["data"]["split_path"], str(FULL_DATABASE_SCAFFOLD_SPLIT_PATH))
            self.assertEqual(config["data"]["processed_artifact_path"], "data/processed/geom_drugs_full_database/molecules.json")
            self.assertEqual(config["train"]["max_steps"], FORMAL_10000_MAX_STEPS)
            self.assertEqual(config["train"]["hard_max_steps"], FORMAL_10000_MAX_STEPS)
            self.assertEqual(config["train"]["optimizer"], "SGD")
            self.assertEqual(config["train"]["learning_rate"], 1.0e-2)
            self.assertEqual(config["train"]["early_stop"]["patience"], 200)
            self.assertEqual(config["sample"]["sample_step_count"], 1)
            self.assertEqual(config["sample"]["sample_step_size"], 0.1)

    def _write_scaffold_test_artifact(self, root: Path) -> Path:
        smiles_values = [
            "CCO",
            "CCN",
            "CCC",
            "CCCl",
            "CCBr",
            "CC(=O)N",
            "CCOC",
            "CCS",
            "CC(C)O",
            "CC(C)N",
            "c1ccccc1",
            "Cc1ccccc1",
            "Oc1ccccc1",
            "Nc1ccccc1",
            "c1ccncc1",
            "Cc1ccncc1",
            "C1CCCCC1",
            "CC1CCCCC1",
            "C1CCNCC1",
            "CC1CCNCC1",
            "c1ccoc1",
            "Cc1ccoc1",
            "c1ccsc1",
            "Cc1ccsc1",
        ]
        artifact_path = root / "processed" / "molecules.json"
        artifact_path.parent.mkdir(parents=True)
        artifact_path.write_text(
            json.dumps(
                {
                    "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                    "result_kind": "processed_output_dry_run_not_benchmark",
                    "benchmark_result": False,
                    "subset_scope": FULL_DATABASE_SCOPE,
                    "molecule_count": len(smiles_values),
                    "conformer_count": len(smiles_values),
                    "molecules": [
                        {"molecule_id": f"mol_{index:03d}", "smiles": smiles}
                        for index, smiles in enumerate(smiles_values)
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return artifact_path

    def test_formal_full_database_accepts_per_run_gpu_paths_without_changing_settings(self) -> None:
        config = load_config("configs/train/geom_full_database_formal_isotropic.yaml")
        config["experiment"]["seed"] = 1
        config["data"]["processed_root"] = "data/processed/geom_drugs_full_database_runs/isotropic_seed1"
        config["data"]["split_root"] = "data/splits/geom_drugs_full_database_runs/isotropic_seed1/random_seed0"
        config["data"][
            "processed_artifact_path"
        ] = "data/processed/geom_drugs_full_database_runs/isotropic_seed1/molecules.json"
        config["data"][
            "split_path"
        ] = "data/splits/geom_drugs_full_database_runs/isotropic_seed1/random_seed0/split.json"
        config["train"]["device"] = "cuda"
        config["train"]["require_cuda"] = True

        validate_formal_full_database_config(config)
        self.assertEqual(config["train"]["max_steps"], FORMAL_10000_MAX_STEPS)
        self.assertEqual(config["train"]["hard_max_steps"], FORMAL_10000_MAX_STEPS)
        self.assertEqual(config["train"]["optimizer"], "SGD")
        self.assertEqual(config["train"]["learning_rate"], 1.0e-2)
        self.assertEqual(config["train"]["early_stop"]["monitor"], "validation_loss_after_step")
        self.assertEqual(config["sample"]["sample_step_count"], 1)
        self.assertEqual(config["sample"]["sample_step_size"], 0.1)

        bad_split = copy.deepcopy(config)
        bad_split["data"][
            "split_path"
        ] = "data/splits/geom_drugs_full_database_runs/isotropic_seed1/random_seed1/split.json"
        with self.assertRaises(ValueError):
            validate_formal_full_database_config(bad_split)

    def test_formal_full_database_seed0_comparison_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_paths = []
            for mode, label, best_loss, rmsd_base in [
                ("isotropic", "isotropic", 0.40, 0.50),
                ("stiffness_structured", "stiffness", 0.30, 0.40),
            ]:
                payload = {
                    "run_name": f"geom_formal_full_database_train_{label}_steps2000_seed0",
                    "result_kind": "geom_drugs_full_database_formal_training_not_benchmark",
                    "benchmark_result": False,
                    "dataset_subset_scope": FULL_DATABASE_SCOPE,
                    "processed_artifact_path": "data/processed/geom_drugs_full_database/molecules.json",
                    "split_path": "data/splits/geom_drugs_full_database/random_seed0/split.json",
                    "random_seed": 0,
                    "corruption_mode": mode,
                    "model_name": "single_linear_velocity_smoke",
                    "backbone": "minimal_linear_velocity_pilot",
                    "hardware": "cpu_only",
                    "gpu_used": False,
                    "scheduler": "slurm",
                    "slurm_account": "torch_pr_872_general",
                    "formal_training_run": True,
                    "full_database_formal_training_run": True,
                    "max_steps": 2000,
                    "hard_max_steps": 2000,
                    "optimizer_name": "SGD",
                    "learning_rate": 1.0e-2,
                    "optimizer_step_count": 2000,
                    "training_sample_count": 4,
                    "validation_sample_count": 1,
                    "sample_count": 6,
                    "early_stop_monitor": "validation_loss_after_step",
                    "early_stop_patience": 200,
                    "early_stop_min_delta": 1.0e-6,
                    "stopped_early": False,
                    "early_stop_reason": "hard_cap_reached",
                    "checkpoint_reusable": True,
                    "checkpoint_selected_by_validation": True,
                    "checkpoint_scope": "reusable_formal_full_database_validation_selected_checkpoint",
                    "sampling_executed": True,
                    "sample_step_count": 1,
                    "sample_step_size": 0.1,
                    "runtime_scope": "formal_full_database_train_validation_select_save_reload_sample_after_cpu_warmup",
                    "runtime_seconds_reliable": True,
                    "runtime_seconds": 10.0 if mode == "isotropic" else 11.0,
                    "dataset": {
                        "name": "geom_drugs",
                        "kind": "geom_drugs_tiny_processed_dry_run",
                        "subset_scope": FULL_DATABASE_SCOPE,
                        "molecule_count": 6,
                        "split_type": "random",
                        "split_scope": FULL_DATABASE_SCOPE,
                        "split_counts": {"train": 4, "val": 1, "test": 1},
                        "degenerate_tiny_split": False,
                    },
                    "metrics": {
                        "metric_scope": "geom_drugs_formal_full_database_training_checkpoint_sampling",
                        "executed_train_steps": 2000,
                        "best_validation_loss": best_loss,
                        "best_validation_loss_step": 2000,
                        "reloaded_validation_loss": best_loss,
                        "sampled_molecules": [
                            {
                                "molecule_id": "train_a",
                                "split": "train",
                                "sampled_rmsd_to_target": rmsd_base,
                            },
                            {
                                "molecule_id": "train_b",
                                "split": "train",
                                "sampled_rmsd_to_target": rmsd_base + 0.02,
                            },
                            {
                                "molecule_id": "train_c",
                                "split": "train",
                                "sampled_rmsd_to_target": rmsd_base + 0.04,
                            },
                            {
                                "molecule_id": "train_d",
                                "split": "train",
                                "sampled_rmsd_to_target": rmsd_base + 0.06,
                            },
                            {
                                "molecule_id": "val_a",
                                "split": "val",
                                "sampled_rmsd_to_target": rmsd_base + 0.08,
                            },
                            {
                                "molecule_id": "test_a",
                                "split": "test",
                                "sampled_rmsd_to_target": rmsd_base + 0.10,
                            },
                        ],
                    },
                }
                path = root / f"{label}.json"
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                result_paths.append(path)

            first = write_formal_full_database_seed0_comparison(
                result_paths=result_paths,
                summary_path=root / "summary.json",
                report_path=root / "summary.md",
            )
            first_summary = json.loads(first["summary"].read_text(encoding="utf-8"))
            first_report = first["report"].read_text(encoding="utf-8")
            second = write_formal_full_database_seed0_comparison(
                result_paths=result_paths,
                summary_path=root / "summary_again.json",
                report_path=root / "summary_again.md",
            )
            second_summary = json.loads(second["summary"].read_text(encoding="utf-8"))

            self.assertEqual(first_summary, second_summary)
            self.assertEqual(first_summary["result_kind"], FORMAL_FULL_DATABASE_SUMMARY_RESULT_KIND)
            self.assertFalse(first_summary["benchmark_result"])
            self.assertFalse(first_summary["training_executed_by_summary"])
            self.assertFalse(first_summary["sampling_executed_by_summary"])
            self.assertEqual(first_summary["dataset"]["split_counts"], {"train": 4, "val": 1, "test": 1})
            self.assertEqual(
                first_summary["directional_verdict"]["verdict"],
                "stiffness_better_on_all_tracked_metrics",
            )
            self.assertEqual(first_summary["directional_verdict"]["stiffness_win_count"], 6)
            self.assertIn("Full-Database Formal Seed0 Comparison", first_report)
            self.assertIn("stiffness_better_on_all_tracked_metrics", first_report)

    def test_formal_full_database_scaffold_seed0_comparison_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_paths = []
            for mode, label, best_loss, rmsd_base in [
                ("isotropic", "isotropic", 0.40, 0.50),
                ("stiffness_structured", "stiffness", 0.30, 0.40),
            ]:
                payload = {
                    "run_name": f"geom_formal_full_database_scaffold_train_{label}_steps2000_seed0",
                    "result_kind": "geom_drugs_full_database_formal_training_not_benchmark",
                    "benchmark_result": False,
                    "dataset_subset_scope": FULL_DATABASE_SCOPE,
                    "processed_artifact_path": "data/processed/geom_drugs_full_database/molecules.json",
                    "split_path": "data/splits/geom_drugs_full_database/scaffold_seed0/split.json",
                    "random_seed": 0,
                    "corruption_mode": mode,
                    "model_name": "single_linear_velocity_smoke",
                    "backbone": "minimal_linear_velocity_pilot",
                    "hardware": "gpu_enabled",
                    "gpu_used": True,
                    "scheduler": "slurm",
                    "slurm_account": "torch_pr_872_general",
                    "formal_training_run": True,
                    "full_database_formal_training_run": True,
                    "max_steps": 2000,
                    "hard_max_steps": 2000,
                    "optimizer_name": "SGD",
                    "learning_rate": 1.0e-2,
                    "optimizer_step_count": 2000,
                    "training_sample_count": 4,
                    "validation_sample_count": 1,
                    "sample_count": 6,
                    "early_stop_monitor": "validation_loss_after_step",
                    "early_stop_patience": 200,
                    "early_stop_min_delta": 1.0e-6,
                    "stopped_early": False,
                    "early_stop_reason": "hard_cap_reached",
                    "checkpoint_reusable": True,
                    "checkpoint_selected_by_validation": True,
                    "checkpoint_scope": "reusable_formal_full_database_validation_selected_checkpoint",
                    "sampling_executed": True,
                    "sample_step_count": 1,
                    "sample_step_size": 0.1,
                    "runtime_scope": "formal_full_database_train_validation_select_save_reload_sample_after_cpu_warmup",
                    "runtime_seconds_reliable": True,
                    "runtime_seconds": 10.0 if mode == "isotropic" else 11.0,
                    "dataset": {
                        "name": "geom_drugs",
                        "kind": "geom_drugs_tiny_processed_dry_run",
                        "subset_scope": FULL_DATABASE_SCOPE,
                        "molecule_count": 6,
                        "split_type": "scaffold",
                        "split_scope": FULL_DATABASE_SCOPE,
                        "split_counts": {"train": 4, "val": 1, "test": 1},
                        "degenerate_tiny_split": False,
                    },
                    "metrics": {
                        "metric_scope": "geom_drugs_formal_full_database_training_checkpoint_sampling",
                        "executed_train_steps": 2000,
                        "best_validation_loss": best_loss,
                        "best_validation_loss_step": 2000,
                        "reloaded_validation_loss": best_loss,
                        "sampled_molecules": [
                            {
                                "molecule_id": "train_a",
                                "split": "train",
                                "sampled_rmsd_to_target": rmsd_base,
                            },
                            {
                                "molecule_id": "train_b",
                                "split": "train",
                                "sampled_rmsd_to_target": rmsd_base + 0.02,
                            },
                            {
                                "molecule_id": "train_c",
                                "split": "train",
                                "sampled_rmsd_to_target": rmsd_base + 0.04,
                            },
                            {
                                "molecule_id": "train_d",
                                "split": "train",
                                "sampled_rmsd_to_target": rmsd_base + 0.06,
                            },
                            {
                                "molecule_id": "val_a",
                                "split": "val",
                                "sampled_rmsd_to_target": rmsd_base + 0.08,
                            },
                            {
                                "molecule_id": "test_a",
                                "split": "test",
                                "sampled_rmsd_to_target": rmsd_base + 0.10,
                            },
                        ],
                    },
                }
                path = root / f"scaffold_{label}.json"
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                result_paths.append(path)

            first = write_formal_full_database_scaffold_seed0_comparison(
                result_paths=result_paths,
                summary_path=root / "scaffold_summary.json",
                report_path=root / "scaffold_summary.md",
            )
            first_summary = json.loads(first["summary"].read_text(encoding="utf-8"))
            first_report = first["report"].read_text(encoding="utf-8")
            second = write_formal_full_database_scaffold_seed0_comparison(
                result_paths=result_paths,
                summary_path=root / "scaffold_summary_again.json",
                report_path=root / "scaffold_summary_again.md",
            )
            second_summary = json.loads(second["summary"].read_text(encoding="utf-8"))

            self.assertEqual(first_summary, second_summary)
            self.assertEqual(
                first_summary["result_kind"],
                FORMAL_FULL_DATABASE_SCAFFOLD_SEED0_SUMMARY_RESULT_KIND,
            )
            self.assertFalse(first_summary["benchmark_result"])
            self.assertFalse(first_summary["training_executed_by_summary"])
            self.assertFalse(first_summary["sampling_executed_by_summary"])
            self.assertEqual(first_summary["dataset"]["split_type"], "scaffold")
            self.assertEqual(first_summary["dataset"]["split_counts"], {"train": 4, "val": 1, "test": 1})
            self.assertEqual(
                first_summary["directional_verdict"]["verdict"],
                "stiffness_better_on_all_tracked_metrics",
            )
            self.assertEqual(first_summary["directional_verdict"]["stiffness_win_count"], 6)
            self.assertTrue(all(row["gpu_used"] for row in first_summary["rows"]))
            self.assertIn("Full-Database Scaffold Formal Seed0 Comparison", first_report)
            self.assertIn("Split type: `scaffold`", first_report)

    def test_formal_full_database_multiseed_summary_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_paths = []
            for seed in [0, 1, 2]:
                for mode, label, loss_offset, rmsd_offset in [
                    ("isotropic", "isotropic", 0.10, 0.20),
                    ("stiffness_structured", "stiffness", 0.05, 0.10),
                ]:
                    best_loss = 0.30 + 0.01 * seed + loss_offset
                    rmsd_base = 0.40 + 0.02 * seed + rmsd_offset
                    payload = {
                        "run_name": f"geom_formal_full_database_train_{label}_steps2000_seed{seed}",
                        "result_kind": "geom_drugs_full_database_formal_training_not_benchmark",
                        "benchmark_result": False,
                        "dataset_subset_scope": FULL_DATABASE_SCOPE,
                        "processed_artifact_path": f"data/processed/geom_drugs_full_database_runs/{label}_seed{seed}/molecules.json",
                        "split_path": f"data/splits/geom_drugs_full_database_runs/{label}_seed{seed}/random_seed0/split.json",
                        "random_seed": seed,
                        "corruption_mode": mode,
                        "model_name": "single_linear_velocity_smoke",
                        "backbone": "minimal_linear_velocity_pilot",
                        "hardware": "cpu_only" if seed == 0 else "gpu_enabled",
                        "gpu_used": seed != 0,
                        "scheduler": "slurm",
                        "slurm_account": "torch_pr_872_general",
                        "formal_training_run": True,
                        "full_database_formal_training_run": True,
                        "max_steps": 2000,
                        "hard_max_steps": 2000,
                        "optimizer_name": "SGD",
                        "learning_rate": 1.0e-2,
                        "optimizer_step_count": 2000,
                        "training_sample_count": 4,
                        "validation_sample_count": 1,
                        "sample_count": 6,
                        "early_stop_monitor": "validation_loss_after_step",
                        "early_stop_patience": 200,
                        "early_stop_min_delta": 1.0e-6,
                        "stopped_early": False,
                        "early_stop_reason": "hard_cap_reached",
                        "checkpoint_reusable": True,
                        "checkpoint_selected_by_validation": True,
                        "checkpoint_scope": "reusable_formal_full_database_validation_selected_checkpoint",
                        "sampling_executed": True,
                        "sample_step_count": 1,
                        "sample_step_size": 0.1,
                        "runtime_scope": "formal_full_database_train_validation_select_save_reload_sample_after_cpu_warmup",
                        "runtime_seconds_reliable": True,
                        "runtime_seconds": 10.0 + seed,
                        "dataset": {
                            "name": "geom_drugs",
                            "kind": "geom_drugs_tiny_processed_dry_run",
                            "subset_scope": FULL_DATABASE_SCOPE,
                            "molecule_count": 6,
                            "split_type": "random",
                            "split_scope": FULL_DATABASE_SCOPE,
                            "split_counts": {"train": 4, "val": 1, "test": 1},
                            "degenerate_tiny_split": False,
                        },
                        "metrics": {
                            "metric_scope": "geom_drugs_formal_full_database_training_checkpoint_sampling",
                            "executed_train_steps": 2000,
                            "best_validation_loss": best_loss,
                            "best_validation_loss_step": 2000,
                            "reloaded_validation_loss": best_loss,
                            "sampled_molecules": [
                                {
                                    "molecule_id": "train_a",
                                    "split": "train",
                                    "sampled_rmsd_to_target": rmsd_base,
                                },
                                {
                                    "molecule_id": "train_b",
                                    "split": "train",
                                    "sampled_rmsd_to_target": rmsd_base + 0.02,
                                },
                                {
                                    "molecule_id": "train_c",
                                    "split": "train",
                                    "sampled_rmsd_to_target": rmsd_base + 0.04,
                                },
                                {
                                    "molecule_id": "train_d",
                                    "split": "train",
                                    "sampled_rmsd_to_target": rmsd_base + 0.06,
                                },
                                {
                                    "molecule_id": "val_a",
                                    "split": "val",
                                    "sampled_rmsd_to_target": rmsd_base + 0.08,
                                },
                                {
                                    "molecule_id": "test_a",
                                    "split": "test",
                                    "sampled_rmsd_to_target": rmsd_base + 0.10,
                                },
                            ],
                        },
                    }
                    path = root / f"{label}_seed{seed}.json"
                    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    result_paths.append(path)

            first = write_formal_full_database_multiseed_summary(
                result_paths=result_paths,
                summary_path=root / "multiseed_summary.json",
                report_path=root / "multiseed_summary.md",
            )
            first_summary = json.loads(first["summary"].read_text(encoding="utf-8"))
            first_report = first["report"].read_text(encoding="utf-8")
            second = write_formal_full_database_multiseed_summary(
                result_paths=result_paths,
                summary_path=root / "multiseed_summary_again.json",
                report_path=root / "multiseed_summary_again.md",
            )
            second_summary = json.loads(second["summary"].read_text(encoding="utf-8"))

            self.assertEqual(first_summary, second_summary)
            self.assertEqual(first_summary["result_kind"], FORMAL_FULL_DATABASE_MULTI_SEED_SUMMARY_RESULT_KIND)
            self.assertFalse(first_summary["benchmark_result"])
            self.assertFalse(first_summary["training_executed_by_summary"])
            self.assertFalse(first_summary["sampling_executed_by_summary"])
            self.assertEqual(first_summary["seed_set"], [0, 1, 2])
            self.assertEqual(first_summary["dataset"]["split_counts"], {"train": 4, "val": 1, "test": 1})
            self.assertEqual(
                first_summary["directional_verdict"]["verdict"],
                "stiffness_better_on_all_tracked_metrics_all_seeds",
            )
            self.assertEqual(first_summary["directional_verdict"]["stiffness_win_count"], 18)
            self.assertEqual(first_summary["directional_verdict"]["tracked_metric_seed_comparison_count"], 18)
            self.assertIn("Full-Database Formal Multi-Seed Summary", first_report)
            self.assertIn("stiffness_better_on_all_tracked_metrics_all_seeds", first_report)

    def test_formal_full_database_scaffold_multiseed_summary_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_paths = []
            for seed in [0, 1, 2]:
                for mode, label, loss_offset, rmsd_offset in [
                    ("isotropic", "isotropic", 0.10, 0.20),
                    ("stiffness_structured", "stiffness", 0.05, 0.10),
                ]:
                    best_loss = 0.30 + 0.01 * seed + loss_offset
                    rmsd_base = 0.40 + 0.02 * seed + rmsd_offset
                    payload = {
                        "run_name": f"geom_formal_full_database_scaffold_train_{label}_steps2000_seed{seed}",
                        "result_kind": "geom_drugs_full_database_formal_training_not_benchmark",
                        "benchmark_result": False,
                        "dataset_subset_scope": FULL_DATABASE_SCOPE,
                        "processed_artifact_path": "data/processed/geom_drugs_full_database/molecules.json",
                        "split_path": "data/splits/geom_drugs_full_database/scaffold_seed0/split.json",
                        "random_seed": seed,
                        "corruption_mode": mode,
                        "model_name": "single_linear_velocity_smoke",
                        "backbone": "minimal_linear_velocity_pilot",
                        "hardware": "gpu_enabled",
                        "gpu_used": True,
                        "scheduler": "slurm",
                        "slurm_account": "torch_pr_872_general",
                        "formal_training_run": True,
                        "full_database_formal_training_run": True,
                        "max_steps": 2000,
                        "hard_max_steps": 2000,
                        "optimizer_name": "SGD",
                        "learning_rate": 1.0e-2,
                        "optimizer_step_count": 2000,
                        "training_sample_count": 4,
                        "validation_sample_count": 1,
                        "sample_count": 6,
                        "early_stop_monitor": "validation_loss_after_step",
                        "early_stop_patience": 200,
                        "early_stop_min_delta": 1.0e-6,
                        "stopped_early": False,
                        "early_stop_reason": "hard_cap_reached",
                        "checkpoint_reusable": True,
                        "checkpoint_selected_by_validation": True,
                        "checkpoint_scope": "reusable_formal_full_database_validation_selected_checkpoint",
                        "sampling_executed": True,
                        "sample_step_count": 1,
                        "sample_step_size": 0.1,
                        "runtime_scope": "formal_full_database_train_validation_select_save_reload_sample_after_cpu_warmup",
                        "runtime_seconds_reliable": True,
                        "runtime_seconds": 10.0 + seed,
                        "dataset": {
                            "name": "geom_drugs",
                            "kind": "geom_drugs_tiny_processed_dry_run",
                            "subset_scope": FULL_DATABASE_SCOPE,
                            "molecule_count": 6,
                            "split_type": "scaffold",
                            "split_scope": FULL_DATABASE_SCOPE,
                            "split_counts": {"train": 4, "val": 1, "test": 1},
                            "degenerate_tiny_split": False,
                        },
                        "metrics": {
                            "metric_scope": "geom_drugs_formal_full_database_training_checkpoint_sampling",
                            "executed_train_steps": 2000,
                            "best_validation_loss": best_loss,
                            "best_validation_loss_step": 2000,
                            "reloaded_validation_loss": best_loss,
                            "sampled_molecules": [
                                {
                                    "molecule_id": "train_a",
                                    "split": "train",
                                    "sampled_rmsd_to_target": rmsd_base,
                                },
                                {
                                    "molecule_id": "train_b",
                                    "split": "train",
                                    "sampled_rmsd_to_target": rmsd_base + 0.02,
                                },
                                {
                                    "molecule_id": "train_c",
                                    "split": "train",
                                    "sampled_rmsd_to_target": rmsd_base + 0.04,
                                },
                                {
                                    "molecule_id": "train_d",
                                    "split": "train",
                                    "sampled_rmsd_to_target": rmsd_base + 0.06,
                                },
                                {
                                    "molecule_id": "val_a",
                                    "split": "val",
                                    "sampled_rmsd_to_target": rmsd_base + 0.08,
                                },
                                {
                                    "molecule_id": "test_a",
                                    "split": "test",
                                    "sampled_rmsd_to_target": rmsd_base + 0.10,
                                },
                            ],
                        },
                    }
                    path = root / f"scaffold_{label}_seed{seed}.json"
                    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    result_paths.append(path)

            first = write_formal_full_database_scaffold_multiseed_summary(
                result_paths=result_paths,
                summary_path=root / "scaffold_multiseed_summary.json",
                report_path=root / "scaffold_multiseed_summary.md",
            )
            first_summary = json.loads(first["summary"].read_text(encoding="utf-8"))
            first_report = first["report"].read_text(encoding="utf-8")
            second = write_formal_full_database_scaffold_multiseed_summary(
                result_paths=result_paths,
                summary_path=root / "scaffold_multiseed_summary_again.json",
                report_path=root / "scaffold_multiseed_summary_again.md",
            )
            second_summary = json.loads(second["summary"].read_text(encoding="utf-8"))

            self.assertEqual(first_summary, second_summary)
            self.assertEqual(
                first_summary["result_kind"],
                FORMAL_FULL_DATABASE_SCAFFOLD_MULTI_SEED_SUMMARY_RESULT_KIND,
            )
            self.assertFalse(first_summary["benchmark_result"])
            self.assertFalse(first_summary["training_executed_by_summary"])
            self.assertFalse(first_summary["sampling_executed_by_summary"])
            self.assertEqual(first_summary["seed_set"], [0, 1, 2])
            self.assertEqual(first_summary["dataset"]["split_type"], "scaffold")
            self.assertEqual(first_summary["dataset"]["split_counts"], {"train": 4, "val": 1, "test": 1})
            self.assertEqual(
                first_summary["dataset"]["processed_artifact_path"],
                "data/processed/geom_drugs_full_database/molecules.json",
            )
            self.assertEqual(
                first_summary["dataset"]["split_path"],
                "data/splits/geom_drugs_full_database/scaffold_seed0/split.json",
            )
            self.assertEqual(
                first_summary["directional_verdict"]["verdict"],
                "stiffness_better_on_all_tracked_metrics_all_seeds",
            )
            self.assertEqual(first_summary["directional_verdict"]["stiffness_win_count"], 18)
            self.assertIn("Full-Database Scaffold Formal Multi-Seed Summary", first_report)
            self.assertIn("Split type: `scaffold`", first_report)

    def test_formal_full_database_attribution_summary_covers_A_B_D_E(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_paths = []
            mode_rows = [
                ("isotropic", "isotropic", 0.50, 0.60, 0.10),
                ("harmonic_prior", "harmonic", 0.45, 0.55, 0.08),
                ("torsion_only", "torsion_only", 0.42, 0.52, 0.06),
                ("stiffness_structured", "stiffness", 0.35, 0.45, 0.04),
            ]
            for split_type in ["random", "scaffold"]:
                split_offset = 0.00 if split_type == "random" else 0.05
                split_label = "" if split_type == "random" else "_scaffold"
                split_path = (
                    "data/splits/geom_drugs_full_database/random_seed0/split.json"
                    if split_type == "random"
                    else "data/splits/geom_drugs_full_database/scaffold_seed0/split.json"
                )
                for mode, label, best_loss, rmsd_base, bond_delta in mode_rows:
                    sampled = [
                        {
                            "molecule_id": f"{split_type}_train",
                            "split": "train",
                            "sampled_rmsd_to_target": rmsd_base + split_offset,
                            "sampled_finite_coordinate": True,
                            "sampled_mean_abs_bond_length_delta": bond_delta,
                            "rotatable_bond_count": 0,
                        },
                        {
                            "molecule_id": f"{split_type}_val",
                            "split": "val",
                            "sampled_rmsd_to_target": rmsd_base + split_offset + 0.02,
                            "sampled_finite_coordinate": True,
                            "sampled_mean_abs_bond_length_delta": bond_delta + 0.01,
                            "rotatable_bond_count": 2,
                        },
                        {
                            "molecule_id": f"{split_type}_test",
                            "split": "test",
                            "sampled_rmsd_to_target": rmsd_base + split_offset + 0.04,
                            "sampled_finite_coordinate": True,
                            "sampled_mean_abs_bond_length_delta": bond_delta + 0.02,
                            "rotatable_bond_count": 5,
                        },
                    ]
                    payload = {
                        "run_name": f"geom_formal_full_database{split_label}_train_{label}_steps2000_seed0",
                        "result_kind": "geom_drugs_full_database_formal_training_not_benchmark",
                        "benchmark_result": False,
                        "dataset_subset_scope": FULL_DATABASE_SCOPE,
                        "processed_artifact_path": "data/processed/geom_drugs_full_database/molecules.json",
                        "split_path": split_path,
                        "random_seed": 0,
                        "corruption_mode": mode,
                        "model_name": "single_linear_velocity_smoke",
                        "backbone": "minimal_linear_velocity_pilot",
                        "hardware": "gpu_enabled",
                        "gpu_used": True,
                        "scheduler": "slurm",
                        "slurm_account": "torch_pr_872_general",
                        "formal_training_run": True,
                        "full_database_formal_training_run": True,
                        "max_steps": 2000,
                        "hard_max_steps": 2000,
                        "optimizer_name": "SGD",
                        "learning_rate": 1.0e-2,
                        "optimizer_step_count": 2000,
                        "training_sample_count": 1,
                        "validation_sample_count": 1,
                        "sample_count": 3,
                        "early_stop_monitor": "validation_loss_after_step",
                        "early_stop_patience": 200,
                        "early_stop_min_delta": 1.0e-6,
                        "stopped_early": False,
                        "early_stop_reason": "hard_cap_reached",
                        "checkpoint_reusable": True,
                        "checkpoint_selected_by_validation": True,
                        "checkpoint_scope": "reusable_formal_full_database_validation_selected_checkpoint",
                        "sampling_executed": True,
                        "sample_step_count": 1,
                        "sample_step_size": 0.1,
                        "runtime_scope": "formal_full_database_train_validation_select_save_reload_sample_after_cpu_warmup",
                        "runtime_seconds_reliable": True,
                        "runtime_seconds": 10.0,
                        "dataset": {
                            "name": "geom_drugs",
                            "kind": "geom_drugs_tiny_processed_dry_run",
                            "subset_scope": FULL_DATABASE_SCOPE,
                            "molecule_count": 3,
                            "split_type": split_type,
                            "split_scope": FULL_DATABASE_SCOPE,
                            "split_counts": {"train": 1, "val": 1, "test": 1},
                            "degenerate_tiny_split": False,
                        },
                        "metrics": {
                            "metric_scope": "geom_drugs_formal_full_database_training_checkpoint_sampling",
                            "executed_train_steps": 2000,
                            "best_validation_loss": best_loss + split_offset,
                            "best_validation_loss_step": 2000,
                            "reloaded_validation_loss": best_loss + split_offset,
                            "sampled_molecules": sampled,
                        },
                    }
                    path = root / f"{split_type}_{label}.json"
                    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    result_paths.append(path)

            first = write_formal_attribution_phase_summary(
                result_paths=result_paths,
                summary_path=root / "attribution_summary.json",
                report_path=root / "attribution_summary.md",
                expected_split_types=("random", "scaffold"),
                expected_seeds=(0,),
            )
            first_summary = json.loads(first["summary"].read_text(encoding="utf-8"))
            first_report = first["report"].read_text(encoding="utf-8")
            second = write_formal_attribution_phase_summary(
                result_paths=result_paths,
                summary_path=root / "attribution_summary_again.json",
                report_path=root / "attribution_summary_again.md",
                expected_split_types=("random", "scaffold"),
                expected_seeds=(0,),
            )
            second_summary = json.loads(second["summary"].read_text(encoding="utf-8"))

            self.assertEqual(first_summary, second_summary)
            self.assertEqual(first_summary["result_kind"], ATTRIBUTION_SUMMARY_RESULT_KIND)
            self.assertFalse(first_summary["benchmark_result"])
            self.assertFalse(first_summary["training_executed_by_summary"])
            self.assertFalse(first_summary["sampling_executed_by_summary"])
            self.assertEqual(first_summary["row_count"], 8)
            self.assertEqual(
                set(first_summary["attribution_modes"]),
                {"isotropic", "harmonic_prior", "torsion_only", "stiffness_structured"},
            )
            self.assertEqual(first_summary["physical_validity_diagnostics"]["status"], "available")
            self.assertEqual(first_summary["rotatable_bond_stratification"]["status"], "available")
            self.assertEqual(
                first_summary["directional_verdict"]["verdict"],
                "stiffness_better_than_all_A_B_D_baselines_on_all_core_metrics",
            )
            self.assertIn("Full-Database Attribution Phase Summary", first_report)
            self.assertIn("Harmonic", first_report)
            self.assertIn("Rotatable-Bond Stratification", first_report)

    def test_formal_10000_summary_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_paths = []
            sampled_template = [
                ("formal_train", "train", 0),
                ("formal_val", "val", 1),
                ("formal_test", "test", 2),
            ]
            for seed in [0, 1, 2]:
                for mode, label, offset in [
                    ("isotropic", "isotropic", 0.40),
                    ("stiffness_structured", "stiffness", 0.30),
                ]:
                    path = root / f"{label}_seed{seed}.json"
                    payload = {
                        "run_name": f"geom_formal_10000_train_{label}_mols10000_steps2000_seed{seed}",
                        "result_kind": FORMAL_10000_RESULT_KIND,
                        "benchmark_result": False,
                        "hardware": "cpu_only",
                        "gpu_used": False,
                        "scheduler": None,
                        "slurm_account": None,
                        "random_seed": seed,
                        "corruption_mode": mode,
                        "sample_count": 10_000,
                        "training_sample_count": 9000,
                        "validation_sample_count": 500,
                        "max_steps": 2000,
                        "hard_max_steps": 2000,
                        "optimizer_name": "SGD",
                        "learning_rate": 1.0e-2,
                        "optimizer_step_count": 1300 + seed,
                        "stopped_early": True,
                        "early_stop_reason": "validation_patience_exhausted",
                        "checkpoint_reusable": True,
                        "checkpoint_selected_by_validation": True,
                        "checkpoint_scope": FORMAL_10000_CHECKPOINT_SCOPE,
                        "sample_step_count": 1,
                        "sample_step_size": 0.1,
                        "dataset": {
                            "subset_scope": "data_scale_check_10000_processed_dry_run",
                            "split_counts": {"train": 9000, "val": 500, "test": 500},
                        },
                        "metrics": {
                            "best_validation_loss": 0.70 + 0.01 * seed if mode == "isotropic" else 0.60 + 0.01 * seed,
                            "best_validation_loss_step": 900 + seed,
                            "reloaded_validation_loss": 0.71 + 0.01 * seed
                            if mode == "isotropic"
                            else 0.61 + 0.01 * seed,
                            "reloaded_train_loss": 0.72 + 0.01 * seed
                            if mode == "isotropic"
                            else 0.62 + 0.01 * seed,
                            "sampled_molecules": [
                                {
                                    "molecule_id": molecule_id,
                                    "split": split,
                                    "sampled_rmsd_to_target": offset + 0.01 * split_index + 0.001 * seed,
                                }
                                for molecule_id, split, split_index in sampled_template
                            ],
                        },
                    }
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    result_paths.append(path)

            first = write_formal_10000_training_multiseed_summary(
                result_paths=result_paths,
                summary_path=root / "summary.json",
                report_path=root / "summary.md",
            )
            first_summary_text = first["summary"].read_text(encoding="utf-8")
            second = write_formal_10000_training_multiseed_summary(
                result_paths=result_paths,
                summary_path=root / "summary2.json",
                report_path=root / "summary2.md",
            )
            self.assertEqual(first_summary_text, second["summary"].read_text(encoding="utf-8"))
            summary = json.loads(first_summary_text)
            report = first["report"].read_text(encoding="utf-8")
            self.assertEqual(
                summary["result_kind"],
                "tiny_real_data_formal_10000_training_multiseed_summary_not_benchmark",
            )
            self.assertFalse(summary["benchmark_result"])
            self.assertEqual(summary["row_count"], 6)
            self.assertEqual(summary["seed_set"], [0, 1, 2])
            self.assertEqual(summary["stiffness_consistency"]["verdict"], "stiffness_wins_consistently")
            self.assertIn("Formal 10000-Molecule Training Multi-Seed Summary", report)

    def test_data_scale_check_10000_convergence_multiseed_summary_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_paths = []
            sampled_template = [
                ("geom_scale10000_train", "train", 0),
                ("geom_scale10000_val", "val", 1),
                ("geom_scale10000_test", "test", 2),
            ]
            for seed in [0, 1, 2]:
                for mode, label, offset, steps in [
                    ("isotropic", "isotropic", 0.40, 52),
                    ("stiffness_structured", "stiffness", 0.30, 47),
                ]:
                    path = root / f"{label}_seed{seed}.json"
                    payload = {
                        "run_name": f"geom_data_scale_check_ckpt_sample_{label}_mols10000_steps200_seed{seed}",
                        "dataset_name": "geom_drugs",
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "dataset_subset_scope": "data_scale_check_10000_processed_dry_run",
                        "processed_artifact_path": "data/processed/geom_drugs_subset/data_scale_check_10000/molecules.json",
                        "split_path": "data/splits/geom_drugs_data_scale_check_10000/random_seed0/split.json",
                        "random_seed": seed,
                        "corruption_mode": mode,
                        "model_name": "single_linear_velocity_smoke",
                        "backbone": "minimal_linear_velocity_pilot",
                        "execution_mode": "local_cpu_tiny_real_data_scale_check_checkpoint_sampling_convergence",
                        "result_kind": "tiny_real_data_bounded_data_scale_check_checkpoint_sampling_not_benchmark",
                        "benchmark_result": False,
                        "hardware": "cpu_only",
                        "gpu_used": False,
                        "scheduler": None,
                        "slurm_account": None,
                        "pilot_run": True,
                        "bounded_pilot": True,
                        "micro_pilot_run": False,
                        "data_scale_check_run": True,
                        "data_scale_check_10000_run": True,
                        "data_scale_check_10000_convergence_run": True,
                        "dry_run": False,
                        "training_executed": True,
                        "model_training_started": True,
                        "training_loop_executed": True,
                        "bounded_real_training_loop_executed": True,
                        "larger_real_training_run_started": False,
                        "max_steps": 200,
                        "hard_max_steps": 200,
                        "optimizer_created": True,
                        "optimizer_step_count": steps + seed,
                        "backward_executed": True,
                        "backward_pass_count": steps + seed,
                        "checkpoint_path": f"experiments/checkpoints/{label}_seed{seed}.pt",
                        "checkpoint_written": True,
                        "checkpoint_reloaded": True,
                        "checkpoint_scope": "disposable_data_scale_check_10000_checkpoint_sampling_only",
                        "sampling_executed": True,
                        "sample_step_count": 1,
                        "sample_step_size": 0.1,
                        "forward_executed": True,
                        "loss_computed": True,
                        "training_sample_count": 9000,
                        "sample_count": 10_000,
                        "runtime_scope": "checkpoint_train_save_reload_sample_after_full_cpu_checkpoint_sampling_warmup",
                        "runtime_seconds_reliable": True,
                        "configured_max_steps": 200,
                        "early_stop_enabled": True,
                        "early_stop_patience": 20,
                        "early_stop_min_delta": 1.0e-6,
                        "early_stop_monitor": "train_loss_before_step",
                        "stopped_early": True,
                        "early_stop_reason": "patience_exhausted",
                        "dataset": {
                            "name": "geom_drugs",
                            "kind": "geom_drugs_tiny_processed_dry_run",
                            "subset_scope": "data_scale_check_10000_processed_dry_run",
                            "molecule_count": 10_000,
                            "split_type": "random",
                            "split_scope": "data_scale_check_10000_processed_dry_run",
                            "split_counts": {"train": 9000, "val": 500, "test": 500},
                            "degenerate_tiny_split": False,
                        },
                        "flow_matching": {"objective": "cartesian_linear", "interpolation": "linear"},
                        "metrics": {
                            "metric_scope": "geom_drugs_data_scale_check_checkpoint_sampling",
                            "model_name": "single_linear_velocity_smoke",
                            "reloaded_loss": 0.70 + 0.01 * seed if mode == "isotropic" else 0.60 + 0.01 * seed,
                            "executed_train_steps": steps + seed,
                            "early_stop_enabled": True,
                            "stopped_early": True,
                            "stop_reason": "patience_exhausted",
                            "best_train_loss": 0.50 if mode == "isotropic" else 0.45,
                            "best_train_loss_step": 5,
                            "train_steps": [{"optimizer_step": index + 1} for index in range(steps + seed)],
                            "sampled_molecules": [
                                {
                                    "molecule_id": molecule_id,
                                    "split": split,
                                    "atom_count": 3,
                                    "sample_step_count": 1,
                                    "sample_step_size": 0.1,
                                    "sampled_rmsd_to_target": offset + 0.01 * split_index + 0.001 * seed,
                                    "predicted_velocity_norm": 1.0,
                                }
                                for molecule_id, split, split_index in sampled_template
                            ],
                        },
                        "runtime_seconds": 12.0 + 0.1 * seed,
                        "notes": ["Bounded 10000-molecule convergence data-scale-check pilot only, not benchmark results."],
                    }
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    result_paths.append(path)

            first = write_data_scale_check_checkpoint_sampling_multiseed_summary(
                result_paths=result_paths,
                summary_path=root / "summary.json",
                report_path=root / "summary.md",
                expected_seeds=(0, 1, 2),
                max_molecules=10_000,
                max_steps=200,
                summary_name="geom_data_scale_check_ckpt_sample_multiseed_summary_mols10000_steps200_convergence",
                report_title="Bounded 10000-Molecule Checkpoint/Sampling Convergence Multi-Seed Summary",
            )
            first_summary_text = first["summary"].read_text(encoding="utf-8")
            second = write_data_scale_check_checkpoint_sampling_multiseed_summary(
                result_paths=result_paths,
                summary_path=root / "summary2.json",
                report_path=root / "summary2.md",
                expected_seeds=(0, 1, 2),
                max_molecules=10_000,
                max_steps=200,
                summary_name="geom_data_scale_check_ckpt_sample_multiseed_summary_mols10000_steps200_convergence",
                report_title="Bounded 10000-Molecule Checkpoint/Sampling Convergence Multi-Seed Summary",
            )
            self.assertEqual(first_summary_text, second["summary"].read_text(encoding="utf-8"))

            summary = json.loads(first_summary_text)
            report = first["report"].read_text(encoding="utf-8")
            self.assertEqual(
                summary["summary_name"],
                "geom_data_scale_check_ckpt_sample_multiseed_summary_mols10000_steps200_convergence",
            )
            self.assertTrue(summary["data_scale_check_10000_convergence_run"])
            self.assertEqual(summary["max_steps"], 200)
            self.assertEqual(summary["row_count"], 6)
            self.assertEqual(summary["stiffness_consistency"]["verdict"], "stiffness_wins_consistently")
            self.assertIsNotNone(summary["early_stop_summary"])
            self.assertEqual(summary["early_stop_summary"]["early_stop_patience_set"], [20])
            self.assertEqual(summary["early_stop_summary"]["stopped_early_count_by_mode"]["isotropic"], 3)
            self.assertEqual(
                summary["early_stop_summary"]["executed_optimizer_steps_by_mode"]["stiffness_structured"],
                [47, 48, 49],
            )
            self.assertIn("Bounded 10000-Molecule Checkpoint/Sampling Convergence Multi-Seed Summary", report)
            self.assertIn("Early Stop Accounting", report)

    def test_data_scale_check_10000_convergence_diagnostics_detects_downward_hard_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_paths = []
            for seed in [0, 1, 2]:
                for mode, label, base_loss in [
                    ("isotropic", "isotropic", 1.20 + 0.01 * seed),
                    ("stiffness_structured", "stiffness", 1.00 + 0.01 * seed),
                ]:
                    path = root / f"{label}_seed{seed}.json"
                    train_steps = [
                        {
                            "optimizer_step": index + 1,
                            "loss_before_step": base_loss - 0.001 * index,
                            "gradient_l2_norm": 0.5 - 0.001 * min(index, 100),
                        }
                        for index in range(200)
                    ]
                    payload = {
                        "run_name": f"geom_data_scale_check_ckpt_sample_{label}_mols10000_steps200_seed{seed}",
                        "dataset_name": "geom_drugs",
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "dataset_subset_scope": "data_scale_check_10000_processed_dry_run",
                        "processed_artifact_path": "data/processed/geom_drugs_subset/data_scale_check_10000/molecules.json",
                        "split_path": "data/splits/geom_drugs_data_scale_check_10000/random_seed0/split.json",
                        "random_seed": seed,
                        "corruption_mode": mode,
                        "model_name": "single_linear_velocity_smoke",
                        "backbone": "minimal_linear_velocity_pilot",
                        "execution_mode": "local_cpu_tiny_real_data_scale_check_checkpoint_sampling_convergence",
                        "result_kind": "tiny_real_data_bounded_data_scale_check_checkpoint_sampling_not_benchmark",
                        "benchmark_result": False,
                        "hardware": "cpu_only",
                        "gpu_used": False,
                        "scheduler": None,
                        "slurm_account": None,
                        "pilot_run": True,
                        "bounded_pilot": True,
                        "micro_pilot_run": False,
                        "data_scale_check_run": True,
                        "data_scale_check_10000_run": True,
                        "data_scale_check_10000_convergence_run": True,
                        "dry_run": False,
                        "training_executed": True,
                        "model_training_started": True,
                        "training_loop_executed": True,
                        "bounded_real_training_loop_executed": True,
                        "larger_real_training_run_started": False,
                        "max_steps": 200,
                        "hard_max_steps": 200,
                        "optimizer_created": True,
                        "optimizer_step_count": 200,
                        "backward_executed": True,
                        "backward_pass_count": 200,
                        "checkpoint_path": f"experiments/checkpoints/{label}_seed{seed}.pt",
                        "checkpoint_written": True,
                        "checkpoint_reloaded": True,
                        "checkpoint_scope": "disposable_data_scale_check_10000_checkpoint_sampling_only",
                        "sampling_executed": True,
                        "sample_step_count": 1,
                        "sample_step_size": 0.1,
                        "forward_executed": True,
                        "loss_computed": True,
                        "training_sample_count": 9000,
                        "sample_count": 10_000,
                        "runtime_scope": "checkpoint_train_save_reload_sample_after_full_cpu_checkpoint_sampling_warmup",
                        "runtime_seconds_reliable": True,
                        "configured_max_steps": 200,
                        "early_stop_enabled": True,
                        "early_stop_patience": 20,
                        "early_stop_min_delta": 1.0e-6,
                        "early_stop_monitor": "train_loss_before_step",
                        "stopped_early": False,
                        "early_stop_reason": "hard_cap_reached",
                        "dataset": {
                            "name": "geom_drugs",
                            "kind": "geom_drugs_tiny_processed_dry_run",
                            "subset_scope": "data_scale_check_10000_processed_dry_run",
                            "molecule_count": 10_000,
                            "split_type": "random",
                            "split_scope": "data_scale_check_10000_processed_dry_run",
                            "split_counts": {"train": 9000, "val": 500, "test": 500},
                            "degenerate_tiny_split": False,
                        },
                        "flow_matching": {"objective": "cartesian_linear", "interpolation": "linear"},
                        "metrics": {
                            "metric_scope": "geom_drugs_data_scale_check_checkpoint_sampling",
                            "model_name": "single_linear_velocity_smoke",
                            "reloaded_loss": 0.70 + 0.01 * seed if mode == "isotropic" else 0.60 + 0.01 * seed,
                            "executed_train_steps": 200,
                            "early_stop_enabled": True,
                            "stopped_early": False,
                            "stop_reason": "hard_cap_reached",
                            "best_train_loss": train_steps[-1]["loss_before_step"],
                            "best_train_loss_step": 200,
                            "train_steps": train_steps,
                            "sampled_molecules": [],
                        },
                        "runtime_seconds": 12.0 + 0.1 * seed,
                        "notes": ["Bounded 10000-molecule convergence data-scale-check pilot only, not benchmark results."],
                    }
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    result_paths.append(path)

            first = write_data_scale_check_convergence_diagnostics(
                result_paths=result_paths,
                summary_path=root / "diagnostics.json",
                report_path=root / "diagnostics.md",
            )
            first_summary_text = first["summary"].read_text(encoding="utf-8")
            second = write_data_scale_check_convergence_diagnostics(
                result_paths=result_paths,
                summary_path=root / "diagnostics2.json",
                report_path=root / "diagnostics2.md",
            )
            self.assertEqual(first_summary_text, second["summary"].read_text(encoding="utf-8"))

            summary = json.loads(first_summary_text)
            report = first["report"].read_text(encoding="utf-8")
            self.assertEqual(
                summary["result_kind"],
                "tiny_real_data_bounded_data_scale_check_convergence_diagnostics_not_benchmark",
            )
            self.assertFalse(summary["benchmark_result"])
            self.assertFalse(summary["training_executed_by_summary"])
            self.assertFalse(summary["sampling_executed_by_summary"])
            self.assertEqual(summary["row_count"], 6)
            self.assertEqual(summary["diagnostic_verdict"]["verdict"], "still_downward_at_hard_cap")
            self.assertTrue(summary["diagnostic_verdict"]["all_hit_hard_cap"])
            self.assertTrue(summary["diagnostic_verdict"]["all_best_at_final_step"])
            self.assertTrue(summary["diagnostic_verdict"]["all_tail_transitions_improve_more_than_min_delta"])
            self.assertEqual(summary["aggregate_by_mode"]["isotropic"]["hit_hard_cap_count"], 3)
            self.assertEqual(summary["aggregate_by_mode"]["stiffness_structured"]["best_at_final_step_count"], 3)
            self.assertIn("Bounded 10000-Molecule Convergence Diagnostics", report)
            self.assertIn("still_downward_at_hard_cap", report)

    def test_micro_pilot_checkpoint_sampling_comparison_flags_legacy_runtime_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_paths = []
            molecule_ids = [f"geom_micro_{index}" for index in range(6)]
            split_by_index = ["train", "train", "train", "test", "train", "val"]
            rmsd_by_mode = {
                "isotropic": [0.20, 0.30, 0.40, 0.50, 0.35, 0.45],
                "stiffness_structured": [0.10, 0.25, 0.45, 0.40, 0.30, 0.44],
            }
            for mode, label, runtime in [
                ("isotropic", "isotropic", 17.0),
                ("stiffness_structured", "stiffness", 0.07),
            ]:
                path = root / f"{label}.json"
                payload = {
                    "run_name": f"geom_micro_pilot_ckpt_sample_{label}_mols6_steps5_seed0",
                    "dataset_name": "geom_drugs",
                    "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                    "dataset_subset_scope": "micro_pilot_processed_dry_run",
                    "processed_artifact_path": "data/processed/geom_drugs_subset/micro_pilot/molecules.json",
                    "split_path": "data/splits/geom_drugs_micro_pilot/random_seed0/split.json",
                    "random_seed": 0,
                    "corruption_mode": mode,
                    "model_name": "single_linear_velocity_smoke",
                    "backbone": "minimal_linear_velocity_pilot",
                    "execution_mode": "local_cpu_tiny_real_micro_pilot_checkpoint_sampling",
                    "result_kind": "tiny_real_data_bounded_micro_pilot_checkpoint_sampling_not_benchmark",
                    "benchmark_result": False,
                    "hardware": "cpu_only",
                    "gpu_used": False,
                    "scheduler": None,
                    "slurm_account": None,
                    "pilot_run": True,
                    "bounded_pilot": True,
                    "micro_pilot_run": True,
                    "dry_run": False,
                    "training_executed": True,
                    "model_training_started": True,
                    "training_loop_executed": True,
                    "bounded_real_training_loop_executed": True,
                    "larger_real_training_run_started": False,
                    "max_steps": 5,
                    "hard_max_steps": 5,
                    "optimizer_created": True,
                    "optimizer_step_count": 5,
                    "backward_executed": True,
                    "backward_pass_count": 5,
                    "checkpoint_path": f"experiments/checkpoints/{label}.pt",
                    "checkpoint_written": True,
                    "checkpoint_reloaded": True,
                    "checkpoint_scope": "disposable_micro_pilot_checkpoint_sampling_only",
                    "sampling_executed": True,
                    "sample_step_count": 1,
                    "sample_step_size": 0.1,
                    "forward_executed": True,
                    "loss_computed": True,
                    "training_sample_count": 4,
                    "sample_count": 6,
                    "dataset": {
                        "name": "geom_drugs",
                        "kind": "geom_drugs_tiny_processed_dry_run",
                        "subset_scope": "micro_pilot_processed_dry_run",
                        "molecule_count": 6,
                        "split_type": "random",
                        "split_scope": "micro_pilot_processed_dry_run",
                        "split_counts": {"train": 4, "val": 1, "test": 1},
                        "degenerate_tiny_split": False,
                    },
                    "flow_matching": {"objective": "cartesian_linear", "interpolation": "linear"},
                    "metrics": {
                        "metric_scope": "geom_drugs_micro_pilot_checkpoint_sampling",
                        "model_name": "single_linear_velocity_smoke",
                        "reloaded_loss": 0.5 if mode == "isotropic" else 0.4,
                        "train_steps": [{"optimizer_step": index + 1} for index in range(5)],
                        "sampled_molecules": [
                            {
                                "molecule_id": molecule_id,
                                "split": split,
                                "atom_count": 3,
                                "sample_step_count": 1,
                                "sample_step_size": 0.1,
                                "sampled_rmsd_to_target": rmsd,
                                "predicted_velocity_norm": 1.0,
                            }
                            for molecule_id, split, rmsd in zip(molecule_ids, split_by_index, rmsd_by_mode[mode])
                        ],
                    },
                    "runtime_seconds": runtime,
                    "notes": ["Bounded pilot only, not benchmark results."],
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                result_paths.append(path)

            paths = write_micro_pilot_checkpoint_sampling_comparison(
                result_paths=result_paths,
                summary_path=root / "summary.json",
                report_path=root / "summary.md",
            )
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            report = paths["report"].read_text(encoding="utf-8")

            self.assertEqual(
                summary["result_kind"],
                "tiny_real_data_bounded_micro_pilot_checkpoint_sampling_comparison_not_benchmark",
            )
            self.assertFalse(summary["benchmark_result"])
            self.assertFalse(summary["training_executed_by_summary"])
            self.assertFalse(summary["sampling_executed_by_summary"])
            self.assertTrue(summary["runtime_audit"]["suspicious_timing_inconsistency"])
            self.assertFalse(summary["runtime_audit"]["runtime_seconds_comparable"])
            self.assertIn("runtime_ratio_exceeds_threshold", summary["runtime_audit"]["reasons"])
            outcomes = {row["metric"]: row["stiffness_outcome"] for row in summary["metric_comparisons"]}
            self.assertEqual(outcomes["reloaded_loss"], "better")
            self.assertEqual(outcomes["sampled_rmsd_to_target_overall_mean"], "better")
            self.assertEqual(outcomes["sampled_rmsd_to_target_train_mean"], "better")
            self.assertEqual(outcomes["sampled_rmsd_to_target_test_mean"], "better")
            self.assertEqual(outcomes["sampled_rmsd_to_target_val_mean"], "better")
            per_molecule_outcomes = [row["stiffness_outcome"] for row in summary["per_molecule_sampled_rmsd"]]
            self.assertIn("worse", per_molecule_outcomes)
            self.assertIn("Bounded Micro-Pilot Checkpoint/Sampling Comparison", report)
            self.assertIn("suspicious_timing_inconsistency", report)
            self.assertIn("Per-Molecule Sampled RMSD", report)

    def test_micro_pilot_checkpoint_sampling_multiseed_summary_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_paths = []
            molecule_ids = [f"geom_micro_{index}" for index in range(6)]
            split_by_index = ["train", "train", "train", "test", "train", "val"]
            for seed in [0, 1, 2]:
                for mode, label, offset in [
                    ("isotropic", "isotropic", 0.20),
                    ("stiffness_structured", "stiffness", 0.10),
                ]:
                    path = root / f"{label}_seed{seed}.json"
                    sampled = [
                        offset + 0.01 * seed + 0.01 * index
                        for index in range(6)
                    ]
                    payload = {
                        "run_name": f"geom_micro_pilot_ckpt_sample_{label}_mols6_steps5_seed{seed}",
                        "dataset_name": "geom_drugs",
                        "dataset_kind": "geom_drugs_tiny_processed_dry_run",
                        "dataset_subset_scope": "micro_pilot_processed_dry_run",
                        "processed_artifact_path": "data/processed/geom_drugs_subset/micro_pilot/molecules.json",
                        "split_path": "data/splits/geom_drugs_micro_pilot/random_seed0/split.json",
                        "random_seed": seed,
                        "corruption_mode": mode,
                        "model_name": "single_linear_velocity_smoke",
                        "backbone": "minimal_linear_velocity_pilot",
                        "execution_mode": "local_cpu_tiny_real_micro_pilot_checkpoint_sampling",
                        "result_kind": "tiny_real_data_bounded_micro_pilot_checkpoint_sampling_not_benchmark",
                        "benchmark_result": False,
                        "hardware": "cpu_only",
                        "gpu_used": False,
                        "scheduler": None,
                        "slurm_account": None,
                        "pilot_run": True,
                        "bounded_pilot": True,
                        "micro_pilot_run": True,
                        "dry_run": False,
                        "training_executed": True,
                        "model_training_started": True,
                        "training_loop_executed": True,
                        "bounded_real_training_loop_executed": True,
                        "larger_real_training_run_started": False,
                        "max_steps": 5,
                        "hard_max_steps": 5,
                        "optimizer_created": True,
                        "optimizer_step_count": 5,
                        "backward_executed": True,
                        "backward_pass_count": 5,
                        "checkpoint_path": f"experiments/checkpoints/{label}_seed{seed}.pt",
                        "checkpoint_written": True,
                        "checkpoint_reloaded": True,
                        "checkpoint_scope": "disposable_micro_pilot_checkpoint_sampling_only",
                        "sampling_executed": True,
                        "sample_step_count": 1,
                        "sample_step_size": 0.1,
                        "forward_executed": True,
                        "loss_computed": True,
                        "training_sample_count": 4,
                        "sample_count": 6,
                        "runtime_scope": "checkpoint_train_save_reload_sample_after_full_cpu_checkpoint_sampling_warmup",
                        "runtime_seconds_reliable": True,
                        "dataset": {
                            "name": "geom_drugs",
                            "kind": "geom_drugs_tiny_processed_dry_run",
                            "subset_scope": "micro_pilot_processed_dry_run",
                            "molecule_count": 6,
                            "split_type": "random",
                            "split_scope": "micro_pilot_processed_dry_run",
                            "split_counts": {"train": 4, "val": 1, "test": 1},
                            "degenerate_tiny_split": False,
                        },
                        "flow_matching": {"objective": "cartesian_linear", "interpolation": "linear"},
                        "metrics": {
                            "metric_scope": "geom_drugs_micro_pilot_checkpoint_sampling",
                            "model_name": "single_linear_velocity_smoke",
                            "reloaded_loss": 0.50 + 0.01 * seed if mode == "isotropic" else 0.40 + 0.01 * seed,
                            "train_steps": [{"optimizer_step": index + 1} for index in range(5)],
                            "sampled_molecules": [
                                {
                                    "molecule_id": molecule_id,
                                    "split": split,
                                    "atom_count": 3,
                                    "sample_step_count": 1,
                                    "sample_step_size": 0.1,
                                    "sampled_rmsd_to_target": rmsd,
                                    "predicted_velocity_norm": 1.0,
                                }
                                for molecule_id, split, rmsd in zip(molecule_ids, split_by_index, sampled)
                            ],
                        },
                        "runtime_seconds": 0.03 + 0.001 * seed,
                        "notes": ["Bounded pilot only, not benchmark results."],
                    }
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    result_paths.append(path)

            first = write_micro_pilot_checkpoint_sampling_multiseed_summary(
                result_paths=result_paths,
                summary_path=root / "summary.json",
                report_path=root / "summary.md",
                expected_seeds=(0, 1, 2),
            )
            first_summary_text = first["summary"].read_text(encoding="utf-8")
            second = write_micro_pilot_checkpoint_sampling_multiseed_summary(
                result_paths=result_paths,
                summary_path=root / "summary2.json",
                report_path=root / "summary2.md",
                expected_seeds=(0, 1, 2),
            )
            self.assertEqual(first_summary_text, second["summary"].read_text(encoding="utf-8"))

            summary = json.loads(first_summary_text)
            report = first["report"].read_text(encoding="utf-8")
            self.assertEqual(
                summary["result_kind"],
                "tiny_real_data_bounded_micro_pilot_checkpoint_sampling_multiseed_summary_not_benchmark",
            )
            self.assertEqual(summary["seed_set"], [0, 1, 2])
            self.assertEqual(summary["row_count"], 6)
            self.assertEqual(summary["stiffness_consistency"]["verdict"], "stiffness_wins_consistently")
            self.assertTrue(summary["runtime_summary"]["all_runtime_seconds_reliable"])
            aggregate = {row["metric"]: row for row in summary["aggregate_metric_comparisons"]}
            self.assertEqual(aggregate["reloaded_loss"]["stiffness_win_count"], 3)
            self.assertEqual(aggregate["sampled_rmsd_to_target_overall_mean"]["stiffness_win_count"], 3)
            self.assertIn("Bounded Micro-Pilot Checkpoint/Sampling Multi-Seed Summary", report)
            self.assertIn("stiffness_wins_consistently", report)

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

    def test_micro_pilot_multiseed_summary_writes_seed_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            result_paths = []
            for mode, base_loss in [("isotropic", 0.43), ("stiffness_structured", 0.39)]:
                for seed in [0, 1]:
                    path = output_root / f"{mode}_{seed}.json"
                    final_train = base_loss + 0.01 * seed
                    payload = {
                        "run_name": f"micro_{mode}_seed{seed}",
                        "result_kind": "tiny_real_data_bounded_pilot_not_benchmark",
                        "benchmark_result": False,
                        "hardware": "cpu_only",
                        "gpu_used": False,
                        "scheduler": None,
                        "slurm_account": None,
                        "pilot_run": True,
                        "bounded_pilot": True,
                        "micro_pilot_run": True,
                        "random_seed": seed,
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

            paths = write_micro_pilot_multiseed_summary(
                result_paths=result_paths,
                summary_path=output_root / "summary.json",
                report_path=output_root / "summary.md",
                expected_seeds=(0, 1),
            )
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            report = paths["report"].read_text(encoding="utf-8")
            self.assertEqual(
                summary["result_kind"],
                "tiny_real_data_bounded_micro_pilot_multiseed_summary_not_benchmark",
            )
            self.assertFalse(summary["benchmark_result"])
            self.assertFalse(summary["training_executed_by_summary"])
            self.assertEqual(summary["seed_set"], [0, 1])
            self.assertEqual(summary["row_count"], 4)
            self.assertEqual(summary["mode_statistics"]["isotropic"]["seed_count"], 2)
            self.assertEqual(summary["mode_statistics"]["stiffness_structured"]["seed_count"], 2)
            self.assertIn("Tiny Real-Data Bounded Micro-Pilot Multi-Seed Summary", report)
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

    def test_geom_data_scale_check_processed_output_and_split_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "geom_drugs"
            drugs = raw / "rdkit_folder" / "drugs"
            processed = root / "processed"
            drugs.mkdir(parents=True)
            for index in range(30):
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
                with (drugs / f"scale_mol_{index:02d}.pickle").open("wb") as handle:
                    pickle.dump(record, handle)

            paths = run_geom_preflight(
                raw_root=raw,
                processed_root=processed,
                manifest_path=root / "manifest.json",
                report_path=root / "report.md",
                write_data_scale_check_output=True,
                max_processed_conformers_per_mol=1,
            )
            artifact = json.loads(paths["processed_artifact"].read_text(encoding="utf-8"))
            processed_manifest = json.loads(paths["processed_manifest"].read_text(encoding="utf-8"))
            self.assertEqual(artifact["subset_scope"], "data_scale_check_processed_dry_run")
            self.assertEqual(artifact["bounds"]["max_molecules"], 30)
            self.assertEqual(artifact["bounds"]["hard_max_molecules"], 30)
            self.assertEqual(artifact["molecule_count"], 30)
            self.assertEqual(artifact["conformer_count"], 30)
            self.assertFalse(artifact["benchmark_result"])
            self.assertFalse(processed_manifest["full_processed_data_written"])

            split_paths = run_geom_tiny_split_dry_run(
                processed_artifact_path=paths["processed_artifact"],
                split_root=root / "splits",
                split_type="random",
                seed=0,
                max_molecules=30,
                split_scope="data_scale_check_processed_dry_run",
                hard_max_molecules=30,
            )
            split = json.loads(split_paths["split"].read_text(encoding="utf-8"))
            manifest = json.loads(split_paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(split["split_scope"], "data_scale_check_processed_dry_run")
            self.assertFalse(split["benchmark_result"])
            self.assertFalse(split["degenerate_tiny_split"])
            self.assertEqual(len(split["train"]), 28)
            self.assertEqual(len(split["val"]), 1)
            self.assertEqual(len(split["test"]), 1)
            self.assertEqual(manifest["bounds"]["hard_max_molecules"], 30)
            self.assertEqual(manifest["counts"], {"train": 28, "val": 1, "test": 1})

    def test_geom_data_scale_check_500_processed_output_and_split_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "geom_drugs"
            drugs = raw / "rdkit_folder" / "drugs"
            processed = root / "processed"
            drugs.mkdir(parents=True)
            for index in range(500):
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
                with (drugs / f"scale500_mol_{index:03d}.pickle").open("wb") as handle:
                    pickle.dump(record, handle)

            paths = run_geom_preflight(
                raw_root=raw,
                processed_root=processed,
                manifest_path=root / "manifest.json",
                report_path=root / "report.md",
                write_data_scale_check_500_output=True,
                max_processed_conformers_per_mol=1,
            )
            artifact = json.loads(paths["processed_artifact"].read_text(encoding="utf-8"))
            processed_manifest = json.loads(paths["processed_manifest"].read_text(encoding="utf-8"))
            self.assertEqual(artifact["subset_scope"], "data_scale_check_500_processed_dry_run")
            self.assertEqual(artifact["bounds"]["max_molecules"], 500)
            self.assertEqual(artifact["bounds"]["hard_max_molecules"], 500)
            self.assertEqual(artifact["molecule_count"], 500)
            self.assertEqual(artifact["conformer_count"], 500)
            self.assertFalse(artifact["benchmark_result"])
            self.assertFalse(processed_manifest["full_processed_data_written"])

            split_paths = run_geom_tiny_split_dry_run(
                processed_artifact_path=paths["processed_artifact"],
                split_root=root / "splits",
                split_type="random",
                seed=0,
                max_molecules=500,
                split_scope="data_scale_check_500_processed_dry_run",
                hard_max_molecules=500,
            )
            split = json.loads(split_paths["split"].read_text(encoding="utf-8"))
            manifest = json.loads(split_paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(split["split_scope"], "data_scale_check_500_processed_dry_run")
            self.assertFalse(split["benchmark_result"])
            self.assertFalse(split["degenerate_tiny_split"])
            self.assertEqual(len(split["train"]), 450)
            self.assertEqual(len(split["val"]), 25)
            self.assertEqual(len(split["test"]), 25)
            self.assertEqual(manifest["bounds"]["hard_max_molecules"], 500)
            self.assertEqual(manifest["counts"], {"train": 450, "val": 25, "test": 25})


if __name__ == "__main__":
    unittest.main()
