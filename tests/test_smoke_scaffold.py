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
from src.train.pipeline import run_smoke_pipeline
from src.train.smoke import run_smoke
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


if __name__ == "__main__":
    unittest.main()
