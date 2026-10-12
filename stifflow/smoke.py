"""Official QM9 checkpoint smoke test, restricted to Slurm compute allocations."""

import argparse
import importlib.util
import json
import os
import pickle
import time
import traceback
from pathlib import Path

from .preflight import WORKSPACE, audit, fingerprint, provenance


def external_directory(path):
    path = Path(path).resolve()
    for name in (
        "ETFlow-reference",
        "dataset",
        "stifflow-v1",
        "stifflow-v1-alternative-A",
        "stifflow-v1-benchmark",
    ):
        if path.is_relative_to(WORKSPACE / name):
            raise ValueError(
                f"Output must be outside source repositories and datasets: {path}"
            )
    return path


def validate_positions(positions, samples, atoms):
    import numpy as np

    if positions.shape != (samples, atoms, 3):
        raise ValueError(f"Unexpected coordinate shape: {positions.shape}")
    if not np.isfinite(positions).all():
        raise ValueError("Non-finite generated coordinates")
    center_error = float(np.abs(positions.mean(axis=1)).max())
    if center_error > 1e-4:
        raise ValueError(f"Non-centered generated coordinates: {center_error} Angstrom")
    return center_error


def run(args, report):
    import numpy as np
    import torch
    import yaml
    from etflow.commons.covmat import build_conformer
    from etflow.commons.sample import batched_sampling
    from etflow.data import EuclideanDataset
    from etflow.models import BaseFlow
    from pytorch_lightning import seed_everything
    from rdkit import Chem

    reference = WORKSPACE / "ETFlow-reference"
    config_path = reference / "configs/qm9-o3.yaml"
    checkpoint_path = WORKSPACE / "dataset/ET-Flow_zenodo/qm9-o3.ckpt"
    test_path = WORKSPACE / "dataset/torsional_diffusion/QM9/test_mols.pkl"
    cfg = yaml.safe_load(config_path.read_text())
    report.update(
        config=cfg,
        seed=cfg["seed"],
        data=(
            fingerprint(args.data_manifest)
            if args.data_manifest
            else fingerprint(test_path)
        ),
        checkpoint=fingerprint(checkpoint_path),
        units="Angstrom",
        split=(
            "v3 QM9 scaffold val; interface-only"
            if args.data_manifest
            else "official QM9 test; interface-only, no tuning"
        ),
        molecules=[],
    )
    seed_everything(cfg["seed"])
    spec = importlib.util.spec_from_file_location(
        "official_prepare_data", reference / "scripts/prepare_data.py"
    )
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    if args.data_manifest:
        from .official_data import OfficialDataset

        dataset = OfficialDataset.from_manifest(args.data_manifest)
        dataset.validate_splits()
        test_mols = {}
        for entry in dataset.candidates("val", cfg["seed"])[: args.molecules]:
            molecule = dataset.load_molecule(entry, "val")
            test_mols[entry["id"]] = [c["rd_mol"] for c in molecule["conformers"]]
    else:
        # Trusted, existing local GEOM/official artifacts only; never download pickles.
        with test_path.open("rb") as stream:
            test_mols = pickle.load(stream)
    if args.molecules > len(test_mols):
        raise ValueError(
            "Requested more molecules than the official test file contains"
        )
    ids = sorted(test_mols)[: args.molecules]
    processed = args.cache_dir / "qm9/test"
    processed.mkdir(parents=True)
    for index, mol_id in enumerate(ids):
        mols = test_mols[mol_id]
        numbers = [a.GetAtomicNum() for a in mols[0].GetAtoms()]
        bonds = [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType()))
            for b in mols[0].GetBonds()
        ]
        for mol in mols:
            if numbers != [a.GetAtomicNum() for a in mol.GetAtoms()] or bonds != [
                (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType()))
                for b in mol.GetBonds()
            ]:
                raise ValueError(
                    f"Inconsistent reference atom order/topology: {mol_id}"
                )
        data = prepare.process_test_mol(mols, "qm9")
        if data is None or not torch.isfinite(data.pos).all():
            raise ValueError(f"Failed official preprocessing: {mol_id}")
        expected = np.stack([mol.GetConformer().GetPositions() for mol in mols]).astype(
            np.float32
        )
        np.testing.assert_array_equal(data.pos.numpy(), expected)
        torch.save(data, processed / f"{index:04d}.pt")
        report["molecules"].append(
            {
                "id": mol_id,
                "cache_file": f"{index:04d}.pt",
                "atoms": len(numbers),
                "hydrogens": numbers.count(1),
                "references": len(mols),
                "status": "prepared",
            }
        )
    (args.cache_dir / "manifest.json").write_text(json.dumps(report, indent=2))
    dataset = EuclideanDataset(data_dir=args.cache_dir, split="test", partition="qm9")
    model = BaseFlow.from_config(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Strict matching checks every key and shape; no filtering or random fallback.
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    report["strict_checkpoint_loaded"] = True
    model.to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    for index, item in enumerate(report["molecules"]):
        item["status"] = "running"
        graph = dataset[index]
        original = test_mols[item["id"]][0]
        reconstructed = dataset.mol_feat.get_mol(graph.smiles)
        if not torch.equal(
            dataset.mol_feat.get_atomic_numbers(graph.smiles).long(),
            graph.atomic_numbers,
        ):
            raise ValueError("Featurization changed atom order or explicit hydrogens")
        original_bonds = sorted(
            (
                min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                str(b.GetBondType()),
            )
            for b in original.GetBonds()
        )
        rebuilt_bonds = sorted(
            (
                min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                str(b.GetBondType()),
            )
            for b in reconstructed.GetBonds()
        )
        if original_bonds != rebuilt_bonds:
            raise ValueError("Featurization changed indexed bond topology")
        # Rebuild coordinates on a copy, bypassing the upstream SMILES-only coordinate cache.
        graph.mol = Chem.Mol(reconstructed)
        graph.mol.RemoveAllConformers()
        graph.mol.AddConformer(build_conformer(graph.pos))
        calls = []
        hook = model.register_forward_hook(lambda *_, calls=calls: calls.append(1))
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            positions = batched_sampling(
                model,
                graph,
                max_batch_size=cfg["eval_args"]["batch_size"],
                num_samples=args.samples,
                seed=cfg["seed"] + index,
                device="cuda",
                **cfg["eval_args"]["sampler_args"],
            )
            torch.cuda.synchronize()
        finally:
            hook.remove()
        item["seconds"] = time.perf_counter() - start
        item["forward_calls"] = len(calls)
        item["nfe_per_conformer"] = cfg["eval_args"]["sampler_args"]["n_timesteps"]
        expected_calls = item["nfe_per_conformer"] * (
            (args.samples + cfg["eval_args"]["batch_size"] - 1)
            // cfg["eval_args"]["batch_size"]
        )
        if len(calls) != expected_calls:
            raise ValueError(f"Unexpected NFE: {len(calls)} != {expected_calls}")
        item["center_max_abs_angstrom"] = validate_positions(
            positions, args.samples, item["atoms"]
        )
        result = args.output_dir / f"{index:04d}.npz"
        np.savez(
            result,
            positions=positions,
            atomic_numbers=graph.atomic_numbers.cpu().numpy(),
            molecule_id=item["id"],
            smiles=graph.smiles,
            atom_indices=np.arange(item["atoms"]),
            pos_ref=np.stack(
                [m.GetConformer().GetPositions() for m in test_mols[item["id"]]]
            ),
        )
        with np.load(result, allow_pickle=False) as restored:
            np.testing.assert_array_equal(restored["positions"], positions)
            np.testing.assert_array_equal(
                restored["atomic_numbers"], graph.atomic_numbers.cpu().numpy()
            )
            np.testing.assert_array_equal(
                restored["atom_indices"], np.arange(item["atoms"])
            )
            if restored["molecule_id"].item() != item["id"]:
                raise ValueError("Molecule ID changed during round-trip")
            validate_positions(restored["positions"], args.samples, item["atoms"])
        item.update(status="passed", result=fingerprint(result))
    report["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=external_directory, required=True)
    parser.add_argument("--output-dir", type=external_directory, required=True)
    parser.add_argument("--molecules", type=int, default=4)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument(
        "--data-manifest",
        type=Path,
        help="Use frozen official memmap scaffold validation rows",
    )
    args = parser.parse_args()
    if not (1 <= args.molecules <= 4 and 1 <= args.samples <= 2):
        parser.error("Interface smoke is limited to 1–4 molecules and 1–2 conformers")
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Run generation inside an allocated Slurm job")
    if args.cache_dir.exists() or args.output_dir.exists():
        parser.error(
            "Use new cache and output directories; existing runs are never overwritten"
        )
    if args.cache_dir.is_relative_to(args.output_dir) or args.output_dir.is_relative_to(
        args.cache_dir
    ):
        parser.error("Cache and output directories must be separate")
    args.output_dir.mkdir(parents=True)
    report = {
        "status": "started",
        "scope": "official ET-Flow interface smoke only",
        "cache_dir": str(args.cache_dir),
        "output_dir": str(args.output_dir),
    }
    try:
        report["environment"] = audit(gpu=True)
        report["provenance"] = provenance()
        if not report["environment"]["ok"]:
            raise RuntimeError("Environment preflight failed; see manifest.json")
        run(args, report)
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
