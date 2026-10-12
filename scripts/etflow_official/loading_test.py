"""Phase 3b: full-scale loading tests on the official memmap dataset.

Uses clean-sde GEOM/BaseDataModule. A local shim adds `data_dir` plumbing to the
featurized dataset because the upstream EuclideanDataset hardcodes its path.
"""

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from etflow.commons.featurization import MoleculeFeaturizer
from etflow.data.datamodule import BaseDataModule
from etflow.data.geom import GEOM
from torch_geometric.data import Data, Dataset

WS = Path(os.environ.get("STIFFLOW_WORKSPACE", Path(__file__).resolve().parents[3]))
DEST = WS / "cache/etflow-official/geom-preprocessed"
EXPANDED = WS / "cache/etflow-official/splits/expanded"
REPORT = WS / "runs/etflow-official/phase3-loading.json"


class GeomDataset(Dataset):
    """clean-sde EuclideanDataset.get with explicit data_dir."""

    def __init__(self, data_dir, use_ogb_feat=True):
        super().__init__()
        self.geom = GEOM(data_dir=data_dir)
        self.mol_feat = MoleculeFeaturizer()
        self.use_ogb_feat = use_ogb_feat

    def len(self):
        return len(self.geom)

    def get(self, idx):
        bunch = self.geom[idx]
        node_attr = self.mol_feat.get_atom_features(bunch.smiles, self.use_ogb_feat)
        chiral_index, chiral_nbr_index, chiral_tag = self.mol_feat.get_chiral_centers(
            bunch.smiles
        )
        edge_index, edge_attr = self.mol_feat.get_edge_index(bunch.smiles, False)
        mol = self.mol_feat.get_mol_with_conformer(bunch.smiles, bunch.pos)
        return Data(
            pos=bunch.pos,
            atomic_numbers=bunch.atomic_numbers,
            smiles=bunch.smiles,
            edge_index=edge_index,
            chiral_index=chiral_index,
            chiral_nbr_index=chiral_nbr_index,
            chiral_tag=chiral_tag,
            mol=mol,
            node_attr=node_attr,
            edge_attr=edge_attr,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--access", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--full-pass", action="store_true")
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Full loading checks require an SHHPC allocation")
    if REPORT.exists():
        parser.error("Existing loading reports are protected")
    started = time.perf_counter()
    torch.set_num_threads(1)
    report = {
        "dest": str(DEST),
        "workers": args.workers,
        "batch_size": args.batch_size,
    }

    # A. raw random access on the memmap GEOM dataset
    geom = GEOM(data_dir=DEST)
    report["geom_len"] = len(geom)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(geom), size=args.access)
    t0 = time.perf_counter()
    for idx in indices:
        item = geom[int(idx)]
        if item.pos.ndim != 2 or item.pos.shape[1] != 3:
            raise AssertionError("bad pos shape")
        if not torch.isfinite(item.pos).all():
            raise AssertionError("nonfinite pos")
        if item.atomic_numbers.shape[0] != item.pos.shape[0]:
            raise AssertionError("atom mismatch")
    seconds = time.perf_counter() - t0
    report["geom_access"] = {
        "items": args.access,
        "seconds": seconds,
        "items_per_second": args.access / seconds,
    }

    # B. featurized dataloader throughput on the QM9 scaffold train subset
    dataset = GeomDataset(DEST)
    datamodule = BaseDataModule(
        dataset=dataset,
        dataloader_args={
            "batch_size": args.batch_size,
            "num_workers": args.workers,
            "pin_memory": False,
            "persistent_workers": args.workers > 0,
        },
        train_indices_path=str(EXPANDED / "scaffold_qm9_train_indices.npy"),
        val_indices_path=str(EXPANDED / "scaffold_qm9_val_indices.npy"),
        test_indices_path=str(EXPANDED / "scaffold_qm9_test_indices.npy"),
    )
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    report["train_subset"] = len(datamodule.train_dataset)
    seen = 0
    batches = 0
    t0 = time.perf_counter()
    for batch in train_loader:
        if "atomic_numbers" not in batch or "pos" not in batch:
            raise AssertionError("missing batch fields")
        if not torch.isfinite(batch.pos).all():
            raise AssertionError("nonfinite batch pos")
        seen += int(batch.num_graphs)
        batches += 1
        if not args.full_pass and batches >= args.batches:
            break
    seconds = time.perf_counter() - t0
    report["loader"] = {
        "items": seen,
        "batches": batches,
        "seconds": seconds,
        "items_per_second": seen / seconds,
        "estimated_epoch_seconds": len(datamodule.train_dataset) / (seen / seconds),
        "full_pass": bool(args.full_pass),
    }
    for split in ("val", "test"):
        loader = getattr(datamodule, f"{split}_dataloader")()
        count = 0
        for batch in loader:
            count += int(batch.num_graphs)
            break
        report[f"{split}_first_batch_items"] = count
    report["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report["seconds"] = time.perf_counter() - started
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
