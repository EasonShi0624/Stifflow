"""Shared Lightning data/entry point; execution requires working ET-Flow + Slurm."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .official_data import KIND, SAMPLING, validate_label_manifest, validate_label_pair
from .preflight import WORKSPACE, fingerprint, provenance
from .smoke import external_directory


def load_records(manifest_path, split, count, method, source_config):
    from rdkit import Chem

    from .features import graph_from_mol
    from .model import graph_data
    from .source import prepare_source
    from .targets import path_module

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    official = manifest.get("dataset_kind") == KIND
    if official:
        validate_label_manifest(manifest)
        if manifest["split"] != split:
            raise ValueError("Official label split differs from requested split")
    elif manifest.get("schema_version") == 2:
        raise ValueError("Unknown version-2 label schema")
    elif (
        manifest.get("audit_partial")
        or manifest.get("failed")
        or not manifest.get("requested_attempts_completed", False)
    ):
        raise ValueError(
            "Training requires complete top-30 labels and an explicitly accepted shared molecule list"
        )
    if manifest["source_config"] != source_config:
        raise ValueError(
            "Offline label factors were built with different source parameters"
        )
    eligible = [r for r in manifest["records"] if r["split"] == split]
    if len(eligible) < count:
        raise ValueError(
            f"Need {count} complete {split} molecules, found {len(eligible)}"
        )
    records = []
    path_module(method)
    for record in eligible[:count]:
        if record["audit_partial"]:
            raise ValueError("Partial conformer label cache")
        path = manifest_path.parent / record["npz"]
        if fingerprint(path)["sha256"] != record["cache"]["sha256"]:
            raise ValueError("Label cache hash mismatch")
        mol = Chem.MolFromMolBlock(record["molblock"], removeHs=False)
        if mol is None:
            raise ValueError("Invalid cached molblock")
        graph = graph_from_mol(mol)
        source = prepare_source(graph, source_config)
        with np.load(path, allow_pickle=False) as arrays:
            positions = arrays["positions"].copy()
            if (
                positions.shape != (len(record["conformer_ids"]), mol.GetNumAtoms(), 3)
                or not np.isfinite(positions).all()
            ):
                raise ValueError("Invalid target positions or conformer IDs")
            if official and (
                positions.dtype != np.float64
                or not np.array_equal(arrays["row_ids"], record["row_ids"])
            ):
                raise ValueError(
                    "Official coordinate dtype or row/label pairing differs"
                )
            if method == "precision":
                if official:
                    dimension = 3 * mol.GetNumAtoms() - 3
                    if (
                        arrays["precision_values"].shape != (len(positions), dimension)
                        or arrays["precision_vectors"].shape
                        != (len(positions), dimension, dimension)
                        or arrays["precision_values"].dtype != np.float64
                        or arrays["precision_vectors"].dtype != np.float64
                        or not np.isfinite(arrays["precision_values"]).all()
                        or not np.isfinite(arrays["precision_vectors"]).all()
                        or np.any(arrays["precision_values"] <= 0)
                    ):
                        raise ValueError("Invalid float64 official precision factors")
                labels = [
                    {"values": v, "vectors": V}
                    for v, V in zip(
                        arrays["precision_values"], arrays["precision_vectors"]
                    )
                ]
            elif method == "alternative_a":
                if official:
                    dimension = 3 * mol.GetNumAtoms() - 3
                    if (
                        arrays["alt_L_K"].shape
                        != (len(positions), dimension, dimension)
                        or arrays["alt_L_K"].dtype != np.float64
                        or not np.isfinite(arrays["alt_L_K"]).all()
                    ):
                        raise ValueError("Invalid float64 official aligned A_E factors")
                labels = [{"L_K": L} for L in arrays["alt_L_K"]]
            else:
                labels = [None] * len(positions)
            if len(labels) != len(positions):
                raise ValueError("Unpaired target and label arrays")
        records.append(
            {
                "id": record["id"],
                "graph": graph_data(graph),
                "positions": positions,
                "source": source,
                "labels": labels,
                "row_ids": record.get("row_ids"),
            }
        )
    return records


def draw_record(lengths, seed, draw_id, sampling):
    """A stable draw ID identifies one molecule/conformer across interruption."""
    rng = np.random.default_rng([seed, draw_id])
    if sampling == SAMPLING:
        offsets = np.r_[0, np.cumsum(lengths)]
        row = int(rng.integers(offsets[-1]))
        record = int(np.searchsorted(offsets[1:], row, side="right"))
        return record, row - int(offsets[record])
    if sampling != "uniform_molecules":
        raise ValueError("Unknown training sampling law")
    record = int(rng.integers(len(lengths)))
    return record, int(rng.integers(lengths[record]))


def make_loader(
    records, batch_size, steps, seed, start_step=0, sampling="uniform_molecules"
):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torch_geometric.data import Batch

    class StepDataset(Dataset):
        def __len__(self):
            return steps * batch_size

        def __getitem__(self, index):
            # Stateless draw IDs preserve graph/conformer choices across resume.
            record_index, conf = draw_record(
                [len(r["positions"]) for r in records],
                seed,
                start_step * batch_size + index,
                sampling,
            )
            record = records[record_index]
            return record["graph"], {
                "x1": record["positions"][conf],
                "source": record["source"],
                "label": record["labels"][conf],
            }

    def collate(rows):
        graphs, supervision = zip(*rows)
        return Batch.from_data_list(list(graphs)), list(supervision)

    return DataLoader(
        StepDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=WORKSPACE / "stifflow-v1/configs/qm9.yaml"
    )
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--val-labels", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=["precision", "alternative_a", "control"],
        default="precision",
    )
    parser.add_argument("--profile", choices=["smoke", "pilot"], default="smoke")
    parser.add_argument("--output", type=external_directory, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Training belongs in an allocated Slurm job")
    if args.output.exists():
        parser.error("Use a fresh output directory")
    import torch
    import yaml
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    from .model import StifFlow

    cfg = yaml.safe_load(args.config.read_text())
    profile = cfg["profiles"][args.profile]
    max_steps = profile["steps"] if args.stop_after is None else args.stop_after
    if not 0 < max_steps <= profile["steps"]:
        parser.error("stop-after must be in the profile step budget")
    seed_everything(cfg["seed"], workers=True)
    train_manifest = json.loads(args.train_labels.read_text())
    val_manifest = json.loads(args.val_labels.read_text())
    official = train_manifest.get("dataset_kind") == KIND
    if official != (val_manifest.get("dataset_kind") == KIND):
        raise ValueError("Cannot mix legacy and official train/validation labels")
    sampling = SAMPLING if official else "uniform_molecules"
    if official:
        validate_label_pair(train_manifest, val_manifest)
        if train_manifest["selection_seed"] != cfg["seed"]:
            raise ValueError(
                "Official molecule selection seed differs from training seed"
            )
    train = load_records(
        args.train_labels,
        "train",
        profile["train_molecules"],
        args.method,
        cfg["source"],
    )
    val = load_records(
        args.val_labels, "val", profile["val_molecules"], args.method, cfg["source"]
    )
    if {r["id"] for r in train} & {r["id"] for r in val}:
        raise ValueError("Train/validation overlap")
    start_step = 0
    if args.resume:
        # Only trusted local checkpoints produced by this entry point.
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_step = int(saved["global_step"])
        if saved["hyper_parameters"]["method"] != args.method:
            raise ValueError("Resume method differs")
    if start_step >= max_steps:
        raise ValueError("Resume checkpoint already reaches requested budget")
    model = StifFlow(
        method=args.method,
        seed=cfg["seed"],
        source_config=cfg["source"],
        **cfg["model_args"],
    )
    model.training_signature = {
        "train": fingerprint(args.train_labels)["sha256"],
        "val": fingerprint(args.val_labels)["sha256"],
        "config": cfg,
        "profile": args.profile,
    }
    if official:
        model.training_signature.update(
            sampling=sampling, official_data=train_manifest["data_manifest"]["sha256"]
        )
    if args.resume and saved.get("training_signature") != model.training_signature:
        raise ValueError("Resume dataset, seed or configuration differs")
    args.output.mkdir(parents=True)
    report = {
        "method": args.method,
        "profile": args.profile,
        "config": cfg,
        "seed": cfg["seed"],
        "sampling": sampling,
        "train_rows": sum(len(r["positions"]) for r in train),
        "val_rows": sum(len(r["positions"]) for r in val),
        "train_manifest": fingerprint(args.train_labels),
        "val_manifest": fingerprint(args.val_labels),
        "train_ids": [r["id"] for r in train],
        "val_ids": [r["id"] for r in val],
        "start_step": start_step,
        "max_steps": max_steps,
        "provenance": provenance(),
        "status": "started",
    }
    try:
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            max_steps=max_steps,
            max_epochs=-1,
            precision="32-true",
            deterministic=True,
            num_sanity_val_steps=0,
            logger=CSVLogger(str(args.output), name="metrics"),
            callbacks=[
                ModelCheckpoint(
                    dirpath=args.output / "checkpoints",
                    save_last=True,
                    every_n_train_steps=min(25, max_steps),
                    save_top_k=-1,
                )
            ],
            enable_progress_bar=False,
        )
        trainer.fit(
            model,
            train_dataloaders=make_loader(
                train,
                cfg["batch_size"],
                max_steps - start_step,
                cfg["seed"],
                start_step,
                sampling,
            ),
            val_dataloaders=make_loader(
                val, cfg["batch_size"], 1, cfg["seed"] + 1, sampling=sampling
            ),
            ckpt_path=str(args.resume) if args.resume else None,
        )
        trainer.save_checkpoint(args.output / "final.ckpt")
        report.update(
            status="completed",
            global_step=trainer.global_step,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (args.output / "manifest.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
