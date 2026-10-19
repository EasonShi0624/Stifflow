"""Shared Lightning data/entry point; execution requires working ET-Flow + Slurm."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .cache_io import atomic_json
from .official_data import KIND, SAMPLING, validate_label_manifest, validate_label_pair
from .preflight import WORKSPACE, fingerprint, provenance
from .research_paths import BASE_METHODS, DELAYED_METHODS, METHODS, label_method
from .smoke import external_directory


def allocator_budget(memory, total, overhead):
    if not (
        memory["budget_gib"] > 0
        and 0 < memory["preferred_fraction"] <= memory["hard_fraction"] < 1
    ):
        raise ValueError("Invalid memory budget/fractions")
    allowed = (
        min(total, memory["budget_gib"] * 1024**3) * memory["hard_fraction"] - overhead
    )
    if allowed <= 0:
        raise MemoryError("Device overhead already exceeds the GPU budget")
    return allowed / total


def gpu_memory_check(memory):
    import torch

    free, total = torch.cuda.mem_get_info()
    limit = min(total, memory["budget_gib"] * 1024**3) * memory["hard_fraction"]
    reserved, used = torch.cuda.max_memory_reserved(), total - free
    valid = torch.tensor(int(max(reserved, used) <= limit), device="cuda")
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(valid, op=torch.distributed.ReduceOp.MIN)
    if not valid:
        raise MemoryError("Exceeds the GPU-memory budget including device overhead")
    return reserved, used


def advance_cursor(consumed, supervision, rank=0, world_size=1):
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid data rank/world size")
    if "draw_id" not in supervision[0]:
        return consumed
    start = consumed + rank * len(supervision)
    if [r["draw_id"] for r in supervision] != list(
        range(start, start + len(supervision))
    ):
        raise RuntimeError("Data cursor repeated/skipped samples after resume")
    return consumed + len(supervision) * world_size


def load_records(
    manifest_path, split, count, method, source_config, packed_cache_root=None
):
    method = label_method(method)
    from rdkit import Chem

    from .features import graph_from_mol
    from .model import graph_data
    from .source import prepare_source
    from .targets import path_module

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if packed_cache_root is not None and manifest.get("schema_version") != 3:
        raise ValueError("Packed relocation requires version-3 labels")
    if manifest.get("speed_only"):
        raise ValueError(
            "Speed-only caches cannot be used for model training/evaluation"
        )
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
    if manifest.get("schema_version") == 3:
        from .formal_cache import signature
        from .packed_data import PackedRecords, relocate_manifest

        current = signature(
            manifest["data_manifest"]["path"],
            split,
            source_config,
            manifest["selection_seed"],
        )
        if manifest["label_implementation"] != current["label_implementation"]:
            raise ValueError(
                "Packed factors were built by different source/label implementations"
            )

        if count not in (None, "all", len(manifest["records"])):
            raise ValueError("Formal training uses the complete shared accepted cohort")
        return PackedRecords(
            (
                relocate_manifest(manifest, manifest_path, packed_cache_root)
                if packed_cache_root is not None
                else manifest
            ),
            method,
        )
    eligible = [r for r in manifest["records"] if r["split"] == split]
    if count in (None, "all"):
        count = len(eligible)
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


def draw_record(lengths, seed, draw_id, sampling, offsets=None, rng_domain=0):
    """A stable draw ID identifies one molecule/conformer across interruption."""
    rng = np.random.default_rng([seed, draw_id, rng_domain])
    if sampling == SAMPLING:
        if offsets is None:
            offsets = np.r_[0, np.cumsum(lengths)]
        row = int(rng.integers(offsets[-1]))
        record = int(np.searchsorted(offsets[1:], row, side="right"))
        return record, row - int(offsets[record])
    if sampling != "uniform_molecules":
        raise ValueError("Unknown training sampling law")
    record = int(rng.integers(len(lengths)))
    return record, int(rng.integers(lengths[record]))


def make_loader(
    records,
    batch_size,
    steps,
    seed,
    start_step=0,
    sampling="uniform_molecules",
    accumulate_grad_batches=1,
    consumed_samples=None,
    num_workers=0,
    rng_domain=0,
    pin_memory=False,
    prefetch_factor=2,
    rank=0,
    world_size=1,
):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torch_geometric.data import Batch

    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid loader rank/world size")

    packed = hasattr(records, "sample")
    lengths = (
        records.lengths
        if packed
        else np.asarray([len(r["positions"]) for r in records])
    )
    offsets = np.r_[0, np.cumsum(lengths)]
    start = (
        start_step * batch_size * accumulate_grad_batches
        if consumed_samples is None
        else consumed_samples
    )

    class StepDataset(Dataset):
        def __len__(self):
            return steps * batch_size * accumulate_grad_batches

        def __getitem__(self, index):
            # Stateless draw IDs preserve graph/conformer choices across resume.
            draw_id = (
                start
                + (index // batch_size * world_size + rank) * batch_size
                + index % batch_size
            )
            record_index, conf = draw_record(
                lengths,
                seed,
                draw_id,
                sampling,
                offsets,
                rng_domain,
            )
            if packed:
                graph, supervision = records.sample(record_index, conf)
                supervision["draw_id"] = draw_id
                return graph, supervision
            record = records[record_index]
            return record["graph"], {
                "x1": record["positions"][conf],
                "source": record["source"],
                "label": record["labels"][conf],
                "draw_id": draw_id,
            }

    def collate(rows):
        graphs, supervision = zip(*rows)
        return Batch.from_data_list(list(graphs)), list(supervision)

    return DataLoader(
        StepDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=pin_memory,
        **({"prefetch_factor": prefetch_factor} if num_workers else {}),
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
        choices=METHODS,
        default="precision",
    )
    parser.add_argument(
        "--profile", choices=["smoke", "pilot", "formal", "research"], default="smoke"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint-dir", type=external_directory)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--io-compatibility", type=Path)
    parser.add_argument("--packed-cache-root", type=Path)
    parser.add_argument("--output", type=external_directory, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--clock", type=Path)
    parser.add_argument("--locality-report", type=Path)
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
    formal = args.profile == "formal"
    research = args.profile == "research"
    scheduled = formal or research
    if not research and args.method not in BASE_METHODS:
        parser.error("New paths require the separate research profile")
    seed = cfg["seed"] if args.seed is None else args.seed
    accumulation = profile.get("accumulate_grad_batches", 1)
    effective_batch = cfg["batch_size"] * accumulation
    if research and (
        effective_batch != 128
        or profile["steps"] != 25000
        or cfg["model_args"]["first_cycle_steps"] != 250000
    ):
        parser.error(
            "Research budget is 25000 updates, effective batch 128, original 250000-step LR schedule"
        )
    if research and (
        seed not in cfg["training_seeds"]
        or cfg["dataset_seed"] != 42
        or cfg["row_rng_domain"] != 2
    ):
        parser.error("Research comparisons retain the formal seeds and row sampling")
    path_clock = json.loads(args.clock.read_text()) if args.clock else None
    if (args.method == "precision_clock") != (path_clock is not None):
        parser.error("Only precision_clock requires --clock")
    if path_clock is not None and (
        path_clock.get("train_manifest", {}).get("sha256")
        != fingerprint(args.train_labels)["sha256"]
        or path_clock.get("source_config") != cfg["source"]
    ):
        parser.error("Clock belongs to different training labels/source")
    if (args.method in DELAYED_METHODS) != (args.locality_report is not None):
        parser.error("Only delayed paths require --locality-report")
    stiffness_start, locality_hash = None, None
    if args.locality_report is not None:
        from .locality import load_locality_report

        if seed != 42:
            parser.error("This locality screen is limited to the two seed-42 runs")
        stiffness_start, locality_hash = load_locality_report(
            args.locality_report, args.train_labels, cfg["source"]
        )
    if formal and (effective_batch != 128 or profile["steps"] != 250000):
        parser.error(
            "Formal budget is 250000 optimizer updates with effective batch 128"
        )
    if formal and seed not in cfg["training_seeds"]:
        parser.error("Formal training seeds are 42, 43 and 44")
    if formal and (cfg["dataset_seed"] != 42 or cfg["row_rng_domain"] != 2):
        parser.error("Formal data seed is 42 and row RNG domain is 2")
    max_steps = profile["steps"] if args.stop_after is None else args.stop_after
    if not 0 < max_steps <= profile["steps"]:
        parser.error("stop-after must be in the profile step budget")
    seed_everything(seed, workers=True)
    train_manifest = json.loads(args.train_labels.read_text())
    val_manifest = json.loads(args.val_labels.read_text())
    official = train_manifest.get("dataset_kind") == KIND
    if official != (val_manifest.get("dataset_kind") == KIND):
        raise ValueError("Cannot mix legacy and official train/validation labels")
    sampling = SAMPLING if official else "uniform_molecules"
    if official:
        validate_label_pair(train_manifest, val_manifest)
        if train_manifest["selection_seed"] != cfg.get("dataset_seed", cfg["seed"]):
            raise ValueError(
                "Official molecule selection seed differs from dataset seed"
            )
    if scheduled and (
        train_manifest.get("schema_version") != 3
        or val_manifest.get("schema_version") != 3
    ):
        raise ValueError(
            "Formal training requires merged, all-candidate mmap label manifests"
        )
    train = load_records(
        args.train_labels,
        "train",
        profile["train_molecules"],
        args.method,
        cfg["source"],
        args.packed_cache_root,
    )
    val = load_records(
        args.val_labels,
        "val",
        profile["val_molecules"],
        args.method,
        cfg["source"],
        args.packed_cache_root,
    )
    train_metadata = train.records if scheduled else train
    val_metadata = val.records if scheduled else val
    if {r["id"] for r in train_metadata} & {r["id"] for r in val_metadata}:
        raise ValueError("Train/validation overlap")
    start_step = 0
    if args.resume:
        # Only trusted local checkpoints produced by this entry point.
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_step = int(saved["global_step"])
        if scheduled and "optimizer_states" not in saved:
            raise ValueError("Inference snapshots cannot resume training")
        if saved["hyper_parameters"]["method"] != args.method:
            raise ValueError("Resume method differs")
    if start_step >= max_steps:
        raise ValueError("Resume checkpoint already reaches requested budget")
    model = StifFlow(
        method=args.method,
        seed=seed,
        source_config=cfg["source"],
        path_clock=path_clock,
        stiffness_start=stiffness_start,
        locality_report_sha256=locality_hash,
        **cfg["model_args"],
    )
    model.gpu_memory = cfg.get("gpu_memory")
    model.training_signature = {
        "train": fingerprint(args.train_labels)["sha256"],
        "val": fingerprint(args.val_labels)["sha256"],
        "config": cfg,
        "profile": args.profile,
    }
    if research:
        model.training_signature.update(
            training_seed=seed,
            dataset_seed=cfg["dataset_seed"],
            effective_batch_size=effective_batch,
            row_rng_domain=cfg["row_rng_domain"],
            path_clock=path_clock,
            research_implementation={
                name: fingerprint(Path(__file__).with_name(name))["sha256"]
                for name in (
                    "training.py",
                    "model.py",
                    "targets.py",
                    "research_paths.py",
                    "research_clock.py",
                )
            },
        )
        if args.locality_report is not None:
            model.training_signature.update(
                stiffness_start=stiffness_start,
                locality_report_sha256=locality_hash,
            )
            model.training_signature["research_implementation"]["locality.py"] = (
                fingerprint(Path(__file__).with_name("locality.py"))["sha256"]
            )
    run_provenance = provenance()
    accepted = None
    compatibility = (
        json.loads(args.io_compatibility.read_text()) if args.io_compatibility else None
    )
    if formal:
        from .formal import acceptance_signature, require_torch_runtime

        if max_steps > 40 or cfg.get("runtime_acceptance") == "torch_runtime":
            if args.acceptance is None:
                parser.error(
                    "Formal training requires --acceptance from the new 40-update/resume checks"
                )
            accepted = json.loads(args.acceptance.read_text())
            if cfg.get("runtime_acceptance") == "torch_runtime":
                require_torch_runtime(accepted, cfg, compatibility)
            elif (
                accepted.get("status") != "completed"
                or accepted.get("methods") != ["precision", "alternative_a", "control"]
                or accepted.get("updates") != 40
                or accepted.get("signature")
                != acceptance_signature(
                    args.train_labels, args.val_labels, args.config, run_provenance
                )
            ):
                raise ValueError(
                    "Formal acceptance belongs to different data/config/code or is incomplete"
                )
        model.training_signature.update(
            training_seed=seed,
            dataset_seed=cfg["dataset_seed"],
            microbatch=cfg["batch_size"],
            effective_batch_size=effective_batch,
            accumulate_grad_batches=accumulation,
            row_rng_domain=cfg["row_rng_domain"],
            code={
                name: {"head": value["head"], "diff": value["diff"]}
                for name, value in run_provenance["repos"].items()
            },
        )
    if official:
        model.training_signature.update(
            sampling=sampling, official_data=train_manifest["data_manifest"]["sha256"]
        )
    if formal and cfg.get("runtime_acceptance") == "torch_runtime":
        from .formal import formal_training_signature

        model.training_signature = formal_training_signature(
            args.train_labels, args.val_labels, cfg, seed, accepted
        )
    model.execution_provenance = {
        "provenance": run_provenance,
        "packed_cache_root": (
            str(args.packed_cache_root.resolve()) if args.packed_cache_root else None
        ),
        "acceptance": fingerprint(args.acceptance) if args.acceptance else None,
        "io_compatibility": (
            fingerprint(args.io_compatibility) if args.io_compatibility else None
        ),
    }
    if args.resume and "checkpoint_migration" in saved:
        model.checkpoint_migration = saved["checkpoint_migration"]
    if args.resume and saved.get("training_signature") != model.training_signature:
        raise ValueError("Resume dataset, seed or configuration differs")
    consumed = (
        saved.get("consumed_samples", start_step * effective_batch)
        if args.resume
        else 0
    )
    if consumed != start_step * effective_batch:
        raise ValueError("Resume must be at a complete optimizer/accumulation boundary")
    model.consumed_samples = consumed
    args.output.mkdir(parents=True)
    report = {
        "method": args.method,
        "profile": args.profile,
        "config": cfg,
        "seed": seed,
        "sampling": sampling,
        "train_rows": (
            int(train.offsets[-1])
            if scheduled
            else sum(len(r["positions"]) for r in train)
        ),
        "val_rows": (
            int(val.offsets[-1]) if scheduled else sum(len(r["positions"]) for r in val)
        ),
        "train_manifest": fingerprint(args.train_labels),
        "val_manifest": fingerprint(args.val_labels),
        "train_ids": [r["id"] for r in train_metadata],
        "val_ids": [r["id"] for r in val_metadata],
        "start_step": start_step,
        "max_steps": max_steps,
        "provenance": run_provenance,
        "effective_batch_size": effective_batch,
        "accumulate_grad_batches": accumulation,
        "allocated_resources": {
            key: os.getenv(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_ACCOUNT",
                "SLURM_JOB_PARTITION",
                "SLURM_JOB_NODELIST",
                "SLURM_CPUS_PER_TASK",
                "SLURM_MEM_PER_NODE",
                "SLURM_JOB_GPUS",
            )
        },
        "status": "started",
        "training_signature": model.training_signature,
        "checkpoint_migration": getattr(model, "checkpoint_migration", None),
        "execution_provenance": model.execution_provenance,
    }
    atomic_json(args.output / "manifest.json", report)
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output / "checkpoints",
            save_last=True,
            every_n_train_steps=min(25, max_steps),
            save_top_k=-1,
        )
    ]
    trainer_options = {}
    if scheduled:
        from .formal_checkpoints import FormalCheckpoints

        recovery = args.checkpoint_dir or (
            Path(saved["recovery_directory"])
            if args.resume
            else args.output / "checkpoints"
        )
        callbacks = [
            FormalCheckpoints(
                recovery,
                effective_batch,
                profile["recovery_every"],
                profile["snapshot_every"],
            )
        ]
        trainer_options.update(
            accumulate_grad_batches=accumulation,
            val_check_interval=profile["validation_every"] * accumulation,
            check_val_every_n_epoch=None,
            enable_checkpointing=False,
        )
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
            callbacks=callbacks,
            enable_progress_bar=False,
            **trainer_options,
        )
        trainer.fit(
            model,
            train_dataloaders=make_loader(
                train,
                cfg["batch_size"],
                profile["steps"] if scheduled else max_steps - start_step,
                seed,
                start_step,
                sampling,
                accumulate_grad_batches=accumulation,
                consumed_samples=consumed,
                num_workers=profile.get("num_workers", 0),
                rng_domain=cfg.get("row_rng_domain", 0),
                pin_memory=profile.get("pin_memory", False),
                prefetch_factor=profile.get("prefetch_factor", 2),
            ),
            val_dataloaders=make_loader(
                val,
                cfg["batch_size"],
                profile.get("validation_rows", cfg["batch_size"]) // cfg["batch_size"],
                cfg.get("dataset_seed", seed) + 1,
                sampling=sampling,
                rng_domain=cfg.get("row_rng_domain", 0),
            ),
            ckpt_path=str(args.resume) if args.resume else None,
        )
        trainer.save_checkpoint(args.output / "final.ckpt")
        report.update(
            status=(
                "completed"
                if not scheduled or trainer.global_step == profile["steps"]
                else "stopped"
            ),
            budget_completed=trainer.global_step == profile["steps"],
            global_step=trainer.global_step,
            consumed_samples=model.consumed_samples,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if scheduled:
            train.close()
            val.close()
        atomic_json(args.output / "manifest.json", report)


if __name__ == "__main__":
    main()
