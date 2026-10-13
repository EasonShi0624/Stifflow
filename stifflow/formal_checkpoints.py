"""Bounded full recovery checkpoints and sparse inference snapshots."""

import os
import time
import uuid
from pathlib import Path

import torch
from lightning.pytorch import Callback

from .cache_io import atomic_json, read_json


class FormalCheckpoints(Callback):
    def __init__(
        self, directory, effective_batch, recovery_every=500, snapshot_every=25000
    ):
        self.directory = Path(directory).resolve()
        self.effective_batch = effective_batch
        self.recovery_every = recovery_every
        self.snapshot_every = snapshot_every
        self.last_step = None
        self.started = None

    def on_train_start(self, trainer, module):
        self.directory.mkdir(parents=True, exist_ok=True)
        owner = self.directory / "owner.json"
        signature = {
            "training": module.training_signature,
            "method": module.method,
            "seed": module.seed,
        }
        if owner.exists() and read_json(owner) != signature:
            raise ValueError("Checkpoint directory belongs to a different run")
        if not owner.exists():
            if any(self.directory.iterdir()):
                raise ValueError(
                    "Nonempty checkpoint directory without its owner receipt"
                )
            atomic_json(owner, signature)
        self.last_step = trainer.global_step
        self.started = time.perf_counter()

    def on_save_checkpoint(self, trainer, module, checkpoint):
        if module.consumed_samples != trainer.global_step * self.effective_batch:
            raise ValueError("Checkpoint is not at a complete accumulation boundary")
        checkpoint["recovery_directory"] = str(self.directory)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == self.last_step:
            return
        self.last_step = step
        if module.consumed_samples != step * self.effective_batch:
            raise ValueError("Optimizer step and consumed sample cursor differ")
        if step % self.recovery_every == 0 or step == trainer.max_steps:
            destination = self.directory / f"step-{step:08d}.ckpt"
            if destination.exists():
                raise FileExistsError(
                    "Recovery step already exists; resume the newest checkpoint"
                )
            temporary = self.directory / ("." + uuid.uuid4().hex + ".ckpt")
            trainer.save_checkpoint(temporary)
            temporary.replace(destination)
            link = self.directory / ("." + uuid.uuid4().hex + ".link")
            link.symlink_to(destination.name)
            os.replace(link, self.directory / "latest.ckpt")
            old = sorted(self.directory.glob("step-????????.ckpt"))
            for path in old[:-2]:
                path.unlink()
        if step % self.snapshot_every == 0:
            snapshots = self.directory.parent / "snapshots"
            snapshots.mkdir(exist_ok=True)
            destination = snapshots / f"step-{step:08d}.ckpt"
            if destination.exists():
                raise FileExistsError("Existing inference snapshot is protected")
            temporary = snapshots / ("." + uuid.uuid4().hex + ".ckpt")
            torch.save(
                {
                    "state_dict": module.state_dict(),
                    "hyper_parameters": dict(module.hparams),
                    "global_step": step,
                    "training_signature": module.training_signature,
                    "snapshot_kind": "weights_only",
                },
                temporary,
            )
            temporary.replace(destination)
        module.log("train/optimizer_updates", float(step), on_step=True, on_epoch=False)
        module.log(
            "train/consumed_samples",
            float(module.consumed_samples),
            on_step=True,
            on_epoch=False,
        )
        module.log(
            "train/elapsed_seconds",
            time.perf_counter() - self.started,
            on_step=True,
            on_epoch=False,
        )
