"""Scalar CPU optimizer checks; no StifFlow model training or real data."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lightning.pytorch import LightningModule, Trainer, seed_everything

from stifflow.formal import summarize
from stifflow.formal_checkpoints import FormalCheckpoints
from stifflow.model import StifFlow, graph_data
from stifflow.source import prepare_source
from stifflow.training import advance_cursor, make_loader
from stifflow.verify_resume import compare_states


class ScalarModule(LightningModule):
    on_save_checkpoint = StifFlow.on_save_checkpoint
    on_load_checkpoint = StifFlow.on_load_checkpoint

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.seed, self.method = 42, "control"
        self.training_signature = {"unit_fixture": True}
        self.target_rng = np.random.default_rng(42)
        self.gradnorm_queue = SimpleNamespace(items=[1.0])
        self.consumed_samples = 0

    def training_step(self, payload, batch_idx):
        _, rows = payload
        self.consumed_samples = advance_cursor(self.consumed_samples, rows)
        x = self.weight.new_tensor([r["x1"][0, 0] for r in rows])
        y = self.weight.new_tensor(self.target_rng.normal(size=len(rows)))
        return ((self.weight * x - y) ** 2).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9),
                "interval": "step",
            },
        }

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch


def scalar_records():
    graph = {
        "atomic_numbers": [6, 1, 1],
        "bonds": [[0, 1], [1, 0], [0, 2], [2, 0]],
        "node_attr": np.zeros((3, 10)),
        "bond_attr": np.ones((4, 3)),
    }
    return [
        {
            "graph": graph_data(graph),
            "source": prepare_source(graph),
            "positions": np.arange(18, dtype=float).reshape(2, 3, 3),
            "labels": [None, None],
        }
    ]


def fit_scalar(root, stop, resume=None, directory=None):
    seed_everything(42)
    model = ScalarModule()
    start = 0
    if resume:
        start = torch.load(resume, map_location="cpu", weights_only=False)[
            "consumed_samples"
        ]
    callback = FormalCheckpoints(
        directory or root / "checkpoints",
        effective_batch=4,
        recovery_every=1,
        snapshot_every=2,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=stop,
        max_epochs=-1,
        accumulate_grad_batches=2,
        deterministic=True,
        callbacks=[callback],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        num_sanity_val_steps=0,
        default_root_dir=root,
    )
    loader = make_loader(
        scalar_records(),
        2,
        4,
        42,
        sampling="uniform_conformer_rows",
        accumulate_grad_batches=2,
        consumed_samples=start,
        num_workers=1,
    )
    trainer.fit(model, train_dataloaders=loader, ckpt_path=resume)
    root.mkdir(exist_ok=True)
    path = root / "final.ckpt"
    trainer.save_checkpoint(path)
    return path


def test_lightning_accumulation_resume_cursor_optimizer_and_retention(tmp_path):
    full = fit_scalar(tmp_path / "continuous", 4)
    partial = fit_scalar(tmp_path / "partial", 2)
    # Resume from a callback checkpoint saved inside an epoch, including its
    # prefetch/loop progress, rather than only the post-fit final checkpoint.
    recovery = tmp_path / "partial/checkpoints"
    resumed = fit_scalar(tmp_path / "resumed", 4, recovery / "latest.ckpt", recovery)
    left = torch.load(full, map_location="cpu", weights_only=False)
    right = torch.load(resumed, map_location="cpu", weights_only=False)
    for key in (
        "global_step",
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "stifflow_rng",
        "torch_rng",
        "consumed_samples",
        "gradient_queue",
    ):
        compare_states(left[key], right[key], key)
    assert left["consumed_samples"] == 16 and left["global_step"] == 4
    assert len(list(recovery.glob("step-????????.ckpt"))) == 2
    assert (recovery / "latest.ckpt").resolve().name == "step-00000004.ckpt"
    snapshot = torch.load(
        tmp_path / "partial/snapshots/step-00000004.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert "optimizer_states" not in snapshot and snapshot["global_step"] == 4
    assert partial.exists()


def reports(root):
    paths = []
    for seed in (42, 43, 44):
        for method in ("precision", "alternative_a", "control"):
            aggregate = {
                "successful_only": {
                    k: (
                        float(seed - 41) if k.startswith("mat") else [0.1 * (seed - 41)]
                    )
                    for k in (
                        "mat_r_mean",
                        "mat_r_median",
                        "mat_p_mean",
                        "mat_p_median",
                        "cov_r_mean",
                        "cov_r_median",
                        "cov_p_mean",
                        "cov_p_median",
                    )
                },
                "chirality": {"fraction": 0.2},
            }
            report = {
                "complete": True,
                "protocol": {"thresholds": [0.5]},
                "records": [{"id": "same-molecule"}],
                "aggregate": aggregate,
                "run": {
                    "method": method,
                    "baseline_kind": "controlled",
                    "seed": 42,
                    "split": "test",
                    "training_budget": {"steps": 250000},
                    "training_data": {"training_seed": seed, "data": "frozen"},
                    "input_manifest": {"sha256": "reference"},
                    "config": {
                        "model": {"method": method, "seed": seed, "width": 16},
                        "steps": 50,
                    },
                    "generation": {},
                    "formal": {"training_seed": seed, "stage": "final"},
                },
            }
            path = root / f"{method}-{seed}.json"
            path.write_text(json.dumps(report))
            paths.append(path)
    return paths


def test_nine_run_summary_allows_only_method_and_training_seed(tmp_path):
    paths = reports(tmp_path)
    result = summarize(paths)
    assert result["methods"]["precision"]["mat_r_mean"]["mean"] == 2.0
    assert result["methods"]["precision"]["mat_r_mean"]["sample_std"] == 1.0
    with pytest.raises(ValueError, match="nine"):
        summarize(paths[:-1])
    changed = json.loads(paths[-1].read_text())
    changed["run"]["config"]["steps"] = 100
    paths[-1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="beyond"):
        summarize(paths)


def test_failed_or_early_report_cannot_enter_primary_comparison(tmp_path):
    paths = reports(tmp_path)
    changed = json.loads(paths[0].read_text())
    changed["complete"] = False
    paths[0].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="complete 250k"):
        summarize(paths)
