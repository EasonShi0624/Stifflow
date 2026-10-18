"""Entry-point wiring with a fake Trainer; no model run or GPU allocation."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from stifflow import training


def test_research_config_preserves_formal_optimizer_and_data_settings():
    configs = Path(__file__).resolve().parents[1] / "configs"
    formal = yaml.safe_load((configs / "qm9-formal-4090d.yaml").read_text())
    research = yaml.safe_load((configs / "qm9-research-25k.yaml").read_text())
    formal.pop("runtime_acceptance")
    profile = formal.pop("profiles")["formal"]
    profile["steps"] = 25000
    assert research.pop("profiles") == {"research": profile}
    assert formal == research


@pytest.mark.parametrize(
    "module,arguments",
    [
        (
            "stifflow.research_clock",
            ["--train-labels", "labels.json", "--config", "config.yaml"],
        ),
        (
            "stifflow.research_path_statistics",
            [
                "--train-labels",
                "labels.json",
                "--config",
                "config.yaml",
                "--method",
                "precision",
            ],
        ),
        (
            "stifflow_benchmark.research_sampling",
            ["references.json", "--checkpoint", "model.ckpt"],
        ),
        ("stifflow_benchmark.research_diagnostics", ["ensembles", "generated.json"]),
    ],
)
def test_heavy_research_entries_require_allocation(
    module, arguments, tmp_path, monkeypatch
):
    import importlib

    entry = importlib.import_module(module)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(
        sys, "argv", [module, *arguments, "--output", str(tmp_path / "output")]
    )
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "profile,method,allowed",
    [("research", "a_rot_null", True), ("formal", "a_rot_null", False)],
)
def test_training_entry_keeps_accumulation_and_new_path_boundary(
    tmp_path, monkeypatch, profile, method, allowed
):
    import lightning.pytorch as lightning
    import stifflow.model as model_module
    import torch

    config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / (
            "qm9-research-25k.yaml"
            if profile == "research"
            else "qm9-formal-4090d.yaml"
        )
    )
    manifests = []
    for split in ("train", "val"):
        path = tmp_path / f"{split}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "dataset_kind": training.KIND,
                    "selection_seed": 42,
                    "data_manifest": {"sha256": "data"},
                }
            )
        )
        manifests.append(path)
    observed = {}

    class FakeModel:
        def __init__(self, **kwargs):
            observed["model"] = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs):
            observed["trainer"] = kwargs
            self.global_step = kwargs["max_steps"]

        def fit(self, model, **kwargs):
            observed["fit"] = kwargs
            model.consumed_samples = self.global_step * 128

        def save_checkpoint(self, path):
            observed["saved"] = path

    def records(path, split, count, method, source, packed_cache_root):
        return SimpleNamespace(
            records=[{"id": split}],
            offsets=np.array([0, 2]),
            close=lambda: observed.setdefault("closed", []).append(split),
        )

    def loader(*args, **kwargs):
        observed.setdefault("loaders", []).append((args, kwargs))

    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-unit-test")
    monkeypatch.setattr(lightning, "Trainer", FakeTrainer)
    monkeypatch.setattr(lightning, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_module, "StifFlow", FakeModel)
    monkeypatch.setattr(training, "load_records", records)
    monkeypatch.setattr(training, "make_loader", loader)
    monkeypatch.setattr(training, "validate_label_pair", lambda *args: None)
    monkeypatch.setattr(training, "provenance", dict)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training",
            "--config",
            str(config),
            "--train-labels",
            str(manifests[0]),
            "--val-labels",
            str(manifests[1]),
            "--method",
            method,
            "--profile",
            profile,
            "--stop-after",
            "40",
            "--output",
            str(tmp_path / "run"),
        ],
    )
    if not allowed:
        with pytest.raises(SystemExit):
            training.main()
        assert "trainer" not in observed
        return
    training.main()
    assert observed["trainer"]["accumulate_grad_batches"] == 4
    assert observed["trainer"]["val_check_interval"] == 20000
    assert observed["model"]["first_cycle_steps"] == 250000
    assert observed["model"]["method"] == method
    assert observed["loaders"][0][0][2] == 25000
    assert observed["loaders"][0][1]["rng_domain"] == 2
    assert observed["closed"] == ["train", "val"]
    report = json.loads((tmp_path / "run/manifest.json").read_text())
    assert report["status"] == "stopped" and not report["budget_completed"]
    assert report["consumed_samples"] == 5120
    assert report["training_signature"]["research_implementation"]
