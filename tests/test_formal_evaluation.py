"""Tiny reference files and failed subprocess stubs; never run generation."""

import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml
from test_formal_cache import frozen as frozen  # noqa: PLC0414
from test_official_data import official as official  # noqa: PLC0414

from stifflow.cache_io import atomic_json, read_json
from stifflow.formal import evaluate_generated, export_references, generate
from stifflow.official_data import OfficialDataset
from stifflow.preflight import fingerprint


def references(frozen):
    dataset, _manifest, root = frozen
    spec = copy.deepcopy(dataset.specification)
    for split, rows in {"train": [0, 1], "val": [2, 3, 4, 5, 6]}.items():
        path = root / f"eval-{split}.npy"
        np.save(path, np.asarray(rows, dtype=np.int64))
        spec["splits"][split] = fingerprint(path)
    spec["variant"]["counts"] = OfficialDataset(spec).validate_splits()
    manifest = root / "eval-data.json"
    atomic_json(manifest, spec)
    config = Path(__file__).parents[1] / "configs/qm9-formal.yaml"
    with patch(
        "stifflow.stiffness.estimate_stiffness",
        side_effect=AssertionError("reference export must not read stiffness"),
    ):
        export_references(OfficialDataset(spec), manifest, config, root / "references")
    return manifest, config, root / "references/val/manifest.json"


def test_reference_coverage_is_independent_of_hessian_availability(frozen):
    manifest, _, path = references(frozen)
    report = read_json(path)
    assert len(report["records"]) == 2
    assert [r["id"] for r in report["records"]] == [
        r["id"] for r in OfficialDataset.from_manifest(manifest).candidates("val")
    ]
    for item in report["records"]:
        with np.load(path.parent / item["reference_npz"]) as saved:
            assert set(saved.files) == {"pos_ref", "atomic_numbers"}
    assert report["selection_frozen"]


def test_failed_generation_preserves_all_ids_and_cannot_be_primary(frozen):
    manifest, config, reference = references(frozen)
    cfg = yaml.safe_load(config.read_text())
    root = frozen[2]
    checkpoint = root / "weights.ckpt"
    state = {
        "global_step": 250000,
        "hyper_parameters": {"method": "precision", "seed": 42},
        "training_signature": {
            "official_data": fingerprint(manifest)["sha256"],
            "config": cfg,
            "effective_batch_size": 128,
            "training_seed": 42,
        },
    }
    torch.save(state, checkpoint)

    def fail(_module, _arguments, _output, workflow):
        workflow["commands"].append({"seconds": 0.01})
        return 1

    with patch("stifflow.formal.run_command", side_effect=fail), patch(
        "stifflow.formal.provenance", return_value={}
    ):
        with pytest.raises(RuntimeError, match="all expected IDs"):
            generate(checkpoint, reference, config, root / "failed", "final")
        output = root / "failed/generated/manifest.json"
        failed = read_json(output)
        assert [r["id"] for r in failed["records"]] == [
            r["id"] for r in read_json(reference)["records"]
        ]
        assert all(r["status"] == "failed" for r in failed["records"])
        evaluation = root / "failed/evaluation.json"
        with pytest.raises(ValueError, match="Incomplete"):
            evaluate_generated(output, evaluation)
        assert read_json(evaluation)["aggregate"]["failed"] == 2
        assert not read_json(evaluation)["complete"]
    bad = read_json(reference)
    bad["records"][0]["id"] = "wrong-molecule"
    atomic_json(reference, bad)
    with pytest.raises(ValueError, match="Reference IDs"):
        generate(checkpoint, reference, config, root / "bad-ref", "final")
    state["global_step"] = 1000
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="250000"):
        generate(checkpoint, reference, config, root / "early", "final")


def test_row_streams_do_not_collide_with_training_or_validation_targets():
    cfg = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/qm9-formal.yaml").read_text()
    )
    target_states = [
        np.random.default_rng(s).bit_generator.state for s in cfg["training_seeds"]
    ]
    target_states += [
        np.random.default_rng([s, b, 1]).bit_generator.state
        for s in cfg["training_seeds"]
        for b in range(2)
    ]
    for seed in (*cfg["training_seeds"], cfg["dataset_seed"] + 1):
        for draw in range(18):
            state = np.random.default_rng(
                [seed, draw, cfg["row_rng_domain"]]
            ).bit_generator.state
            assert all(state != target for target in target_states)
