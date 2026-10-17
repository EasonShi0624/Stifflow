"""Tiny reference files and failed subprocess stubs; never run generation."""

import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml
from stifflow.cache_io import atomic_json, read_json
from stifflow.formal import (
    evaluate_generated,
    export_references,
    generate,
    summarize,
)
from stifflow.official_data import OfficialDataset
from stifflow.preflight import fingerprint
from test_formal_cache import frozen as frozen  # noqa: PLC0414
from test_official_data import official as official  # noqa: PLC0414


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


def _summary_report(method, seed, train, val, official="data"):
    metrics = {
        "mat_r_mean": 0.5,
        "mat_r_median": 0.5,
        "mat_p_mean": 0.5,
        "mat_p_median": 0.5,
        "cov_r_mean": [0.4, 0.6],
        "cov_r_median": [0.4, 0.6],
        "cov_p_mean": [0.4, 0.6],
        "cov_p_median": [0.4, 0.6],
    }
    return {
        "complete": True,
        "run": {
            "method": method,
            "baseline_kind": "controlled",
            "seed": 42,
            "split": "test",
            "training_budget": {"steps": 250000},
            "training_data": {
                "train": train,
                "val": val,
                "official_data": official,
                "training_seed": seed,
                "config": {"x": 1},
                "code": {"stifflow-v1": {"head": "head", "diff": ""}},
            },
            "input_manifest": {"sha256": "references"},
            "config": {"model": {"method": method, "seed": seed, "x": 1}, "steps": 50},
            "formal": {
                "stage": "final",
                "training_seed": seed,
                "generation_seed": 42,
                "expected_molecules": 1,
            },
            "generation": {},
        },
        "protocol": {"thresholds": [0.05, 0.5]},
        "records": [{"id": "molecule"}],
        "aggregate": {
            "attempted": 1,
            "succeeded": 1,
            "failed": 0,
            "successful_only": metrics,
            "chirality": {"fraction": 0.5},
        },
    }


def test_summary_normalizes_staged_label_manifest_hashes(tmp_path):
    methods = ("precision", "alternative_a", "control")
    paths = []
    for seed in (42, 43, 44):
        for method in methods:
            report = _summary_report(
                method,
                seed,
                f"staged-train-{method}-{seed}",
                f"staged-val-{method}-{seed}",
            )
            path = tmp_path / f"{method}-{seed}.json"
            atomic_json(path, report)
            paths.append(path)
    with pytest.raises(ValueError, match="differ beyond method/training seed"):
        summarize(paths)
    summary = summarize(paths, normalize_staged_labels=True)
    assert summary["complete"]
    assert len(summary["staged_label_manifests"]) == 9
    assert summary["data_identity"]["canonical"].startswith("training_signature")
    assert summary["per_seed"]["42"]["runs"][0]["staged_label_manifests"]

    changed = _summary_report(
        "precision",
        42,
        "staged-train-precision-42",
        "staged-val-precision-42",
        official="different-data",
    )
    changed_path = tmp_path / "changed.json"
    atomic_json(changed_path, changed)
    with pytest.raises(ValueError, match="differ beyond method/training seed"):
        summarize([changed_path, *paths[1:]], normalize_staged_labels=True)


def test_evaluate_generated_requests_multi_fragment_support(tmp_path):
    generated = tmp_path / "generated.json"
    output = tmp_path / "evaluation.json"
    generated.write_text("{}")

    def fake_evaluate(path, allow_disconnected=False):
        assert allow_disconnected is True
        return {
            "run": {"formal": {"expected_molecules": 1}},
            "aggregate": {"attempted": 1, "failed": 0},
        }

    with patch(
        "stifflow_benchmark.evaluate.evaluate", side_effect=fake_evaluate
    ), patch("stifflow.formal.provenance", return_value={}):
        report = evaluate_generated(generated, output)
    assert report["complete"]
    assert read_json(output)["complete"]


def test_generate_resume_reuses_existing_manifest(frozen):
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
    output = root / "resumed"
    records = [
        {"id": item["id"], "molblock": item["molblock"], "status": "failed"}
        for item in read_json(reference)["records"]
    ]
    atomic_json(
        output / "generated/manifest.json",
        {"records": records, "generation": {}},
    )
    atomic_json(output / "workflow.json", {"commands": [], "status": "failed"})
    seen = {}

    def resume(_module, arguments, out, workflow):
        seen["resume"] = "--resume" in arguments
        path = out / "generated/manifest.json"
        report = read_json(path)
        report["records"] = [record | {"status": "ok"} for record in report["records"]]
        atomic_json(path, report)
        workflow["commands"].append({"seconds": 0.01})
        return 0

    with patch("stifflow.formal.run_command", side_effect=resume), patch(
        "stifflow.formal.provenance", return_value={}
    ):
        workflow = generate(checkpoint, reference, config, output, "final", resume=True)
    assert seen["resume"]
    assert workflow["status"] == "completed"
    assert all(
        record["status"] == "ok"
        for record in read_json(output / "generated/manifest.json")["records"]
    )
