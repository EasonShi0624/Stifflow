"""Scalar recovery and tiny frozen caches; never load a real model checkpoint."""

import copy
import hashlib
import json
import shutil
import subprocess
import types
from pathlib import Path

import pytest
import torch
from test_formal_cache import build, pack
from test_formal_cache import frozen as frozen  # noqa: PLC0414
from test_formal_training import fit_scalar
from test_official_data import official as official  # noqa: PLC0414

from stifflow import formal
from stifflow.cache_io import atomic_json, read_json
from stifflow.migrate_run import (
    LEGACY_COMMITS,
    METADATA,
    legacy_manifest_matches,
    migrate,
)
from stifflow.packed_data import PackedRecords, relocate_manifest
from stifflow.preflight import fingerprint
from stifflow.verify_resume import compare_states

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def migration_case(frozen, monkeypatch):
    train_manifest = pack(frozen, build(frozen))
    val_manifest = pack(frozen, build(frozen, split="val"), split="val")
    root = frozen[2]
    train, val = root / "train-packed/manifest.json", root / "val-packed/manifest.json"
    acceptance = REPO / "results/torch-runtime-batch-4090d-20261014.json"
    accepted = read_json(acceptance)
    compatibility = root / "compatibility.json"
    atomic_json(
        compatibility,
        {
            "schema_version": 1,
            "kind": "io_compatibility",
            "status": "completed",
            "acceptance_content_sha256": formal.content_sha256(accepted),
            "baseline_runtime_code": accepted["runtime_code"],
            "runtime_code": formal.runtime_code_signature(),
            "checks": dict.fromkeys(formal.IO_CHECKS, True),
            "legacy_source_commits": list(LEGACY_COMMITS),
        },
    )
    signature = formal.formal_training_signature(
        train, val, accepted["selected_config"], 42, accepted
    )
    code = {
        "stifflow-v1": {"head": LEGACY_COMMITS[-1], "diff": ""},
        **{
            name: {"head": head, "diff": ""}
            for name, head in accepted["runtime_code"]["dependencies"].items()
        },
    }
    legacy = {
        k: v
        for k, v in signature.items()
        if k not in ("signature_version", "runtime_scope")
    }
    legacy["code"] = code
    report = {
        "method": "control",
        "seed": 42,
        "config": signature["config"],
        "profile": "formal",
        "provenance": {"repos": code},
        "train_manifest": fingerprint(train),
        "val_manifest": fingerprint(val),
        "train_ids": [r["id"] for r in train_manifest["records"]],
        "val_ids": [r["id"] for r in val_manifest["records"]],
    }
    legacy_run = root / "legacy-run.json"
    atomic_json(legacy_run, report)
    monkeypatch.setattr(
        "stifflow.migrate_run.provenance", lambda: {"unit_fixture": True}
    )
    return root, train, val, acceptance, compatibility, signature, legacy, legacy_run


@pytest.mark.parametrize("staged", [False, True])
def test_migration_and_path_relocation_preserve_scalar_recovery(migration_case, staged):
    root, train, val, acceptance, compatibility, signature, legacy, legacy_run = (
        migration_case
    )
    if staged:
        report = read_json(legacy_run)
        module = legacy_staging_module()
        for split, path in (("train", train), ("val", val)):
            manifest = read_json(path)
            old_path = (
                Path("/old-node/nvme/stifflow-packed-v3-control-seed42")
                / split
                / "manifest.json"
            )
            module.rewrite_paths(
                manifest, str(path.parent) + "/", str(old_path.parent) + "/"
            )
            serialized = (json.dumps(manifest, indent=2) + "\n").encode()
            legacy[split] = hashlib.sha256(serialized).hexdigest()
            report[split + "_manifest"] = {
                "path": str(old_path),
                "sha256": legacy[split],
                "bytes": len(serialized),
            }
        atomic_json(legacy_run, report)
    records = PackedRecords(read_json(train), "control")
    try:
        full = fit_scalar(
            root / "full", 4, signature=signature, records=records, num_workers=0
        )
        fit_scalar(
            root / "partial", 2, signature=legacy, records=records, num_workers=0
        )
    finally:
        records.close()
    source = root / "immutable-old.ckpt"
    saved = torch.load(
        root / "partial/checkpoints/latest.ckpt", map_location="cpu", weights_only=False
    )
    saved["cuda_rng"] = [
        torch.arange(16, dtype=torch.uint8)
    ]  # Inert GPU metadata in a CPU fixture.
    torch.save(saved, source)
    before = fingerprint(source)
    output = root / "migrated"
    receipt = migrate(source, legacy_run, train, val, acceptance, compatibility, output)
    assert receipt["state_preserved_exactly"] is True
    migrated = output / "checkpoints/latest.ckpt"
    written = torch.load(migrated, map_location="cpu", weights_only=False)
    compare_states(
        {k: v for k, v in saved.items() if k not in METADATA},
        {k: v for k, v in written.items() if k not in METADATA},
        exact=True,
    )
    assert read_json(output / "checkpoints/owner.json")["training"] == signature
    shutil.copytree(train.parent, root / "different-node/train")
    relocated = PackedRecords(
        relocate_manifest(read_json(train), train, root / "different-node"), "control"
    )
    try:
        resumed = fit_scalar(
            root / "resumed",
            4,
            migrated,
            output / "checkpoints",
            signature=signature,
            records=relocated,
            num_workers=0,
        )
    finally:
        relocated.close()
    left = torch.load(full, map_location="cpu", weights_only=False)
    right = torch.load(resumed, map_location="cpu", weights_only=False)
    for key in (
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "torch_rng",
        "stifflow_rng",
        "gradient_queue",
        "consumed_samples",
        "global_step",
    ):
        compare_states(left[key], right[key], key)
    assert right["global_step"] == 4 and right["consumed_samples"] == 512
    assert fingerprint(source) == before
    with pytest.raises(FileExistsError):
        migrate(source, legacy_run, train, val, acceptance, compatibility, output)


def legacy_staging_module():
    code = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO),
            "show",
            LEGACY_COMMITS[-1] + ":scripts/shhpc/stage_packed_cache.py",
        ],
        text=True,
    )
    module = types.ModuleType("legacy_staging_fixture")
    exec(  # noqa: S102 -- pinned Git source
        compile(code, "legacy-staging.py", "exec"), module.__dict__
    )
    return module


def test_legacy_staging_hash_is_reconstructed_exactly(migration_case):
    _root, train, _val, _acceptance, _compatibility, _signature, legacy, _run = (
        migration_case
    )
    module = legacy_staging_module()
    staged = read_json(train)
    old_path = "/old-node/nvme/stifflow-packed-v3-control-seed42/train/manifest.json"
    module.rewrite_paths(
        staged, str(train.parent) + "/", str(Path(old_path).parent) + "/"
    )
    digest = hashlib.sha256((json.dumps(staged, indent=2) + "\n").encode()).hexdigest()
    assert digest != legacy["train"]
    legacy_manifest_matches(train, {"path": old_path, "sha256": digest}, digest)
    with pytest.raises(ValueError, match="reconstructed"):
        legacy_manifest_matches(
            train,
            {"path": old_path.replace("old-node", "guessed-node"), "sha256": digest},
            digest,
        )
    legacy_manifest_matches(train, fingerprint(train), fingerprint(train)["sha256"])


def test_migration_rejects_unknown_or_incomplete_state_before_writing(migration_case):
    root, train, val, acceptance, compatibility, _signature, legacy, legacy_run = (
        migration_case
    )
    path = fit_scalar(root / "partial", 2, signature=legacy, num_workers=0)
    original = torch.load(path, map_location="cpu", weights_only=False)
    original["cuda_rng"] = [torch.zeros(16, dtype=torch.uint8)]
    for case in (
        "cursor",
        "dirty",
        "unknown",
        "optimizer",
        "data",
        "world",
        "sampling",
    ):
        changed = copy.deepcopy(original)
        if case == "cursor":
            changed["consumed_samples"] += 1
        if case == "dirty":
            changed["training_signature"]["code"]["stifflow-v1"][
                "diff"
            ] = "unreviewed patch"
        if case == "unknown":
            changed["training_signature"]["code"]["stifflow-v1"]["head"] = "f" * 40
        if case == "optimizer":
            changed.pop("optimizer_states")
        if case == "data":
            changed["training_signature"]["train"] = "f" * 64
        if case == "world":
            changed["distributed_states"] = {"world_size": 2}
        if case == "sampling":
            changed["training_signature"]["sampling"] = "uniform_molecules"
        source = root / f"invalid-{case}.ckpt"
        torch.save(changed, source)
        before = fingerprint(source)
        output = root / f"rejected-{case}"
        with pytest.raises(ValueError):
            migrate(source, legacy_run, train, val, acceptance, compatibility, output)
        assert not output.exists() and fingerprint(source) == before
