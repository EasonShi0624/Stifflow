"""Tiny packed arrays exercise I/O equivalence and directory/data-loss boundaries."""

import copy
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.shhpc import stage_packed_cache as staging
from stifflow import formal
from stifflow.cache_io import atomic_json, read_json
from stifflow.formal_checkpoints import FormalCheckpoints
from stifflow.packed_data import PackedRecords, relocate_manifest
from stifflow.preflight import fingerprint
from stifflow.verify_resume import compare_states

BASELINE = "e463b16543f1365baab731e4da909d6f6cc8e2e6"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def packed(tmp_path):
    root = tmp_path / "canonical"
    directory = root / "train"
    directory.mkdir(parents=True)
    manifest = {
        "schema_version": 3,
        "split": "train",
        "records": [],
        "graph_shards": [],
        "factor_shards": [],
    }
    index = []
    for part in range(18):
        n, d = 3, 6
        bonds = np.zeros((1, n, n, 3), dtype=np.float32)
        bonds[0, 0, 1:] = bonds[0, 1:, 0] = 1
        groups = {
            "graph": {
                "numbers": np.array([[6, 1, 1]], dtype=np.int64),
                "nodes": np.zeros((1, n, 10), dtype=np.float32),
                "bonds": bonds,
                "S": np.eye(d)[None],
                "C0": np.eye(d)[None],
            },
            "factor": {
                "positions": np.arange(18, dtype=np.float64).reshape(2, n, 3) + part,
                "row_ids": np.array([2 * part, 2 * part + 1], dtype=np.int64),
                "precision_values": np.full((2, d), 2.0),
                "precision_vectors": np.tile(np.eye(d), (2, 1, 1)),
                "alt_L_K": np.tile(np.eye(d) * 0.5, (2, 1, 1)),
            },
        }
        for kind, arrays in groups.items():
            files = {}
            for name, array in arrays.items():
                path = directory / f"{kind}-{part}-{name}.npy"
                np.save(path, array)
                files[name] = {**fingerprint(path), "shape": list(array.shape)}
            manifest[kind + "_shards"].append({"atoms": n, "files": files})
        manifest["records"].append(
            {
                "id": f"unit-{part}",
                "row_ids": [2 * part, 2 * part + 1],
                "graph_shard": part,
                "graph_row": 0,
            }
        )
        index.extend([[part, 0, 2 * part], [part, 1, 2 * part + 1]])
    path = directory / "row_index.npy"
    np.save(path, np.array(index, dtype=np.int64))
    manifest["row_index"] = fingerprint(path)
    atomic_json(directory / "manifest.json", manifest)
    return root, directory / "manifest.json", manifest


def test_old_and_new_reader_are_identical_across_cache_evictions(packed, tmp_path):
    root, path, manifest = packed
    source = subprocess.check_output(
        ["git", "-C", str(REPO), "show", BASELINE + ":stifflow/packed_data.py"],
        text=True,
    )
    module = types.ModuleType("stifflow._legacy_packed")
    module.__package__ = "stifflow"
    exec(  # noqa: S102 -- pinned Git source
        compile(source, "legacy-packed-data.py", "exec"), module.__dict__
    )
    original = path.read_bytes()
    staging.stage(root, tmp_path / "local", splits=["train"], min_free_gib=0)
    relocated = relocate_manifest(manifest, path, tmp_path / "local")
    for method in formal.METHODS:
        old = module.PackedRecords(manifest, method)
        new = PackedRecords(relocated, method, max_open_parts=3, graph_cache_size=1)
        try:
            for part in [*range(18), *reversed(range(18))]:
                for row in (0, 1):
                    a, x = old.sample(part, row)
                    b, y = new.sample(part, row)
                    compare_states(a.to_dict(), b.to_dict(), exact=True)
                    np.testing.assert_array_equal(x["x1"], y["x1"])
                    for key in ("Q", "S", "C0"):
                        np.testing.assert_array_equal(
                            getattr(x["source"], key), getattr(y["source"], key)
                        )
                    compare_states(x["label"], y["label"], exact=True)
                    assert x["row_id"] == y["row_id"]
                    assert len(new.parts) <= 3
        finally:
            old.close()
            new.close()
    assert (
        path.read_bytes()
        == original
        == (tmp_path / "local/train/manifest.json").read_bytes()
    )
    assert manifest == read_json(path)


def test_relocation_confines_paths_and_still_detects_corruption(packed, tmp_path):
    root, path, manifest = packed
    staging.stage(root, tmp_path / "local", splits=["train"], min_free_gib=0)
    runtime = relocate_manifest(manifest, path, tmp_path / "local")
    target = Path(runtime["factor_shards"][0]["files"]["positions"]["path"])
    raw = bytearray(target.read_bytes())
    raw[-1] ^= 1
    target.write_bytes(raw)
    reader = PackedRecords(runtime, "control")
    with pytest.raises(ValueError, match="hash mismatch"):
        reader.sample(0, 0)
    reader.close()
    invalid = copy.deepcopy(manifest)
    invalid["row_index"]["path"] = str(tmp_path / "outside.npy")
    with pytest.raises(ValueError, match="canonical split"):
        relocate_manifest(invalid, path, tmp_path / "local")
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "train").symlink_to(root / "train", target_is_directory=True)
    with pytest.raises(ValueError, match="staging root"):
        relocate_manifest(manifest, path, linked)


@pytest.mark.parametrize(
    "case", ["equal", "nested", "ancestor", "existing", "split", "zero", "symlink"]
)
def test_staging_rejects_unsafe_inputs_without_touching_data(packed, tmp_path, case):
    source, path, _ = packed
    before = path.read_bytes()
    destination, options = tmp_path / "new", {}
    if case == "equal":
        destination = source
    if case == "nested":
        destination = source / "nested"
    if case == "ancestor":
        destination = source.parent
    if case == "existing":
        destination.mkdir()
        (destination / "keep").write_text("old cache")
    if case == "split":
        options["splits"] = ["../train"]
    if case == "zero":
        options["verify_samples"] = 0
    if case == "symlink":
        destination.symlink_to(source, target_is_directory=True)
    with pytest.raises((ValueError, FileExistsError)):
        staging.stage(source, destination, min_free_gib=0, **options)
    assert path.read_bytes() == before
    if case == "existing":
        assert (destination / "keep").read_text() == "old cache"


def test_interrupted_copy_is_not_published(packed, tmp_path, monkeypatch):
    source, path, _ = packed
    before = path.read_bytes()
    run = subprocess.run

    def interrupted(command, **kwargs):
        if command[0] == "cp":
            raise subprocess.CalledProcessError(1, command)
        return run(command, **kwargs)

    monkeypatch.setattr(staging.subprocess, "run", interrupted)
    target = tmp_path / "partial"
    with pytest.raises(subprocess.CalledProcessError):
        staging.stage(source, target, splits=["train"], min_free_gib=0)
    assert not (target / ".staged.json").exists()
    assert path.read_bytes() == before


def test_checkpoint_directory_has_only_one_writer(tmp_path):
    first = FormalCheckpoints(tmp_path / "checkpoints", 128)
    second = FormalCheckpoints(tmp_path / "checkpoints", 128)
    first.setup(None, None, "fit")
    try:
        with pytest.raises(RuntimeError, match="Another process"):
            second.setup(None, None, "fit")
    finally:
        first.teardown(None, None, "fit")
    second.setup(None, None, "fit")
    second.teardown(None, None, "fit")


def test_concurrent_staging_has_separate_outputs_and_serialized_copies(
    packed, tmp_path, monkeypatch
):
    source, path, _ = packed
    before = path.read_bytes()
    entered, release = threading.Event(), threading.Event()
    lock = tmp_path / "shared-disk.lock"
    run = subprocess.run
    copies = []

    def copy_command(command, **kwargs):
        if command[0] == "cp":
            copies.append(command[-1])
            if len(copies) == 1:
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("Unit fixture did not release first copy")
        return run(command, **kwargs)

    monkeypatch.setattr(staging.subprocess, "run", copy_command)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            staging.stage, source, tmp_path / "job-a", ["train"], 8, 0, lock
        )
        try:
            assert entered.wait(10)
            second = pool.submit(
                staging.stage, source, tmp_path / "job-b", ["train"], 8, 0, lock
            )
            with lock.open("a") as stream, pytest.raises(BlockingIOError):
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            release.set()
        first.result(timeout=10)
        second.result(timeout=10)
    assert len(set(copies)) == 2
    assert all((tmp_path / job / ".staged.json").exists() for job in ("job-a", "job-b"))
    assert path.read_bytes() == before


def test_io_acceptance_preserves_algorithm_config_and_dependency_guards(monkeypatch):
    report = read_json(REPO / "results/torch-runtime-batch-4090d-20261014.json")
    cfg = report["selected_config"]
    current = copy.deepcopy(report["runtime_code"])
    current["stifflow"].update(
        packed_data="reader",
        training="identity",
        formal_checkpoints="metadata",
        formal="guard",
        official_data="references",
    )
    compatibility = {
        "schema_version": 1,
        "kind": "io_compatibility",
        "status": "completed",
        "baseline_runtime_code": report["runtime_code"],
        "runtime_code": current,
        "acceptance_content_sha256": formal.content_sha256(report),
        "checks": dict.fromkeys(formal.IO_CHECKS, True),
    }
    monkeypatch.setattr(formal, "runtime_code_signature", lambda: current)
    with pytest.raises(ValueError, match="compatibility"):
        formal.require_torch_runtime(report, cfg)
    formal.require_torch_runtime(report, cfg, compatibility)
    for field in ("acceptance_content_sha256", "checks", "runtime_code"):
        changed = copy.deepcopy(compatibility)
        changed[field] = {}
        with pytest.raises(ValueError):
            formal.require_torch_runtime(report, cfg, changed)
    changed_cfg = copy.deepcopy(cfg)
    changed_cfg["batch_size"] = 64
    with pytest.raises(ValueError):
        formal.require_torch_runtime(report, changed_cfg, compatibility)
    for section, name in (("stifflow", "model"), ("dependencies", "ETFlow-reference")):
        original = current[section][name]
        current[section][name] = "unreviewed"
        with pytest.raises(ValueError):
            formal.require_torch_runtime(report, cfg, compatibility)
        current[section][name] = original


def test_exact_migration_comparison_does_not_use_resume_tolerance():
    left = torch.tensor([1.0], dtype=torch.float64)
    changed = left + 1e-8
    compare_states(left, changed)
    with pytest.raises(AssertionError):
        compare_states(left, changed, exact=True)


def test_runtime_signature_rejects_dirty_dependencies(monkeypatch):
    monkeypatch.setattr(formal, "git", lambda *args: "unreviewed dependency patch")
    with pytest.raises(ValueError, match="uncommitted changes"):
        formal.runtime_code_signature()


def test_staged_wrapper_restarts_same_job_in_private_attempts(tmp_path):
    workspace = tmp_path / "workspace"
    repo = workspace / "stifflow-v1"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(REPO / "scripts/formal.sbatch", scripts / "formal.sbatch")
    binary = tmp_path / "bin"
    binary.mkdir()
    (scripts / "shhpc_env.sh").write_text(
        f"export PATH={shlex.quote(str(binary))}:$PATH\ncd {shlex.quote(str(repo))}\n"
    )
    python = binary / "python"
    python.write_text(f"#!{sys.executable}\n" + """import json, os, sys, uuid
from pathlib import Path
args = sys.argv[1:]
def value(flag): return args[args.index(flag) + 1]
if args[0] == "-c":
    print(uuid.uuid4().hex)
elif args[0].endswith("stage_packed_cache.py"):
    Path(value("--dst-root")).mkdir()
    with Path(os.environ["UNIT_STAGE_LOG"]).open("a") as stream:
        stream.write(json.dumps(args) + "\\n")
elif args[:2] == ["-m", "stifflow.preflight"]:
    path = Path(value("--output"))
    with path.open("x") as stream: stream.write("{}")
elif args[:2] == ["-m", "stifflow.training"]:
    path = Path(value("--output"))
    path.mkdir(parents=True)
    (path / "args.json").write_text(json.dumps(args))
    checkpoints = Path(value("--checkpoint-dir"))
    checkpoints.mkdir(exist_ok=True)
    (checkpoints / "latest.ckpt").touch()
else:
    raise SystemExit("Unexpected fake command: " + str(args))
""")
    python.chmod(0o700)
    df = binary / "df"
    df.write_text("#!/bin/sh\nprintf 'Filesystem Avail\\next4 999999999999\\n'\n")
    df.chmod(0o700)
    local = tmp_path / "local"
    local.mkdir()
    source = tmp_path / "canonical"
    source.mkdir()
    (source / "keep").write_text("read-only fixture")
    env = {
        **os.environ,
        "STIFFLOW_WORKSPACE": str(workspace),
        "STIFFLOW_ACCEPTANCE": "/fixture/acceptance.json",
        "STIFFLOW_IO_COMPATIBILITY": "/fixture/compatibility.json",
        "STIFFLOW_RUN_ROOT": str(tmp_path / "runs"),
        "STIFFLOW_PACKED_SOURCE_ROOT": str(source),
        "SLURM_JOB_ID": "unit4321",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_TMPDIR": str(local),
        "UNIT_STAGE_LOG": str(tmp_path / "stage.jsonl"),
    }
    for restart in (0, 1):
        env["SLURM_RESTART_COUNT"] = str(restart)
        subprocess.run(
            ["bash", str(REPO / "scripts/shhpc/formal_staged.sbatch")],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    base = tmp_path / "runs/precision-seed42"
    assert len(list(base.glob("preflight-*.json"))) == 2
    attempts = list((base / "segments").iterdir())
    assert len(attempts) == 2 and attempts[0].name != attempts[1].name
    roots = []
    for attempt in attempts:
        args = json.loads((attempt / "args.json").read_text())
        assert args[args.index("--train-labels") + 1] == str(
            source / "train/manifest.json"
        )
        assert (
            args[args.index("--io-compatibility") + 1] == "/fixture/compatibility.json"
        )
        roots.append(args[args.index("--packed-cache-root") + 1])
    assert roots[0] != roots[1] and all(not Path(root).exists() for root in roots)
    assert (source / "keep").read_text() == "read-only fixture"
