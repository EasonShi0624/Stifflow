"""Small CPU contracts of DDP IDs, memory budgets and legacy configuration."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from stifflow.model import StifFlow, graph_data
from stifflow.source import prepare_source
from stifflow.speed_benchmark import case
from stifflow.training import advance_cursor, allocator_budget, make_loader


def tiny_records():
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


def test_rank_draw_blocks_resume_and_effective_batch():
    assert case("precision", 32, devices=2)["accumulation"] == 2
    assert case("precision", 16, devices=2)["accumulation"] == 4
    assert case("precision", 32, devices=2)["gradient_as_bucket_view"] is False
    assert case("precision", 32, devices=2, gradient_as_bucket_view=True) != case(
        "precision", 32, devices=2
    )  # Different strategy settings also differ in the checkpoint case signature.
    records = tiny_records()
    ids = []
    for rank in (0, 1):
        loader = make_loader(
            records, 2, 2, 42, accumulate_grad_batches=2, rank=rank, world_size=2
        )
        cursor = 0
        sequence = []
        for _, rows in loader:
            sequence.extend(r["draw_id"] for r in rows)
            cursor = advance_cursor(cursor, rows, rank, 2)
        assert cursor == 16
        ids.extend(sequence)
        resumed = make_loader(
            records,
            2,
            1,
            42,
            accumulate_grad_batches=2,
            rank=rank,
            world_size=2,
            consumed_samples=8,
        )
        assert [r["draw_id"] for _, rows in resumed for r in rows] == sequence[4:]
    assert sorted(ids) == list(range(16))
    with pytest.raises(RuntimeError):
        advance_cursor(0, [{"draw_id": 1}], 0, 2)
    with pytest.raises(ValueError):
        make_loader(records, 2, 1, 42, rank=2, world_size=2)


def test_resume_failure_keeps_path_and_numeric_diagnostics():
    from stifflow.verify_resume import compare_states

    with pytest.raises(AssertionError) as error:
        compare_states(torch.tensor([1.0]), torch.tensor([2.0]), "state_dict.weight")
    assert "state_dict.weight" in str(error.value)
    assert "Greatest absolute difference" in str(error.value)
    with pytest.raises(ValueError, match="gradient_queue: 1.0 vs 2.0"):
        compare_states(1.0, 2.0, "gradient_queue")


def test_memory_ceiling_counts_overhead_and_validates_fractions():
    cfg = {"budget_gib": 24, "preferred_fraction": 0.9, "hard_fraction": 0.95}
    assert allocator_budget(cfg, 48 * 1024**3, 1024**3) == pytest.approx(
        (22.8 - 1) / 48
    )
    with pytest.raises(MemoryError):
        allocator_budget(cfg, 24 * 1024**3, 24 * 1024**3)
    with pytest.raises(ValueError):
        allocator_budget({**cfg, "preferred_fraction": 0.99}, 24 * 1024**3, 0)


def test_global_draw_targets_ignore_rank_partition_and_keep_default_stream():
    records = tiny_records()
    model = StifFlow(
        method="control",
        target_rng_mode="global_draw",
        hidden_channels=8,
        num_layers=1,
        num_rbf=8,
        num_heads=2,
        clip_during_norm=True,
        lr_scheduler_type=None,
    )
    model.log_helper = lambda *args, **kwargs: None
    full = next(iter(make_loader(records, 4, 1, 42, sampling="uniform_conformer_rows")))
    expected = model.generic_step(full, 0, "val")
    actual = (
        sum(
            model.generic_step(
                next(
                    iter(
                        make_loader(
                            records,
                            2,
                            1,
                            42,
                            sampling="uniform_conformer_rows",
                            rank=r,
                            world_size=2,
                        )
                    )
                ),
                0,
                "val",
            )
            for r in (0, 1)
        )
        / 2
    )
    torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-5)
    assert (
        StifFlow(
            method="control",
            hidden_channels=8,
            num_layers=1,
            num_rbf=8,
            num_heads=2,
            clip_during_norm=True,
        ).target_rng_mode
        == "stream"
    )
    with pytest.raises(ValueError):
        StifFlow(target_rng_mode="invalid")


def test_formal_4090_config_preserves_training_budget():
    base = yaml.safe_load(Path("configs/qm9-formal.yaml").read_text())
    cfg = yaml.safe_load(Path("configs/qm9-formal-4090d.yaml").read_text())
    assert (
        cfg["batch_size"] * cfg["profiles"]["formal"]["accumulate_grad_batches"] == 128
    )
    assert cfg["model_args"] == base["model_args"] and cfg["source"] == base["source"]
    assert cfg["profiles"]["formal"]["steps"] == 250000
    assert cfg["training_seeds"] == [42, 43, 44]
    assert cfg["runtime_acceptance"] == "torch_runtime"


def test_runtime_receipt_scope_code_recovery_and_memory_gate(monkeypatch):
    import copy

    from stifflow import formal

    cfg = yaml.safe_load(Path("configs/qm9-formal-4090d.yaml").read_text())
    signature = {"verified": "sha"}
    monkeypatch.setattr(formal, "runtime_code_signature", lambda: signature)
    r = {
        "kind": "torch_runtime",
        "status": "completed",
        "devices": 1,
        "selected_config": cfg,
        "runtime_code": signature,
        "maximum_atoms": 29,
        "gpu": {"total_bytes": 48 * 1024**3},
        "checks": {
            m
            + "-formal-stream-resume": {
                "passed": True,
                "updates": 40,
                "consumed_samples": 5120,
            }
            for m in formal.METHODS
        },
        "cases": [
            {
                "maximum": True,
                "status": "completed",
                "case": {"method": m, "microbatch": cfg["batch_size"]},
                "peak_reserved_whole_case_bytes": 10 * 1024**3,
                "peak_device_used_bytes": 11 * 1024**3,
            }
            for m in formal.METHODS
        ],
    }
    formal.require_torch_runtime(r, cfg)
    for key, value in (
        ("runtime_code", {}),
        ("devices", 2),
        ("maximum_atoms", 25),
        ("checks", {}),
    ):
        bad = copy.deepcopy(r)
        bad[key] = value
        with pytest.raises(ValueError):
            formal.require_torch_runtime(bad, cfg)
    bad = copy.deepcopy(r)
    bad["cases"][0]["peak_device_used_bytes"] = 23 * 1024**3
    with pytest.raises(ValueError):
        formal.require_torch_runtime(bad, cfg)


def test_batch_selection_requires_gain_in_every_method_and_repeat():
    import copy

    from stifflow.runtime_benchmark import choose_batch
    from stifflow.speed_benchmark import METHODS

    rows = [
        {
            "stage": f"batch-throughput-{repeat}",
            "case": case(method, micro),
            "status": "completed",
            "measured_updates": 40,
            "seconds_per_update": 1.0 if micro == 32 else 0.9,
        }
        for repeat in range(1, 4)
        for method in METHODS
        for micro in (32, 64)
    ]
    selected, comparison = choose_batch(rows)
    assert selected == 64
    assert comparison["precision"]["speedup"] == pytest.approx(1 / 0.9)
    changed = copy.deepcopy(rows)
    changed[-1]["seconds_per_update"] = 1.01
    assert choose_batch(changed)[0] == 32
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["case"]["method"] == "control" and row["case"]["microbatch"] == 64:
            row["seconds_per_update"] = 0.97
    assert choose_batch(changed)[0] == 32
    baseline = [r for r in rows if r["case"]["microbatch"] == 32]
    assert choose_batch(baseline) == (32, {})
    with pytest.raises(ValueError, match="three complete"):
        choose_batch(rows[:-1])
    broken = copy.deepcopy(rows)
    broken[-1]["status"] = "budget_exhausted"
    with pytest.raises(ValueError, match="three complete"):
        choose_batch(broken)


def test_batch_memory_failure_skips_candidate_and_keeps_full_acceptance(
    monkeypatch, tmp_path
):
    from stifflow import runtime_benchmark as runtime

    cfg = yaml.safe_load(Path("configs/qm9-formal-4090d.yaml").read_text())
    report = {
        "cases": [],
        "checks": {},
        "config": cfg,
        "gpu": {"total_bytes": 48 * 1024**3},
    }
    recovered = []

    def run(spec, stage, maximum=False, **options):
        result = {
            "case": spec,
            "stage": stage,
            "maximum": maximum,
            "status": "completed",
            "measured_updates": options.get("updates", 6),
            "seconds_per_update": 1.0,
            "peak_reserved_whole_case_bytes": 10 * 1024**3,
            "peak_device_used_bytes": 11 * 1024**3,
        }
        if spec["microbatch"] == 64:
            assert maximum and stage == "batch-maximum"
            result.update(status="failed", error_type="MemoryError")
        if "capture" in options:
            torch.save(torch.ones(3), options["capture"])
        report["cases"].append(result)
        return result

    def recovery(spec, *args):
        recovered.append(spec)
        return {"passed": True, "updates": 40, "consumed_samples": 5120}

    monkeypatch.setattr(runtime, "recovery_probe", recovery)
    assert runtime.optimize_batch(run, report, tmp_path, lambda: None) == 32
    assert len(recovered) == 3
    assert all(r["microbatch"] == 32 and r["workers"] == 2 for r in recovered)
    assert sum(r["stage"] == "maximum-sustained" for r in report["cases"]) == 3
    assert report["maximum_memory_fraction"] == pytest.approx(11 / 24)
    assert "candidate_rejected" in report
