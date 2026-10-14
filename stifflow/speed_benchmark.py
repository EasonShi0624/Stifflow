"""Allocated single-GPU speed tests; no formal training or quality claims."""

import argparse
import gc
import importlib.metadata
import os
import signal
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from etflow.models.loss import batchwise_mse_loss
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.strategies import DDPStrategy
from torch_geometric.data import Batch

from . import precision
from .cache_io import atomic_json
from .model import StifFlow
from .official_data import SAMPLING
from .packed_data import PackedRecords
from .preflight import WORKSPACE, fingerprint, provenance
from .smoke import external_directory
from .speed_data import prepare_speed_cache
from .targets import draw_target
from .training import draw_record, make_loader
from .verify_resume import compare_states

METHODS = ("precision", "alternative_a", "control")


class SpeedBudgetReached(RuntimeError):
    pass


def legacy_finite(tensors, what):
    """The audited SHHPC implementation, retained only for same-card comparison."""
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise FloatingPointError(f"Non-finite {what}")


def case(method, microbatch, effective=128, **options):
    devices = options.get("devices", 1)
    if microbatch < 1 or devices not in (1, 2) or effective % (microbatch * devices):
        raise ValueError("Microbatch must divide the effective batch")
    if devices == 2:
        # Bucket aliases change gradient storage after warmup and break exact resume.
        options.setdefault("gradient_as_bucket_view", False)
    return {
        "method": method,
        "microbatch": microbatch,
        "effective_batch": effective,
        "accumulation": effective // (microbatch * devices),
        "runtime": "optimized",
        "precision": "32-true",
        "workers": 2,
        "prefetch": 2,
        "pin_memory": False,
        **options,
    }


class UpdateTiming(Callback):
    def __init__(self, warmup, deadline, case_started, effective, memory=None):
        self.warmup, self.deadline, self.case_started = warmup, deadline, case_started
        self.effective = effective
        self.rows = []
        self.last_step = 0
        self.samples = self.atoms = 0
        self.loss_sum = 0.0
        self.startup_seconds = None
        self.memory = memory
        self.peak_device_used_bytes = self.peak_reserved_bytes = 0

    def memory_check(self):
        reserved = torch.cuda.max_memory_reserved()
        free, total = torch.cuda.mem_get_info()
        used = total - free
        self.peak_reserved_bytes = max(self.peak_reserved_bytes, reserved)
        self.peak_device_used_bytes = max(self.peak_device_used_bytes, used)
        limit = (
            min(
                torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).total_memory,
                self.memory["budget_gib"] * 1024**3,
            )
            * self.memory["hard_fraction"]
            if self.memory
            else 0.8
            * torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        )
        valid = torch.tensor(int(max(reserved, used) <= limit), device="cuda")
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(valid, op=torch.distributed.ReduceOp.MIN)
        if not valid:
            raise MemoryError("Exceeds the GPU-memory budget including device overhead")

    def on_before_backward(self, trainer, pl_module, loss):
        self.memory_check()
        self.loss_sum += float(loss.detach())

    def on_train_start(self, trainer, pl_module):
        torch.cuda.synchronize()
        self.last_step = trainer.global_step
        self.last_stamp = time.perf_counter()
        if not self.warmup:
            self.startup_seconds = self.last_stamp - self.case_started
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        expired = torch.tensor(
            int(time.monotonic() >= self.deadline), device=pl_module.device
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(expired, op=torch.distributed.ReduceOp.MAX)
        if expired:
            raise SpeedBudgetReached("Reached the speed-test deadline")
        self.samples += len(batch[1])
        self.atoms += batch[0].num_nodes

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.memory_check()
        step = trainer.global_step
        if step == self.last_step:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        elapsed = now - self.last_stamp
        totals = torch.tensor(
            [self.samples, self.atoms, self.loss_sum],
            device=pl_module.device,
            dtype=torch.float64,
        )
        duration = torch.tensor(elapsed, device=pl_module.device, dtype=torch.float64)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
            torch.distributed.all_reduce(duration, op=torch.distributed.ReduceOp.MAX)
        samples, atoms = map(int, totals[:2].tolist())
        update_loss = float(totals[2]) / trainer.world_size
        if samples != self.effective:
            raise RuntimeError("Wrong sample count at optimizer boundary")
        if step > self.warmup:
            self.rows.append(
                {
                    "step": step,
                    "seconds": float(duration),
                    "samples": samples,
                    "atoms": atoms,
                    "last_microbatch_loss": float(outputs["loss"].detach().cpu()),
                    "update_mean_loss": update_loss,
                }
            )
        if step == self.warmup:
            self.startup_seconds = now - self.case_started
            torch.cuda.reset_peak_memory_stats()
        self.last_step = step
        self.samples = self.atoms = 0
        self.loss_sum = 0.0
        self.last_stamp = time.perf_counter()


class UnclippedGradient(Callback):
    def __init__(self, path):
        self.path = path
        self.saved = False

    def on_before_optimizer_step(self, trainer, module, optimizer):
        if not self.saved and trainer.global_rank == 0:
            gradient = torch.cat(
                [
                    p.grad.detach().flatten()
                    for p in module.parameters()
                    if p.grad is not None
                ]
            )
            torch.save(gradient.cpu(), self.path)
        self.saved = True


def timing_summary(rows):
    seconds = np.asarray([r["seconds"] for r in rows])
    if not len(seconds):
        return {"measured_updates": 0}
    elapsed = float(seconds.sum())
    return {
        "measured_updates": len(rows),
        "seconds_per_update": float(seconds.mean()),
        "median_seconds_per_update": float(np.median(seconds)),
        "p90_seconds_per_update": float(np.percentile(seconds, 90)),
        "std_seconds_per_update": float(seconds.std()),
        "samples_per_second": sum(r["samples"] for r in rows) / elapsed,
        "atoms_per_second": sum(r["atoms"] for r in rows) / elapsed,
    }


def run_case(
    spec,
    manifest,
    cfg,
    output,
    deadline,
    *,
    warmup=2,
    updates=6,
    stress=False,
    profile=False,
    resume=None,
    save=None,
    budget_steps=None,
    capture=None,
):
    """Same Lightning optimizer, hooks and data cursor as training; I/O is tiny."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world = spec.get("devices", 1)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    started = time.perf_counter()
    store = PackedRecords(manifest, spec["method"])
    model = trainer = loader = None
    result = {"case": spec, "stress": stress, "profiled": profile, "status": "started"}
    if rank == 0:
        atomic_json(output / "result.json", result)
    seed_everything(42, workers=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        records = store
        if stress:
            largest = int(np.argmax(manifest["atoms"]))
            graph, row = store.sample(largest, 0)
            records = [
                {
                    "graph": graph,
                    "positions": row["x1"][None],
                    "source": row["source"],
                    "labels": [row["label"]],
                }
            ]
        consumed, start_step = 0, 0
        if resume:
            saved = torch.load(resume, map_location="cpu", weights_only=False)
            consumed, start_step = saved["consumed_samples"], saved["global_step"]
        stop = warmup + updates
        model = StifFlow(
            method=spec["method"],
            seed=42,
            source_config=cfg["source"],
            target_rng_mode=spec.get("target_rng_mode", "stream"),
            **cfg["model_args"],
        )
        model.consumed_samples = consumed
        model.training_signature = {
            "speed_only": True,
            "case": spec,
            "data": manifest["row_index"]["sha256"],
            "config": cfg,
        }
        if resume and saved.get("training_signature") != model.training_signature:
            raise ValueError("Benchmark resume belongs to different data/config/case")
        loader = make_loader(
            records,
            spec["microbatch"],
            budget_steps or stop,
            42,
            sampling=SAMPLING,
            accumulate_grad_batches=spec["accumulation"],
            consumed_samples=consumed,
            num_workers=spec["workers"],
            rng_domain=2,
            pin_memory=spec["pin_memory"],
            prefetch_factor=spec["prefetch"],
            rank=rank,
            world_size=world,
        )
        timer = UpdateTiming(
            warmup, deadline, started, spec["effective_batch"], cfg.get("gpu_memory")
        )
        trainer = Trainer(
            accelerator="gpu",
            devices=world,
            strategy=(
                DDPStrategy(
                    find_unused_parameters=False,
                    gradient_as_bucket_view=spec["gradient_as_bucket_view"],
                )
                if world > 1
                else "auto"
            ),
            use_distributed_sampler=False,
            plugins=[SLURMEnvironment(auto_requeue=False)],
            max_steps=stop,
            max_epochs=-1,
            accumulate_grad_batches=spec["accumulation"],
            precision=spec["precision"],
            deterministic=True,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[timer] + ([UnclippedGradient(capture)] if capture else []),
            default_root_dir=output,
        )
        with ExitStack() as context:
            if spec["runtime"] == "baseline":
                context.enter_context(
                    patch("stifflow.model.require_finite", legacy_finite)
                )
                context.enter_context(
                    patch("stifflow.targets.precision_target", precision.sample_target)
                )
                context.enter_context(
                    patch.object(
                        model,
                        "transfer_batch_to_device",
                        lambda payload, device, dataloader_idx: (
                            payload[0].to(device),
                            payload[1],
                        ),
                    )
                )
            profiler = (
                context.enter_context(
                    torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    )
                )
                if profile
                else None
            )
            trainer.fit(
                model,
                train_dataloaders=loader,
                ckpt_path=str(resume) if resume else None,
            )
        if profiler:
            averages = profiler.key_averages()
            phases = []
            for event in averages:
                if event.key.startswith("stifflow/") or any(
                    s in event.key
                    for s in ("item", "Synchronize", "Optimizer", "Backward")
                ):
                    phases.append(
                        {
                            "name": event.key,
                            "calls": event.count,
                            "cpu_ms": event.cpu_time_total / 1000,
                            "self_cpu_ms": event.self_cpu_time_total / 1000,
                            "device_ms": getattr(event, "device_time_total", 0) / 1000,
                        }
                    )
            atomic_json(output / "phases.json", phases)
            (output / "profile.txt").write_text(
                averages.table(sort_by="self_cpu_time_total", row_limit=40)
            )
        result.update(
            status="completed",
            startup_seconds=timer.startup_seconds,
            start_step=start_step,
            global_step=trainer.global_step,
            consumed_samples=model.consumed_samples,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_device_used_bytes=timer.peak_device_used_bytes,
            peak_reserved_whole_case_bytes=timer.peak_reserved_bytes,
            updates=timer.rows,
            **timing_summary(timer.rows),
        )
        if (
            trainer.global_step != stop
            or model.consumed_samples != stop * spec["effective_batch"]
        ):
            raise RuntimeError(
                "Benchmark stopped before its requested update/sample budget"
            )
        if save:
            trainer.save_checkpoint(save)
    except SpeedBudgetReached:
        result.update(status="budget_exhausted")
        raise
    except (RuntimeError, MemoryError, FloatingPointError) as exc:
        result.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
        )
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        if torch.distributed.is_initialized():
            memory = torch.tensor(
                [
                    result.get("peak_reserved_whole_case_bytes", 0),
                    result.get("peak_device_used_bytes", 0),
                ],
                device="cuda",
                dtype=torch.int64,
            )
            memories = [torch.empty_like(memory) for _ in range(world)]
            torch.distributed.all_gather(memories, memory)
            result["per_rank_memory"] = [list(map(int, v.tolist())) for v in memories]
        if rank == 0:
            atomic_json(output / "result.json", result)
        store.close()
        del loader, trainer, model
        gc.collect()
        torch.cuda.empty_cache()
    return result


def fixed_batch_check(manifest, cfg):
    """Full backbone: compare equal supervision partitioned as 16x1 and 4x4."""
    seed_everything(42)
    store = PackedRecords(manifest, "precision")
    rows, graphs = [], []
    rng = np.random.default_rng(42)
    for draw_id in range(16):
        i, j = draw_record(store.lengths, 42, draw_id, SAMPLING, store.offsets, 2)
        graph, row = store.sample(i, j)
        rows.append(
            draw_target(row["x1"], row["source"], row["label"], "precision", rng)
        )
        graphs.append(graph)
    model = StifFlow(
        method="precision", source_config=cfg["source"], **cfg["model_args"]
    ).cuda()

    def gradients(model, microbatch, bf16=False):
        model.zero_grad(set_to_none=True)
        loss_total, predictions = 0.0, []
        for start in range(0, 16, microbatch):
            selected = rows[start : start + microbatch]
            graph = Batch.from_data_list(graphs[start : start + microbatch]).to("cuda")
            x = torch.as_tensor(
                np.concatenate([r[0] for r in selected]),
                dtype=torch.float32,
                device="cuda",
            )
            u = torch.as_tensor(
                np.concatenate([r[1] for r in selected]),
                dtype=torch.float32,
                device="cuda",
            )
            t = torch.tensor([r[2] for r in selected], device="cuda").reshape(-1, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                predicted = model(
                    graph.atomic_numbers,
                    t,
                    x,
                    graph.edge_index,
                    graph.edge_attr,
                    graph.node_attr,
                    graph.batch,
                )
                loss = batchwise_mse_loss(
                    predicted.float(), u, graph.batch, reduce="sum"
                ) / (16 // microbatch)
            loss.backward()
            loss_total += float(loss.detach())
            predictions.append(predicted.detach().float())
        model.on_after_backward()
        gradient = torch.cat(
            [
                p.grad.detach().reshape(-1)
                for p in model.parameters()
                if p.grad is not None
            ]
        )
        return loss_total, gradient, torch.cat(predictions)

    try:
        expected_loss, expected_grad, expected_prediction = gradients(model, 16)
        actual_loss, actual_grad, _ = gradients(model, 4)
        relative = float(
            torch.linalg.vector_norm(expected_grad - actual_grad)
            / torch.linalg.vector_norm(expected_grad)
        )
        if relative > 1e-4 or not np.isclose(
            expected_loss, actual_loss, rtol=1e-5, atol=1e-5
        ):
            raise AssertionError("Accumulation changes the equal-sample gradient/loss")
        report = {
            "samples": 16,
            "accumulation_gradient_relative_l2": relative,
            "full_batch_loss": expected_loss,
            "accumulated_loss": actual_loss,
            "passed": True,
        }
        try:
            loss, gradient, prediction = gradients(model, 16, bf16=True)
            report["bf16"] = {
                "loss_relative_difference": abs(loss - expected_loss)
                / abs(expected_loss),
                "gradient_relative_l2": float(
                    torch.linalg.vector_norm(expected_grad - gradient)
                    / torch.linalg.vector_norm(expected_grad)
                ),
                "prediction_relative_l2": float(
                    torch.linalg.vector_norm(expected_prediction - prediction)
                    / torch.linalg.vector_norm(expected_prediction)
                ),
                "finite": bool(
                    torch.isfinite(gradient).all() and torch.isfinite(prediction).all()
                ),
            }
        except (RuntimeError, FloatingPointError) as exc:
            report["bf16"] = {"supported": False, "error": str(exc)}
        return report
    finally:
        store.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def resume_check(spec, run, output):
    paths = [output / f"{name}.ckpt" for name in ("continuous", "partial", "resumed")]
    for name, stop, resume, save in (
        ("continuous", 4, None, paths[0]),
        ("partial", 2, None, paths[1]),
        ("resumed", 4, paths[1], paths[2]),
    ):
        result = run(
            spec,
            "resume-" + name,
            warmup=0,
            updates=stop,
            resume=resume,
            save=save,
            budget_steps=4,
        )
        if result["status"] != "completed":
            raise RuntimeError(f"GPU recovery probe failed: {result}")
    left, right = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in (paths[0], paths[2])
    ]
    for key in (
        "global_step",
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "stifflow_rng",
        "torch_rng",
        "cuda_rng",
        "consumed_samples",
        "gradient_queue",
    ):
        compare_states(left[key], right[key], key)
    return {"passed": True, "updates": 4, "consumed_samples": left["consumed_samples"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=WORKSPACE / "stifflow-v1/configs/qm9-formal.yaml"
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=external_directory, required=True)
    parser.add_argument("--output", type=external_directory, required=True)
    parser.add_argument("--max-seconds", type=int, default=3300)
    args = parser.parse_args()
    if (
        not os.getenv("SLURM_JOB_ID")
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        parser.error(
            "Speed tests require an allocated Slurm job with exactly one visible GPU"
        )
    if not 0 < args.max_seconds <= 3300:
        parser.error("The approved GPU test limit is 55 minutes plus shutdown time")
    args.output.mkdir(parents=True, exist_ok=False)
    elapsed = time.time() - float(os.getenv("STIFFLOW_SPEED_START_EPOCH", time.time()))
    deadline = time.monotonic() + max(0, args.max_seconds - elapsed)
    cfg = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    report = {
        "speed_only": True,
        "status": "started",
        "cases": [],
        "checks": {},
        "gpu": {
            "name": props.name,
            "total_bytes": props.total_memory,
            "capability": torch.cuda.get_device_capability(),
        },
        "python": sys.executable,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "lightning",
                "numpy",
                "scipy",
                "rdkit",
                "torch-geometric",
                "torch-cluster",
            )
        },
        "cuda_toolkit": torch.version.cuda,
        "config": cfg,
        "config_file": fingerprint(args.config),
        "provenance": provenance(),
        "allocated_resources": {
            key: os.getenv(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_ACCOUNT",
                "SLURM_JOB_PARTITION",
                "SLURM_CPUS_PER_TASK",
                "SLURM_JOB_GPUS",
            )
        },
        "timing_protocol": "2 warmup + 6 measured optimizer updates, synchronized per update; profiling/checkpoints separate",
        "limitations": "Tiny local legacy-training cache, not SHHPC strict-v3 storage or final training quality; last-microbatch loss is not update-mean loss",
    }
    report_path = args.output / "summary.json"
    atomic_json(report_path, report)

    def interrupted(signum, frame):
        report.update(status="interrupted", signal=signum)
        atomic_json(report_path, report)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)

    def run(spec, stage, **options):
        if time.monotonic() >= deadline:
            raise SpeedBudgetReached("Reached the approved test budget")
        number = len(report["cases"])
        report["active_case"] = {"case": spec, "stage": stage, "index": number}
        atomic_json(report_path, report)
        result = run_case(
            spec,
            manifest,
            cfg,
            args.output / f"{number:02d}-{stage}",
            deadline,
            **options,
        )
        result["stage"] = stage
        report["cases"].append(result)
        report.pop("active_case", None)
        atomic_json(report_path, report)
        print(
            f"case {number:02d} {stage}: {result['status']} {spec}; samples/s={result.get('samples_per_second')}",
            flush=True,
        )
        return result

    try:
        manifest = prepare_speed_cache(args.split_manifest, args.cache, cfg["source"])
        report["data"] = {
            "manifest": fingerprint(args.cache / "manifest.json"),
            "atoms": manifest["atoms"],
            "preparation_seconds": manifest["preparation_seconds"],
            "stiffness_preprocessing_seconds": manifest[
                "stiffness_preprocessing_seconds"
            ],
        }
        report["checks"]["fixed_supervision"] = fixed_batch_check(manifest, cfg)
        atomic_json(report_path, report)
        for method in METHODS:
            run(case(method, 8, runtime="baseline"), "baseline")
        candidates = [
            run(case("precision", micro), "batch128") for micro in (8, 16, 32, 64, 128)
        ]
        valid = sorted(
            (r for r in candidates if r["status"] == "completed"),
            key=lambda r: r["samples_per_second"],
            reverse=True,
        )
        chosen = None
        for candidate in valid:
            if (
                run(candidate["case"], "stress128", warmup=0, updates=1, stress=True)[
                    "status"
                ]
                == "completed"
            ):
                chosen = candidate["case"]
                break
        if chosen is None:
            raise RuntimeError(
                "No batch-128 setting passes the largest-molecule memory probe"
            )
        report["batch128_candidate"] = chosen
        for effective in (256, 512):
            larger = case("precision", chosen["microbatch"], effective)
            run(larger, f"batch{effective}")
            biggest = chosen["microbatch"]
            probe = biggest * 2
            while probe <= effective:
                test = run(
                    case("precision", probe, probe),
                    f"memory{probe}",
                    warmup=0,
                    updates=1,
                    stress=True,
                )
                if test["status"] != "completed":
                    break
                biggest, probe = probe, probe * 2
            if biggest != chosen["microbatch"]:
                run(
                    case("precision", biggest, effective),
                    f"batch{effective}-larger-micro",
                )
        loader_cases = [
            run({**chosen, "workers": workers}, "loader") for workers in (0, 2, 4)
        ]
        loader_valid = [r for r in loader_cases if r["status"] == "completed"]
        tuned = max(loader_valid, key=lambda r: r["samples_per_second"])["case"]
        prefetch_case = {**tuned, "workers": tuned["workers"] or 2, "prefetch": 4}
        loader_cases.append(run(prefetch_case, "prefetch"))
        loader_cases.append(run({**tuned, "pin_memory": True}, "pinned"))
        loader_valid = [r for r in loader_cases if r["status"] == "completed"]
        tuned = max(loader_valid, key=lambda r: r["samples_per_second"])["case"]
        report["batch128_candidate"] = tuned
        for method in METHODS:
            run({**tuned, "method": method}, "confirm128", updates=12)
            if torch.cuda.is_bf16_supported():
                run({**tuned, "method": method, "precision": "bf16-mixed"}, "bf16")
        for effective in (256, 512):
            matched = [
                r
                for r in report["cases"]
                if r["stage"].startswith(f"batch{effective}")
                and r["status"] == "completed"
            ]
            if matched and torch.cuda.is_bf16_supported():
                expanded = max(matched, key=lambda r: r["samples_per_second"])["case"]
                run({**expanded, "precision": "bf16-mixed"}, f"bf16-{effective}")
        run(
            case("precision", 8, runtime="baseline"),
            "profile-baseline",
            warmup=0,
            updates=1,
            profile=True,
        )
        run(tuned, "profile-candidate", warmup=0, updates=1, profile=True)
        report["checks"]["gpu_resume"] = resume_check(tuned, run, args.output)
        report.update(status="completed", formal_batch_unchanged=True)
    except SpeedBudgetReached as exc:
        report.update(status="budget_exhausted", error=str(exc))
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = args.max_seconds - max(
            0, deadline - time.monotonic()
        )
        atomic_json(report_path, report)
        print(f"Speed report: {report_path}; status={report['status']}", flush=True)


if __name__ == "__main__":
    main()
