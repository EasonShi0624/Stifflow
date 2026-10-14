"""24 GiB proxy memory acceptance and optional two-rank DDP measurements."""

import argparse
import copy
import importlib.metadata
import os
import signal
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import yaml

from .cache_io import atomic_json, read_json
from .formal import runtime_code_signature
from .preflight import WORKSPACE, fingerprint, provenance
from .smoke import external_directory
from .speed_benchmark import METHODS, SpeedBudgetReached, case, run_case
from .training import allocator_budget
from .verify_resume import compare_states


def recovery_probe(spec, run, output, rank):
    mode = spec.get("target_rng_mode", "stream")
    steps = 4 if spec.get("devices", 1) == 1 and mode == "global_draw" else 40
    paths = [
        output / f"{spec['method']}-{mode}-{name}.ckpt"
        for name in ("continuous", "partial", "resumed")
    ]
    for name, stop, resume, save in (
        ("continuous", steps, None, paths[0]),
        ("partial", steps // 2, None, paths[1]),
        ("resumed", steps, paths[1], paths[2]),
    ):
        result = run(
            spec,
            "resume-" + name,
            warmup=0,
            updates=stop,
            resume=resume,
            save=save,
            budget_steps=steps,
        )
        if result["status"] != "completed":
            raise RuntimeError(f"Recovery case failed: {result}")
    error = None
    if rank == 0:
        try:
            left, right = [
                torch.load(p, map_location="cpu", weights_only=False)
                for p in (paths[0], paths[2])
            ]
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
            key = "distributed_states" if spec.get("devices", 1) == 2 else "cuda_rng"
            compare_states(left[key], right[key], key)
        except (AssertionError, ValueError, TypeError, KeyError, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    if torch.distributed.is_initialized():
        message = [error]
        torch.distributed.broadcast_object_list(message, src=0)
        error = message[0]
    if error:
        raise AssertionError(error)
    return {"passed": True, "updates": steps, "consumed_samples": steps * 128}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE / "stifflow-v1/configs/qm9-formal-4090d.yaml",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stress-manifest", type=Path, required=True)
    parser.add_argument("--output", type=external_directory, required=True)
    parser.add_argument("--devices", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--single-report", type=Path)
    args = parser.parse_args()
    if (
        not os.getenv("SLURM_JOB_ID")
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != args.devices
    ):
        parser.error("Require a Slurm allocation with the requested visible GPU count")
    if not 0 < args.max_seconds <= (630 if args.devices == 1 else 990):
        parser.error("Deadline must leave shutdown time within the approved allocation")
    if args.devices == 2 and args.single_report is None:
        parser.error("DDP requires the matching single-card baseline")
    rank = int(os.getenv("SLURM_PROCID", "0")) if args.devices == 2 else 0
    if rank not in range(args.devices):
        parser.error("Invalid rank")
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.devices == 2:
        os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
        torch.distributed.init_process_group(
            "nccl", rank=rank, world_size=2, timeout=timedelta(seconds=60)
        )
    cfg = yaml.safe_load(args.config.read_text())
    manifest, stress_manifest = read_json(args.manifest), read_json(
        args.stress_manifest
    )
    if (
        not manifest.get("speed_only")
        or not stress_manifest.get("speed_only")
        or stress_manifest["atoms"] != [29, 29]
    ):
        raise ValueError("Require the audited speed and maximum-atom caches")
    if (
        manifest["source_config"] != cfg["source"]
        or stress_manifest["source_config"] != cfg["source"]
    ):
        raise ValueError("Source/cache mismatch")
    free, total = torch.cuda.mem_get_info()
    fraction = allocator_budget(
        cfg["gpu_memory"], total, total - free - torch.cuda.memory_reserved()
    )
    torch.cuda.set_per_process_memory_fraction(fraction)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    if args.devices == 2:
        torch.distributed.barrier()
    elapsed = time.time() - float(os.getenv("STIFFLOW_SPEED_START_EPOCH", time.time()))
    deadline = time.monotonic() + max(0, args.max_seconds - elapsed)
    report = {
        "status": "started",
        "speed_only": True,
        "devices": args.devices,
        "cases": [],
        "checks": {},
        "config": cfg,
        "config_file": fingerprint(args.config),
        "data": fingerprint(args.manifest),
        "maximum_data": fingerprint(args.stress_manifest),
        "gpu": {
            "name": torch.cuda.get_device_name(rank),
            "total_bytes": total,
            "virtual_budget_gib": 24,
        },
        "allocator_fraction": fraction,
        "provenance": provenance() if rank == 0 else None,
        "versions": {
            n: importlib.metadata.version(n)
            for n in (
                "torch",
                "lightning",
                "numpy",
                "scipy",
                "rdkit",
                "torch-geometric",
                "torch-cluster",
            )
        },
        "allocation": {
            k: os.getenv(k)
            for k in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NODELIST",
                "SLURM_JOB_GPUS",
                "SLURM_JOB_PARTITION",
            )
        },
        "limitations": "L40S 24 GiB proxy; local legacy train, not SHHPC strict cohort; DDP assessment only",
        "runtime_code": runtime_code_signature() if rank == 0 else None,
    }

    def persist():
        if rank == 0:
            atomic_json(args.output / "summary.json", report)

    def interrupted(signum, frame):
        report.update(status="interrupted", signal=signum)
        persist()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    persist()

    def run(spec, stage, maximum=False, **options):
        expired = torch.tensor(int(time.monotonic() >= deadline), device="cuda")
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(expired, op=torch.distributed.ReduceOp.MAX)
        if expired:
            raise SpeedBudgetReached("Runtime test deadline")
        report["active_case"] = {"stage": stage, "spec": spec}
        persist()
        result = run_case(
            spec,
            stress_manifest if maximum else manifest,
            cfg,
            args.output / f"{len(report['cases']):02d}-{spec['method']}-{stage}",
            deadline,
            **options,
        )
        result.update(stage=stage, maximum=maximum)
        report["cases"].append(result)
        report.pop("active_case", None)
        persist()
        if rank == 0:
            print(
                stage,
                spec,
                result["status"],
                result.get("samples_per_second"),
                flush=True,
            )
        return result

    try:
        micro = 32
        if args.devices == 2:
            baseline = read_json(args.single_report)
            if (
                baseline["status"] != "completed"
                or baseline["config_file"]["sha256"] != report["config_file"]["sha256"]
            ):
                raise ValueError(
                    "Single-card baseline is incomplete or has different configuration"
                )
            micro = baseline["selected_microbatch"]
        while True:
            results = [
                run(
                    case(
                        method,
                        micro,
                        devices=args.devices,
                        target_rng_mode="global_draw",
                        workers=0,
                    ),
                    "maximum",
                    maximum=True,
                )
                for method in METHODS
            ]
            failures = [r for r in results if r["status"] != "completed"]
            if not failures:
                break
            if micro == 16 or any(
                r.get("error_type") not in ("OutOfMemoryError", "MemoryError")
                for r in failures
            ):
                raise RuntimeError(f"Maximum-atom pressure failed: {failures}")
            micro = 16
        report["selected_microbatch"] = micro
        pressure_bytes = max(
            max(r["peak_reserved_whole_case_bytes"], r["peak_device_used_bytes"])
            for r in results
        )
        report["maximum_memory_fraction"] = pressure_bytes / min(
            total, cfg["gpu_memory"]["budget_gib"] * 1024**3
        )
        report["within_preferred_memory"] = (
            report["maximum_memory_fraction"] <= cfg["gpu_memory"]["preferred_fraction"]
        )
        for method in METHODS:
            spec = case(
                method,
                micro,
                devices=args.devices,
                target_rng_mode="global_draw",
                workers=0,
            )
            result = run(spec, "throughput", updates=12)
            if result["status"] != "completed":
                raise RuntimeError(f"Throughput failed: {result}")
            gradient_case = run(
                spec,
                "gradient",
                warmup=0,
                updates=1,
                capture=args.output / f"{method}-gradient.pt",
            )
            if gradient_case["status"] != "completed":
                raise RuntimeError("Gradient capture failed")
            if args.devices == 2:
                error = None
                if rank == 0:
                    left = torch.load(
                        args.single_report.parent / f"{method}-gradient.pt",
                        weights_only=True,
                    )
                    right = torch.load(
                        args.output / f"{method}-gradient.pt", weights_only=True
                    )
                    relative = float(
                        torch.linalg.vector_norm(left - right)
                        / torch.linalg.vector_norm(left)
                    )
                    old = next(
                        v
                        for v in baseline["cases"]
                        if v["stage"] == "gradient" and v["case"]["method"] == method
                    )["updates"][0]["update_mean_loss"]
                    new = gradient_case["updates"][0]["update_mean_loss"]
                    passed = relative <= 1e-4 and np.isclose(
                        old, new, rtol=1e-5, atol=1e-5
                    )
                    report["checks"][method + "-gradient"] = {
                        "relative_l2": relative,
                        "loss_relative": abs(new - old) / abs(old),
                        "passed": bool(passed),
                    }
                    if not passed:
                        error = "Single/DDP gradient or loss differs"
                message = [error]
                torch.distributed.broadcast_object_list(message, src=0)
                if message[0]:
                    raise AssertionError(message[0])
            report["checks"][method + "-resume"] = recovery_probe(
                spec, run, args.output, rank
            )
            persist()
        if args.devices == 1:
            for effective in (256, 512):
                result = run(
                    case(
                        "precision",
                        micro,
                        effective,
                        target_rng_mode="global_draw",
                        workers=0,
                    ),
                    "larger-batch",
                )
                if result["status"] != "completed":
                    raise RuntimeError("Expanded batch case failed")
            for method in METHODS:
                report["checks"][method + "-formal-stream-resume"] = recovery_probe(
                    case(method, micro, devices=1), run, args.output, rank
                )
        report["status"] = "completed"
        report["selected_config"] = copy.deepcopy(cfg)
        report["selected_config"]["batch_size"] = micro
        report["selected_config"]["profiles"]["formal"]["accumulate_grad_batches"] = (
            128 // micro
        )
        report["maximum_atoms"] = 29
        if rank == 0 and args.devices == 1:
            with (args.output / "selected-config.yaml").open("x") as stream:
                yaml.safe_dump(report["selected_config"], stream, sort_keys=False)
            report["kind"] = "torch_runtime"
    except SpeedBudgetReached as exc:
        report.update(status="budget_exhausted", error=str(exc))
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = args.max_seconds - max(
            0, deadline - time.monotonic()
        )
        persist()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
