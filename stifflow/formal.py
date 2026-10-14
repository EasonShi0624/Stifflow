"""SHHPC acceptance, independent references and fixed-budget formal evaluation."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem

from .cache_io import atomic_json, read_json
from .freeze_data import require_frozen
from .official_data import OfficialDataset, validate_label_pair, verified_path
from .preflight import WORKSPACE, fingerprint, git, provenance
from .smoke import external_directory

METHODS = ("precision", "alternative_a", "control")
SEEDS = (42, 43, 44)


def runtime_code_signature():
    """Hash executed algorithm files, so a results-only commit does not invalidate evidence."""
    files = (
        "model",
        "training",
        "targets",
        "source",
        "precision",
        "geometry",
        "features",
        "packed_data",
        "formal_checkpoints",
    )
    return {
        "stifflow": {
            name: fingerprint(WORKSPACE / "stifflow-v1/stifflow" / (name + ".py"))[
                "sha256"
            ]
            for name in files
        },
        "dependencies": {
            name: git(WORKSPACE / name, "rev-parse", "HEAD").strip()
            for name in (
                "ETFlow-reference",
                "stifflow-v1-alternative-A",
                "stifflow-v1-benchmark",
            )
        },
    }


def require_torch_runtime(report, cfg):
    """Runtime-only evidence; production label/data validation remains in training."""
    if (
        report.get("kind") != "torch_runtime"
        or report.get("status") != "completed"
        or report.get("devices") != 1
        or report.get("selected_config") != cfg
        or report.get("runtime_code") != runtime_code_signature()
        or report.get("maximum_atoms") != 29
    ):
        raise ValueError(
            "Torch runtime acceptance scope/config/code differs or is incomplete"
        )
    for method in METHODS:
        check = report.get("checks", {}).get(method + "-formal-stream-resume", {})
        if check != {"passed": True, "updates": 40, "consumed_samples": 5120}:
            raise ValueError(
                "Missing formal-stream 40-update runtime recovery evidence"
            )
        pressure = [
            r
            for r in report.get("cases", [])
            if r.get("maximum")
            and r["case"]["method"] == method
            and r["case"]["microbatch"] == cfg["batch_size"]
        ]
        if not pressure or pressure[-1].get("status") != "completed":
            raise ValueError("Missing maximum-atom memory evidence")
        limit = (
            min(report["gpu"]["total_bytes"], cfg["gpu_memory"]["budget_gib"] * 1024**3)
            * cfg["gpu_memory"]["hard_fraction"]
        )
        memory = max(
            pressure[-1].get("peak_reserved_whole_case_bytes", 0),
            pressure[-1].get("peak_device_used_bytes", 0),
        )
        if not 0 < memory <= limit:
            raise ValueError(
                "Maximum-atom memory evidence exceeds the configured ceiling"
            )


def acceptance_signature(train, val, config, prov):
    return {
        "train": fingerprint(train)["sha256"],
        "val": fingerprint(val)["sha256"],
        "config": fingerprint(config)["sha256"],
        "code": {
            name: {"head": item["head"], "diff": item["diff"]}
            for name, item in prov["repos"].items()
        },
    }


def run_command(module, arguments, output, report):
    command = [sys.executable, "-m", module, *map(str, arguments)]
    log = output / f"command-{len(report['commands']) + 1:02d}.log"
    item = {"command": command, "log": str(log), "status": "started"}
    report["commands"].append(item)
    atomic_json(output / "workflow.json", report)
    started = time.perf_counter()
    with log.open("x") as stream:
        result = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    item.update(
        returncode=result.returncode,
        seconds=time.perf_counter() - started,
        status="passed" if result.returncode == 0 else "failed",
    )
    atomic_json(output / "workflow.json", report)
    return result.returncode


def accept(train, val, config, output):
    validate_label_pair(read_json(train), read_json(val))
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "started",
        "signature": acceptance_signature(train, val, config, provenance()),
        "commands": [],
        "methods": list(METHODS),
        "updates": 40,
    }
    try:
        if run_command(
            "stifflow.preflight",
            ["--gpu", "--warmup", "--output", output / "preflight.json"],
            output,
            report,
        ):
            raise RuntimeError("GPU preflight failed")
        for method in METHODS:
            base = [
                "--config",
                config,
                "--train-labels",
                train,
                "--val-labels",
                val,
                "--profile",
                "formal",
                "--method",
                method,
                "--seed",
                42,
            ]
            paths = {
                key: output / f"{method}-{key}"
                for key in ("continuous", "partial", "resumed")
            }
            commands = [
                (
                    "stifflow.training",
                    [*base, "--stop-after", 40, "--output", paths["continuous"]],
                ),
                (
                    "stifflow.training",
                    [*base, "--stop-after", 20, "--output", paths["partial"]],
                ),
                (
                    "stifflow.training",
                    [
                        *base,
                        "--stop-after",
                        40,
                        "--resume",
                        paths["partial"] / "final.ckpt",
                        "--output",
                        paths["resumed"],
                    ],
                ),
                (
                    "stifflow.verify_resume",
                    [
                        paths["continuous"] / "final.ckpt",
                        paths["resumed"] / "final.ckpt",
                        "--output",
                        output / f"{method}-resume.json",
                    ],
                ),
            ]
            for module, arguments in commands:
                if run_command(module, arguments, output, report):
                    raise RuntimeError(
                        f"{method} acceptance failed; inspect workflow.json"
                    )
            if read_json(output / f"{method}-resume.json")["passed"] is not True:
                raise ValueError("Resume comparison did not pass")
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_json(output / "workflow.json", report)
    return report


def export_references(dataset, data_manifest, config, output):
    import yaml
    from stifflow_benchmark.evaluate import tetrahedral_chirality

    from .features import graph_from_mol

    require_frozen(dataset.specification)
    cfg = yaml.safe_load(config.read_text())
    output.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for split in ("val", "test"):
        directory = output / split
        directory.mkdir()
        reference = {
            "schema_version": 1,
            "split": split,
            "data_manifest": fingerprint(data_manifest),
            "variant": dataset.specification["variant"]["name"],
            "records": [],
            "selection_frozen": True,
            "formal_config": fingerprint(config),
            "selection": "all frozen split molecules; independent of stiffness availability",
        }
        for entry in dataset.candidates(split, cfg["dataset_seed"]):
            molecule = dataset.load_molecule(entry, split, allow_stereo_variation=True)
            selected = molecule["conformers"]
            mol = selected[0]["rd_mol"]
            graph_from_mol(mol)  # Check graph-only model features before training.
            positions = np.stack([c["positions"] for c in selected])
            filename = hashlib.sha256(entry["id"].encode()).hexdigest() + ".npz"
            with (directory / filename).open("xb") as stream:
                np.savez_compressed(
                    stream,
                    pos_ref=positions,
                    atomic_numbers=molecule["graph"]["atomic_numbers"],
                )
            reference["records"].append(
                {
                    "id": entry["id"],
                    "molblock": Chem.MolToMolBlock(mol),
                    "reference_count": len(selected),
                    "reference_npz": filename,
                    "cache": fingerprint(directory / filename),
                    "reference_chirality": tetrahedral_chirality(mol, positions),
                    "stereo_variation_rows": molecule["stereo_variation_rows"],
                    "conditioning_graph": "first official conformer; reference stereo differences recorded",
                }
            )
        atomic_json(directory / "manifest.json", reference)
        manifests[split] = reference
    monitor = copy.deepcopy(manifests["val"])
    monitor["records"] = monitor["records"][: cfg["evaluation"]["monitor_molecules"]]
    monitor["selection"] = "seed42 identity-SHA256 prefix of the full validation split"
    # Same directory keeps the existing evaluator's relative reference paths valid.
    atomic_json(output / "val/monitor.json", monitor)
    return {split: len(r["records"]) for split, r in manifests.items()}


def generate(checkpoint, references, config, output, stage):
    import torch
    import yaml

    cfg = yaml.safe_load(config.read_text())
    reference = read_json(references)
    if reference["formal_config"]["sha256"] != fingerprint(config)["sha256"]:
        raise ValueError("Reference and formal configuration differ")
    data = read_json(verified_path(reference["data_manifest"]))
    require_frozen(data)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    step = saved["global_step"]
    training = saved["training_signature"]
    if (
        training["official_data"] != reference["data_manifest"]["sha256"]
        or training["config"] != cfg
    ):
        raise ValueError("Checkpoint/reference data or configuration differs")
    if (
        training["effective_batch_size"] != 128
        or training["training_seed"] not in SEEDS
    ):
        raise ValueError("Wrong formal training budget/seed")
    if stage == "final" and step != 250000:
        raise ValueError("Primary evaluation requires the 250000-update checkpoint")
    if stage == "monitor" and (
        reference["split"] != "val"
        or len(reference["records"]) != cfg["evaluation"]["monitor_molecules"]
        or not 0 < step <= 250000
        or step % 25000
    ):
        raise ValueError(
            "Monitoring requires the frozen 512-molecule validation subset and a 25k snapshot"
        )
    if reference["split"] == "test" and (
        stage != "final" or not reference["selection_frozen"]
    ):
        raise ValueError("Test is evaluated once with the fixed final checkpoint")
    if reference["split"] not in ("val", "test"):
        raise ValueError("Formal geometry evaluation uses validation or test")
    expected_groups = OfficialDataset(data).candidates(
        reference["split"], cfg["dataset_seed"]
    )
    if stage == "monitor":
        expected_groups = expected_groups[: cfg["evaluation"]["monitor_molecules"]]
    if [r["id"] for r in reference["records"]] != [
        r["id"] for r in expected_groups
    ] or any(
        r["reference_count"] != len(g["row_ids"])
        for r, g in zip(reference["records"], expected_groups)
    ):
        raise ValueError(
            "Reference IDs/counts differ from the complete frozen split or monitor subset"
        )
    if (
        stage == "final"
        and len(reference["records"])
        != data["variant"]["counts"][reference["split"]]["molecules"]
    ):
        raise ValueError("Final evaluation requires the entire frozen split")
    for item in reference["records"]:
        verified_path(item["cache"])
    output.mkdir(parents=True, exist_ok=False)
    workflow = {"commands": [], "status": "started"}
    directory = output / "generated"
    result = run_command(
        "stifflow_benchmark.generate",
        [
            references,
            "--checkpoint",
            checkpoint,
            "--steps",
            50,
            "--seed",
            cfg["generation_seed"],
            "--output",
            directory,
        ],
        output,
        workflow,
    )
    path = directory / "manifest.json"
    if not path.exists():
        directory.mkdir(exist_ok=True)
        atomic_json(
            path,
            {
                "schema_version": 1,
                "method": saved["hyper_parameters"]["method"],
                "baseline_kind": "controlled",
                "seed": cfg["generation_seed"],
                "split": reference["split"],
                "config": {"model": saved["hyper_parameters"], "steps": 50},
                "training_budget": {"steps": step},
                "training_data": training,
                "input_manifest": fingerprint(references),
                "checkpoint": fingerprint(checkpoint),
                "provenance": provenance(),
                "generation": {
                    "wall_seconds": workflow["commands"][-1]["seconds"],
                    "expected": len(reference["records"]),
                },
                "records": [],
            },
        )
    report = read_json(path)
    actual = {item["id"]: item for item in report["records"]}
    expected = {item["id"] for item in reference["records"]}
    if len(actual) != len(report["records"]) or set(actual) - expected:
        raise ValueError("Unexpected/duplicate generated molecule IDs")
    report["records"] = [
        actual.get(
            item["id"],
            {
                "id": item["id"],
                "molblock": item["molblock"],
                "status": "failed",
                "reason": "generation terminated before this molecule",
            },
        )
        for item in reference["records"]
    ]
    report["generation"]["attempted"] = len(expected)
    report["formal"] = {
        "stage": stage,
        "training_seed": training["training_seed"],
        "generation_seed": cfg["generation_seed"],
        "references": fingerprint(references),
        "expected_molecules": len(expected),
    }
    atomic_json(path, report)
    workflow.update(
        status="completed" if result == 0 else "failed",
        generated_manifest=fingerprint(path),
    )
    atomic_json(output / "workflow.json", workflow)
    if result:
        raise RuntimeError(
            "Generation failed; all expected IDs remain in the manifest for evaluation"
        )
    return workflow


def evaluate_generated(generated, output):
    from stifflow_benchmark.evaluate import evaluate

    if output.exists():
        raise FileExistsError(output)
    report = evaluate(generated)
    expected = report["run"]["formal"]["expected_molecules"]
    report["complete"] = (
        report["aggregate"]["attempted"] == expected
        and report["aggregate"]["failed"] == 0
    )
    report["evaluation_provenance"] = provenance()
    atomic_json(output, report)
    if not report["complete"]:
        raise ValueError(
            "Incomplete formal evaluation; retained failures are not a valid primary comparison"
        )
    return report


def summarize(paths):
    from stifflow_benchmark.summarize import summarize as compare_methods

    reports = [read_json(path) for path in paths]
    grouped, common = {}, None
    for path, report in zip(paths, reports):
        run = report["run"]
        seed = run["formal"]["training_seed"]
        method = run["method"]
        if (
            not report.get("complete")
            or run["formal"]["stage"] != "final"
            or run["training_budget"]["steps"] != 250000
        ):
            raise ValueError(
                "Only complete 250k final reports enter the primary summary"
            )
        if seed not in SEEDS or method not in METHODS or (seed, method) in grouped:
            raise ValueError("Unexpected or duplicate method/seed")
        signature = copy.deepcopy(
            {
                "ids": [r["id"] for r in report["records"]],
                "split": run["split"],
                "protocol": report["protocol"],
                "data": run["training_data"],
                "reference": run["input_manifest"]["sha256"],
                "config": run["config"],
                "generation_seed": run["seed"],
            }
        )
        signature["data"].pop("training_seed")
        signature["config"]["model"].pop("seed")
        signature["config"]["model"].pop("method")
        if common is not None and signature != common:
            raise ValueError("Formal experiments differ beyond method/training seed")
        common = signature
        grouped[seed, method] = (path, report)
    if set(grouped) != {(s, m) for s in SEEDS for m in METHODS}:
        raise ValueError("All nine method/seed reports are required")
    per_seed = {
        str(seed): compare_methods([grouped[seed, method][0] for method in METHODS])
        for seed in SEEDS
    }
    thresholds = np.asarray(reports[0]["protocol"]["thresholds"])
    at_half = int(np.argmin(abs(thresholds - 0.5)))
    if not np.isclose(thresholds[at_half], 0.5):
        raise ValueError("Missing COV@0.5 threshold")
    methods = {}
    for method in METHODS:
        metrics = {}
        for seed in SEEDS:
            aggregate = grouped[seed, method][1]["aggregate"]
            for name in (
                "mat_r_mean",
                "mat_r_median",
                "mat_p_mean",
                "mat_p_median",
                "cov_r_mean",
                "cov_r_median",
                "cov_p_mean",
                "cov_p_median",
            ):
                value = aggregate["successful_only"][name]
                if name.startswith("cov"):
                    value = value[at_half]
                metrics.setdefault(name, []).append(value)
            metrics.setdefault("chirality_fraction", []).append(
                aggregate["chirality"]["fraction"]
            )
        methods[method] = {
            name: {
                "per_seed": values,
                "mean": (
                    float(np.mean(values))
                    if all(v is not None for v in values)
                    else None
                ),
                "sample_std": (
                    float(np.std(values, ddof=1))
                    if all(v is not None for v in values)
                    else None
                ),
            }
            for name, values in metrics.items()
        }
    return {
        "complete": True,
        "seeds": list(SEEDS),
        "cov_threshold": 0.5,
        "std_ddof": 1,
        "per_seed": per_seed,
        "methods": methods,
        "signature": common,
        "reports": [fingerprint(path) for path in paths],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("acceptance", "references", "generate", "evaluate", "summarize"),
    )
    parser.add_argument("--output", type=external_directory, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-labels", type=Path)
    parser.add_argument("--val-labels", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--stage", choices=("monitor", "final"), default="final")
    parser.add_argument("--reports", type=Path, nargs="+")
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Formal execution/evaluation belongs in an SHHPC allocation")
    required = {
        "acceptance": ("config", "train_labels", "val_labels"),
        "references": ("config", "manifest"),
        "generate": ("config", "checkpoint", "references"),
        "evaluate": ("manifest",),
        "summarize": ("reports",),
    }
    for name in required[args.action]:
        if getattr(args, name) is None:
            parser.error(f"{args.action} requires --{name.replace('_', '-')}")
    if args.action == "acceptance":
        report = accept(args.train_labels, args.val_labels, args.config, args.output)
    elif args.action == "references":
        report = export_references(
            OfficialDataset.from_manifest(args.manifest),
            args.manifest,
            args.config,
            args.output,
        )
    elif args.action == "generate":
        report = generate(
            args.checkpoint, args.references, args.config, args.output, args.stage
        )
    elif args.action == "evaluate":
        report = evaluate_generated(args.manifest, args.output)
    else:
        if args.output.exists():
            parser.error("Existing summary is protected")
        report = summarize(args.reports)
        atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "action": args.action,
                "output": str(args.output),
                "complete": report.get("complete", report.get("status")),
            }
        )
    )


if __name__ == "__main__":
    main()
