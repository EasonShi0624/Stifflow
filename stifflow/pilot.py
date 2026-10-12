"""SHHPC smoke/resume acceptance, then the fixed three-method QM9 pilot."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .official_data import KIND, validate_label_pair
from .preflight import WORKSPACE, fingerprint, provenance
from .smoke import external_directory

METHODS = ("precision", "alternative_a", "control")


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--val-labels", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument(
        "--config", type=Path, default=WORKSPACE / "stifflow-v1/configs/qm9.yaml"
    )
    parser.add_argument("--output", type=external_directory, required=True)
    parser.add_argument("--smoke-acceptance", type=Path)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Pilot execution belongs in an SHHPC GPU allocation")
    if args.output.exists():
        parser.error("Existing runs are never overwritten")
    train = json.loads(args.train_labels.read_text())
    val = json.loads(args.val_labels.read_text())
    validate_label_pair(train, val)
    cfg = yaml.safe_load(args.config.read_text())
    profile = cfg["profiles"][args.profile]
    if (
        min(
            len(train["records"]) - profile["train_molecules"],
            len(val["records"]) - profile["val_molecules"],
        )
        < 0
    ):
        parser.error("Insufficient complete shared molecules for the requested profile")
    prov = provenance()
    signature = {
        "train": fingerprint(args.train_labels)["sha256"],
        "val": fingerprint(args.val_labels)["sha256"],
        "config": fingerprint(args.config)["sha256"],
        "repos": {
            name: {k: repo[k] for k in ("head", "diff")}
            for name, repo in prov["repos"].items()
        },
    }
    if args.profile == "pilot":
        if args.smoke_acceptance is None:
            parser.error(
                "Pilot requires --smoke-acceptance from this data/config/code version"
            )
        accepted = json.loads(args.smoke_acceptance.read_text())
        if (
            accepted.get("status") != "completed"
            or accepted.get("profile") != "smoke"
            or accepted.get("signature") != signature
        ):
            parser.error(
                "Smoke acceptance is incomplete or belongs to different data/config/code"
            )
    args.output.mkdir(parents=True)
    report = {
        "profile": args.profile,
        "signature": signature,
        "provenance": prov,
        "dataset_kind": KIND,
        "status": "started",
        "commands": [],
    }
    path = args.output / "manifest.json"

    def save():
        temporary = args.output / "manifest.tmp"
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(path)

    def run(module, arguments):
        command = [sys.executable, "-m", module, *map(str, arguments)]
        item = {"command": command, "status": "started"}
        report["commands"].append(item)
        save()
        started = time.perf_counter()
        log = args.output / f"command-{len(report['commands']):02d}.log"
        item["log"] = log.name
        with log.open("x") as stream:
            result = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        item.update(
            returncode=result.returncode,
            seconds=time.perf_counter() - started,
            status="passed" if result.returncode == 0 else "failed",
        )
        save()
        if result.returncode:
            raise RuntimeError(f"{module} failed; see {log}")

    try:
        run("stifflow.preflight", ["--gpu", "--output", args.output / "preflight.json"])
        if args.profile == "smoke":
            run(
                "stifflow.smoke",
                [
                    "--data-manifest",
                    train["data_manifest"]["path"],
                    "--cache-dir",
                    args.output / "official-cache",
                    "--output-dir",
                    args.output / "official-generated",
                ],
            )
        for method in METHODS:
            directory = args.output / method
            base = [
                "--method",
                method,
                "--profile",
                args.profile,
                "--config",
                args.config,
                "--train-labels",
                args.train_labels,
                "--val-labels",
                args.val_labels,
            ]
            run("stifflow.training", [*base, "--output", directory])
            if args.profile == "smoke":
                partial = args.output / f"{method}-partial"
                resumed = args.output / f"{method}-resumed"
                run(
                    "stifflow.training",
                    [*base, "--stop-after", profile["steps"] // 2, "--output", partial],
                )
                run(
                    "stifflow.training",
                    [*base, "--resume", partial / "final.ckpt", "--output", resumed],
                )
                run(
                    "stifflow.verify_resume",
                    [
                        directory / "final.ckpt",
                        resumed / "final.ckpt",
                        "--output",
                        args.output / f"{method}-resume.json",
                    ],
                )
            else:
                generated = args.output / f"{method}-generated"
                references = args.val_labels.parent / "references/manifest.json"
                run(
                    "stifflow_benchmark.generate",
                    [
                        references,
                        "--checkpoint",
                        directory / "final.ckpt",
                        "--steps",
                        50,
                        "--seed",
                        cfg["seed"],
                        "--output",
                        generated,
                    ],
                )
                run(
                    "stifflow_benchmark.evaluate",
                    [
                        generated / "manifest.json",
                        "--output",
                        generated / "evaluation.json",
                    ],
                )
        if args.profile == "pilot":
            run(
                "stifflow_benchmark.summarize",
                [
                    *[
                        args.output / f"{method}-generated/evaluation.json"
                        for method in METHODS
                    ],
                    "--output",
                    args.output / "comparison.json",
                ],
            )
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
