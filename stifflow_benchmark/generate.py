"""Generate controlled StifFlow runs from graph-only records, then attach references."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from stifflow.cache_io import atomic_json
from stifflow.preflight import fingerprint, provenance
from stifflow.smoke import external_directory


def validate_resume(
    report,
    specification,
    output,
    checkpoint_fingerprint,
    input_fingerprint,
    checkpoint,
    method,
    seed,
    steps,
    current_provenance,
):
    """Return completed records from a validated partial generation manifest."""
    if report.get("schema_version") != 1:
        raise ValueError("Resume manifest has unsupported schema")
    if report.get("baseline_kind") != "controlled" or report.get("method") != method:
        raise ValueError("Resume manifest belongs to a different baseline/method")
    if report.get("split") != specification["split"] or report.get("seed") != seed:
        raise ValueError("Resume manifest split/seed differs")
    expected_config = {"model": checkpoint["hyper_parameters"], "steps": steps}
    if report.get("config") != expected_config:
        raise ValueError("Resume manifest sampler/model configuration differs")
    if report.get("training_budget") != {"steps": checkpoint["global_step"]}:
        raise ValueError("Resume manifest training budget differs")
    if report.get("training_data") != checkpoint["training_signature"]:
        raise ValueError("Resume manifest training data signature differs")
    if report.get("checkpoint", {}).get("sha256") != checkpoint_fingerprint["sha256"]:
        raise ValueError("Resume manifest checkpoint differs")
    if report.get("input_manifest", {}).get("sha256") != input_fingerprint["sha256"]:
        raise ValueError("Resume manifest reference input differs")
    if report.get("provenance", {}).get("repos") != current_provenance["repos"]:
        raise ValueError("Resume manifest was produced by different code revisions")
    records = report.get("records")
    if not isinstance(records, list):
        raise TypeError("Resume manifest records are missing")
    expected_ids = {item["id"] for item in specification["records"]}
    ids = [record.get("id") for record in records]
    if len(ids) != len(set(ids)) or not set(ids) <= expected_ids:
        raise ValueError("Resume manifest has duplicate or unexpected molecule IDs")
    done = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        npz = record.get("npz")
        if not isinstance(npz, str) or not (output / npz).is_file():
            continue
        saved = record.get("npz_fingerprint")
        if saved is not None and saved != fingerprint(output / npz):
            continue
        done[record["id"]] = record
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=external_directory, required=True)
    parser.add_argument("--steps", type=int, choices=[25, 50, 100], default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse an existing output directory and skip completed molecules",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=50,
        help="atomically rewrite manifest.json after this many processed molecules",
    )
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Neural generation requires a Slurm allocation")
    if args.flush_every < 1:
        parser.error("--flush-every must be positive")
    if args.output.exists() and not args.resume:
        parser.error("Use a fresh output directory or --resume")
    import torch
    from rdkit import Chem
    from stifflow.features import graph_from_mol
    from stifflow.model import StifFlow, generate

    input_path = args.input_manifest.resolve()
    specification = json.loads(input_path.read_text())
    if specification["split"] == "test" and not specification.get(
        "selection_frozen", False
    ):
        raise ValueError(
            "Freeze validation-selected configuration before test evaluation"
        )
    ids = [r["id"] for r in specification["records"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate molecule IDs")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = StifFlow(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.cuda().eval()
    checkpoint_fingerprint = fingerprint(args.checkpoint)
    input_fingerprint = fingerprint(input_path)
    current_provenance = provenance()
    if args.output.exists():
        manifest_path = args.output / "manifest.json"
        if not args.resume or not manifest_path.is_file():
            parser.error("Resume requires an existing generated manifest.json")
        report = json.loads(manifest_path.read_text())
        done = validate_resume(
            report,
            specification,
            args.output,
            checkpoint_fingerprint,
            input_fingerprint,
            checkpoint,
            model.method,
            args.seed,
            args.steps,
            current_provenance,
        )
        # Preserve every validated record in the manifest before generating new
        # ones, so a crash does not discard the completed part of a resumed run.
        report["records"] = [
            done[item["id"]] for item in specification["records"] if item["id"] in done
        ]
        report["provenance"] = current_provenance
    else:
        args.output.mkdir(parents=True)
        done = {}
        report = {
            "schema_version": 1,
            "method": model.method,
            "baseline_kind": "controlled",
            "seed": args.seed,
            "split": specification["split"],
            "config": {"model": checkpoint["hyper_parameters"], "steps": args.steps},
            "training_budget": {"steps": checkpoint["global_step"]},
            "training_data": checkpoint["training_signature"],
            "generation": {},
            "provenance": current_provenance,
            "input_manifest": input_fingerprint,
            "checkpoint": checkpoint_fingerprint,
            "records": [],
        }
    manifest_path = args.output / "manifest.json"
    started = time.perf_counter()

    def flush():
        report["generation"] = {
            "wall_seconds": time.perf_counter() - started,
            "postprocessing": "none",
            "attempted": len(report["records"]),
            "expected": len(ids),
            "resumed": args.resume,
            "resumed_records": len(done),
            "flush_every": args.flush_every,
        }
        atomic_json(manifest_path, report)

    try:
        processed = 0
        for item in specification["records"]:
            if item["id"] in done:
                continue
            record = {
                "id": item["id"],
                "status": "started",
                "molblock": item["molblock"],
            }
            report["records"].append(record)
            try:
                mol = Chem.MolFromMolBlock(item["molblock"], removeHs=False)
                if mol is None:
                    raise ValueError("Invalid input molecular graph")
                mol.RemoveAllConformers()
                graph = graph_from_mol(mol)
                if (
                    not isinstance(item["reference_count"], int)
                    or item["reference_count"] < 1
                ):
                    raise ValueError("Invalid reference count")
                positions, metrics = generate(
                    model,
                    graph,
                    sampler_config={
                        "steps": args.steps,
                        "samples": 2 * item["reference_count"],
                        "seed": args.seed,
                    },
                )
                # Reference positions are read only after generation; no label cache is opened.
                with np.load(
                    input_path.parent / item["reference_npz"], allow_pickle=False
                ) as data:
                    refs = data["pos_ref"]
                    if not np.array_equal(
                        data["atomic_numbers"], graph["atomic_numbers"]
                    ):
                        raise ValueError("Reference atom order mismatch")
                if (
                    refs.shape
                    != (item["reference_count"], len(graph["atomic_numbers"]), 3)
                    or not np.isfinite(refs).all()
                ):
                    raise ValueError("Reference count/shape/finiteness mismatch")
                name = hashlib.sha256(item["id"].encode()).hexdigest()[:20] + ".npz"
                saved = args.output / name
                np.savez_compressed(
                    saved,
                    pos_ref=refs,
                    pos_gen=positions,
                    atomic_numbers=graph["atomic_numbers"],
                )
                record.update(
                    status="ok",
                    npz=name,
                    npz_fingerprint=fingerprint(saved),
                    generation=metrics,
                )
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                record.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            processed += 1
            if processed % args.flush_every == 0:
                flush()
    finally:
        flush()
    raise SystemExit(1 if any(r["status"] != "ok" for r in report["records"]) else 0)


if __name__ == "__main__":
    main()
