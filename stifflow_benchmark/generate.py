"""Generate controlled StifFlow runs from graph-only records, then attach references."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from stifflow.preflight import fingerprint, provenance
from stifflow.smoke import external_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=external_directory, required=True)
    parser.add_argument("--steps", type=int, choices=[25, 50, 100], default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Neural generation requires a Slurm allocation")
    if args.output.exists():
        parser.error("Use a fresh output directory")
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
    args.output.mkdir(parents=True)
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
        "provenance": provenance(),
        "input_manifest": fingerprint(input_path),
        "checkpoint": fingerprint(args.checkpoint),
        "records": [],
    }
    started = time.perf_counter()
    try:
        for item in specification["records"]:
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
                np.savez_compressed(
                    args.output / name,
                    pos_ref=refs,
                    pos_gen=positions,
                    atomic_numbers=graph["atomic_numbers"],
                )
                record.update(status="ok", npz=name, generation=metrics)
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                record.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    finally:
        report["generation"] = {
            "wall_seconds": time.perf_counter() - started,
            "postprocessing": "none",
            "attempted": len(report["records"]),
            "expected": len(ids),
        }
        (args.output / "manifest.json").write_text(json.dumps(report, indent=2))
    raise SystemExit(1 if any(r["status"] != "ok" for r in report["records"]) else 0)


if __name__ == "__main__":
    main()
