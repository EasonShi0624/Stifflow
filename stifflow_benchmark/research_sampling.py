"""SHHPC-only matched-noise solver/first-step experiment on a fixed validation subset."""

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

from .evaluate import evaluate_record
from .research_diagnostics import read_ensemble, select_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("references", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-steps", type=int, default=200)
    parser.add_argument("--output", type=external_directory, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Neural solver diagnostics require an SHHPC GPU allocation")
    if args.output.exists():
        parser.error("Use a fresh diagnostic directory")
    import torch
    from stifflow.features import graph_from_mol
    from stifflow.model import StifFlow
    from stifflow.research_solver import diagnose

    spec = json.loads(args.references.read_text())
    if spec["split"] != "val":
        raise ValueError("Solver decisions use validation only")
    selected = select_records(spec["records"], args.limit)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if spec.get("data_manifest", {}).get("sha256") != saved["training_signature"].get(
        "official_data"
    ) or not saved["training_signature"].get("official_data"):
        raise ValueError(
            "Reference/checkpoint frozen data identities differ or are missing"
        )
    model = StifFlow(**saved["hyper_parameters"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.cuda().eval()
    args.output.mkdir(parents=True)
    report = {
        "checkpoint": fingerprint(args.checkpoint),
        "references": fingerprint(args.references),
        "hyper_parameters": saved["hyper_parameters"],
        "selection": "SHA256(id) ascending",
        "selected_ids": [r["id"] for r in selected],
        "seed": args.seed,
        "reference_steps": args.reference_steps,
        "provenance": provenance(),
        "records": [],
    }
    atomic_json(args.output / "diagnostics.json", report)
    for item in selected:
        record = {"id": item["id"], "status": "started"}
        report["records"].append(record)
        started = time.perf_counter()
        try:
            # Only the graph reaches diagnose; references enter the evaluator below.
            mol, ref, _, identity = read_ensemble(item, args.references.parent)
            if len(ref) != item["reference_count"]:
                raise ValueError("Reference count mismatch")
            mol.RemoveAllConformers()
            graph = graph_from_mol(mol)
            arrays, diagnostics = diagnose(
                model, graph, 2 * len(ref), args.seed, args.reference_steps
            )
            record.update(diagnostics=diagnostics, reference=identity, evaluations={})
            name = hashlib.sha256(item["id"].encode()).hexdigest()[:20]
            for variant, positions in arrays.items():
                filename = f"{name}-{variant}.npz"
                np.savez_compressed(
                    args.output / filename,
                    pos_ref=ref,
                    pos_gen=positions,
                    atomic_numbers=graph["atomic_numbers"],
                )
                evaluation = evaluate_record(
                    {
                        "id": item["id"],
                        "status": "ok",
                        "npz": filename,
                        "molblock": item["molblock"],
                    },
                    args.output,
                    np.array([0.5]),
                    allow_disconnected=True,
                )
                record["evaluations"][variant] = evaluation
            record["status"] = (
                "ok"
                if all(r["status"] == "ok" for r in record["evaluations"].values())
                else "failed"
            )
        except (
            ValueError,
            OSError,
            KeyError,
            RuntimeError,
            FloatingPointError,
        ) as error:
            record.update(status="failed", reason=f"{type(error).__name__}: {error}")
        finally:
            record["wall_seconds"] = time.perf_counter() - started
            atomic_json(args.output / "diagnostics.json", report)
    report["complete"] = len(report["records"]) == len(selected) and all(
        r["status"] == "ok" for r in report["records"]
    )
    atomic_json(args.output / "diagnostics.json", report)
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
