"""SHHPC label audit/selection: all official rows, whole-molecule rejection."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from rdkit import Chem

from .official_data import KIND, SAMPLING, OfficialDataset
from .preflight import WORKSPACE, fingerprint, provenance
from .smoke import external_directory
from .source import prepare_source
from .stiffness import DEFAULTS, build_label_arrays


def label_record(dataset, entry, split, source_config, audit=False):
    """One complete molecule uses the existing physical-label implementation."""
    molecule = dataset.load_molecule(entry, split, limit=1 if audit else None)
    selected = molecule["conformers"]
    source = prepare_source(molecule["graph"], source_config)
    arrays, audits, residuals = build_label_arrays(selected, source)
    rows = [c["row_id"] for c in selected]
    arrays["row_ids"] = np.asarray(rows, dtype=np.int64)
    record = {
        "id": entry["id"],
        "split": split,
        "row_ids": rows,
        "raw_indices": [c["raw_index"] for c in selected],
        "conformer_ids": [c["conformer_id"] for c in selected],
        "retained_conformers": len(entry["row_ids"]),
        "audit_partial": audit,
        "molblock": Chem.MolToMolBlock(selected[0]["rd_mol"]),
        "graph": molecule["graph"],
        "source": molecule["source"],
        "original_weights": [c["boltzmann_weight"] for c in selected],
        "original_energies": [c["energy"] for c in selected],
        "labels": audits,
        "alternative_factor_residuals": residuals,
    }
    return record, arrays, source


def build_cache(
    dataset,
    data_manifest,
    output,
    split,
    count,
    source_config,
    seed=42,
    audit=False,
    max_candidates=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {
        "schema_version": 2,
        "dataset_kind": KIND,
        "sampling": SAMPLING,
        "split": split,
        "selection_seed": seed,
        "requested_molecules": count,
        "data_manifest": fingerprint(data_manifest),
        "source_config": source_config,
        "stiffness_config": DEFAULTS,
        "audit_partial": audit,
        "requested_attempts_completed": False,
        "status": "started",
        "selection": "seeded SHA256 ordering; whole molecule rejected if any official row fails",
        "records": [],
        "attempts": [],
        "provenance": provenance(),
    }
    manifest = output / "manifest.json"

    def save_progress():
        # Only this invocation owns the fresh output directory and report.
        temporary = output / "manifest.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(manifest)

    save_progress()
    try:
        candidates = dataset.candidates(split, seed)
        report["candidate_molecules"] = len(candidates)
        for entry in candidates:
            if len(report["records"]) == count or (
                max_candidates is not None and len(report["attempts"]) >= max_candidates
            ):
                break
            attempt = {
                "id": entry["id"],
                "split": split,
                "official_conformers": len(entry["row_ids"]),
                "status": "started",
            }
            report["attempts"].append(attempt)
            attempt_started = time.perf_counter()
            try:
                record, arrays, _source = label_record(
                    dataset, entry, split, source_config, audit
                )
                rows = record["row_ids"]
                filename = (
                    hashlib.sha256(entry["id"].encode()).hexdigest()[:20] + ".npz"
                )
                with (output / filename).open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                record.update(npz=filename, cache=fingerprint(output / filename))
                report["records"].append(record)
                attempt.update(
                    status="success",
                    atoms=len(record["graph"]["atomic_numbers"]),
                    labeled_rows=len(rows),
                )
            except (
                ValueError,
                RuntimeError,
                OSError,
                KeyError,
                np.linalg.LinAlgError,
                FloatingPointError,
            ) as exc:
                attempt.update(status="rejected", reason=f"{type(exc).__name__}: {exc}")
            finally:
                attempt["seconds"] = time.perf_counter() - attempt_started
                save_progress()
        report["requested_attempts_completed"] = len(report["records"]) == count
        report["status"] = (
            "completed"
            if report["requested_attempts_completed"]
            else "insufficient_labels"
        )
        if split == "val" and not audit and report["status"] == "completed":
            export_references(report, output)
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        report["rejected_molecules"] = sum(
            a["status"] == "rejected" for a in report["attempts"]
        )
        report["labeled_rows"] = sum(len(r["row_ids"]) for r in report["records"])
        report["label_bytes"] = sum(r["cache"]["bytes"] for r in report["records"])
        if audit:
            report["cost_estimates"] = [
                {
                    "id": a["id"],
                    "official_rows": a["official_conformers"],
                    "estimated_all_rows_seconds": a["seconds"]
                    * a["official_conformers"],
                }
                for a in report["attempts"]
                if a["status"] == "success"
            ]
            report["cost_estimate_limit"] = (
                "One-row timing extrapolation only; not measured full-molecule runtime"
            )
        save_progress()
    return report


def export_references(report, output):
    """Separate graph/reference files so generation never opens Hessian caches."""
    directory = output / "references"
    directory.mkdir()
    reference = {
        "schema_version": 1,
        "split": "val",
        "data_manifest": report["data_manifest"],
        "records": [],
    }
    for record in report["records"]:
        with np.load(output / record["npz"], allow_pickle=False) as arrays:
            positions = arrays["positions"]
        with (directory / record["npz"]).open("xb") as stream:
            np.savez_compressed(
                stream,
                pos_ref=positions,
                atomic_numbers=record["graph"]["atomic_numbers"],
            )
        reference["records"].append(
            {
                "id": record["id"],
                "molblock": record["molblock"],
                "reference_count": len(record["row_ids"]),
                "reference_npz": record["npz"],
            }
        )
    with (directory / "manifest.json").open("x") as stream:
        json.dump(reference, stream, indent=2, allow_nan=False)


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=external_directory, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--molecules", type=int, default=128)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument(
        "--audit",
        action="store_true",
        help="One row per molecule; never a training-ready cache",
    )
    parser.add_argument(
        "--config", type=Path, default=WORKSPACE / "stifflow-v1/configs/qm9.yaml"
    )
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Official label preprocessing belongs in an SHHPC allocation")
    if args.molecules < 1 or (
        args.max_candidates is not None and args.max_candidates < 1
    ):
        parser.error("Counts must be positive")
    if args.output.exists():
        parser.error("Existing label caches are never overwritten")
    cfg = yaml.safe_load(args.config.read_text())
    dataset = OfficialDataset.from_manifest(args.manifest)
    dataset.validate_splits()
    report = build_cache(
        dataset,
        args.manifest,
        args.output,
        args.split,
        args.molecules,
        cfg["source"],
        cfg["seed"],
        args.audit,
        args.max_candidates,
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "labeled_rows",
                    "rejected_molecules",
                    "seconds",
                    "label_bytes",
                )
            }
        )
    )
    raise SystemExit(0 if report["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
