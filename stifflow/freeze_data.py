"""Audit existing v3 mappings without rerunning the large memmap validation."""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from .cache_io import atomic_json
from .identity import SplitMapper, molecule_identity
from .official_data import OfficialDataset
from .preflight import fingerprint, provenance
from .smoke import external_directory

VARIANT = "qm9-v3-clean-v1"


def require_frozen(specification):
    variant = specification.get("variant", {})
    if (
        variant.get("name") != VARIANT
        or variant.get("mapping_matches_existing") is not True
    ):
        raise ValueError("Formal work requires the audited qm9-v3-clean-v1 manifest")
    receipt = variant["mapping_receipt"]
    if fingerprint(receipt["path"])["sha256"] != receipt["sha256"]:
        raise ValueError("Frozen mapping receipt hash mismatch")


def freeze(dataset, manifest, scaffold_dir, smiles_npz, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(smiles_npz, allow_pickle=False) as saved:
        uniques = saved["uniques"]
        inverse = saved["inv_indices"]
    if len(inverse) != dataset.row_offsets[-1]:
        raise ValueError("SMILES rows differ from the verified inventory")
    uid_groups = [set() for _ in uniques]
    for index, (start, end) in enumerate(
        zip(dataset.row_offsets[:-1], dataset.row_offsets[1:])
    ):
        for uid in np.unique(inverse[start:end]):
            uid_groups[int(uid)].add(index)
    mapper = SplitMapper(uniques, uid_groups)
    identities, checks, sources = {}, {}, {}
    receipt_path = output / "mapping.jsonl"
    with receipt_path.open("x") as stream:
        for split in ("train", "val", "test"):
            path = Path(scaffold_dir) / f"scaffold_{split}_smiles.pkl"
            sources[split] = fingerprint(path)
            with path.open("rb") as file:
                keys = pickle.load(file)
            seen, indices, exclusions = set(), [], {}
            unresolved = 0
            for position, key in enumerate(keys):
                uid, method = mapper.resolve(str(key))
                item = {
                    "split": split,
                    "list_index": position,
                    "smiles": str(key),
                    "method": method,
                }
                reason = None
                if uid is None:
                    reason = method
                    unresolved += 1
                else:
                    index = next(iter(uid_groups[uid]))
                    if str(dataset.partitions[index]) != "qm9":
                        raise ValueError("QM9 list maps to another partition")
                    identity = molecule_identity(str(dataset.keys[index]))
                    item.update(unique_id=uid, inventory_index=index, identity=identity)
                    if identity in seen:
                        reason = "duplicate"
                    elif any(
                        identity in identities.get(prior, set())
                        for prior in ("train", "val")
                        if prior != split
                    ):
                        reason = "prior_split_overlap"
                    seen.add(identity)
                    if reason is None:
                        indices.append(index)
                item["status"] = "excluded" if reason else "accepted"
                if reason:
                    item["reason"] = reason
                    exclusions[reason] = exclusions.get(reason, 0) + 1
                stream.write(json.dumps(item) + "\n")
            identities[split] = seen
            rows = (
                np.sort(
                    np.concatenate(
                        [
                            np.arange(
                                dataset.row_offsets[i],
                                dataset.row_offsets[i + 1],
                                dtype=np.int64,
                            )
                            for i in indices
                        ]
                    )
                )
                if indices
                else np.empty(0, dtype=np.int64)
            )
            existing = np.sort(
                np.load(
                    dataset.specification["splits"][split]["path"], allow_pickle=False
                )
            )
            matches = np.array_equal(rows, existing)
            checks[split] = {
                "source_molecules": len(keys),
                "molecules": len(indices),
                "rows": len(rows),
                "exclusions": exclusions,
                "unresolved": unresolved,
                "matches_existing": bool(matches),
                "new_only_rows": len(np.setdiff1d(rows, existing)),
                "existing_only_rows": len(np.setdiff1d(existing, rows)),
            }
    passed = (
        all(
            c["matches_existing"] and not c["exclusions"].get("ambiguous_group", 0)
            for c in checks.values()
        )
        and not checks["test"]["unresolved"]
    )
    report = {
        "passed": passed,
        "checks": checks,
        "sources": sources,
        "mapping_receipt": fingerprint(receipt_path),
        "input_manifest": fingerprint(manifest),
        "provenance": provenance(),
    }
    atomic_json(output / "audit.json", report)
    if not passed:
        raise ValueError(
            "Strict mapping differs or is ambiguous; inspect audit.json, preserve the existing split"
        )
    dataset.validate_splits()
    specification = dict(dataset.specification)
    specification["split_scheme"] = VARIANT
    specification["variant"] = {
        "name": VARIANT,
        "mapping_matches_existing": True,
        "mapping_receipt": report["mapping_receipt"],
        "audit": fingerprint(output / "audit.json"),
        "source_lists": sources,
        "counts": checks,
        "recipe": "list-order identity deduplication; train wins val; test excludes prior identities; unique indexed inventory group",
        "identity": "canonical isomeric graph; ordinary H normalized; maps removed; isotopes and charge retained",
    }
    atomic_json(output / "manifest.json", specification)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scaffold-dir", type=Path, required=True)
    parser.add_argument("--smiles-npz", type=Path, required=True)
    parser.add_argument("--output", type=external_directory, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Full metadata audit belongs in an SHHPC allocation")
    report = freeze(
        OfficialDataset.from_manifest(args.manifest),
        args.manifest,
        args.scaffold_dir,
        args.smiles_npz,
        args.output,
    )
    print(json.dumps(report["checks"]))


if __name__ == "__main__":
    main()
