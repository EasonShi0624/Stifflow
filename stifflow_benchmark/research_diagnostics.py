"""P0 audits of saved ensembles, reference mappings and clipping logs; no fitting."""

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from stifflow.preflight import fingerprint, provenance
from stifflow.smoke import external_directory

from .evaluate import conformer_mol


def select_records(records, limit):
    ids = [r["id"] for r in records]
    if not ids or len(ids) != len(set(ids)) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Expected unique nonempty IDs and positive limit")
    return sorted(records, key=lambda r: hashlib.sha256(r["id"].encode()).digest())[
        :limit
    ]


def cip_labels(mol, positions=None):
    candidate = Chem.Mol(mol) if positions is None else conformer_mol(mol, positions)
    if positions is not None:
        Chem.RemoveStereochemistry(candidate)
        Chem.AssignAtomChiralTagsFromStructure(
            candidate, confId=0, replaceExistingTags=True
        )
    Chem.AssignStereochemistry(candidate, cleanIt=True, force=True)
    return {
        a.GetIdx(): a.GetProp("_CIPCode")
        for a in candidate.GetAtoms()
        if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED and a.HasProp("_CIPCode")
    }


def stereo_audit(mol, positions):
    expected = cip_labels(mol)
    patterns, per_center = [], {str(i): Counter() for i in expected}
    for pos in positions:
        actual = cip_labels(mol, pos)
        matches = [actual.get(i) == code for i, code in expected.items()]
        if not expected:
            pattern = "unspecified"
        elif all(matches):
            pattern = "correct"
        elif any(
            actual.get(i) not in ("R", "S") or code not in ("R", "S")
            for i, code in expected.items()
        ):
            pattern = "missing_or_non_RS"
        elif not any(matches):
            pattern = "all_inverted"
        else:
            pattern = "partially_inverted"
        patterns.append(pattern)
        for i, code in expected.items():
            per_center[str(i)][
                "correct"
                if actual.get(i) == code
                else ("wrong" if i in actual else "missing")
            ] += 1
    counts = dict(Counter(patterns))
    return {
        "expected": expected,
        "patterns": patterns,
        "counts": counts,
        "per_center": per_center,
        "assessed": len(positions) if expected else 0,
        "correct": counts.get("correct", 0),
        "fraction": counts.get("correct", 0) / len(positions)
        if expected and len(positions)
        else None,
    }


def read_ensemble(record, base):
    if record.get("status", "ok") != "ok":
        raise ValueError(record.get("reason", "Generation failed"))
    mol = Chem.MolFromMolBlock(record["molblock"], removeHs=False, sanitize=True)
    if mol is None:
        raise ValueError("Invalid molblock")
    path = base / record.get("npz", record.get("reference_npz", ""))
    with np.load(path, allow_pickle=False) as arrays:
        ref = arrays["pos_ref"].copy()
        gen = arrays["pos_gen"].copy() if "pos_gen" in arrays else None
        if not np.array_equal(
            arrays["atomic_numbers"], [a.GetAtomicNum() for a in mol.GetAtoms()]
        ):
            raise ValueError("Atomic number/order mismatch")
    for positions in (ref, gen):
        if positions is not None and (
            positions.ndim != 3
            or positions.shape[1:] != (mol.GetNumAtoms(), 3)
            or not len(positions)
            or not np.isfinite(positions).all()
        ):
            raise ValueError("Invalid ensemble shape/finiteness")
    return mol, ref, gen, fingerprint(path)


def audit_ensemble(mol, ref, generated):
    result = {"reference_stereo": stereo_audit(mol, ref)}
    if generated is None:
        return result
    if len(generated) < 2 * len(ref):
        raise ValueError("Need 2*N_ref conformers, as in the formal evaluation")
    generated = generated[: 2 * len(ref)]
    stereo = stereo_audit(mol, generated)
    references = [Chem.RemoveHs(conformer_mol(mol, p)) for p in ref]
    mirror_pairs = []
    for index, (pos, pattern) in enumerate(zip(generated, stereo["patterns"])):
        if pattern in ("correct", "unspecified"):
            continue
        mirrored = pos * np.array([-1.0, 1.0, 1.0])
        distances = []
        for positions in (pos, mirrored):
            probe = Chem.RemoveHs(conformer_mol(mol, positions))
            distances.append(
                min(rdMolAlign.GetBestRMS(Chem.Mol(probe), r) for r in references)
            )
        mirror_pairs.append(
            {
                "index": index,
                "pattern": pattern,
                "original_rmsd": distances[0],
                "mirror_rmsd": distances[1],
                "improvement": distances[0] - distances[1],
            }
        )
    return result | {
        "generated_stereo": stereo,
        "wrong_stereo_mirror_pairs": mirror_pairs,
    }


def audit_mapping(item, ref, dataset, group, split):
    from stifflow.data import graph_from_mol
    from stifflow.official_data import graph_signatures

    molecule = dataset.load_molecule(
        group, split, allow_stereo_variation=True, allow_topology_variation=True
    )
    canonical = np.asarray([c["positions"] for c in molecule["conformers"]])
    if canonical.shape != ref.shape:
        raise ValueError("Reference row count differs from frozen official rows")
    if "row_ids" in item and item["row_ids"] != group["row_ids"]:
        raise ValueError("Reference row order differs from frozen official rows")
    matches = np.all(np.isclose(ref, canonical, rtol=0, atol=1e-6), axis=(1, 2))
    stored_graph = Chem.MolFromMolBlock(item["molblock"], removeHs=False)
    if stored_graph is None:
        raise ValueError("Invalid reference conditioning graph")
    actual = graph_signatures(graph_from_mol(stored_graph))
    expected = graph_signatures(molecule["graph"])
    return {
        "matches_current_canonical_coordinates": bool(matches.all()),
        "conditioning_topology_matches": actual[0] == expected[0],
        "conditioning_stereochemistry_matches": actual[1] == expected[1],
        "different_rows": [row for row, ok in zip(group["row_ids"], matches) if not ok],
        "topology_variation_rows": molecule["topology_variation_rows"],
        "stereo_variation_rows": molecule["stereo_variation_rows"],
        "reference_atom_mappings": molecule["reference_atom_mappings"],
        "raw_source": molecule["source"],
    }


def clipping_log(path, optimizer_updates=None):
    pattern = re.compile(
        r"Clipped gradient with value ([\d.eE+-]+) while allowed ([\d.eE+-]+)"
    )
    events = []
    with Path(path).open(errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            match = pattern.search(line)
            if match:
                norm, allowed = map(float, match.groups())
                events.append({"line": line_number, "norm": norm, "allowed": allowed})
    if optimizer_updates is not None and (
        optimizer_updates < 1 or len(events) > optimizer_updates
    ):
        raise ValueError("Invalid optimizer count or log includes multiple runs/ranks")
    return {
        "input": fingerprint(path),
        "events": events,
        "logged_clipping_events": len(events),
        "optimizer_updates": optimizer_updates,
        "logged_event_fraction": len(events) / optimizer_updates
        if optimizer_updates
        else None,
        "caveat": "Rounded log values; absent events do not prove no clipping unless INFO logging and this run's full log are present.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("ensembles", "logs"))
    parser.add_argument("input", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--optimizer-updates", type=int)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--output", type=external_directory, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Existing audit output is protected")
    if args.action == "logs":
        report = clipping_log(args.input, args.optimizer_updates)
    else:
        if not os.getenv("SLURM_JOB_ID"):
            parser.error("Ensemble/mapping audits belong in an SHHPC allocation")
        manifest = json.loads(args.input.read_text())
        selected = select_records(manifest["records"], args.limit)
        dataset, groups = None, {}
        if args.data_manifest:
            from stifflow.official_data import OfficialDataset

            dataset = OfficialDataset.from_manifest(args.data_manifest)
            groups = {g["id"]: g for g in dataset.groups(manifest["split"])}
        records = []
        for item in selected:
            record = {"id": item["id"]}
            try:
                mol, ref, gen, identity = read_ensemble(item, args.input.parent)
                record.update(
                    audit_ensemble(mol, ref, gen), input=identity, status="ok"
                )
                if dataset:
                    record["mapping"] = audit_mapping(
                        item, ref, dataset, groups[item["id"]], manifest["split"]
                    )
            except (ValueError, OSError, KeyError, RuntimeError, TypeError) as error:
                record.update(
                    status="failed", reason=f"{type(error).__name__}: {error}"
                )
            records.append(record)
        stereo = [
            r["generated_stereo"]
            for r in records
            if r["status"] == "ok" and r.get("generated_stereo", {}).get("assessed")
        ]
        assessed = sum(s["assessed"] for s in stereo)
        report = {
            "input_manifest": fingerprint(args.input),
            "selection": "SHA256(id) ascending, before reading outcomes",
            "limit": args.limit,
            "records": records,
            "failed": sum(r["status"] != "ok" for r in records),
            "chirality_macro": float(np.mean([s["fraction"] for s in stereo]))
            if stereo
            else None,
            "chirality_micro": sum(s["correct"] for s in stereo) / assessed
            if assessed
            else None,
            "mapping_data": fingerprint(args.data_manifest)
            if args.data_manifest
            else None,
            "caveat": "All-center inversion alone does not establish a global geometric mirror; inspect paired RMSD. No E/Z metric. Mapping differences are flagged, never silently repaired.",
        }
    report["provenance"] = provenance()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    if report.get("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
