"""Frozen GEOM-QM9 validation panel and offline 250k torsion analysis.

SHHPC runs the real-data commands. No model, label generation or minimization.
Local Taylor checks use core formulation v2 eq:hessian/rotspace/calibration;
regularized curvature is reported separately (eq:stiffness), never as raw H.
"""

import argparse
import copy
import csv
import hashlib
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import TorsionFingerprints as TF
from rdkit.Chem import rdMolDescriptors, rdMolTransforms
from stifflow.cache_io import atomic_json, read_json
from stifflow.freeze_data import VARIANT, require_frozen
from stifflow.geometry import centered, proper_align, rigid_subspaces
from stifflow.official_data import validate_label_manifest, verified_path
from stifflow.preflight import fingerprint, provenance
from stifflow.smoke import external_directory

from .evaluate import conformer_mol
from .research_compare import bootstrap_delta, label_identity
from .research_diagnostics import read_ensemble
from .stiffness_diagnostics import (
    MMFF94,
    load_archive,
    rms_force,
    stereo_matches,
    taylor_metrics,
)

GROUPS = ("zero", "one", "multiple", "conjugated")
SEEDS = (42, 43, 44)
METHODS = ("alternative_a", "control", "precision")
PROTOCOL = {
    "selection": "SHA256('42:' + id), first 3 per group before reading outcomes",
    "heavy_atoms": [7, 9],
    "reference_count": {"zero": [1, 32], "other": [3, 32]},
    "group_precedence": "conjugated first; otherwise Strict rotor count 0/1/>=2",
    "conjugated": "measurable non-ring conjugated single bond",
    "tfd": {"maxDev": "equal", "symmRadius": 2, "ignoreColinearBonds": True},
    "tfd_weights": "equal non-ring torsion weights; rings excluded",
    "angles_degrees": [0, -2, 2, -5, 5, -15, 15],
    "physics": "MMFF94; at most 2 archived rows and 8 generated conformers/run/molecule",
    "bootstrap": {
        "unit": "molecule after equal training-seed mean",
        "replicates": 2000,
        "seed": 42,
    },
    "scope": "exploratory validation only; no advancement or significance gate",
}
ERRORS = (
    ValueError,
    RuntimeError,
    OSError,
    KeyError,
    FloatingPointError,
    np.linalg.LinAlgError,
)


def indexed(records):
    result = {r["id"]: r for r in records}
    if (
        not result
        or len(result) != len(records)
        or any(not isinstance(k, str) or not k for k in result)
    ):
        raise ValueError("Expected unique nonempty molecule IDs")
    return result


def references(path):
    spec = read_json(path)
    if (
        spec.get("schema_version") != 1
        or spec.get("split") != "val"
        or spec.get("selection_frozen") is not True
        or spec.get("variant") != VARIANT
    ):
        raise ValueError("Need frozen qm9-v3-clean-v1 validation references")
    data = read_json(verified_path(spec["data_manifest"]))
    require_frozen(data)
    records = indexed(spec["records"])
    if len(records) != data["variant"]["counts"]["val"]["molecules"]:
        raise ValueError("Need the complete frozen validation split, not monitor/test")
    return spec, records


def heavy_torsions(mol):
    copy_mol = Chem.Mol(mol)
    for atom in copy_mol.GetAtoms():
        atom.SetIntProp("original_index", atom.GetIdx())
    heavy = Chem.RemoveHs(copy_mol)
    mapping = [a.GetIntProp("original_index") for a in heavy.GetAtoms()]
    groups, _ = TF.CalculateTorsionLists(heavy, **PROTOCOL["tfd"])
    groups = [
        (sorted(tuple(t) for t in angles), deviation)
        for angles, deviation in groups
        if heavy.GetBondBetweenAtoms(*angles[0][1:3]).GetBondType()
        == Chem.BondType.SINGLE
    ]
    return heavy, mapping, groups


def describe(record):
    mol = Chem.MolFromMolBlock(record["molblock"], removeHs=False)
    if mol is None:
        raise ValueError("Invalid reference molblock")
    heavy, mapping, torsions = heavy_torsions(mol)
    strict = rdMolDescriptors.CalcNumRotatableBonds(
        heavy, rdMolDescriptors.NumRotatableBondsOptions.Strict
    )
    central = [list(angles[0][1:3]) for angles, _ in torsions]
    restricted = [b for b in central if heavy.GetBondBetweenAtoms(*b).GetIsConjugated()]
    group = (
        "conjugated"
        if restricted
        else ("zero" if strict == 0 else "one" if strict == 1 else "multiple")
    )
    Chem.AssignStereochemistry(heavy, force=True, cleanIt=True)
    n_ref = record["reference_count"]
    if type(n_ref) is not int or n_ref < 1:
        raise ValueError("Invalid reference_count")
    return {
        "id": record["id"],
        "status": "ok",
        "smiles": Chem.MolToSmiles(heavy),
        "group": group,
        "heavy_atoms": heavy.GetNumHeavyAtoms(),
        "strict_rotors": strict,
        "reference_count": n_ref,
        "connected": len(Chem.GetMolFrags(heavy)) == 1,
        "specified_centers": sum(
            a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
            and a.HasProp("_CIPCode")
            for a in heavy.GetAtoms()
        ),
        "rings": heavy.GetRingInfo().NumRings(),
        "nonaromatic_rings": sum(
            not all(heavy.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
            for ring in heavy.GetRingInfo().AtomRings()
        ),
        "torsions": [
            [[mapping[i] for i in t] for t in angles] for angles, _ in torsions
        ],
        "conjugated_bonds": [[mapping[i] for i in b] for b in restricted],
        "bond_types": [
            "-".join(heavy.GetAtomWithIdx(i).GetSymbol() for i in b) for b in central
        ],
        "adjacent_torsions": [
            [i, j]
            for i, j in itertools.combinations(range(len(central)), 2)
            if set(central[i]) & set(central[j])
        ],
    }


def choose_panel(inventory):
    selected, counts = [], {}
    for group in GROUPS:
        lower = 1 if group == "zero" else 3
        candidates = [
            r
            for r in inventory
            if r["status"] == "ok"
            and r["group"] == group
            and r["connected"]
            and 7 <= r["heavy_atoms"] <= 9
            and lower <= r["reference_count"] <= 32
        ]
        candidates.sort(
            key=lambda r: hashlib.sha256(("42:" + r["id"]).encode()).digest()
        )
        counts[group] = {
            "eligible": len(candidates),
            "selected": min(3, len(candidates)),
            "shortfall": max(0, 3 - len(candidates)),
        }
        selected.extend(r["id"] for r in candidates[:3])
    return selected, counts


def select(reference_path, output):
    started = time.monotonic()
    spec, records = references(reference_path)
    inventory = []
    for identity, record in records.items():
        try:
            inventory.append(describe(record))
        except ERRORS as exc:
            inventory.append({"id": identity, "status": "failed", "reason": str(exc)})
    ids, counts = choose_panel(inventory)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "inventory.json", {"records": inventory})
    fields = [
        "id",
        "status",
        "smiles",
        "group",
        "heavy_atoms",
        "strict_rotors",
        "reference_count",
        "connected",
        "specified_centers",
        "rings",
        "nonaromatic_rings",
        "bond_types",
        "torsions",
        "conjugated_bonds",
        "adjacent_torsions",
        "reason",
    ]
    with (output / "inventory.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(inventory)
    panel = {
        "schema_version": 1,
        "kind": "stifflow-torsion-panel",
        "protocol": PROTOCOL,
        "rdkit": rdBase.rdkitVersion,
        "references": fingerprint(reference_path),
        "official_data": spec["data_manifest"]["sha256"],
        "inventory": fingerprint(output / "inventory.json"),
        "ids": ids,
        "groups": counts,
        "inventory_counts": {
            "status": dict(Counter(r["status"] for r in inventory)),
            "groups": dict(
                Counter(r["group"] for r in inventory if r["status"] == "ok")
            ),
            "multi_fragment": sum(
                not r["connected"] for r in inventory if r["status"] == "ok"
            ),
        },
        "provenance": provenance(),
    }
    from rdkit.Chem.Draw import rdMolDraw2D

    for identity in ids:
        mol = Chem.MolFromMolBlock(records[identity]["molblock"], removeHs=False)
        heavy, mapping, torsions = heavy_torsions(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(480, 320)
        for i, original in enumerate(mapping):
            drawer.drawOptions().atomLabels[
                i
            ] = f"{heavy.GetAtomWithIdx(i).GetSymbol()}{original}"
        bonds = [
            heavy.GetBondBetweenAtoms(*angles[0][1:3]).GetIdx()
            for angles, _ in torsions
        ]
        heavy = rdMolDraw2D.PrepareMolForDrawing(heavy, forceCoords=True)
        drawer.DrawMolecule(
            heavy, highlightAtoms=[], highlightBonds=bonds, legend=identity
        )
        drawer.FinishDrawing()
        (output / (hashlib.sha256(identity.encode()).hexdigest() + ".svg")).write_text(
            drawer.GetDrawingText()
        )
    panel["seconds"] = time.monotonic() - started
    atomic_json(output / "panel.json", panel)
    return panel


def load_panel(path):
    panel = read_json(path)
    if (
        panel.get("kind") != "stifflow-torsion-panel"
        or panel.get("schema_version") != 1
        or panel.get("protocol") != PROTOCOL
        or panel.get("rdkit") != rdBase.rdkitVersion
    ):
        raise ValueError("Panel protocol or RDKit changed; do not silently reselect")
    spec, records = references(verified_path(panel["references"]))
    inventory = read_json(verified_path(panel["inventory"]))["records"]
    if (
        set(indexed(inventory)) != set(records)
        or panel["official_data"] != spec["data_manifest"]["sha256"]
    ):
        raise ValueError("Panel inventory/reference identity differs")
    if choose_panel(inventory) != (panel["ids"], panel["groups"]):
        raise ValueError("Panel selection was changed")
    return panel, records, inventory


def load_runs(pairs, panel, ref_records, labels=()):
    runs, common, label_paths = {}, None, {}
    for method, seed, train, val in labels:
        key = (method, int(seed))
        if key in label_paths:
            raise ValueError("Duplicate archived label arguments")
        label_paths[key] = (Path(train), Path(val))
    for manifest_path, evaluation_path in pairs:
        manifest_path, evaluation_path = Path(manifest_path), Path(evaluation_path)
        manifest, report = read_json(manifest_path), read_json(evaluation_path)
        if report["input_manifest"]["sha256"] != fingerprint(manifest_path)[
            "sha256"
        ] or report["run"] != {k: v for k, v in manifest.items() if k != "records"}:
            raise ValueError("Evaluation is not bound to this saved ensemble")
        training, cfg, protocol = (
            manifest["training_data"],
            manifest["config"],
            report["protocol"],
        )
        key = (manifest["method"], training["training_seed"])
        if (
            key[0] not in METHODS
            or key[1] not in SEEDS
            or key in runs
            or manifest["split"] != "val"
            or manifest["baseline_kind"] != "controlled"
            or manifest["training_budget"]["steps"] != 250000
            or manifest["seed"] != 42
            or cfg["steps"] != 50
            or manifest["generation"].get("postprocessing") != "none"
        ):
            raise ValueError(
                "Need distinct controlled val 250k/Euler50/seed42 runs, training seeds 42/43/44"
            )
        if (
            protocol["generated_reference_ratio"] != 2
            or protocol["force_field_optimization"]
            or not protocol["heavy_atoms"]
            or protocol["coordinates"] != "angstrom"
            or protocol["coverage_comparison"] != "strict <"
            or training["official_data"] != panel["official_data"]
            or manifest["input_manifest"]["sha256"] != panel["references"]["sha256"]
        ):
            raise ValueError("Reference, dataset or evaluation protocol differs")
        if cfg["model"]["method"] != key[0] or cfg["model"]["seed"] != key[1]:
            raise ValueError("Model method/seed and training metadata differ")
        generated, evaluated = indexed(manifest["records"]), indexed(report["records"])
        if (
            set(generated) != set(ref_records)
            or set(evaluated) != set(ref_records)
            or manifest["generation"].get("expected") != len(ref_records)
        ):
            raise ValueError(
                "Saved runs must cover all frozen validation IDs, retaining failures"
            )
        signature = copy.deepcopy(
            {"training": training, "config": cfg, "protocol": protocol}
        )
        signature["training"].pop("training_seed")
        signature["config"]["model"].pop("method")
        signature["config"]["model"].pop("seed")
        if label_paths:
            if key not in label_paths:
                raise ValueError("Provide --labels METHOD SEED TRAIN VAL for every run")
            for split, path in zip(("train", "val"), label_paths[key]):
                signature["training"][split] = label_identity(
                    path, training[split], split
                )
        if common is not None and signature != common:
            raise ValueError(
                "Runs differ beyond method/training seed; staged paths require archived --labels evidence"
            )
        common = signature
        runs[key] = {
            "manifest": manifest,
            "evaluation": report,
            "records": generated,
            "metrics": evaluated,
            "base": manifest_path.parent,
            "inputs": {
                "manifest": fingerprint(manifest_path),
                "evaluation": fingerprint(evaluation_path),
            },
        }
    required = {(m, s) for m in METHODS[:2] for s in SEEDS}
    if any(k[0] == "precision" for k in runs):
        required |= {("precision", s) for s in SEEDS}
    if set(runs) != required or label_paths and set(label_paths) != required:
        raise ValueError(
            "Need A/control for all three training seeds; optional precision also requires all three"
        )
    return runs


def geometry_value(record, metric, thresholds, has_stereo):
    if record["status"] != "ok":
        raise ValueError(record.get("reason", "failed geometry record"))
    if metric == "chirality":
        if not has_stereo:
            return None
        if not record["chirality"]["assessed"]:
            raise ValueError("Missing specified-center chirality assessment")
        value = record["chirality"]["fraction"]
    elif metric.startswith("cov_"):
        name = metric[:-3]
        matches = np.flatnonzero(np.isclose(thresholds, 0.5, rtol=0, atol=1e-12))
        if len(matches) != 1:
            raise ValueError("Need a unique COV threshold of 0.5 angstrom")
        value = record[name][int(matches[0])]
    else:
        value = record[metric]
    if value is None or not np.isfinite(value):
        raise ValueError("Nonfinite geometry metric")
    return float(value)


def geometry_summary(runs, inventory):
    summaries = {}
    for candidate in (m for m in ("alternative_a", "precision") if (m, 42) in runs):
        for group in GROUPS:
            members = [
                r for r in inventory if r["status"] == "ok" and r["group"] == group
            ]
            strata = {"all": members}
            for r in members:
                label = f"heavy={r['heavy_atoms']},stereo={bool(r['specified_centers'])},nonaromatic_ring={bool(r['nonaromatic_rings'])}"
                strata.setdefault(label, []).append(r)
            for stratum, rows in strata.items():
                metrics = {}
                for metric in ("mat_p", "mat_r", "cov_p_05", "cov_r_05", "chirality"):
                    deltas, per_seed, failures = [], {s: [] for s in SEEDS}, []
                    applicable = [
                        r
                        for r in rows
                        if metric != "chirality" or r["specified_centers"]
                    ]
                    for r in applicable:
                        values = []
                        for seed in SEEDS:
                            try:
                                pair = []
                                for method in (candidate, "control"):
                                    run = runs[method, seed]
                                    pair.append(
                                        geometry_value(
                                            run["metrics"][r["id"]],
                                            metric,
                                            run["evaluation"]["protocol"]["thresholds"],
                                            r["specified_centers"],
                                        )
                                    )
                                values.append(pair[0] - pair[1])
                                per_seed[seed].append(values[-1])
                            except ERRORS as exc:
                                failures.append(
                                    {"id": r["id"], "seed": seed, "reason": str(exc)}
                                )
                        if len(values) == len(SEEDS):
                            deltas.append(float(np.mean(values)))
                    metrics[metric] = {
                        "expected_molecules": len(applicable),
                        "failures": failures,
                        "delta": (
                            bootstrap_delta(deltas) if deltas and not failures else None
                        ),
                        "per_seed_delta": {
                            str(s): (
                                float(np.mean(v))
                                if v and len(v) == len(applicable)
                                else None
                            )
                            for s, v in per_seed.items()
                        },
                    }
                summaries[f"{candidate}-control/{group}/{stratum}"] = {
                    "molecules": len(rows),
                    "metrics": metrics,
                }
    return summaries


def torsion_angles(mol, positions):
    heavy, mapping, groups = heavy_torsions(mol)
    angles = []
    for point in positions:
        values = TF.CalculateTorsionAngles(
            conformer_mol(heavy, point[mapping]), groups, []
        )
        if any(not np.isfinite(a).all() for a, _ in values):
            raise ValueError("Undefined/collinear torsion angle")
        angles.append(values)
    return angles


def torsion_metrics(ref_angles, gen_angles):
    if not ref_angles[0]:
        return {"status": "not_applicable", "reason": "no measurable non-ring torsion"}
    distances = np.array(
        [[TF.CalculateTFD(a, b) for b in gen_angles] for a in ref_angles]
    )
    if not np.isfinite(distances).all():
        raise ValueError("Nonfinite torsion distance")
    return {
        "status": "ok",
        "mat_r": float(distances.min(axis=1).mean()),
        "mat_p": float(distances.min(axis=0).mean()),
    }


def same_graph(left, right):
    from stifflow.features import graph_from_mol

    a, b = graph_from_mol(left), graph_from_mol(right)
    edges = lambda g: sorted(
        (tuple(e), tuple(v)) for e, v in zip(g["bonds"], g["bond_attr"])
    )
    return all(
        np.array_equal(a[k], b[k]) for k in ("atomic_numbers", "node_attr")
    ) and edges(a) == edges(b)


def read_saved(run, identity, ref_record, reference_base):
    record = run["records"][identity]
    mol, ref, gen, saved = read_ensemble(record, run["base"])
    expected = run["metrics"][identity]
    if (
        expected["status"] != "ok"
        or expected["input"]["sha256"] != saved["sha256"]
        or record.get("npz_fingerprint", saved)["sha256"] != saved["sha256"]
    ):
        raise ValueError(
            "Saved coordinates changed since geometry evaluation or evaluation failed"
        )
    ref_mol, frozen, _, ref_file = read_ensemble(ref_record, reference_base)
    if (
        ref_record["cache"]["sha256"] != ref_file["sha256"]
        or len(frozen) != ref_record["reference_count"]
        or not same_graph(mol, ref_mol)
        or not np.array_equal(ref, frozen)
    ):
        raise ValueError(
            "Saved reference graph, atom order, coordinates or count changed"
        )
    if gen is None or len(gen) < 2 * len(ref):
        raise ValueError("Need 2*N_ref saved generated conformers")
    return mol, ref, gen[: 2 * len(ref)], saved


def displaced(mol, origin, quartet, degrees):
    candidate = conformer_mol(mol, origin)
    conf = candidate.GetConformer()
    angle = rdMolTransforms.GetDihedralDeg(conf, *quartet)
    rdMolTransforms.SetDihedralDeg(conf, *quartet, angle + degrees)
    return proper_align(conf.GetPositions(), origin)


def local_checks(mol, ref, label_record, source_config, emit):
    label_mol, source, arrays, inputs = load_archive(label_record, source_config)
    if not same_graph(mol, label_mol) or not all(
        any(np.allclose(x, r, rtol=0, atol=1e-7) for r in ref)
        for x in arrays["positions"]
    ):
        raise ValueError(
            "Archived label graph/target pairing differs from saved references"
        )
    receipt = read_json(verified_path(label_record["audit_receipt"]))["record"]
    torsions = describe(
        {
            "id": label_record["id"],
            "molblock": Chem.MolToMolBlock(mol),
            "reference_count": len(ref),
        }
    )["torsions"]
    if not torsions:
        return {
            "status": "not_applicable",
            "reason": "no measurable non-ring torsion",
            "inputs": inputs,
        }
    checks, failures = [], []
    for row in np.argsort(arrays["row_ids"])[:2]:
        origin, H = centered(arrays["reference"][row]), arrays["H_raw"][row]
        potential = MMFF94(conformer_mol(mol, origin))
        context = {
            "kind": "local",
            "id": label_record["id"],
            "row_id": int(arrays["row_ids"][row]),
        }
        value0 = query(potential, origin, emit, context | {"anchor": True})
        target = centered(arrays["positions"][row])
        target_value = query(
            potential, target, emit, context | {"original_target": True}
        )
        Qrot, Qvib = rigid_subspaces(origin, source.Q)
        values, vectors = np.linalg.eigh(Qvib.T @ H @ Qvib)
        if not np.allclose(values, arrays["spectrum"][row], rtol=1e-7, atol=1e-7):
            raise ValueError("Raw vibrational spectrum differs from archived spectrum")
        audit = receipt["labels"][row]
        settings = audit["settings"]
        kmin = settings["kappa_min"]
        if (
            not np.isfinite(kmin)
            or kmin <= 0
            or np.any(values < -settings["negative_tolerance"])
        ):
            raise ValueError("Invalid stable-reference spectrum or soft-mode floor")
        floor_mask = arrays["floored_mask"][row]
        if floor_mask.dtype != np.dtype(bool) or not np.array_equal(
            floor_mask, arrays["spectrum"][row] < kmin
        ):
            raise ValueError("Archived floored modes differ from raw spectrum/settings")
        for equivalent in torsions:
            quartet = min(tuple(t) for t in equivalent)
            try:
                step = 1e-4
                tangent = (
                    source.Q.T
                    @ (
                        (
                            displaced(mol, origin, quartet, step)
                            - displaced(mol, origin, quartet, -step)
                        )
                        / (2 * np.deg2rad(step))
                    ).ravel()
                )
                tangent -= Qrot @ (Qrot.T @ tangent)
                weights = np.square(vectors.T @ Qvib.T @ tangent)
                if weights.sum() <= 1e-20:
                    raise ValueError("Degenerate internal torsion tangent")
                rows = []
                for angle in PROTOCOL["angles_degrees"]:
                    point = displaced(mol, origin, quartet, angle)
                    val = query(
                        potential,
                        point,
                        emit,
                        context | {"quartet": quartet, "degrees": angle},
                    )
                    rows.append(
                        {
                            "degrees": angle,
                            "stereo_preserved": stereo_matches(mol, point),
                            **taylor_metrics(point, origin, H, source.Q, val, value0),
                        }
                    )
                checks.append(
                    {
                        **context,
                        "quartet": quartet,
                        "target_reference_rms_angstrom": rms_force(target - origin),
                        "reference_force": rms_force(value0[1]),
                        "target_force": rms_force(target_value[1]),
                        "raw_curvature_kcal_mol_rad2": float(values @ weights),
                        "regularized_curvature_kcal_mol_rad2": float(
                            np.maximum(values, kmin) @ weights
                        ),
                        "floored_tangent_fraction": float(
                            weights[arrays["floored_mask"][row]].sum() / weights.sum()
                        ),
                        "audit": audit,
                        "scan": rows,
                    }
                )
            except ERRORS as exc:
                failures.append({**context, "quartet": quartet, "reason": str(exc)})
    return {
        "status": "failed" if failures else "ok",
        "inputs": inputs,
        "checks": checks,
        "failures": failures,
    }


def query(potential, point, emit, context):
    try:
        energy, gradient = potential(point)
        emit(
            {
                **context,
                "status": "ok",
                "positions": point.tolist(),
                "energy_kcal_mol": energy,
                "gradient": gradient.tolist(),
            }
        )
        return energy, gradient
    except ERRORS as exc:
        emit({**context, "status": "failed", "reason": str(exc)})
        raise


def force_audit(mol, ref, gen, emit, context):
    potential, forces, failures = MMFF94(mol), [], []
    references_force = []
    for kind, points, result in (
        ("generated", gen[:8], forces),
        ("reference", ref[:2], references_force),
    ):
        for i, point in enumerate(points):
            try:
                _, grad = query(
                    potential, point, emit, context | {"kind": kind, "conformer": i}
                )
                result.append(rms_force(grad))
            except ERRORS as exc:
                failures.append({"kind": kind, "conformer": i, "reason": str(exc)})
    return {
        "status": "failed" if failures else "ok",
        "force_median": float(np.median(forces)) if forces and not failures else None,
        "generated_forces": forces,
        "reference_forces": references_force,
        "failures": failures,
    }


def plots(output, identity, meta, ensembles):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = hashlib.sha256(identity.encode()).hexdigest()
    angles = {
        label: [a for seed_angles in samples for a in seed_angles]
        for label, samples in ensembles.items()
    }
    colors = {
        "reference": "black",
        "alternative_a": "tab:blue",
        "control": "tab:orange",
        "precision": "tab:red",
    }
    bins = np.linspace(-180, 180, 25)
    if not meta["torsions"] or not angles:
        return
    fig, axes = plt.subplots(
        len(meta["torsions"]),
        1,
        figsize=(7, 2.7 * len(meta["torsions"])),
        squeeze=False,
    )
    for i, ax in enumerate(axes[:, 0]):
        for label, samples in angles.items():
            a = [((v + 180) % 360 - 180) for sample in samples for v in sample[i][0]]
            weights = [
                1 / (len(samples) * len(sample[i][0]))
                for sample in samples
                for _ in sample[i][0]
            ]
            ax.hist(
                a,
                bins=bins,
                weights=weights,
                histtype="step",
                label=label,
                color=colors[label],
            )
        ax.set(
            xlabel="dihedral (degrees)",
            ylabel="fraction",
            title=str(meta["torsions"][i]),
        )
        ax.legend()
    fig.suptitle(identity + " (equal seed pooling for display only)")
    fig.tight_layout()
    fig.savefig(output / f"{prefix}-angles.png")
    plt.close(fig)
    for i, j in meta["adjacent_torsions"]:
        fig, axes = plt.subplots(
            1, len(angles), figsize=(4 * len(angles), 4), squeeze=False
        )
        for ax, (label, samples) in zip(axes[0], angles.items()):
            pairs, weights = [], []
            for sample in samples:
                a, b = sample[i][0], sample[j][0]
                pairs.extend(itertools.product(a, b))
                weights.extend(
                    [1 / (len(samples) * len(a) * len(b))] * (len(a) * len(b))
                )
            data = (np.asarray(pairs) + 180) % 360 - 180
            ax.hist2d(
                data[:, 0], data[:, 1], bins=bins, weights=weights, vmin=0, vmax=1
            )
            ax.set(
                title=label,
                xlabel=f"torsion {i} (degrees)",
                ylabel=f"torsion {j} (degrees)",
            )
        fig.tight_layout()
        fig.savefig(output / f"{prefix}-joint-{i}-{j}.png")
        plt.close(fig)


def paired_panel(records, runs):
    """Keep three-seed molecule deltas and incomplete denominators explicit."""
    result = {}
    for method in (m for m in ("alternative_a", "precision") if (m, 42) in runs):
        for group in GROUPS:
            members = [r for r in records if r["group"] == group]
            metrics = ["tfd_mat_p", "tfd_mat_r"]
            if any("physics" in r["runs"].get(f"{method}/42", {}) for r in members):
                metrics.append("force_median")
            for metric in metrics:
                rows = []
                for item in members:
                    values, failures, not_applicable = {}, [], False
                    for seed in SEEDS:
                        try:
                            pair = []
                            for name in (method, "control"):
                                run = item["runs"][f"{name}/{seed}"]
                                if run["status"] != "ok":
                                    raise ValueError(run["reason"])
                                if metric.startswith("tfd_"):
                                    measurement = run["torsion"]
                                    if measurement["status"] == "not_applicable":
                                        not_applicable = True
                                        break
                                    pair.append(measurement[metric[4:]])
                                else:
                                    measurement = run.get("physics", {})
                                    if measurement.get("status") != "ok":
                                        raise ValueError(
                                            "Physical force unavailable or failed"
                                        )
                                    pair.append(measurement[metric])
                            if len(pair) == 2:
                                values[str(seed)] = pair[0] - pair[1]
                        except ERRORS as exc:
                            failures.append({"seed": seed, "reason": str(exc)})
                    rows.append(
                        {
                            "id": item["id"],
                            "status": (
                                "not_applicable"
                                if not_applicable
                                else "failed" if failures else "ok"
                            ),
                            "per_seed_delta": values,
                            "delta": (
                                float(np.mean(list(values.values())))
                                if len(values) == 3
                                and not failures
                                and not not_applicable
                                else None
                            ),
                            "failures": failures,
                        }
                    )
                applicable = [r for r in rows if r["status"] != "not_applicable"]
                complete = applicable and all(r["status"] == "ok" for r in applicable)
                result[f"{method}-control/{group}/{metric}"] = {
                    "expected_molecules": len(applicable),
                    "molecules": rows,
                    "delta": (
                        bootstrap_delta([r["delta"] for r in applicable])
                        if complete
                        else None
                    ),
                }
    return result


def analyze(
    panel_path, pairs, output, labels=(), val_labels=None, physics=False, limit=None
):
    started = time.monotonic()
    panel, ref_records, inventory = load_panel(panel_path)
    runs = load_runs(pairs, panel, ref_records, labels)
    if limit is not None and limit != 2:
        raise ValueError(
            "Only --limit 2 is supported for acceptance; omit for full panel"
        )
    if physics and val_labels is None:
        raise ValueError(
            "--physics requires the existing canonical --val-labels manifest"
        )
    label_records, label_spec = {}, None
    if val_labels is not None:
        label_spec = read_json(val_labels)
        if (
            label_spec["split"] != "val"
            or label_spec["data_manifest"]["sha256"] != panel["official_data"]
            or label_spec["source_config"]
            != runs["control", 42]["manifest"]["training_data"]["config"]["source"]
        ):
            raise ValueError("Physical labels differ in split, frozen data or source")
        validate_label_manifest(label_spec)
        label_records = indexed(label_spec["records"])
    meta = indexed(inventory)
    selected = panel["ids"]
    if limit:
        # Include a torsion-bearing case in the bounded acceptance when available.
        first = selected[:1]
        second = next((i for i in selected[1:] if meta[i]["torsions"]), None)
        selected = first + ([second] if second is not None else selected[1:2])
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": 1,
        "kind": "stifflow-torsion-analysis",
        "protocol": PROTOCOL,
        "panel": fingerprint(panel_path),
        "acceptance_only": limit is not None,
        "selected_ids": selected,
        "expected_panel_ids": panel["ids"],
        "inputs": {f"{m}/{s}": r["inputs"] for (m, s), r in runs.items()},
        "label_manifest": fingerprint(val_labels) if val_labels else None,
        "geometry": geometry_summary(runs, inventory),
        "records": [],
        "provenance": provenance(),
        "environment": {
            "python": sys.executable,
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "status": "started",
    }
    queries = 0
    try:
        with (output / "queries.jsonl").open("x") as stream:

            def emit(record):
                nonlocal queries
                queries += 1
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()

            reference_base = Path(panel["references"]["path"]).parent
            for identity in selected:
                item = {
                    "id": identity,
                    "group": meta[identity]["group"],
                    "runs": {},
                    "local": {"status": "not_requested"},
                }
                plot_data, frozen_ref = {}, None
                for (method, seed), run in runs.items():
                    key = f"{method}/{seed}"
                    try:
                        mol, ref, gen, saved = read_saved(
                            run, identity, ref_records[identity], reference_base
                        )
                        ref_angles, gen_angles = torsion_angles(
                            mol, ref
                        ), torsion_angles(mol, gen)
                        item["runs"][key] = {
                            "status": "ok",
                            "input": saved,
                            "torsion": torsion_metrics(ref_angles, gen_angles),
                        }
                        plot_data.setdefault(method, []).append(gen_angles)
                        if frozen_ref is None:
                            frozen_ref = mol, ref
                            plot_data["reference"] = [ref_angles]
                        if physics and method in METHODS[:2]:
                            try:
                                item["runs"][key]["physics"] = force_audit(
                                    mol,
                                    ref,
                                    gen,
                                    emit,
                                    {"id": identity, "method": method, "seed": seed},
                                )
                            except ERRORS as exc:
                                item["runs"][key]["physics"] = {
                                    "status": "failed",
                                    "reason": str(exc),
                                }
                    except ERRORS as exc:
                        item["runs"][key] = {"status": "failed", "reason": str(exc)}
                if physics:
                    try:
                        if frozen_ref is None:
                            raise ValueError("No verified saved ensemble")
                        if identity not in label_records:
                            raise ValueError(
                                "No existing full Hessian archive; molecule is retained"
                            )
                        item["local"] = local_checks(
                            *frozen_ref,
                            label_records[identity],
                            label_spec["source_config"],
                            emit,
                        )
                    except ERRORS as exc:
                        item["local"] = {"status": "unavailable", "reason": str(exc)}
                item["plot_omitted_methods"] = [
                    m
                    for m in METHODS
                    if (m, 42) in runs and len(plot_data.get(m, [])) != len(SEEDS)
                ]
                plot_data = {
                    m: samples
                    for m, samples in plot_data.items()
                    if m not in item["plot_omitted_methods"]
                }
                plots(output, identity, meta[identity], plot_data)
                result["records"].append(item)
                atomic_json(output / "report.json", result)
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["paired_panel"] = paired_panel(result["records"], runs)
        result["geometry_complete"] = all(
            r["status"] == "ok" for r in inventory
        ) and all(
            not metric["failures"]
            for group in result["geometry"].values()
            for metric in group["metrics"].values()
        )
        result["panel_complete"] = (
            bool(selected)
            and selected == panel["ids"]
            and not any(g["shortfall"] for g in panel["groups"].values())
        )
        result["all_saved_runs_valid"] = all(
            r["status"] == "ok"
            for item in result["records"]
            for r in item["runs"].values()
        ) and len(result["records"]) == len(selected)
        result["physics_complete"] = (
            physics
            and bool(selected)
            and len(result["records"]) == len(selected)
            and all(
                item["local"]["status"] in ("ok", "not_applicable")
                and all(
                    item["runs"].get(f"{m}/{s}", {}).get("physics", {}).get("status")
                    == "ok"
                    for m in METHODS[:2]
                    for s in SEEDS
                )
                for item in result["records"]
            )
        )
        result.update(
            seconds=time.monotonic() - started, energy_gradient_queries=queries
        )
        atomic_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="task", required=True)
    selection = commands.add_parser(
        "select",
        help="freeze structure-only inventory and panel before opening outcomes",
    )
    selection.add_argument("--references", required=True, type=Path)
    analysis = commands.add_parser(
        "analyze", help="read saved 250k runs; no model execution"
    )
    analysis.add_argument("--panel", required=True, type=Path)
    analysis.add_argument(
        "--run",
        nargs=2,
        action="append",
        required=True,
        metavar=("ENSEMBLE_MANIFEST", "EVALUATION"),
    )
    analysis.add_argument(
        "--labels",
        nargs=4,
        action="append",
        default=[],
        metavar=("METHOD", "SEED", "ARCHIVED_TRAIN", "ARCHIVED_VAL"),
    )
    analysis.add_argument("--val-labels", type=Path)
    analysis.add_argument("--physics", action="store_true")
    analysis.add_argument("--limit", type=int, choices=[2])
    for command in (selection, analysis):
        command.add_argument("--output", required=True, type=external_directory)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh external output directory")
    if args.task == "select":
        report = select(args.references, args.output)
        print(json.dumps(report["groups"], indent=2))
    else:
        report = analyze(
            args.panel,
            args.run,
            args.output,
            args.labels,
            args.val_labels,
            args.physics,
            args.limit,
        )
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "status",
                        "panel_complete",
                        "geometry_complete",
                        "all_saved_runs_valid",
                        "physics_complete",
                        "energy_gradient_queries",
                    )
                },
                indent=2,
            )
        )
        if (
            not report["geometry_complete"]
            or not report["all_saved_runs_valid"]
            or args.physics
            and not report["physics_complete"]
        ):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
