"""SHHPC-only train locality gate and paired-cohort physics audit of saved ensembles."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation
from stifflow.features import graph_from_mol
from stifflow.geometry import centered, proper_align
from stifflow.locality import LOCALITY_PROTOCOL, choose_start
from stifflow.official_data import validate_label_manifest, verified_path
from stifflow.preflight import fingerprint, provenance
from stifflow.research_paths import curvature_schedule, sample_target
from stifflow.smoke import external_directory
from stifflow.source import prepare_source

from .evaluate import conformer_mol
from .research_compare import bootstrap_delta
from .research_diagnostics import cip_labels, read_ensemble, select_records

PHYSICS_PROTOCOL = {
    "molecules": 64,
    "conformers": 8,
    "selection": "SHA256(id), first 64; first min(8,2*N_ref) generated conformers",
    "potential": "MMFF94",
    "force_units": "kcal/mol/angstrom",
    "force_statistic": "sqrt(sum_a ||F_a||^2/N); conformer median then equal molecule mean",
    "relaxation": {"maxIts": 2000, "forceTol": 1e-6, "energyTol": 1e-9},
    "strain": "E_raw-E_relaxed; only converged, stereo-preserving rows; failures counted separately",
    "oracle": {
        "method": "GFN2-xTB",
        "molecules": 16,
        "spin": "closed shell only; explicit --uhf 0",
    },
    "seed": 42,
}
HARTREE_KCAL_MOL = 627.5094740631
BOHR_ANGSTROM = 0.529177210903
PHYSICAL_ERRORS = (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError)


def rms_force(gradient):
    gradient = np.asarray(gradient, dtype=np.float64).reshape(-1, 3)
    if not len(gradient) or not np.isfinite(gradient).all():
        raise ValueError("Nonfinite/empty force")
    scale = float(np.abs(gradient).max())
    if scale == 0:
        return 0.0
    value = scale * float(np.sqrt(np.square(gradient / scale).sum() / len(gradient)))
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite RMS force")
    return value


class MMFF94:
    def __init__(self, mol):
        self.mol = Chem.Mol(mol)
        if not AllChem.MMFFHasAllMoleculeParams(self.mol):
            raise ValueError("MMFF94 parameters unavailable")
        props = AllChem.MMFFGetMoleculeProperties(self.mol, mmffVariant="MMFF94")
        self.ff = AllChem.MMFFGetMoleculeForceField(self.mol, props)
        if self.ff is None:
            raise ValueError("MMFF94 force field unavailable")
        self.queries = 0

    def __call__(self, positions):
        self.queries += 1
        point = tuple(np.asarray(positions, dtype=np.float64).ravel())
        # RDKit caches distance data: energy must be evaluated at this same point.
        energy = float(self.ff.CalcEnergy(point))
        gradient = np.asarray(self.ff.CalcGrad(point)).reshape(-1, 3)
        if not np.isfinite(energy) or not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite MMFF94 energy/gradient")
        return energy, gradient


def stereo_matches(mol, positions):
    expected, actual = cip_labels(mol), cip_labels(mol, positions)
    return all(actual.get(i) == code for i, code in expected.items())


def taylor_metrics(point, origin, hessian, basis, value, origin_value):
    """Actual first derivative, raw reference Hessian; no K-floor or abs spectrum."""
    delta = basis.T @ (point - origin).ravel()
    dg = (basis @ (hessian @ delta)).reshape(-1, 3)
    quadratic = float(0.5 * delta @ hessian @ delta)
    remainder = float(
        value[0] - origin_value[0] - np.vdot(origin_value[1], point - origin)
    )
    actual_dg = value[1] - origin_value[1]
    ef = LOCALITY_PROTOCOL["energy_floor_kcal_mol"]
    ff = LOCALITY_PROTOCOL["force_floor_kcal_mol_angstrom"]
    energy_error = abs(remainder - quadratic) / max(abs(remainder), abs(quadratic), ef)
    force_error = rms_force(actual_dg - dg) / max(
        rms_force(actual_dg), rms_force(dg), ff
    )
    if not np.isfinite([quadratic, remainder, energy_error, force_error]).all():
        raise FloatingPointError("Nonfinite Taylor diagnostic")
    return {
        "energy_relative_error": float(energy_error),
        "force_relative_error": float(force_error),
        "quadratic_kcal_mol": quadratic,
        "actual_remainder_kcal_mol": remainder,
        "displacement_rms_angstrom": rms_force(point - origin),
    }


def load_archive(record, source_config):
    """Read the hash-verified full receipt/archive, never rebuild missing labels."""
    receipt = json.loads(verified_path(record["audit_receipt"]).read_text())
    full = receipt["record"]
    if (
        receipt["attempt"]["status"] != "success"
        or full["id"] != record["id"]
        or full["split"] != record["split"]
        or full["row_ids"] != record["row_ids"]
        or full["audit_partial"]
    ):
        raise ValueError("Full receipt and accepted packed record differ")
    if len(full["labels"]) != len(full["row_ids"]) or any(
        audit["potential"] != "MMFF94"
        or audit["units"]
        != {
            "length": "angstrom",
            "energy": "kcal/mol",
            "stiffness": "kcal/mol/angstrom^2",
        }
        for audit in full["labels"]
    ):
        raise ValueError("Raw Hessian potential/units differ from the diagnostic")
    archive = verified_path(full["cache"])
    mol = Chem.MolFromMolBlock(full["molblock"], removeHs=False)
    if mol is None:
        raise ValueError("Invalid label molblock")
    source = prepare_source(graph_from_mol(mol), source_config)
    with np.load(archive, allow_pickle=False) as saved:
        arrays = {
            k: saved[k]
            for k in (
                "positions",
                "reference",
                "H_raw",
                "alt_L_K",
                "floored_mask",
                "spectrum",
                "row_ids",
            )
        }
        for key, expected in [("source_S", source.S), ("source_C0", source.C0)]:
            if not np.allclose(saved[key], expected, rtol=1e-12, atol=1e-12):
                raise ValueError("Archived source differs from configured graph source")
    n, d, c = mol.GetNumAtoms(), len(source.S), len(full["row_ids"])
    expected_shapes = {
        "positions": (c, n, 3),
        "reference": (c, n, 3),
        "H_raw": (c, d, d),
        "alt_L_K": (c, d, d),
    }
    for key, shape in expected_shapes.items():
        if (
            arrays[key].shape != shape
            or arrays[key].dtype != np.float64
            or not np.isfinite(arrays[key]).all()
        ):
            raise ValueError(f"Invalid full archive {key}")
    if not np.array_equal(arrays["row_ids"], full["row_ids"]):
        raise ValueError("Archived row/target pairing differs")
    return (
        mol,
        source,
        arrays,
        {"receipt": record["audit_receipt"], "archive": full["cache"]},
    )


def locality_molecule(identity, mol, source, arrays, row_index, emit):
    row_id = int(arrays["row_ids"][row_index])
    x1, reference = centered(arrays["positions"][row_index]), centered(
        arrays["reference"][row_index]
    )
    H, L = arrays["H_raw"][row_index], arrays["alt_L_K"][row_index]
    seed = int.from_bytes(
        hashlib.sha256(f"{identity}:{row_id}".encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng([42, seed])
    noises = rng.standard_normal((LOCALITY_PROTOCOL["noise_draws"], len(source.S)))
    R = Rotation.random(random_state=np.random.default_rng([42, seed, 1])).as_matrix()
    W = source.Q.T @ np.kron(np.eye(len(x1)), R) @ source.Q
    null_H = W @ H @ W.T
    potential = MMFF94(conformer_mol(mol, x1))
    target_value, reference_value = potential(x1), potential(reference)
    transfer = taylor_metrics(x1, reference, H, source.Q, target_value, reference_value)
    nodes, weights = np.polynomial.legendre.leggauss(
        LOCALITY_PROTOCOL["quadrature_nodes"]
    )
    nodes, weights = (nodes + 1) / 2, weights / 2
    candidates = []
    for start in LOCALITY_PROTOCOL["starts"]:
        valid_weight, total_weight, difference_weight, failed = 0.0, 0.0, 0.0, 0
        for noise_index, z in enumerate(noises):
            signal = float(np.square((L - source.S) @ z).sum())
            for node, weight in zip(nodes, weights):
                t = start + (1 - start) * node
                _, gp = curvature_schedule(t, start)
                w = float((1 - start) * weight * gp * gp * signal)
                total_weight += w
                item = {
                    "id": identity,
                    "row_id": row_id,
                    "noise": noise_index,
                    "stiffness_start": start,
                    "t": float(t),
                    "signal_weight": w,
                }
                try:
                    point = sample_target(
                        x1, t, z, source, {"L_K": L}, "a_delayed", stiffness_start=start
                    )[0]
                    point = proper_align(point, x1)
                    value = potential(point)
                    true = taylor_metrics(point, x1, H, source.Q, value, target_value)
                    null = taylor_metrics(
                        point, x1, null_H, source.Q, value, target_value
                    )
                    relaxed = taylor_metrics(
                        point, reference, H, source.Q, value, reference_value
                    )
                    valid = (
                        stereo_matches(mol, point)
                        and true["energy_relative_error"]
                        <= LOCALITY_PROTOCOL["relative_error_limit"]
                        and true["force_relative_error"]
                        <= LOCALITY_PROTOCOL["relative_error_limit"]
                    )
                    valid_weight += w * valid
                    difference_weight += w * (
                        true["force_relative_error"] - null["force_relative_error"]
                    )
                    item.update(
                        status="ok",
                        valid=bool(valid),
                        target=true,
                        reference=relaxed,
                        rotation_null=null,
                    )
                except PHYSICAL_ERRORS as exc:
                    failed += 1
                    item.update(
                        status="failed",
                        valid=False,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                emit(item)
        candidates.append(
            {
                "stiffness_start": start,
                "valid_signal_fraction": (
                    float(valid_weight / total_weight) if total_weight else 0.0
                ),
                "force_prediction_difference": (
                    float(difference_weight / total_weight)
                    if total_weight and not failed
                    else None
                ),
                "failed_queries": failed,
                "signal_integral": total_weight / len(noises),
            }
        )
    return {
        "id": identity,
        "row_id": row_id,
        "status": "ok",
        "candidates": candidates,
        "reference_transfer": transfer,
        "reference_rmsd_angstrom": rms_force(reference - x1),
        "floored_mode_fraction": float(np.mean(arrays["floored_mask"][row_index])),
        "raw_spectrum_min": float(np.min(arrays["spectrum"][row_index])),
        "energy_gradient_queries": potential.queries,
    }


def run_locality(args, emit):
    import yaml

    manifest = json.loads(args.train_labels.read_text())
    if manifest["split"] != "train" or manifest["schema_version"] != 3:
        raise ValueError("Locality fitting requires completed packed train labels")
    validate_label_manifest(manifest)
    config = yaml.safe_load(args.config.read_text())
    if config["source"] != manifest["source_config"]:
        raise ValueError("Config and label source differ")
    selected = select_records(manifest["records"], LOCALITY_PROTOCOL["molecules"])
    if len(selected) != LOCALITY_PROTOCOL["molecules"]:
        raise ValueError("Need all 128 preregistered training molecules")
    records = []
    for record in selected:
        try:
            mol, source, arrays, inputs = load_archive(record, config["source"])
            i = min(
                range(len(arrays["row_ids"])),
                key=lambda j: hashlib.sha256(
                    f'{record["id"]}:{arrays["row_ids"][j]}'.encode()
                ).digest(),
            )
            result = locality_molecule(record["id"], mol, source, arrays, i, emit)
            result["inputs"] = inputs
        except (*PHYSICAL_ERRORS, OSError, KeyError) as exc:
            result = {
                "id": record["id"],
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        records.append(result)
    candidates = []
    for i, start in enumerate(LOCALITY_PROTOCOL["starts"]):
        rows = [r["candidates"][i] for r in records if r["status"] == "ok"]
        differences = [r["force_prediction_difference"] for r in rows]
        complete = len(rows) == len(records) and all(v is not None for v in differences)
        candidates.append(
            {
                "stiffness_start": start,
                "valid_signal_fraction": sum(r["valid_signal_fraction"] for r in rows)
                / len(records),
                "failed_queries": sum(r["failed_queries"] for r in rows)
                + len(records)
                - len(rows),
                "force_prediction_delta": (
                    bootstrap_delta(differences) if complete else None
                ),
            }
        )
    return {
        "schema_version": 1,
        "kind": "stifflow-locality",
        "split": "train",
        "status": "completed",
        "protocol": LOCALITY_PROTOCOL,
        "train_manifest": fingerprint(args.train_labels),
        "config": fingerprint(args.config),
        "source_config": config["source"],
        "candidates": candidates,
        "selected_start": choose_start(candidates),
        "records": records,
        "energy_gradient_queries": sum(
            r.get("energy_gradient_queries", 0) for r in records
        ),
        "stiffness_preprocessing": "Reused cached labels; historical preprocessing cost not included in query count",
    }


def relaxation(mol, point, energy):
    potential = MMFF94(conformer_mol(mol, point))
    started = time.perf_counter()
    status = int(potential.ff.Minimize(**PHYSICS_PROTOCOL["relaxation"]))
    relaxed = potential.mol.GetConformer().GetPositions()
    e, grad = potential(relaxed)
    changed = cip_labels(mol, point) != cip_labels(mol, relaxed)
    strain = float(energy - e)
    return {
        "status": "ok" if status == 0 and strain >= -1e-6 else "not_converged",
        "minimize_status": status,
        "stereo_changed": changed,
        "strain_kcal_mol": strain,
        "valid_strain": status == 0 and not changed and strain >= -1e-6,
        "relaxed_force_rms": rms_force(grad),
        "displacement_rms_angstrom": rms_force(
            proper_align(relaxed, point) - centered(point)
        ),
        "energy_gradient_queries": potential.queries,
        "seconds": time.perf_counter() - started,
    }


def parse_xtb_gradient(text, atoms):
    """xTB --grad writes Turbomole $grad: energy in Eh, gradient in Eh/bohr."""
    match = re.search(r"SCF energy\s*=\s*([-+\d.eEdD]+)", text)
    vectors = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 3:
            try:
                vectors.append(
                    [float(f.replace("D", "E").replace("d", "e")) for f in fields]
                )
            except ValueError:
                pass
    if match is None or len(vectors) != atoms:
        raise ValueError("Unexpected xTB gradient format/atom count")
    energy = float(match[1].replace("D", "E").replace("d", "e")) * HARTREE_KCAL_MOL
    gradient = np.asarray(vectors) * HARTREE_KCAL_MOL / BOHR_ANGSTROM
    if not np.isfinite(energy) or not np.isfinite(gradient).all():
        raise ValueError("Nonfinite xTB gradient")
    return energy, gradient


def xtb_value(mol, point, executable, directory):
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    electrons = sum(a.GetAtomicNum() for a in mol.GetAtoms()) - charge
    if electrons % 2 or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError(
            "Oracle requires explicit closed-shell graph; open-shell state is unresolved"
        )
    with tempfile.TemporaryDirectory(prefix="xtb-", dir=directory) as name:
        path = Path(name)
        xyz = [str(len(point)), "fixed raw geometry; charge from graph; uhf=0"]
        xyz += [
            f"{a.GetSymbol()} {x:.16g} {y:.16g} {z:.16g}"
            for a, (x, y, z) in zip(mol.GetAtoms(), point)
        ]
        (path / "input.xyz").write_text("\n".join(xyz) + "\n")
        completed = subprocess.run(
            [
                executable,
                "input.xyz",
                "--gfn",
                "2",
                "--grad",
                "--chrg",
                str(charge),
                "--uhf",
                "0",
            ],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        if completed.returncode or not (path / "gradient").is_file():
            raise ValueError(
                f"xTB gradient failed: {(completed.stderr+completed.stdout)[-1000:]}"
            )
        return parse_xtb_gradient((path / "gradient").read_text(), len(point))


def audit_molecule(
    identity, mol, ref, generated, emit, label_data=None, xtb=None, directory=None
):
    expected = min(PHYSICS_PROTOCOL["conformers"], 2 * len(ref))
    if generated is None or len(generated) < 2 * len(ref):
        raise ValueError("Need the saved 2*N_ref generated conformers")
    potential = MMFF94(conformer_mol(mol, ref[0]))
    rows, reference_rows = [], []
    for kind, points in [("reference", ref[:8]), ("generated", generated[:expected])]:
        for i, point in enumerate(points):
            item = {"id": identity, "kind": kind, "conformer": i}
            try:
                e, grad = potential(point)
                item.update(
                    status="ok",
                    energy_kcal_mol=e,
                    force_rms=rms_force(grad),
                    stereo_ok=stereo_matches(mol, point),
                )
                try:
                    item["relaxation"] = relaxation(mol, point, e)
                except PHYSICAL_ERRORS as exc:
                    item["relaxation"] = {
                        "status": "failed",
                        "reason": str(exc),
                        "valid_strain": False,
                    }
                if kind == "generated" and label_data is not None:
                    try:
                        source, arrays = label_data
                        best = min(
                            range(len(arrays["positions"])),
                            key=lambda j: rms_force(
                                proper_align(point, arrays["positions"][j])
                                - centered(arrays["positions"][j])
                            ),
                        )
                        origin = centered(arrays["positions"][best])
                        aligned = proper_align(point, origin)
                        item["taylor"] = taylor_metrics(
                            aligned,
                            origin,
                            arrays["H_raw"][best],
                            source.Q,
                            potential(aligned),
                            potential(origin),
                        )
                        item["taylor"]["row_id"] = int(arrays["row_ids"][best])
                    except PHYSICAL_ERRORS as exc:
                        item["taylor_error"] = str(exc)
            except PHYSICAL_ERRORS as exc:
                item.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            if xtb is not None:
                oracle_started = time.perf_counter()
                try:
                    e, grad = xtb_value(mol, point, xtb, directory)
                    item["oracle"] = {
                        "status": "ok",
                        "energy_kcal_mol": e,
                        "force_rms": rms_force(grad),
                    }
                except (*PHYSICAL_ERRORS, OSError, subprocess.TimeoutExpired) as exc:
                    item["oracle"] = {"status": "failed", "reason": str(exc)}
                item["oracle"]["seconds"] = time.perf_counter() - oracle_started
            (rows if kind == "generated" else reference_rows).append(item)
            emit(item)
    complete = len(rows) == expected and all(r["status"] == "ok" for r in rows)
    oracle_complete = xtb is not None and all(
        r["oracle"]["status"] == "ok" for r in rows
    )
    relax = [r["relaxation"] for r in rows if "relaxation" in r]
    valid_strain = [r["strain_kcal_mol"] for r in relax if r["valid_strain"]]
    return {
        "id": identity,
        "status": "ok" if complete else "failed",
        "expected_conformers": expected,
        "failed_conformers": sum(r["status"] != "ok" for r in rows),
        "force_median": (
            float(np.median([r["force_rms"] for r in rows])) if complete else None
        ),
        "reference_force_median": (
            float(np.median([r["force_rms"] for r in reference_rows]))
            if all(r["status"] == "ok" for r in reference_rows)
            else None
        ),
        "strain_median_stereo_preserving": (
            float(np.median(valid_strain)) if valid_strain else None
        ),
        "relaxation_failures": sum(r["status"] != "ok" for r in relax),
        "relaxation_stereo_changes": sum(r.get("stereo_changed", False) for r in relax),
        "oracle_force_median": (
            float(np.median([r["oracle"]["force_rms"] for r in rows]))
            if oracle_complete
            else None
        ),
        "oracle_attempted": xtb is not None,
        "oracle_failed_conformers": sum(
            r.get("oracle", {}).get("status") == "failed" for r in rows
        ),
        "taylor_covered_conformers": sum("taylor" in r for r in rows),
        "energy_gradient_queries": potential.queries
        + sum(
            r.get("relaxation", {}).get("energy_gradient_queries", 0)
            for r in rows + reference_rows
        ),
        "oracle_queries": sum("oracle" in r for r in rows + reference_rows),
        "oracle_seconds": sum(
            r.get("oracle", {}).get("seconds", 0) for r in rows + reference_rows
        ),
        "minimization_calls": sum("relaxation" in r for r in rows + reference_rows),
    }


def run_physics(args, emit):
    manifest = json.loads(args.manifest.read_text())
    evaluation = json.loads(args.evaluation.read_text())
    if (
        manifest["split"] != "val"
        or manifest["baseline_kind"] != "controlled"
        or manifest["training_budget"]["steps"] != args.budget
        or manifest["seed"] != 42
        or manifest["config"]["steps"] != 50
        or evaluation["input_manifest"]["sha256"]
        != fingerprint(args.manifest)["sha256"]
        or evaluation["run"] != {k: v for k, v in manifest.items() if k != "records"}
        or evaluation["protocol"]["generated_reference_ratio"] != 2
        or evaluation["protocol"]["force_field_optimization"]
    ):
        raise ValueError(
            f"Need matching raw validation-only {args.budget}-update/Euler50/seed42 "
            "evaluation and ensembles"
        )
    selected = select_records(manifest["records"], PHYSICS_PROTOCOL["molecules"])
    evaluated = {r["id"]: r for r in evaluation["records"]}
    if len(evaluated) != len(evaluation["records"]) or set(evaluated) != {
        r["id"] for r in manifest["records"]
    }:
        raise ValueError("Geometry evaluation and saved-ensemble molecule IDs differ")
    if (
        len(manifest["records"]) != 512
        or len(selected) != PHYSICS_PROTOCOL["molecules"]
    ):
        raise ValueError(
            "Physics cohort must come from the complete fixed 512-molecule monitor"
        )
    labels = json.loads(args.val_labels.read_text())
    if labels["split"] != "val":
        raise ValueError("Physical audit labels must belong to validation")
    validate_label_manifest(labels)
    label_records = {r["id"]: r for r in labels["records"]}
    oracle_ids = {r["id"] for r in selected[: PHYSICS_PROTOCOL["oracle"]["molecules"]]}
    executable, oracle_provenance = None, {"status": "not_requested"}
    if args.xtb:
        executable = shutil.which(args.xtb)
        if executable is None:
            raise ValueError(
                "Requested xTB executable is unavailable; no installation is attempted"
            )
        version = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        oracle_provenance = {
            "status": "requested",
            "binary": fingerprint(executable),
            "version": version.stdout + version.stderr,
        }
    records = []
    for record in selected:
        label_error, label_data, label_inputs = None, None, None
        try:
            mol, ref, generated, input_file = read_ensemble(
                record, args.manifest.parent
            )
            if (
                evaluated[record["id"]].get("input", {}).get("sha256")
                != input_file["sha256"]
                or record.get("npz_fingerprint", {}).get("sha256", input_file["sha256"])
                != input_file["sha256"]
            ):
                raise ValueError("Saved ensemble changed since geometry evaluation")
            try:
                label_mol, source, arrays, label_inputs = load_archive(
                    label_records[record["id"]], labels["source_config"]
                )
                left_graph, right_graph = graph_from_mol(label_mol), graph_from_mol(mol)

                def indexed_edges(graph):
                    return sorted(
                        (tuple(edge), tuple(attr))
                        for edge, attr in zip(graph["bonds"], graph["bond_attr"])
                    )

                if (
                    Chem.MolToSmiles(label_mol) != Chem.MolToSmiles(mol)
                    or not np.array_equal(
                        left_graph["atomic_numbers"], right_graph["atomic_numbers"]
                    )
                    or not np.array_equal(
                        left_graph["node_attr"], right_graph["node_attr"]
                    )
                    or indexed_edges(left_graph) != indexed_edges(right_graph)
                ):
                    raise ValueError("Label and generated graph differ")
                if not all(
                    any(np.allclose(p, r, atol=1e-7, rtol=0) for r in ref)
                    for p in arrays["positions"]
                ):
                    raise ValueError(
                        "Cached targets and ensemble references differ in coordinates/order"
                    )
                label_data = source, arrays
            except (*PHYSICAL_ERRORS, OSError, KeyError) as exc:
                label_error = f"{type(exc).__name__}: {exc}"
            result = audit_molecule(
                record["id"],
                mol,
                ref,
                generated,
                emit,
                label_data,
                executable if record["id"] in oracle_ids else None,
                args.output,
            )
            result.update(
                input=input_file, label_inputs=label_inputs, label_error=label_error
            )
        except (*PHYSICAL_ERRORS, OSError, KeyError) as exc:
            result = {
                "id": record["id"],
                "status": "failed",
                "force_median": None,
                "oracle_force_median": None,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        records.append(result)
    complete = all(r["status"] == "ok" for r in records)
    return {
        "schema_version": 1,
        "kind": "stifflow-physics",
        "status": "completed",
        "split": "val",
        "method": manifest["method"],
        "protocol": PHYSICS_PROTOCOL,
        "ensemble_manifest": fingerprint(args.manifest),
        "evaluation": fingerprint(args.evaluation),
        "val_manifest": fingerprint(args.val_labels),
        "run": evaluation["run"],
        "oracle": oracle_provenance,
        "oracle_ids": sorted(oracle_ids),
        "records": records,
        "aggregate": {
            "expected_molecules": len(selected),
            "failed_molecules": sum(r["status"] != "ok" for r in records),
            "force_mean_molecule_median": (
                float(np.mean([r["force_median"] for r in records]))
                if complete
                else None
            ),
        },
        "energy_gradient_queries": sum(
            r.get("energy_gradient_queries", 0) for r in records
        ),
        "minimization_calls": sum(r.get("minimization_calls", 0) for r in records),
        "oracle_queries": sum(r.get("oracle_queries", 0) for r in records),
        "oracle_seconds": sum(r.get("oracle_seconds", 0) for r in records),
        "cost_caveat": "Query count covers direct MMFF94 energy/gradient calls; internal minimizer calls are not exposed. Oracle calls and wall time are separate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="task", required=True)
    local = commands.add_parser("locality")
    local.add_argument("--train-labels", type=Path, required=True)
    local.add_argument("--config", type=Path, required=True)
    physics = commands.add_parser("physics")
    physics.add_argument("manifest", type=Path)
    physics.add_argument("--evaluation", type=Path, required=True)
    physics.add_argument("--val-labels", type=Path, required=True)
    physics.add_argument(
        "--budget",
        type=int,
        default=25000,
        choices=(25000, 250000),
        help="Training budget recorded in the ensemble manifest (default 25000)",
    )
    physics.add_argument(
        "--xtb",
        help="Existing xTB executable; closed-shell single-point gradients only",
    )
    for command in (local, physics):
        command.add_argument("--output", type=external_directory, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error(
            "Physical diagnostics and saved-ensemble processing belong in an SHHPC allocation"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with (args.output / "records.jsonl").open("x") as stream:

        def emit(item):
            stream.write(json.dumps(item, allow_nan=False) + "\n")

        report = (
            run_locality(args, emit)
            if args.task == "locality"
            else run_physics(args, emit)
        )
    report.update(
        provenance=provenance(),
        environment={
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "python": sys.executable,
            "python_version": sys.version,
        },
        allocation={
            "job_id": os.getenv("SLURM_JOB_ID"),
            "cpus": os.getenv("SLURM_CPUS_PER_TASK"),
            "threads": {
                key: os.getenv(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        seconds=time.perf_counter() - started,
        detail_records=fingerprint(args.output / "records.jsonl"),
    )
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "report": str(args.output / "report.json"),
                "selected_start": report.get("selected_start"),
                "failed_molecules": sum(r["status"] != "ok" for r in report["records"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
