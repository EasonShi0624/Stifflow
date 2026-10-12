"""MMFF94 curvature labels: core v2 eq:hessian/rotspace/stiffness/calibration.

Units: Angstrom and kcal/mol, with non-mass-weighted Cartesian coordinates.
Relaxation never replaces the stored GEOM target. All matrix arithmetic is float64.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .geometry import centered, centered_basis, proper_align, rigid_subspaces

EPSILON_E = 0.00198720425864083 * 298.15  # R T, kcal/mol at 298.15 K
DEFAULTS = {
    "epsilon_E": EPSILON_E,
    "kappa_min": EPSILON_E,
    "kappa_rot": EPSILON_E,
    "difference_step": 1e-4,
    "hessian_relative_tolerance": 1e-4,
    "negative_tolerance": 1e-3,
    "gradient_tolerance": 1e-4,
    "max_iterations": 2000,
}


def _stereo_from_coordinates(mol):
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    Chem.AssignStereochemistryFrom3D(copy, replaceExistingTags=True)
    Chem.AssignStereochemistry(copy, force=True, cleanIt=True)
    return (
        tuple(
            a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else None
            for a in copy.GetAtoms()
        ),
        tuple(int(b.GetStereo()) for b in copy.GetBonds()),
    )


def estimate_stiffness(mol, config=None):
    """Return unchanged target, aligned reference, shared K and physical audit.

    A failed stable-reference assumption raises an explicit error; no UFF fallback,
    absolute-value eigenvalues, upper clipping or target replacement is allowed.
    """
    started = time.perf_counter()
    settings = dict(DEFAULTS)
    if config:
        if set(config) - settings.keys():
            raise ValueError("Unknown stiffness configuration")
        settings.update(config)
    if not np.isfinite(list(settings.values())).all() or min(settings.values()) <= 0:
        raise ValueError("Stiffness parameters must be finite and positive")
    if not isinstance(settings["max_iterations"], int):
        raise TypeError("max_iterations must be an integer")
    target = np.array(mol.GetConformer().GetPositions(), dtype=np.float64)
    if len(Chem.GetMolFrags(mol)) != 1 or not np.isfinite(target).all():
        raise ValueError("Need connected molecule and finite target coordinates")
    Q = centered_basis(len(target))
    reference_mol = Chem.Mol(mol)
    original_stereo = _stereo_from_coordinates(reference_mol)
    if not AllChem.MMFFHasAllMoleculeParams(reference_mol):
        raise ValueError("MMFF94 parameters missing; no force-field fallback")
    props = AllChem.MMFFGetMoleculeProperties(reference_mol, mmffVariant="MMFF94")
    ff = AllChem.MMFFGetMoleculeForceField(reference_mol, props)
    status = ff.Minimize(
        maxIts=settings["max_iterations"], forceTol=1e-8, energyTol=1e-12
    )
    if status != 0:
        raise ValueError(f"MMFF94 relaxation did not converge (status={status})")
    reference = proper_align(reference_mol.GetConformer().GetPositions(), target)
    for i, xyz in enumerate(reference):
        reference_mol.GetConformer().SetAtomPosition(i, xyz)
    if _stereo_from_coordinates(reference_mol) != original_stereo:
        raise ValueError("Stereochemistry changed during MMFF94 relaxation")
    ff = AllChem.MMFFGetMoleculeForceField(reference_mol, props)
    x = reference.ravel()

    def gradient(flat):
        # RDKit caches pair distances: CalcGrad(pos) alone can reuse stale values.
        # Recompute energy at the SAME coordinates before every analytic gradient.
        ff.CalcEnergy(tuple(flat))
        return np.asarray(ff.CalcGrad(tuple(flat)), dtype=np.float64)

    grad = gradient(x)
    gradient_norm = float(np.linalg.norm(Q.T @ grad))
    if gradient_norm > settings["gradient_tolerance"]:
        raise ValueError(
            f"Nonstationary relaxed reference: reduced gradient norm {gradient_norm:.6g}"
        )

    def hessian(step):
        return np.column_stack(
            [
                Q.T
                @ (gradient(x + step * direction) - gradient(x - step * direction))
                / (2 * step)
                for direction in Q.T
            ]
        )

    coarse = hessian(settings["difference_step"])
    fine = hessian(settings["difference_step"] / 2)
    if not np.isfinite(coarse).all() or not np.isfinite(fine).all():
        raise ValueError("Nonfinite finite-difference Hessian")
    scale = max(float(np.linalg.norm(fine)), np.finfo(np.float64).tiny)
    half_step_error = float(np.linalg.norm(coarse - fine) / scale)
    asymmetry = float(np.linalg.norm(fine - fine.T) / scale)
    if max(half_step_error, asymmetry) > settings["hessian_relative_tolerance"]:
        raise ValueError(
            f"Unstable Hessian: half-step={half_step_error:.3g}, asymmetry={asymmetry:.3g}"
        )
    H = (fine + fine.T) / 2
    Qrot, Qvib = rigid_subspaces(reference, Q)
    spectrum, vectors = np.linalg.eigh(Qvib.T @ H @ Qvib)
    if len(spectrum) and spectrum[0] < -settings["negative_tolerance"]:
        raise ValueError(
            f"Significant negative vibrational curvature: {spectrum[0]:.6g} kcal/mol/angstrom^2"
        )
    modes = Qvib @ vectors
    floored = spectrum < settings["kappa_min"]
    K = (modes * np.maximum(spectrum, settings["kappa_min"])) @ modes.T
    K += settings["kappa_rot"] * (Qrot @ Qrot.T)
    K = (K + K.T) / 2
    np.linalg.cholesky(K)
    energy = float(ff.CalcEnergy(tuple(x)))
    calibration = []
    retained = np.flatnonzero(~floored)
    if len(retained):
        j = int(retained[len(retained) // 2])
        direction = Q @ modes[:, j]
        for amplitude in (1e-2, 5e-3, 2.5e-3):
            displacement = amplitude * direction
            remainder = (
                ff.CalcEnergy(tuple(x + displacement)) - energy - grad @ displacement
            )
            calibration.append(
                {
                    "amplitude_angstrom": amplitude,
                    "mode": j,
                    "ratio": float(2 * remainder / (amplitude**2 * spectrum[j])),
                }
            )
    audit = {
        "potential": "MMFF94",
        "units": {
            "length": "angstrom",
            "energy": "kcal/mol",
            "stiffness": "kcal/mol/angstrom^2",
        },
        "settings": settings,
        "relaxation_status": status,
        "gradient_norm": gradient_norm,
        "alignment_rmsd_angstrom": float(
            np.sqrt(np.mean(np.sum((reference - centered(target)) ** 2, axis=1)))
        ),
        "relaxed_energy_kcal_mol": energy,
        "stereochemistry_unchanged": True,
        "rotation_rank": Qrot.shape[1],
        "rotation_hessian_residual": float(np.linalg.norm(H @ Qrot)),
        "rot_vib_cross_residual": float(np.linalg.norm(Qrot.T @ H @ Qvib)),
        "half_step_relative_error": half_step_error,
        "antisymmetric_relative_error": asymmetry,
        "floored_modes": int(floored.sum()),
        "calibration": calibration,
        "curvature_transfer": "Hessian of separately relaxed/aligned MMFF94 reference; original target unchanged",
        "seconds": time.perf_counter() - started,
    }
    return {
        "K": K,
        "Q": Q,
        "H_raw": H,
        "H_unsymmetrized": fine,
        "target": target,
        "reference": reference,
        "gradient": grad,
        "spectrum": spectrum,
        "floored_mask": floored,
        "epsilon_E": settings["epsilon_E"],
        "audit": audit,
    }


def build_label_arrays(conformers, source):
    """Use one unchanged target/K pair for both core and Alternative A factors."""
    from stifflow_alt_a.path import prepare_label as prepare_alt_label

    from .precision import prepare_label

    labels = []
    for conformer in conformers:
        try:
            labels.append(estimate_stiffness(conformer["rd_mol"]))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"{conformer['conformer_id']}: {exc}") from exc
    precision = [prepare_label(source, l["K"], l["epsilon_E"]) for l in labels]
    alternative = [prepare_alt_label(source, l["K"], l["epsilon_E"]) for l in labels]
    arrays = {
        "positions": np.stack([c["positions"] for c in conformers]),
        "K": np.stack([l["K"] for l in labels]),
        "epsilon_E": labels[0]["epsilon_E"],
        "reference": np.stack([l["reference"] for l in labels]),
        "H_raw": np.stack([l["H_raw"] for l in labels]),
        "spectrum": np.stack([l["spectrum"] for l in labels]),
        "floored_mask": np.stack([l["floored_mask"] for l in labels]),
        "precision_values": np.stack([l["values"] for l in precision]),
        "precision_vectors": np.stack([l["vectors"] for l in precision]),
        "alt_L_K": np.stack([l["L_K"] for l in alternative]),
    }
    if any(
        not np.array_equal(l["target"], c["positions"])
        for l, c in zip(labels, conformers)
    ):
        raise ValueError("Curvature preprocessing changed the original target")
    return (
        arrays,
        [l["audit"] for l in labels],
        [l["covariance_residual"] for l in alternative],
    )


def main():
    from .data import load_molecule
    from .preflight import fingerprint, provenance
    from .smoke import external_directory
    from .source import prepare_source

    parser = argparse.ArgumentParser(
        description="Bounded train/validation MMFF94 label preprocessing"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=external_directory, required=True)
    parser.add_argument("--molecules", type=int, default=2)
    parser.add_argument("--max-atoms", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--conformers", type=int, default=1)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    args = parser.parse_args()
    if min(args.molecules, args.max_atoms, args.conformers, args.max_candidates) < 1:
        parser.error("Counts must be positive")
    import os

    if (args.molecules > 4 or args.conformers > 2) and not os.getenv("SLURM_JOB_ID"):
        parser.error("Larger label preprocessing requires a Slurm allocation")
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    records, attempts = [], []
    report = {
        "purpose": "small train/validation label audit; max-atoms selection is not a pilot filter",
        "audit_partial": True,
        "source_config": {"sigma0": 1.0, "rho": 1.0},
        "manifest": fingerprint(args.manifest),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "records": records,
        "attempts": attempts,
        "provenance": provenance(),
    }
    for entry in manifest["entries"]:
        if entry["split"] != args.split:
            continue
        if len(attempts) >= args.max_candidates:
            break
        attempt = {"id": entry["molecule_id"], "split": args.split}
        attempts.append(attempt)
        try:
            molecule = load_molecule(entry, manifest["raw_dir"])
            if len(molecule["graph"]["atomic_numbers"]) > args.max_atoms:
                attempt.update(
                    status="audit_filter",
                    reason="exceeds --max-atoms; not a pilot exclusion",
                )
                continue
            selected = molecule["conformers"][: args.conformers]
            source = prepare_source(molecule["graph"])
            arrays, audits, residuals = build_label_arrays(selected, source)
            filename = (
                hashlib.sha256(entry["molecule_id"].encode()).hexdigest()[:20] + ".npz"
            )
            np.savez_compressed(args.output / filename, **arrays)
            record = {
                "id": entry["molecule_id"],
                "split": args.split,
                "npz": filename,
                "molblock": Chem.MolToMolBlock(selected[0]["rd_mol"]),
                "conformer_ids": [c["conformer_id"] for c in selected],
                "original_weights": [c["boltzmann_weight"] for c in selected],
                "original_energies": [c["energy"] for c in selected],
                "raw_indices": [c["raw_index"] for c in selected],
                "retained_conformers": len(molecule["conformers"]),
                "audit_partial": len(selected) != len(molecule["conformers"]),
                "graph": molecule["graph"],
                "source": molecule["source"],
                "labels": audits,
                "alternative_factor_residuals": residuals,
                "cache": fingerprint(args.output / filename),
            }
            records.append(record)
            attempt.update(status="success", npz=filename)
        except (ValueError, RuntimeError, KeyError) as exc:
            attempt.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        if sum(a["status"] != "audit_filter" for a in attempts) >= args.molecules:
            break
    report["audit_partial"] = any(r["audit_partial"] for r in records)
    report["requested_attempts_completed"] = (
        sum(a["status"] != "audit_filter" for a in attempts) == args.molecules
    )
    report["failed"] = sum(a["status"] == "failed" for a in attempts)
    with (args.output / "manifest.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "successful": len(records),
                "attempts": len(attempts),
            }
        )
    )
    raise SystemExit(
        1
        if report["failed"] or not records or not report["requested_attempts_completed"]
        else 0
    )


if __name__ == "__main__":
    main()
