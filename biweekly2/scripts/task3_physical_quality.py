"""Task 3: physical quality of raw generated conformers on the shared monitor.

For the frozen 512-molecule validation monitor subset, evaluate the RAW
generated conformers of the three formal methods (seed 42, 250k) plus the
reference ensembles:

- local geometry: heavy-atom bond-length deviation from the reference ensemble
  mean, per conformer;
- MMFF94 single-point energy and gradient (force) norm;
- relaxable strain energy: E(gen) - E(minimized from gen), with minimization
  status and stereochemistry-change bookkeeping.  Optimization is diagnostic
  only; it never replaces the raw generated coordinates in any metric.

Every failure (missing MMFF params, non-converged minimization, stereo change
during relaxation) is recorded, not silently dropped.

Outputs runs/biweekly2-20261019/results/task3-physical-quality.json
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from stifflow_benchmark.evaluate import conformer_mol

RDLogger.DisableLog("rdApp.*")

WS = Path("/gpfsnyu/scratch/ys6132/stifflow")
OUT = WS / "runs/biweekly2-20261019/results/task3-physical-quality.json"
MONITOR = WS / "cache/qm9-v3-references-4090d/val/monitor.json"
EVAL = WS / "runs/formal-eval-20261018"
METHODS = ("precision", "alternative_a", "control")
SEED = 42


def stereo_from_coordinates(mol):
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


def heavy_bonds(mol):
    return [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in mol.GetBonds()
        if b.GetBeginAtom().GetAtomicNum() > 1 and b.GetEndAtom().GetAtomicNum() > 1
    ]


def bond_lengths(positions, bonds):
    return np.array([np.linalg.norm(positions[i] - positions[j]) for i, j in bonds])


def mmff_energies(mol, positions):
    """Single-point energy, gradient norm, and relaxed strain with bookkeeping."""
    probe = conformer_mol(mol, positions)
    if not AllChem.MMFFHasAllMoleculeParams(probe):
        return {"status": "failed", "reason": "missing MMFF94 parameters"}
    props = AllChem.MMFFGetMoleculeProperties(probe, mmffVariant="MMFF94")
    ff = AllChem.MMFFGetMoleculeForceField(probe, props)
    e_gen = float(ff.CalcEnergy())
    grad = np.asarray(ff.CalcGrad(), dtype=np.float64).reshape(-1, 3)
    force_norm = float(np.linalg.norm(grad))

    original_stereo = stereo_from_coordinates(probe)
    relaxed = Chem.Mol(probe)
    ff2 = AllChem.MMFFGetMoleculeForceField(relaxed, props)
    status = ff2.Minimize(maxIts=2000, forceTol=1e-6, energyTol=1e-9)
    stereo_changed = stereo_from_coordinates(relaxed) != original_stereo
    e_min = float(ff2.CalcEnergy())
    return {
        "status": "ok" if status == 0 else "minimization_not_converged",
        "minimize_status": int(status),
        "energy_kcal_mol": e_gen,
        "force_norm_kcal_mol_a": force_norm,
        "relaxed_energy_kcal_mol": e_min,
        "relaxable_strain_kcal_mol": e_gen - e_min,
        "stereochemistry_changed_during_relaxation": stereo_changed,
    }


def process_method(method):
    manifest = json.loads(
        (EVAL / f"monitor/{method}-seed{SEED}/generated/manifest.json").read_text()
    )
    evaluation = json.loads(
        (EVAL / f"monitor/{method}-seed{SEED}/evaluation.json").read_text()
    )
    evaluated = {r["id"]: r for r in evaluation["records"]}
    rows = []
    failures = []
    for record in manifest["records"]:
        if record["status"] != "ok":
            failures.append({"id": record["id"], "reason": "generation failed"})
            continue
        mol = Chem.MolFromMolBlock(record["molblock"], removeHs=False, sanitize=True)
        if mol is None:
            failures.append({"id": record["id"], "reason": "invalid molblock"})
            continue
        with np.load(
            EVAL / f"monitor/{method}-seed{SEED}/generated" / record["npz"],
            allow_pickle=False,
        ) as saved:
            refs = saved["pos_ref"]
            generated = saved["pos_gen"][: 2 * len(saved["pos_ref"])]
        bonds = heavy_bonds(mol)
        reference_lengths = (
            np.mean([bond_lengths(p, bonds) for p in refs], axis=0) if bonds else None
        )
        eval_record = evaluated.get(record["id"], {})
        chirality_fraction = eval_record.get("chirality", {}).get("fraction")
        cov_r = None
        if eval_record.get("status") == "ok":
            thresholds = evaluation["protocol"]["thresholds"]
            cov_r = eval_record["cov_r"][thresholds.index(0.5)]
        for index, positions in enumerate(generated):
            entry = {"id": record["id"], "conformer": index}
            if reference_lengths is not None:
                entry["bond_deviation_angstrom"] = float(
                    np.abs(bond_lengths(positions, bonds) - reference_lengths).mean()
                )
            energy = mmff_energies(mol, positions)
            entry.update(energy)
            rows.append(entry)
            if energy["status"] != "ok":
                failures.append(
                    {"id": record["id"], "conformer": index, "reason": energy["status"]}
                )
    return rows, failures


def process_references():
    """Reference-ensemble baseline: same quantities on the official references."""
    monitor = json.loads(MONITOR.read_text())
    rows, failures = [], []
    for record in monitor["records"]:
        mol = Chem.MolFromMolBlock(record["molblock"], removeHs=False, sanitize=True)
        if mol is None:
            failures.append({"id": record["id"], "reason": "invalid molblock"})
            continue
        with np.load(
            MONITOR.parent / record["reference_npz"], allow_pickle=False
        ) as saved:
            refs = saved["pos_ref"]
        for index, positions in enumerate(refs):
            energy = mmff_energies(mol, positions)
            rows.append({"id": record["id"], "conformer": index, **energy})
            if energy["status"] != "ok":
                failures.append(
                    {"id": record["id"], "conformer": index, "reason": energy["status"]}
                )
    return rows, failures


def summarize(rows):
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return {"n": 0}
    strain = np.array([r["relaxable_strain_kcal_mol"] for r in ok])
    energy = np.array([r["energy_kcal_mol"] for r in ok])
    force = np.array([r["force_norm_kcal_mol_a"] for r in ok])
    bonds = [r["bond_deviation_angstrom"] for r in ok if "bond_deviation_angstrom" in r]
    stereo_changes = sum(
        1 for r in ok if r.get("stereochemistry_changed_during_relaxation")
    )
    result = {
        "n_conformers": len(rows),
        "n_ok": len(ok),
        "n_minimization_not_converged": sum(
            1 for r in rows if r.get("status") == "minimization_not_converged"
        ),
        "n_stereo_changed_during_relaxation": stereo_changes,
        "relaxable_strain_kcal_mol": {
            "mean": float(strain.mean()),
            "median": float(np.median(strain)),
            "p90": float(np.percentile(strain, 90)),
        },
        "energy_kcal_mol": {
            "mean": float(energy.mean()),
            "median": float(np.median(energy)),
        },
        "force_norm_kcal_mol_a": {
            "mean": float(force.mean()),
            "median": float(np.median(force)),
            "p90": float(np.percentile(force, 90)),
        },
    }
    if bonds:
        b = np.array(bonds)
        result["bond_deviation_angstrom"] = {
            "mean": float(b.mean()),
            "median": float(np.median(b)),
            "p90": float(np.percentile(b, 90)),
        }
    return result


def main():
    result = {"schema_version": 1, "seed": SEED, "split": "val monitor (512)", "methods": {}}
    for method in METHODS:
        rows, failures = process_method(method)
        result["methods"][method] = {
            "per_conformer": rows,
            "failures": failures,
            "summary": summarize(rows),
        }
        print(
            f"{method}: {len(rows)} conformers, "
            f"{result['methods'][method]['summary'].get('n_ok', 0)} ok, "
            f"{len(failures)} failures",
            flush=True,
        )
    ref_rows, ref_failures = process_references()
    result["references"] = {
        "per_conformer": ref_rows,
        "failures": ref_failures,
        "summary": summarize(ref_rows),
    }
    print(
        f"references: {len(ref_rows)} conformers, "
        f"{result['references']['summary'].get('n_ok', 0)} ok",
        flush=True,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    compact = {
        m: result["methods"][m]["summary"] for m in METHODS
    } | {"references": result["references"]["summary"]}
    print(json.dumps(compact, indent=1))


if __name__ == "__main__":
    main()
