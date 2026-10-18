"""Stage D diagnostics: MMFF94 relaxable strain for the four 25k candidates.

Same protocol as task3 in biweekly2 (raw generated conformers on the frozen
512-molecule monitor, seed 42), restricted to the research campaign's 25k
candidate generations.  Diagnostic only; no metric replaces raw coordinates.

Outputs runs/research-20261018/compare/strain-25k-candidates.json
"""

import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from stifflow_benchmark.evaluate import conformer_mol

RDLogger.DisableLog("rdApp.*")

WS = Path("/gpfsnyu/scratch/ys6132/stifflow")
ROOT = WS / "runs/research-20261018"
OUT = ROOT / "compare/strain-25k-candidates.json"
CANDIDATES = ("a_rot_null-25k", "a_trace_null-25k", "linear-25k", "aligned_linear-25k")
BASELINES = ("alternative_a-25k",)


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
        "energy_kcal_mol": e_gen,
        "force_norm_kcal_mol_a": force_norm,
        "relaxed_energy_kcal_mol": e_min,
        "relaxable_strain_kcal_mol": e_gen - e_min,
        "stereochemistry_changed_during_relaxation": stereo_changed,
    }


def process(name):
    manifest = json.loads((ROOT / f"generated/{name}/generated/manifest.json").read_text())
    rows = []
    for record in manifest["records"]:
        if record["status"] != "ok":
            continue
        mol = Chem.MolFromMolBlock(record["molblock"], removeHs=False, sanitize=True)
        if mol is None:
            continue
        with np.load(
            ROOT / f"generated/{name}/generated" / record["npz"], allow_pickle=False
        ) as saved:
            refs = saved["pos_ref"]
            generated = saved["pos_gen"][: 2 * len(saved["pos_ref"])]
        bonds = heavy_bonds(mol)
        reference_lengths = (
            np.mean([bond_lengths(p, bonds) for p in refs], axis=0) if bonds else None
        )
        for positions in generated:
            entry = {"id": record["id"]}
            if reference_lengths is not None:
                entry["bond_deviation_angstrom"] = float(
                    np.abs(bond_lengths(positions, bonds) - reference_lengths).mean()
                )
            entry.update(mmff_energies(mol, positions))
            rows.append(entry)
    return rows


def summarize(rows):
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return {"n": 0}
    strain = np.array([r["relaxable_strain_kcal_mol"] for r in ok])
    bonds = [r["bond_deviation_angstrom"] for r in ok if "bond_deviation_angstrom" in r]
    result = {
        "n_conformers": len(rows),
        "n_ok": len(ok),
        "n_stereo_changed_during_relaxation": sum(
            1 for r in ok if r.get("stereochemistry_changed_during_relaxation")
        ),
        "relaxable_strain_kcal_mol": {
            "mean": float(strain.mean()),
            "median": float(np.median(strain)),
            "p90": float(np.percentile(strain, 90)),
        },
    }
    if bonds:
        b = np.array(bonds)
        result["bond_deviation_angstrom"] = {
            "mean": float(b.mean()),
            "median": float(np.median(b)),
        }
    return result


def main():
    result = {"schema_version": 1, "seed": 42, "methods": {}}
    for name in CANDIDATES + BASELINES:
        if not (ROOT / f"generated/{name}/evaluation.json").is_file():
            print(f"skip {name}: no evaluation yet")
            continue
        rows = process(name)
        # keep per-conformer rows out of the committed artifact
        result["methods"][name] = {
            "summary": summarize(rows),
            "per_conformer_file": None,
        }
        print(name, json.dumps(result["methods"][name]["summary"]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"output": str(OUT)}))


if __name__ == "__main__":
    main()
