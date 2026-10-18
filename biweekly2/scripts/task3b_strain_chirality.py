"""Task 3b: pair physical quality with chirality and coverage per molecule.

For the shared monitor subset, aggregates per-molecule mean relaxable strain
by method, together with the molecule's chirality fraction and COV-R@0.5 from
the existing evaluation; computes per-molecule paired strain differences
(A - control) with a molecule bootstrap CI, stratified by chirality-correct vs
incorrect.  Answers: does Alternative A buy any physical-quality or
chirality/coverage benefit over the stiffness-free control?

Outputs runs/biweekly2-20261019/results/task3b-strain-chirality.json
"""

import json
from pathlib import Path

import numpy as np

WS = Path("/gpfsnyu/scratch/ys6132/stifflow")
BW = WS / "runs/biweekly2-20261019"
OUT = BW / "results/task3b-strain-chirality.json"
EVAL = WS / "runs/formal-eval-20261018"
METHODS = ("precision", "alternative_a", "control")
SEED = 42


def main():
    physical = json.loads((BW / "results/task3-physical-quality.json").read_text())
    per_method = {}
    for method in METHODS:
        rows = [
            r
            for r in physical["methods"][method]["per_conformer"]
            if r.get("status") == "ok"
        ]
        per_method[method] = {r["id"]: r for r in rows}
    chirality, covr = {}, {}
    for method in METHODS:
        report = json.loads(
            (EVAL / f"monitor/{method}-seed{SEED}/evaluation.json").read_text()
        )
        thresholds = report["protocol"]["thresholds"]
        at_half = thresholds.index(0.5)
        chirality[method], covr[method] = {}, {}
        for record in report["records"]:
            if record.get("status") != "ok":
                continue
            chirality[method][record["id"]] = record["chirality"]["fraction"]
            covr[method][record["id"]] = record["cov_r"][at_half]

    common = sorted(set(per_method["alternative_a"]) & set(per_method["control"]))
    a_strain = np.array(
        [
            per_method["alternative_a"][i]["relaxable_strain_kcal_mol"]
            for i in common
        ]
    )
    c_strain = np.array(
        [per_method["control"][i]["relaxable_strain_kcal_mol"] for i in common]
    )

    # Molecule-level means for the paired comparison (conformers are paired
    # per molecule; average first, then bootstrap molecules).
    ids = common
    a_by_mol, c_by_mol = {}, {}
    for i in ids:
        a_by_mol.setdefault(i, []).append(
            per_method["alternative_a"][i]["relaxable_strain_kcal_mol"]
        )
        c_by_mol.setdefault(i, []).append(
            per_method["control"][i]["relaxable_strain_kcal_mol"]
        )
    a_mean = np.array([np.mean(a_by_mol[i]) for i in ids])
    c_mean = np.array([np.mean(c_by_mol[i]) for i in ids])
    delta = a_mean - c_mean
    rng = np.random.default_rng(42)
    n = len(ids)
    draws = delta[rng.integers(0, n, size=(10000, n))].mean(axis=1)

    # Stratify by chirality-correct molecules (chiral A conformers all-correct
    # per molecule fraction == 1 vs < 1) for the A method.
    chiral_ids = [i for i in ids if chirality["alternative_a"][i] is not None]
    correct_ids = [i for i in chiral_ids if chirality["alternative_a"][i] == 1.0]
    partial_ids = [i for i in chiral_ids if chirality["alternative_a"][i] < 1.0]
    index_of = {i: k for k, i in enumerate(ids)}

    def stats_for(subset):
        if not subset:
            return None
        values = delta[[index_of[i] for i in subset]]
        return {
            "molecules": len(subset),
            "mean_strain_delta": float(values.mean()),
        }

    result = {
        "schema_version": 1,
        "seed": SEED,
        "paired_strain_delta_A_minus_control": {
            "molecules": n,
            "mean": float(delta.mean()),
            "ci95_molecule_bootstrap": [
                float(np.percentile(draws, 2.5)),
                float(np.percentile(draws, 97.5)),
            ],
        },
        "strain_delta_by_A_chirality": {
            "all_correct_molecules": stats_for(correct_ids),
            "some_wrong_molecules": stats_for(partial_ids),
        },
        "method_summaries": {},
    }
    for method in METHODS:
        strains = np.array(
            [
                r["relaxable_strain_kcal_mol"]
                for r in physical["methods"][method]["per_conformer"]
                if r.get("status") == "ok"
            ]
        )
        result["method_summaries"][method] = {
            "strain_mean": float(strains.mean()),
            "strain_median": float(np.median(strains)),
        }
    ref_strains = np.array(
        [
            r["relaxable_strain_kcal_mol"]
            for r in physical["references"]["per_conformer"]
            if r.get("status") == "ok"
        ]
    )
    result["method_summaries"]["references"] = {
        "strain_mean": float(ref_strains.mean()),
        "strain_median": float(np.median(ref_strains)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=1)[:1200])


if __name__ == "__main__":
    main()
