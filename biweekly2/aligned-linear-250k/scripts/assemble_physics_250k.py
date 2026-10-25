"""Assemble the 250k physics audit including aligned_linear."""

import json
from pathlib import Path

import numpy as np

OUT = Path("/gpfsnyu/scratch/ys6132/stifflow/runs/locality-20261019")
METHODS = ("precision", "alternative_a", "control", "aligned_linear")
SEEDS = (42, 43, 44)
N_BOOT = 2000


def load(method, seed):
    report = json.loads((OUT / f"physics-250k/{method}-seed{seed}/report.json").read_text())
    assert report["status"] == "completed"
    assert report["run"]["training_budget"]["steps"] == 250000
    assert report["aggregate"]["failed_molecules"] == 0
    return {r["id"]: r for r in report["records"]}


def summary(rows):
    out = {}
    for method in METHODS:
        entry = {}
        for seed in SEEDS:
            records = rows[(method, seed)]
            entry[str(seed)] = {
                "force_mean_molecule_median": float(
                    np.mean([r["force_median"] for r in records.values()])
                ),
                "reference_force_mean_molecule_median": float(
                    np.mean([r["reference_force_median"] for r in records.values()])
                ),
                "strain_mean_molecule_median": float(
                    np.mean(
                        [
                            r["strain_median_stereo_preserving"]
                            for r in records.values()
                            if r["strain_median_stereo_preserving"] is not None
                        ]
                    )
                ),
            }
        values = list(entry.values())
        out[method] = {
            "per_seed": entry,
            "mean": {k: float(np.mean([v[k] for v in values])) for k in values[0]},
            "seed_std": {
                k: float(np.std([v[k] for v in values], ddof=1)) for k in values[0]
            },
        }
    return out


def paired(rows, left, right, seed=42):
    ids = sorted(rows[(left, SEEDS[0])])
    pooled = np.array(
        [
            np.mean(
                [
                    rows[(left, s)][i]["force_median"]
                    - rows[(right, s)][i]["force_median"]
                    for s in SEEDS
                ]
            )
            for i in ids
        ]
    )
    rng = np.random.default_rng(seed)
    draws = pooled[rng.integers(0, len(pooled), size=(N_BOOT, len(pooled)))].mean(axis=1)
    per_seed = [
        float(
            np.mean(
                [
                    rows[(left, s)][i]["force_median"]
                    - rows[(right, s)][i]["force_median"]
                    for i in ids
                ]
            )
        )
        for s in SEEDS
    ]
    return {
        "molecules": len(ids),
        "seeds": list(SEEDS),
        "mean": float(pooled.mean()),
        "ci95_molecule_bootstrap": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
        "per_seed_mean": per_seed,
        "seed_std_of_paired_mean": float(np.std(per_seed, ddof=1)),
        "fraction_molecules_lower": float((pooled < 0).mean()),
    }


rows = {(m, s): load(m, s) for m in METHODS for s in SEEDS}
ids = sorted(rows[("alternative_a", 42)])
assert all(sorted(rows[(m, s)]) == ids for m in METHODS for s in SEEDS)

report = {
    "schema_version": 1,
    "kind": "stifflow-physics-250k-including-aligned-linear",
    "protocol": {
        "molecules": 64,
        "selection": "SHA256(id), first 64 of the frozen 512-molecule monitor",
        "conformers": "min(8, 2*N_ref) generated per molecule",
        "potential": "MMFF94",
        "force_units": "kcal/mol/angstrom",
        "force_statistic": "conformer median then equal molecule mean",
        "budget": 250000,
        "generation": "Euler50, seed 42, 2x references, no postprocessing",
    },
    "methods": summary(rows),
    "paired_force_deltas": {
        f"{a}-{b}": paired(rows, a, b)
        for a, b in (
            ("aligned_linear", "control"),
            ("aligned_linear", "alternative_a"),
            ("aligned_linear", "precision"),
            ("alternative_a", "control"),
            ("alternative_a", "precision"),
            ("control", "precision"),
        )
    },
    "caveats": [
        "MMFF94-only; no GFN2-xTB oracle.",
        "Raw force is not a thermodynamic/basin-population metric.",
        "Frozen 512-molecule validation monitor; no test selection.",
    ],
}
target = OUT / "analysis/physics-250k-all.json"
target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(target)
for method, value in report["methods"].items():
    print(
        method,
        "force %.3f" % value["mean"]["force_mean_molecule_median"],
        "strain %.3f" % value["mean"]["strain_mean_molecule_median"],
    )
for key, value in report["paired_force_deltas"].items():
    print(key, "%+.3f" % value["mean"], "[%+.3f, %+.3f]" % tuple(value["ci95_molecule_bootstrap"]))
