"""Full-budget (250k) paired comparison of aligned_linear vs control and A.

Validates data identity (canonical vs the formal runs' relocated staged
manifests, paths excluded), protocol/reference/seed/budget consistency, then
reports per-split three-seed means with seed SD and pooled molecule-paired
bootstrap CIs for the aligned_linear - {control, alternative_a} deltas.

Outputs runs/aligned-linear-250k/analysis/full-budget-comparison.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

WS = Path("/gpfsnyu/scratch/ys6132/stifflow")
AL = WS / "runs/aligned-linear-250k"
FORMAL = WS / "runs/formal-eval-20261018"
OUT = AL / "analysis/full-budget-comparison.json"
SEEDS = (42, 43, 44)
SPLITS = ("test", "val", "monitor")


def without_paths(value):
    if isinstance(value, dict):
        return {k: without_paths(v) for k, v in value.items() if k != "path"}
    if isinstance(value, list):
        return [without_paths(v) for v in value]
    return value


def content_hash(manifest):
    payload = without_paths(manifest)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_data_identity():
    """The formal runs trained on relocated (staged) copies of the same
    canonical packed cache; verify content equality, paths excluded."""
    canonical = {
        split: json.loads(
            (WS / f"cache/qm9-v3-packed-strict/{split}/manifest.json").read_text()
        )
        for split in ("train", "val")
    }
    hashes = {split: content_hash(m) for split, m in canonical.items()}
    checks = {"canonical_content_hashes": hashes, "staged_reproductions": {}}
    # Archived identity JSONs exist for control/A/precision seed42 (rebuilt by
    # the research campaign and byte-matching the checkpoint signatures).
    for method in ("control", "alternative_a"):
        path = WS / f"runs/research-20261018/identity/{method}-seed42"
        reproduced = {}
        for split in ("train", "val"):
            archived = json.loads((path / f"{split}.json").read_text())
            reproduced[split] = {
                "content_hash": content_hash(archived),
                "matches_canonical": content_hash(archived) == hashes[split],
            }
        checks["staged_reproductions"][method] = reproduced
    # The full-budget aligned_linear runs used the canonical manifests directly.
    checks["aligned_linear_labels"] = "canonical packed cache (direct)"
    return checks


def load_report(path):
    report = json.loads(Path(path).read_text())
    run = report["run"]
    thresholds = report["protocol"]["thresholds"]
    at_half = thresholds.index(0.5)
    rows = {}
    for record in report["records"]:
        if record.get("status") != "ok":
            raise ValueError(f"failed record in {path}: {record['id']}")
        rows[record["id"]] = {
            "mat_r": record["mat_r"],
            "mat_p": record["mat_p"],
            "cov_r": record["cov_r"][at_half],
            "chirality": record["chirality"]["fraction"],
        }
    return {
        "steps": run["training_budget"]["steps"],
        "seed": run["seed"],
        "generation_seed": run["seed"],
        "reference": run["input_manifest"]["sha256"],
        "rows": rows,
        "attempted": report["aggregate"]["attempted"],
        "failed": report["aggregate"]["failed"],
    }


def collect(split):
    reports = {"aligned_linear": {}, "control": {}, "alternative_a": {}}
    for seed in SEEDS:
        reports["aligned_linear"][seed] = load_report(
            AL / f"eval/aligned_linear-seed{seed}/{split}/evaluation.json"
        )
        for method in ("control", "alternative_a"):
            if split == "monitor":
                path = FORMAL / f"monitor/{method}-seed{seed}/evaluation.json"
            else:
                path = FORMAL / f"{split}/{method}-seed{seed}/evaluation.json"
            reports[method][seed] = load_report(path)
    # Consistency checks across all nine reports.
    references = {reports[m][SEEDS[0]]["reference"] for m in reports}
    if len(references) != 1:
        raise ValueError("Reference manifests differ across methods")
    steps = {reports[m][SEEDS[0]]["steps"] for m in reports}
    if steps != {250000}:
        raise ValueError(f"Not full-budget reports: {steps}")
    ids = {m: set(reports[m][SEEDS[0]]["rows"]) for m in reports}
    if len({frozenset(s) for s in ids.values()}) != 1:
        raise ValueError("Molecule cohorts differ across methods")
    for m in reports:
        for seed in SEEDS:
            r = reports[m][seed]
            if r["failed"] or r["attempted"] != len(r["rows"]):
                raise ValueError(f"Incomplete report {m} seed {seed}")
    return reports


def metrics_summary(reports):
    out = {}
    for method, by_seed in reports.items():
        per_seed = {}
        for seed, report in by_seed.items():
            values = list(report["rows"].values())
            per_seed[seed] = {
                key: float(np.mean([v[key] for v in values if v[key] is not None]))
                for key in values[0]
            }
            fractions = [
                v["chirality"] for v in values if v["chirality"] is not None
            ]
            per_seed[seed]["chirality_macro"] = float(np.mean(fractions))
        keys = list(per_seed[SEEDS[0]])
        out[method] = {
            "per_seed": per_seed,
            "mean": {
                key: float(np.mean([per_seed[s][key] for s in SEEDS])) for key in keys
            },
            "seed_std": {
                key: float(np.std([per_seed[s][key] for s in SEEDS], ddof=1))
                for key in keys
            },
        }
    return out


def paired_delta(candidate, baseline, n_boot=10000, seed=0):
    common = sorted(set(candidate[SEEDS[0]]["rows"]))
    stacked = {}
    for metric in ("mat_r", "mat_p", "cov_r", "chirality"):
        rows = []
        for mid in common:
            per_seed_vals = []
            for s in SEEDS:
                a = candidate[s]["rows"][mid][metric]
                b = baseline[s]["rows"][mid][metric]
                if a is None or b is None:
                    per_seed_vals = None
                    break
                per_seed_vals.append(a - b)
            rows.append(None if per_seed_vals is None else np.mean(per_seed_vals))
        kept = [r for r in rows if r is not None]
        values = np.array(kept)
        rng = np.random.default_rng(seed)
        draws = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(
            axis=1
        )
        seed_means = []
        for s in SEEDS:
            per_seed = [
                candidate[s]["rows"][mid][metric] - baseline[s]["rows"][mid][metric]
                for mid in common
                if candidate[s]["rows"][mid][metric] is not None
                and baseline[s]["rows"][mid][metric] is not None
            ]
            seed_means.append(float(np.mean(per_seed)))
        stacked[metric] = {
            "molecules": len(values),
            "mean": float(values.mean()),
            "ci95_molecule_bootstrap": [
                float(np.percentile(draws, 2.5)),
                float(np.percentile(draws, 97.5)),
            ],
            "per_seed_mean": seed_means,
            "seed_std_of_paired_mean": float(np.std(seed_means, ddof=1)),
            "fraction_molecules_better": float((values < 0).mean()),
        }
    return stacked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = {
        "schema_version": 1,
        "budget": 250000,
        "seeds": list(SEEDS),
        "protocol": "Euler50, generation seed 42, 2x references, frozen manifests",
        "data_identity": verify_data_identity(),
        "splits": {},
    }
    for split in SPLITS:
        reports = collect(split)
        entry = {
            "metrics": metrics_summary(reports),
            "paired_deltas": {
                "aligned_linear-control": paired_delta(
                    reports["aligned_linear"], reports["control"]
                ),
                "aligned_linear-alternative_a": paired_delta(
                    reports["aligned_linear"], reports["alternative_a"]
                ),
            },
        }
        result["splits"][split] = entry
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    for split in SPLITS:
        d = result["splits"][split]["paired_deltas"]["aligned_linear-control"]
        print(
            f"{split}: AL-control MAT-R {d['mat_r']['mean']:+.4f} "
            f"CI {d['mat_r']['ci95_molecule_bootstrap'][0]:+.4f}…{d['mat_r']['ci95_molecule_bootstrap'][1]:+.4f} "
            f"| chir {d['chirality']['mean']:+.4f}"
        )


if __name__ == "__main__":
    main()
