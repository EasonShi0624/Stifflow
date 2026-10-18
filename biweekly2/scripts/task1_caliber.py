"""Task 1: evaluation-caliber audit and pooled three-seed paired bootstrap.

Reads only existing formal-evaluation artifacts; no new generation.
Outputs runs/biweekly2-20261019/results/task1-caliber.json
"""

import hashlib
import json
from pathlib import Path

import numpy as np

WS = Path("/gpfsnyu/scratch/ys6132/stifflow")
EVAL = WS / "runs/formal-eval-20261018"
OUT = WS / "runs/biweekly2-20261019/results/task1-caliber.json"
METHODS = ("precision", "alternative_a", "control")
SEEDS = (42, 43, 44)


def reference_mapping_audit():
    """Scan the frozen reference manifests for topology/mapping reorders.

    Commit 793d502 (topology tolerance) records per-conformer atom mappings
    whenever a reference row needs reordering to the conditioning graph.  Any
    non-empty entry would mean the exported references (and hence every
    evaluation) used a reordered atom order that the mapping audit must check.
    """
    result = {}
    for split in ("val", "test"):
        manifest = json.loads(
            (WS / f"cache/qm9-v3-references-4090d/{split}/manifest.json").read_text()
        )
        records = manifest["records"]
        mapped = [
            r["id"]
            for r in records
            if r.get("reference_atom_mappings")
        ]
        topology = [
            r["id"]
            for r in records
            if r.get("topology_variation_rows")
        ]
        stereo = [
            r["id"]
            for r in records
            if r.get("stereo_variation_rows")
        ]
        result[split] = {
            "molecules": len(records),
            "molecules_with_atom_mappings": len(mapped),
            "ids_with_atom_mappings": mapped[:20],
            "molecules_with_topology_variation": len(topology),
            "molecules_with_stereo_variation": len(stereo),
        }
    monitor = json.loads(
        (WS / "cache/qm9-v3-references-4090d/val/monitor.json").read_text()
    )
    result["monitor"] = {
        "molecules": len(monitor["records"]),
        "molecules_with_atom_mappings": sum(
            1 for r in monitor["records"] if r.get("reference_atom_mappings")
        ),
        "molecules_with_topology_variation": sum(
            1 for r in monitor["records"] if r.get("topology_variation_rows")
        ),
        "molecules_with_stereo_variation": sum(
            1 for r in monitor["records"] if r.get("stereo_variation_rows")
        ),
    }
    # P0 ensembles already re-verified conditioning vs the trusted raw GEOM
    # files for 64 molecules per method (mapping fields live per record).
    p0 = {}
    for method in METHODS:
        path = WS / f"runs/research-20261018/p0/ensembles-{method}.json"
        if path.is_file():
            report = json.loads(path.read_text())
            records = report.get("records", [])
            summary = {
                "source": str(path),
                "molecules": len(records),
                "ok_records": sum(1 for r in records if r.get("status") == "ok"),
            }
            for key in (
                "matches_current_canonical_coordinates",
                "conditioning_topology_matches",
                "conditioning_stereochemistry_matches",
            ):
                values = [r.get("mapping", {}).get(key) for r in records]
                summary[key] = {
                    "n": len(values),
                    "all_true": bool(values) and all(v is True for v in values),
                }
            for key in ("different_rows", "reference_atom_mappings"):
                counts = [len(r.get("mapping", {}).get(key, [])) for r in records]
                summary[key] = {"total_nonempty_entries": int(sum(counts))}
            p0[method] = summary
        else:
            p0[method] = {"source": None, "missing": True}
    result["p0_raw_data_recheck"] = p0
    return result


def per_molecule_metrics(evaluation_path):
    report = json.loads(Path(evaluation_path).read_text())
    thresholds = report["protocol"]["thresholds"]
    at_half = thresholds.index(0.5)
    rows = {}
    for record in report["records"]:
        if record.get("status") != "ok":
            continue
        rows[record["id"]] = {
            "mat_r": record["mat_r"],
            "mat_p": record["mat_p"],
            "cov_r": record["cov_r"][at_half],
            "cov_p": record["cov_p"][at_half],
            "chirality": record["chirality"]["fraction"],
        }
    return rows


def paired_bootstrap(delta_by_id, metric, n_boot=10000, seed=0):
    ids = sorted(delta_by_id)
    values = np.array([delta_by_id[i][metric] for i in ids])
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return {
        "molecules": n,
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "fraction_better": float((values < 0).mean()),
        "fraction_tied": float((values == 0).mean()),
    }


def pooled_paired_bootstrap(method_rows, baseline_rows, metric, n_boot=10000, seed=0):
    """Average per-molecule deltas across seeds, then bootstrap molecules."""
    common = None
    for seed in SEEDS:
        ids = (
            set(method_rows[seed]) & set(baseline_rows[seed])
            if baseline_rows[seed]
            else set(method_rows[seed])
        )
        common = ids if common is None else (common & ids)
    ids = sorted(common)
    per_seed_delta = {
        seed: {
            i: method_rows[seed][i][metric] - baseline_rows[seed][i][metric]
            for i in ids
            if method_rows[seed][i][metric] is not None
            and baseline_rows[seed][i][metric] is not None
        }
        for seed in SEEDS
    }
    # Keep molecules defined for every seed (chirality is None when a molecule
    # has no specified tetrahedral center).
    ids = sorted(
        i for i in ids if all(i in per_seed_delta[seed] for seed in SEEDS)
    )
    stacked = np.array(
        [[per_seed_delta[seed][i] for i in ids] for seed in SEEDS]
    )  # [seed, molecule]
    mean_delta = stacked.mean(axis=0)
    rng = np.random.default_rng(seed)
    n = len(ids)
    draws = mean_delta[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    # Seed-level variability of the paired mean (training-seed std).
    seed_means = stacked.mean(axis=1)
    return {
        "molecules": n,
        "seeds": list(SEEDS),
        "mean": float(mean_delta.mean()),
        "ci95_molecule_bootstrap": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
        "per_seed_mean": [float(m) for m in seed_means],
        "seed_std_of_paired_mean": float(seed_means.std(ddof=1)),
        "fraction_molecules_better": float((mean_delta < 0).mean()),
    }


def main():
    result = {"schema_version": 1, "reference_mapping_audit": reference_mapping_audit()}

    comparisons = {}
    for split in ("test", "val"):
        rows = {
            method: {
                seed: per_molecule_metrics(
                    EVAL / f"{split}/{method}-seed{seed}/evaluation.json"
                )
                for seed in SEEDS
            }
            for method in METHODS
        }
        split_result = {}
        for candidate, baseline in (
            ("alternative_a", "control"),
            ("precision", "control"),
        ):
            pair = {}
            for metric in ("mat_r", "mat_p", "cov_r", "cov_p", "chirality"):
                pair[metric] = pooled_paired_bootstrap(
                    rows[candidate], rows[baseline], metric
                )
            split_result[f"{candidate}-{baseline}"] = pair
        comparisons[split] = split_result
    result["pooled_paired_comparisons"] = comparisons

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"output": str(OUT), "splits": list(comparisons)}))


if __name__ == "__main__":
    main()
