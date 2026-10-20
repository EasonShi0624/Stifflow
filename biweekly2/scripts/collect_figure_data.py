"""Assemble figure-ready datasets from existing + new biweekly-2 results.

Produces runs/biweekly2-20261019/results/figure-data.json, a compact JSON that
the plotting script consumes.  Pure CPU aggregation of finished analyses.
"""

import json
from pathlib import Path

import numpy as np

WS = Path("/gpfsnyu/scratch/ys6132/stifflow")
EVAL = WS / "runs/formal-eval-20261018"
BW = WS / "runs/biweekly2-20261019"
METHODS = ("alternative_a", "control")
SEEDS = (42, 43, 44)


def metric_at_half(report, record):
    thresholds = report["protocol"]["thresholds"]
    at_half = thresholds.index(0.5)
    return {
        "mat_r": record["mat_r"],
        "mat_p": record["mat_p"],
        "cov_r": record["cov_r"][at_half],
        "cov_p": record["cov_p"][at_half],
        "chirality": record["chirality"]["fraction"],
    }


def formal_metrics(split):
    """Per-seed aggregate metrics from the frozen formal evaluation."""
    out = {}
    for method in METHODS:
        per_seed = {}
        for seed in SEEDS:
            report = json.loads(
                (EVAL / f"{split}/{method}-seed{seed}/evaluation.json").read_text()
            )
            records = [r for r in report["records"] if r.get("status") == "ok"]
            values = [metric_at_half(report, r) for r in records]
            per_seed[seed] = {
                key: float(np.mean([v[key] for v in values if v[key] is not None]))
                for key in values[0]
            }
            # macro chirality: molecule-mean fraction (None excluded), matching
            # the paired-comparison convention.
            fractions = [
                r["chirality"]["fraction"]
                for r in records
                if r["chirality"]["fraction"] is not None
            ]
            per_seed[seed]["chirality_macro"] = float(np.mean(fractions))
        keys = list(per_seed[SEEDS[0]])
        means = {
            key: float(np.mean([per_seed[s][key] for s in SEEDS])) for key in keys
        }
        stds = {
            key: float(np.std([per_seed[s][key] for s in SEEDS], ddof=1))
            for key in keys
        }
        out[method] = {"per_seed": per_seed, "mean": means, "seed_std": stds}
    return out


def learning_curve():
    """Monitor MAT-R/COV-R/chirality vs training step for the three methods."""
    curve = {m: [] for m in METHODS}

    def read(path):
        report = json.loads(Path(path).read_text())
        thresholds = report["protocol"]["thresholds"]
        at_half = thresholds.index(0.5)
        records = [r for r in report["records"] if r.get("status") == "ok"]
        return {
            "mat_r": float(np.mean([r["mat_r"] for r in records])),
            "cov_r": float(np.mean([r["cov_r"][at_half] for r in records])),
            "chirality": report["aggregate"]["chirality"]["fraction"],
            "n": len(records),
        }

    # control: existing 25k + 250k + new snapshots
    curve["control"].append(
        {
            "step": 25000,
            **read(EVAL / "monitor-snapshots/control-seed42-step025000/evaluation.json"),
        }
    )
    for step in (50000, 100000, 175000):
        path = BW / f"learning-curves/control-seed42-step{step:06d}/evaluation.json"
        if path.is_file():
            curve["control"].append({"step": step, **read(path)})
    curve["control"].append(
        {"step": 250000, **read(EVAL / "monitor/control-seed42/evaluation.json")}
    )
    # alternative A: new snapshots + existing 250k
    for step in (25000, 50000, 100000, 175000):
        path = BW / f"learning-curves/alternative_a-seed42-step{step:06d}/evaluation.json"
        if path.is_file():
            curve["alternative_a"].append({"step": step, **read(path)})
    curve["alternative_a"].append(
        {"step": 250000, **read(EVAL / "monitor/alternative_a-seed42/evaluation.json")}
    )
    return curve


def per_t_error():
    data = json.loads((EVAL / "analysis/per-t-error.json").read_text())
    out = {}
    for method in METHODS:
        bins = data["methods"][method]["bins"]
        out[method] = {
            "t_mid": [(b["t_low"] + b["t_high"]) / 2 for b in bins],
            "target_energy": [b["target_energy_mean"] for b in bins],
            "mse": [b["mse_mean"] for b in bins],
            "r2": [b["r2"] for b in bins],
            "overall_mse": data["methods"][method]["overall"]["mse_mean"],
        }
    return out


def p0_paths():
    out = {}
    for method in METHODS:
        data = json.loads(
            (WS / f"runs/research-20261018/p0/paths-{method}.json").read_text()
        )
        out[method] = {
            "times": data["times"],
            "target_energy": data["mean_per_molecule_target_energy"],
            "velocity_rms": data["velocity_rms"],
        }
    return out


def main():
    result = {
        "schema_version": 1,
        "test_metrics": formal_metrics("test"),
        "val_metrics": formal_metrics("val"),
        "learning_curve": learning_curve(),
        "per_t_error": per_t_error(),
        "p0_paths": p0_paths(),
    }
    optional = {
        "task1_caliber": BW / "results/task1-caliber.json",
        "task2a_gaussian": BW / "results/task2a-conditional-gaussian.json",
        "task2b_equivariance": BW / "results/task2b-equivariance.json",
        "task3_physical_quality": BW / "results/task3-physical-quality.json",
        "task3b_strain_chirality": BW / "results/task3b-strain-chirality.json",
        "task4a_signal": BW / "results/task4a-stiffness-signal.json",
        "task4b_learning_curves": BW / "results/task4b-learning-curves.json",
    }
    for key, path in optional.items():
        if path.is_file():
            result[key] = json.loads(path.read_text())
        else:
            result[key] = None
    out = BW / "results/figure-data.json"
    if out.exists():
        out.unlink()
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"output": str(out), "parts": list(result)}))


if __name__ == "__main__":
    main()
