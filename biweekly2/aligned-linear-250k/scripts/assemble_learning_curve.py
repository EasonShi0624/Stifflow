"""Assemble the seed-42 monitor learning curves (aligned_linear, A, control, precision)."""

import json
from pathlib import Path

import numpy as np

WS = Path("/gpfsnyu/scratch/ys6132/stifflow")
AL = WS / "runs/aligned-linear-250k"
FORMAL = WS / "runs/formal-eval-20261018"
BIW2 = WS / "runs/biweekly2-20261019/learning-curves"

SOURCES = {
    "aligned_linear": [
        AL / f"learning-curves/aligned_linear-seed42-step{step:08d}/evaluation.json"
        for step in range(25000, 250000, 25000)
    ]
    + [AL / "eval/aligned_linear-seed42/monitor/evaluation.json"],
    "alternative_a": [
        BIW2 / f"alternative_a-seed42-step{step:06d}/evaluation.json"
        for step in (25000, 50000, 100000, 175000)
    ]
    + [FORMAL / "monitor/alternative_a-seed42/evaluation.json"],
    "control": [
        FORMAL / "monitor-snapshots/control-seed42-step025000/evaluation.json",
        BIW2 / "control-seed42-step050000/evaluation.json",
        BIW2 / "control-seed42-step100000/evaluation.json",
        BIW2 / "control-seed42-step175000/evaluation.json",
        FORMAL / "monitor/control-seed42/evaluation.json",
    ],
    "precision": [
        FORMAL / f"monitor-snapshots/precision-seed42-step{step:06d}/evaluation.json"
        for step in (25000, 50000, 100000, 175000)
    ]
    + [FORMAL / "monitor/precision-seed42/evaluation.json"],
}


def metrics(path):
    report = json.loads(path.read_text())
    run, protocol = report["run"], report["protocol"]
    if (
        run["split"] != "val"
        or run["config"]["steps"] != 50
        or run["seed"] != 42
        or len(report["records"]) != 512
        or report["aggregate"]["failed"] != 0
        or protocol["generated_reference_ratio"] != 2
    ):
        raise ValueError(f"Unexpected protocol in {path}")
    thresholds = protocol["thresholds"]
    index = [i for i, value in enumerate(thresholds) if abs(value - 0.5) < 1e-12]
    if len(index) != 1:
        raise ValueError(f"No unique 0.5 coverage threshold in {path}")
    index = index[0]
    records = report["records"]
    chir = [r["chirality"]["fraction"] for r in records if r["chirality"]["assessed"]]
    return {
        "budget": run["training_budget"]["steps"],
        "reference": run["input_manifest"]["sha256"],
        "mat_r": float(np.mean([r["mat_r"] for r in records])),
        "mat_p": float(np.mean([r["mat_p"] for r in records])),
        "cov_r_0.5": float(np.mean([r["cov_r"][index] for r in records])),
        "chirality_macro": float(np.mean(chir)) if chir else None,
    }


curves, references = {}, set()
for method, paths in SOURCES.items():
    points = [metrics(p) for p in paths]
    references.update(p["reference"] for p in points)
    curves[method] = {
        str(p["budget"]): {k: v for k, v in p.items() if k not in ("budget", "reference")}
        for p in sorted(points, key=lambda x: x["budget"])
    }
if len(references) != 1:
    raise ValueError(f"Reference manifests differ: {references}")

out = {
    "schema_version": 1,
    "kind": "stifflow-learning-curve-seed42",
    "protocol": "monitor 512 molecules; Euler50; seed42; 2x references; raw generations",
    "reference_manifest": references.pop(),
    "methods": curves,
}
target = AL / "analysis/learning-curve-seed42.json"
target.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
print(target)
for method, points in curves.items():
    print(method, {b: round(v["mat_r"], 4) for b, v in points.items()})
