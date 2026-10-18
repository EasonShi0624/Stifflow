"""Stage D: exploratory (non-gating) comparisons + screening summary.

Runs the four preregistered gating comparisons via run_compare.sh logic is
kept separate: this script only ADDS the exploratory pairs
(nulls vs control-25k, linear/aligned_linear vs A-25k) and assembles the
final screening summary (gating verdicts, chirality localization, strain).

Gating comparisons themselves come from run_compare.sh; this script refuses
to duplicate them.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

WS = Path("/gpfsnyu/scratch/ys6132/stifflow")
ROOT = WS / "runs/research-20261018"
PY = "/gpfsnyu/home/ys6132/anaconda3/envs/etflow/bin/python"
CAN_TRAIN = WS / "cache/qm9-v3-packed-strict/train/manifest.json"
CAN_VAL = WS / "cache/qm9-v3-packed-strict/val/manifest.json"
A_TRAIN = ROOT / "identity/alternative_a-seed42/train.json"
A_VAL = ROOT / "identity/alternative_a-seed42/val.json"
C_TRAIN = ROOT / "identity/control-seed42/train.json"
C_VAL = ROOT / "identity/control-seed42/val.json"
BASE_A = ROOT / "generated/alternative_a-25k/evaluation.json"
BASE_C = (
    WS / "runs/formal-eval-20261018/monitor-snapshots/control-seed42-step025000/evaluation.json"
)
OUT = ROOT / "compare/screening-summary.json"

EXPLORATORY = {
    "a_rot_null-25k-vs-control-25k": ("a_rot_null-25k", BASE_C, C_TRAIN, C_VAL),
    "a_trace_null-25k-vs-control-25k": ("a_trace_null-25k", BASE_C, C_TRAIN, C_VAL),
    "linear-25k-vs-alternative_a-25k": ("linear-25k", BASE_A, A_TRAIN, A_VAL),
    "aligned_linear-25k-vs-alternative_a-25k": (
        "aligned_linear-25k",
        BASE_A,
        A_TRAIN,
        A_VAL,
    ),
}
GATING = {
    "a_rot_null-25k": "a_rot_null-25k-vs-alternative_a-25k",
    "a_trace_null-25k": "a_trace_null-25k-vs-alternative_a-25k",
    "linear-25k": "linear-25k-vs-control-seed42-step025000",
    "aligned_linear-25k": "aligned_linear-25k-vs-control-seed42-step025000",
}


def run_exploratory():
    outputs = {}
    for name, (cand, baseline, btrain, bval) in EXPLORATORY.items():
        out = ROOT / f"compare/{name}-paired.json"
        if out.exists():
            outputs[name] = json.loads(out.read_text())
            continue
        cmd = [
            sys.executable,
            "-m",
            "stifflow_benchmark.research_compare",
            str(ROOT / f"generated/{cand}/evaluation.json"),
            str(baseline),
            "--label-manifests",
            str(CAN_TRAIN),
            str(btrain),
            str(CAN_VAL),
            str(bval),
            "--output",
            str(out),
        ]
        subprocess.run(cmd, check=True)
        outputs[name] = json.loads(out.read_text())
    return outputs


def evaluate_gates():
    verdicts = {}
    for cand, gate_file in GATING.items():
        path = ROOT / "compare" / f"{gate_file}-paired.json"
        if not path.is_file():
            verdicts[cand] = {"missing": True}
            continue
        report = json.loads(path.read_text())
        verdicts[cand] = {
            "criteria": report["criteria"],
            "advance_to_second_seed": report["advance_to_second_seed"],
            "relative_mat_p_change": report["relative_mat_p_change"],
            "deltas": report["deltas_candidate_minus_baseline"],
        }
    return verdicts


def chirality_localization():
    """A vs control at 25k: is A's small chirality deficit already present?"""
    a = json.loads(BASE_A.read_text())
    c = json.loads(BASE_C.read_text())
    left = {r["id"]: r for r in a["records"]}
    right = {r["id"]: r for r in c["records"]}
    pairs = []
    for mid in sorted(left):
        ra, rb = left[mid]["chirality"], right[mid]["chirality"]
        if ra["assessed"] and rb["assessed"]:
            pairs.append((ra["fraction"], rb["fraction"]))
    values = np.asarray(pairs)
    delta = values[:, 0] - values[:, 1]
    rng = np.random.default_rng(42)
    draws = [
        float(delta[rng.integers(len(delta), size=len(delta))].mean())
        for _ in range(2000)
    ]
    return {
        "molecules": len(delta),
        "A_minus_control_macro_chirality": float(delta.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def main():
    exploratory = run_exploratory()
    gates = evaluate_gates()
    strain_path = ROOT / "compare/strain-25k-candidates.json"
    strain = json.loads(strain_path.read_text()) if strain_path.is_file() else None
    result = {
        "schema_version": 1,
        "gating_comparisons": gates,
        "exploratory_comparisons": {
            name: {
                "deltas": report["deltas_candidate_minus_baseline"],
                "means": report["means"],
                "note": "exploratory only; not a gating comparison",
            }
            for name, report in exploratory.items()
        },
        "chirality_localization_A_vs_control_25k": chirality_localization(),
        "strain_25k": strain["methods"] if strain else None,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result["chirality_localization_A_vs_control_25k"], indent=1))
    for cand, verdict in gates.items():
        print(cand, "advance:", verdict.get("advance_to_second_seed"))


if __name__ == "__main__":
    main()
