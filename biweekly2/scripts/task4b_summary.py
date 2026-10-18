"""Task 4b summary: learning-curve table for A/control/precision on monitor."""

import json
from pathlib import Path

WS = Path("/gpfsnyu/scratch/ys6132/stifflow")
EVAL = WS / "runs/formal-eval-20261018"
BW = WS / "runs/biweekly2-20261019"
OUT = BW / "results/task4b-learning-curves.json"
METHODS = ("precision", "alternative_a", "control")


def read(path):
    report = json.loads(Path(path).read_text())
    thresholds = report["protocol"]["thresholds"]
    at_half = thresholds.index(0.5)
    records = [r for r in report["records"] if r.get("status") == "ok"]
    return {
        "attempted": report["aggregate"]["attempted"],
        "succeeded": report["aggregate"]["succeeded"],
        "failed": report["aggregate"]["failed"],
        "mat_r_mean": report["aggregate"]["successful_only"]["mat_r_mean"],
        "cov_r_at_0.5": report["aggregate"]["successful_only"]["cov_r_mean"][at_half],
        "mat_p_mean": report["aggregate"]["successful_only"]["mat_p_mean"],
        "chirality": report["aggregate"]["chirality"]["fraction"],
    }


def main():
    result = {"schema_version": 1, "seed": 42, "methods": {}}
    curves = {
        "precision": {
            25000: EVAL / "monitor-snapshots/precision-seed42-step025000/evaluation.json",
            50000: EVAL / "monitor-snapshots/precision-seed42-step050000/evaluation.json",
            100000: EVAL / "monitor-snapshots/precision-seed42-step100000/evaluation.json",
            175000: EVAL / "monitor-snapshots/precision-seed42-step175000/evaluation.json",
            250000: EVAL / "monitor/precision-seed42/evaluation.json",
        },
        "control": {
            25000: EVAL / "monitor-snapshots/control-seed42-step025000/evaluation.json",
            50000: BW / "learning-curves/control-seed42-step050000/evaluation.json",
            100000: BW / "learning-curves/control-seed42-step100000/evaluation.json",
            175000: BW / "learning-curves/control-seed42-step175000/evaluation.json",
            250000: EVAL / "monitor/control-seed42/evaluation.json",
        },
        "alternative_a": {
            25000: BW / "learning-curves/alternative_a-seed42-step025000/evaluation.json",
            50000: BW / "learning-curves/alternative_a-seed42-step050000/evaluation.json",
            100000: BW / "learning-curves/alternative_a-seed42-step100000/evaluation.json",
            175000: BW / "learning-curves/alternative_a-seed42-step175000/evaluation.json",
            250000: EVAL / "monitor/alternative_a-seed42/evaluation.json",
        },
    }
    for method, points in curves.items():
        rows = []
        for step, path in sorted(points.items()):
            if not path.is_file():
                continue
            rows.append({"step": step, "source": str(path), **read(path)})
        result["methods"][method] = rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    for method in METHODS:
        for row in result["methods"][method]:
            print(
                "%-15s %6d  MAT-R %.4f  COV-R %.4f  chir %.4f  ok %d/%d"
                % (
                    method,
                    row["step"],
                    row["mat_r_mean"],
                    row["cov_r_at_0.5"],
                    row["chirality"],
                    row["succeeded"],
                    row["attempted"],
                )
            )


if __name__ == "__main__":
    main()
