"""Compare saved reports while refusing mismatched controlled experiments."""

import argparse
import json
from pathlib import Path

from stifflow.preflight import fingerprint


def summarize(paths):
    rows = []
    controlled_signature = None
    for path in paths:
        report = json.loads(Path(path).read_text())
        run = report["run"]
        if run["baseline_kind"] == "controlled":
            if not run.get("training_data") or not run.get("input_manifest"):
                raise ValueError(
                    "Controlled comparison requires training and reference manifest identities"
                )
            configuration = dict(run["config"])
            model = dict(configuration.get("model", {}))
            model.pop("method", None)
            configuration["model"] = model
            signature = {
                "ids": [r["id"] for r in report["records"]],
                "split": run["split"],
                "seed": run["seed"],
                "training_budget": run["training_budget"],
                "training_data": run["training_data"],
                "reference_manifest": run["input_manifest"]["sha256"],
                "config": configuration,
                "protocol": report["protocol"],
            }
            if controlled_signature is not None and signature != controlled_signature:
                raise ValueError(
                    "Controlled runs differ in IDs, split, seed, budget, model/source, sampler or evaluation protocol"
                )
            controlled_signature = signature
        rows.append(
            {
                "method": run["method"],
                "baseline_kind": run["baseline_kind"],
                "report": fingerprint(path),
                "training_budget": run["training_budget"],
                "generation": run["generation"],
                "aggregate": report["aggregate"],
            }
        )
    if not rows:
        raise ValueError("No evaluation reports")
    return {
        "runs": rows,
        "comparison": "External pretrained baselines remain separate; no automatic superiority claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
