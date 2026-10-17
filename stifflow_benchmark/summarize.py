"""Compare saved reports while refusing mismatched controlled experiments."""

import argparse
import copy
import json
from pathlib import Path

from stifflow.preflight import fingerprint


def _normalized_training_data(training_data, normalize_staged_labels):
    """Separate canonical data identity from per-run staged manifest paths.

    Training staged the byte-identical packed label cache onto node-local
    storage and rewrote only absolute ``path`` fields.  The frozen
    ``official_data`` hash is the canonical identity; the per-run ``train`` /
    ``val`` manifest hashes are retained as evidence instead of being compared.
    """
    data = copy.deepcopy(training_data)
    staged = {}
    if normalize_staged_labels:
        missing = [key for key in ("train", "val") if key not in data]
        if missing:
            raise ValueError(
                f"Training data lacks staged manifest hashes: {', '.join(missing)}"
            )
        staged = {key: data.pop(key) for key in ("train", "val")}
    return data, staged


def summarize(paths, normalize_staged_labels=False):
    rows = []
    controlled_signature = None
    for path in paths:
        report = json.loads(Path(path).read_text())
        run = report["run"]
        staged = None
        if run["baseline_kind"] == "controlled":
            if not run.get("training_data") or not run.get("input_manifest"):
                raise ValueError(
                    "Controlled comparison requires training and reference manifest identities"
                )
            configuration = copy.deepcopy(run["config"])
            model = dict(configuration.get("model", {}))
            model.pop("method", None)
            configuration["model"] = model
            training_data, staged = _normalized_training_data(
                run["training_data"], normalize_staged_labels
            )
            signature = {
                "ids": [r["id"] for r in report["records"]],
                "split": run["split"],
                "seed": run["seed"],
                "training_budget": run["training_budget"],
                "training_data": training_data,
                "reference_manifest": run["input_manifest"]["sha256"],
                "config": configuration,
                "protocol": report["protocol"],
            }
            if controlled_signature is not None and signature != controlled_signature:
                raise ValueError(
                    "Controlled runs differ in IDs, split, seed, budget, model/source, sampler or evaluation protocol"
                )
            controlled_signature = signature
        row = {
            "method": run["method"],
            "baseline_kind": run["baseline_kind"],
            "report": fingerprint(path),
            "training_budget": run["training_budget"],
            "generation": run["generation"],
            "aggregate": report["aggregate"],
        }
        if staged is not None:
            row["staged_label_manifests"] = staged
        rows.append(row)
    if not rows:
        raise ValueError("No evaluation reports")
    result = {
        "runs": rows,
        "comparison": "External pretrained baselines remain separate; no automatic superiority claim",
    }
    if normalize_staged_labels:
        result["data_identity"] = {
            "canonical": "training_data.official_data (frozen data manifest)",
            "normalization": "per-run staged train/val manifest path hashes are excluded from equality and recorded per run",
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--normalize-staged-labels",
        action="store_true",
        help="compare frozen data identity while recording per-run staged train/val hashes",
    )
    args = parser.parse_args()
    report = summarize(
        args.reports, normalize_staged_labels=args.normalize_staged_labels
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
