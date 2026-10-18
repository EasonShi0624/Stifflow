"""Paired validation-only 25k screening; no test-set selection or superiority claim."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from stifflow.preflight import fingerprint
from stifflow.smoke import external_directory


def comparison_signature(report):
    run, protocol = report["run"], report["protocol"]
    if (
        run["split"] != "val"
        or run["baseline_kind"] != "controlled"
        or run["training_budget"]["steps"] != 25000
    ):
        raise ValueError(
            "Screening requires controlled 25000-update validation reports"
        )
    training = run["training_data"]
    cfg = training["config"]
    profile = cfg["profiles"][training["profile"]]
    model = dict(run["config"]["model"])
    model.pop("method")
    model.pop("path_clock", None)
    configuration = dict(run["config"])
    configuration["model"] = model
    records = report["records"]
    ids = [r["id"] for r in records]
    if (
        not ids
        or len(ids) != len(set(ids))
        or any(r["status"] != "ok" for r in records)
    ):
        raise ValueError(
            "Incomplete/failed/duplicate cohorts cannot pass a screening gate"
        )
    if run.get("generation", {}).get("expected", len(ids)) != len(ids):
        raise ValueError("Missing attempted molecules")
    return {
        "ids": sorted(ids),
        "protocol": protocol,
        "config": configuration,
        "generation_seed": run["seed"],
        "reference": run["input_manifest"]["sha256"],
        "training": {
            key: training[key]
            for key in (
                "train",
                "val",
                "official_data",
                "sampling",
                "dataset_seed",
                "training_seed",
                "row_rng_domain",
                "effective_batch_size",
            )
        },
        "source": cfg["source"],
        "model_args": cfg["model_args"],
        "microbatch": cfg["batch_size"],
        "accumulation": profile["accumulate_grad_batches"],
    }


def label_identity(path, expected_sha, split):
    """Compare staged/canonical manifests by content, never by data split alone."""
    identity = fingerprint(path)
    if identity["sha256"] != expected_sha:
        raise ValueError("Label manifest does not match the checkpoint's recorded hash")
    manifest = json.loads(Path(path).read_text())
    if (
        manifest.get("schema_version") != 3
        or manifest["split"] != split
        or manifest.get("status") != "completed"
        or manifest.get("all_candidates_processed") is not True
    ):
        raise ValueError("Expected a matching version-3 packed split")

    def without_paths(value):
        if isinstance(value, dict):
            return {k: without_paths(v) for k, v in value.items() if k != "path"}
        if isinstance(value, list):
            return [without_paths(v) for v in value]
        return value

    content = without_paths(
        {
            key: manifest[key]
            for key in (
                "data_manifest",
                "dataset_kind",
                "split",
                "sampling",
                "source_config",
                "stiffness_config",
                "label_implementation",
                "records",
                "graph_shards",
                "factor_shards",
                "row_index",
            )
        }
    )
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def paired(candidate, baseline, bootstrap=2000, seed=42, label_manifests=None):
    signatures = [comparison_signature(report) for report in (candidate, baseline)]
    equivalence = None
    if label_manifests is not None:
        if len(label_manifests) != 4:
            raise ValueError(
                "Supply candidate/baseline train, then candidate/baseline val manifests"
            )
        equivalence = {}
        for offset, split in enumerate(("train", "val")):
            hashes = [
                label_identity(
                    label_manifests[2 * offset + i],
                    signatures[i]["training"][split],
                    split,
                )
                for i in range(2)
            ]
            if hashes[0] != hashes[1]:
                raise ValueError(
                    "Accepted rows, cached factor checksums or label implementations differ"
                )
            equivalence[split] = hashes[0]
            for signature in signatures:
                signature["training"][split] = hashes[0]
    if signatures[0] != signatures[1]:
        raise ValueError(
            "Runs differ in cohort, seed, data, model, LR, batch or evaluation protocol"
        )
    if not isinstance(bootstrap, int) or not 100 <= bootstrap <= 10000:
        raise ValueError("Use 100..10000 molecule bootstrap replicates")
    indices = np.flatnonzero(np.isclose(candidate["protocol"]["thresholds"], 0.5))
    if len(indices) != 1:
        raise ValueError("Requires exactly one COV@0.5 threshold")
    index = int(indices[0])
    left = {r["id"]: r for r in candidate["records"]}
    right = {r["id"]: r for r in baseline["records"]}
    rng = np.random.default_rng(seed)
    deltas, means = {}, {}
    for metric in ("mat_p", "mat_r", "cov_r", "chirality_macro"):
        pairs = []
        for identity in sorted(left):
            a, b = left[identity], right[identity]
            if metric == "chirality_macro":
                if a["chirality"]["assessed"] != b["chirality"]["assessed"]:
                    raise ValueError("Chirality assessment cohorts differ")
                if not a["chirality"]["assessed"]:
                    continue
                values = a["chirality"]["fraction"], b["chirality"]["fraction"]
            else:
                values = (
                    (a[metric][index], b[metric][index])
                    if metric == "cov_r"
                    else (a[metric], b[metric])
                )
            pairs.append(values)
        if not pairs:
            deltas[metric], means[metric] = None, None
            continue
        values = np.asarray(pairs, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite comparison metrics")
        delta = values[:, 0] - values[:, 1]
        # Resample molecules, not conformers or pooled seed rows.
        draws = [
            float(delta[rng.integers(len(delta), size=len(delta))].mean())
            for _ in range(bootstrap)
        ]
        deltas[metric] = {
            "mean": float(delta.mean()),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "molecules": len(delta),
        }
        means[metric] = {
            "candidate": float(values[:, 0].mean()),
            "baseline": float(values[:, 1].mean()),
        }
    relative = (
        deltas["mat_p"]["mean"] / means["mat_p"]["baseline"]
        if means["mat_p"]["baseline"] > 0
        else None
    )
    stereo = deltas["chirality_macro"]
    criteria = {
        "mat_p_reduction_at_least_5_percent": relative is not None
        and relative <= -0.05,
        "mat_p_difference_ci_excludes_zero": deltas["mat_p"]["ci95"][1] < 0,
        "mat_r_increase_at_most_0.005_angstrom": deltas["mat_r"]["mean"] <= 0.005,
        "cov_r_drop_at_most_0.01": deltas["cov_r"]["mean"] >= -0.01,
        "macro_chirality_drop_at_most_0.01": stereo is not None
        and stereo["mean"] >= -0.01,
    }
    return {
        "candidate": candidate["run"]["method"],
        "baseline": baseline["run"]["method"],
        "verified_label_content": equivalence,
        "deltas_candidate_minus_baseline": deltas,
        "means": means,
        "relative_mat_p_change": relative,
        "criteria": criteria,
        "advance_to_second_seed": all(criteria.values()),
        "bootstrap": {"unit": "molecule", "replicates": bootstrap, "seed": seed},
        "caveat": "A preregistered screening rule for one matched training seed. This does not establish across-seed benefit; missing stereo evidence fails the gate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument(
        "--label-manifests",
        nargs=4,
        type=Path,
        metavar=("CAND_TRAIN", "BASE_TRAIN", "CAND_VAL", "BASE_VAL"),
    )
    parser.add_argument("--output", type=external_directory, required=True)
    args = parser.parse_args()
    report = paired(
        json.loads(args.candidate.read_text()),
        json.loads(args.baseline.read_text()),
        label_manifests=args.label_manifests,
    )
    report["inputs"] = {
        "candidate": fingerprint(args.candidate),
        "baseline": fingerprint(args.baseline),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
