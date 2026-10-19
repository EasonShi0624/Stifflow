"""Frozen physics screen: five matched 25k validation runs, no test selection."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from stifflow.locality import LOCALITY_PROTOCOL
from stifflow.preflight import fingerprint, provenance
from stifflow.smoke import external_directory

from .research_compare import bootstrap_delta, comparison_signature, label_identity
from .research_diagnostics import select_records
from .stiffness_diagnostics import PHYSICS_PROTOCOL

METHODS = ("alternative_a", "control", "a_rot_null", "a_delayed", "a_delayed_rot_null")


def validate_comparison(geometry, physics, labels=None):
    if set(geometry) != set(METHODS) or set(physics) != set(METHODS):
        raise ValueError(
            "Supply exactly A, control, A rot-null and the two delayed runs"
        )
    if labels is not None and set(labels) != set(METHODS):
        raise ValueError("Supply train/val label manifests for all five runs")
    signatures = {}
    for method in METHODS:
        report, physical = geometry[method], physics[method]
        run = report["run"]
        signature = comparison_signature(report)
        if (
            run["method"] != method
            or run["seed"] != 42
            or run["config"]["steps"] != 50
            or signature["training"]["training_seed"] != 42
            or len(signature["ids"]) != 512
            or report["protocol"]["generated_reference_ratio"] != 2
            or report["protocol"]["force_field_optimization"]
        ):
            raise ValueError(
                "Requires the complete seed42/25k/512/Euler50/raw validation screen"
            )
        if labels is not None:
            for i, split in enumerate(("train", "val")):
                signature["training"][split] = label_identity(
                    labels[method][i], signature["training"][split], split
                )
        signatures[method] = signature
        expected = sorted(r["id"] for r in select_records(report["records"], 64))
        expected_oracle = sorted(
            r["id"] for r in select_records(report["records"], 64)[:16]
        )
        ids = [r["id"] for r in physical["records"]]
        if (
            physical.get("schema_version") != 1
            or physical.get("kind") != "stifflow-physics"
            or physical.get("status") != "completed"
            or physical.get("split") != "val"
            or physical["method"] != method
            or physical["protocol"] != PHYSICS_PROTOCOL
            or physical["run"] != run
            or sorted(ids) != expected
            or len(set(ids)) != 64
            or physical["oracle_ids"] != expected_oracle
        ):
            raise ValueError(
                "Physical reports differ in cohort, protocol, model or generation provenance"
            )
    reference = signatures["alternative_a"]
    if any(s != reference for s in signatures.values()):
        raise ValueError(
            "Runs differ in data, cohort, seed, model, LR, batch or geometry protocol"
        )
    versions = [
        {k: physics[m]["environment"][k] for k in ("numpy", "scipy", "rdkit")}
        for m in METHODS
    ]
    if any(v != versions[0] for v in versions):
        raise ValueError("Physical audit library versions differ")
    left, right = [
        geometry[m]["run"]["config"]["model"]
        for m in ("a_delayed", "a_delayed_rot_null")
    ]
    if (
        left["stiffness_start"] != right["stiffness_start"]
        or left["locality_report_sha256"] != right["locality_report_sha256"]
        or left["stiffness_start"] not in LOCALITY_PROTOCOL["starts"][1:]
    ):
        raise ValueError(
            "Delayed true/null must use the same frozen start and locality report"
        )
    return reference


def force_contrast(left, right, metric="force_median", identities=None):
    ids = sorted(left) if identities is None else sorted(identities)
    if any(
        left[i].get(metric) is None
        or right[i].get(metric) is None
        or (
            metric == "force_median"
            and (left[i]["status"] != "ok" or right[i]["status"] != "ok")
        )
        for i in ids
    ):
        return None
    a, b = np.array([left[i][metric] for i in ids]), np.array(
        [right[i][metric] for i in ids]
    )
    if (
        not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or np.any(a < 0)
        or np.any(b < 0)
    ):
        raise ValueError("Invalid physical force statistic")
    return {
        **bootstrap_delta(a - b),
        "candidate_mean": float(a.mean()),
        "baseline_mean": float(b.mean()),
        "relative_change": (
            float((a.mean() - b.mean()) / b.mean()) if b.mean() > 0 else None
        ),
    }


def geometry_contrasts(candidate, baseline):
    ids = sorted(r["id"] for r in candidate["records"])
    a, b = ({r["id"]: r for r in report["records"]} for report in (candidate, baseline))
    indices = np.flatnonzero(np.isclose(candidate["protocol"]["thresholds"], 0.5))
    if len(indices) != 1:
        raise ValueError("Requires exactly one coverage threshold at 0.5 angstrom")
    index = int(indices[0])
    result = {}
    for metric in ("mat_r", "mat_p", "cov_r", "cov_p", "chirality_macro"):
        pairs = []
        for i in ids:
            if metric == "chirality_macro":
                if a[i]["chirality"]["assessed"] != b[i]["chirality"]["assessed"]:
                    raise ValueError("Chirality assessment cohorts differ")
                if a[i]["chirality"]["assessed"]:
                    pairs.append(
                        (a[i]["chirality"]["fraction"], b[i]["chirality"]["fraction"])
                    )
            elif metric.startswith("cov_"):
                pairs.append((a[i][metric][index], b[i][metric][index]))
            else:
                pairs.append((a[i][metric], b[i][metric]))
        result[metric] = (
            bootstrap_delta(np.subtract(*np.asarray(pairs).T)) if pairs else None
        )
    return result


def screen(geometry, physics, labels=None):
    signature = validate_comparison(geometry, physics, labels)
    rows = {m: {r["id"]: r for r in physics[m]["records"]} for m in METHODS}
    force = {
        m: force_contrast(rows["a_delayed"], rows[m])
        for m in ("alternative_a", "control", "a_delayed_rot_null")
    }
    geom = {
        m: geometry_contrasts(geometry["a_delayed"], geometry[m])
        for m in ("alternative_a", "control")
    }
    criteria = {}
    for m, contrast in force.items():
        criteria[f"force_ci_below_{m}"] = (
            contrast is not None and contrast["ci95"][1] < 0
        )
    value = force["alternative_a"]
    criteria["force_reduction_at_least_10_percent_vs_A"] = (
        value is not None
        and value["relative_change"] is not None
        and value["relative_change"] <= -0.1
    )
    for m, contrasts in geom.items():
        for metric, contrast in contrasts.items():
            criteria[f"geometry_{metric}_vs_{m}"] = contrast is not None and (
                contrast["mean"] <= 0.005
                if metric.startswith("mat_")
                else contrast["mean"] >= -0.01
            )
    complete = all(
        r["status"] == "ok" and r.get("force_median") is not None
        for m in METHODS
        for r in rows[m].values()
    )
    criteria["all_five_physics_cohorts_complete"] = complete
    interaction = None
    if complete:
        interaction = bootstrap_delta(
            [
                (
                    rows["a_delayed"][i]["force_median"]
                    - rows["a_delayed_rot_null"][i]["force_median"]
                )
                - (
                    rows["alternative_a"][i]["force_median"]
                    - rows["a_rot_null"][i]["force_median"]
                )
                for i in sorted(rows["alternative_a"])
            ]
        )
    oracle = {"status": "not_requested", "contrasts": {}}
    requested = [physics[m]["oracle"]["status"] == "requested" for m in METHODS]
    if any(requested):
        binaries = [
            (
                physics[m]["oracle"].get("binary", {}).get("sha256"),
                physics[m]["oracle"].get("version"),
            )
            for m in METHODS
            if physics[m]["oracle"]["status"] == "requested"
        ]
        if (
            any(
                not isinstance(sha, str) or len(sha) != 64 or not version
                for sha, version in binaries
            )
            or len(set(binaries)) != 1
        ):
            raise ValueError(
                "Oracle executable/version provenance is absent or differs"
            )
        oracle_ids = physics["alternative_a"]["oracle_ids"]
        if len(oracle_ids) != 16 or any(
            physics[m]["oracle_ids"] != oracle_ids for m in METHODS
        ):
            raise ValueError("Oracle cohorts differ")
        contrasts = {
            m: force_contrast(
                rows["a_delayed"], rows[m], "oracle_force_median", oracle_ids
            )
            for m in ("alternative_a", "control", "a_delayed_rot_null")
        }
        valid = all(requested) and all(v is not None for v in contrasts.values())
        reversed_direction = any(
            v is not None and v["mean"] > 0 for v in contrasts.values()
        )
        oracle = {
            "status": (
                "reversed"
                if reversed_direction
                else ("consistent" if valid else "incomplete")
            ),
            "contrasts": contrasts,
        }
    criteria["no_observed_cross_potential_reversal"] = oracle["status"] != "reversed"
    return {
        "schema_version": 1,
        "kind": "stifflow-locality-screen",
        "split": "val",
        "force_candidate_minus_baseline": force,
        "geometry_candidate_minus_baseline": geom,
        "direction_difference_in_differences": interaction,
        "timing_specific_direction_evidence": interaction is not None
        and interaction["ci95"][1] < 0,
        "oracle": oracle,
        "criteria": criteria,
        "advance_candidate": all(criteria.values()),
        "formal_method_upgrade": False,
        "claim_scope": (
            "MMFF94 plus small GFN2 cross-check"
            if oracle["status"] == "consistent"
            else "MMFF94 only"
        ),
        "verified_signature": signature,
        "bootstrap": {
            "unit": "molecule",
            "replicates": 2000,
            "seed": 42,
            "shared_draws": "same sorted IDs per contrast family",
        },
        "caveats": [
            "Single seed, 25k screening; no across-seed or ET-Flow superiority claim.",
            "Geometry margins are point-estimate screening rules, not a statistical noninferiority proof.",
            "Lower forces do not certify a minimum, basin populations or thermodynamic correctness.",
            "Delayed timing also changes total stiffness signal; interaction alone does not isolate timing.",
        ],
    }


def keyed(values):
    result = {}
    for row in values or []:
        if row[0] in result:
            raise ValueError(f"Duplicate method: {row[0]}")
        result[row[0]] = row[1:]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry",
        nargs=2,
        action="append",
        metavar=("METHOD", "REPORT"),
        required=True,
    )
    parser.add_argument(
        "--physics",
        nargs=2,
        action="append",
        metavar=("METHOD", "REPORT"),
        required=True,
    )
    parser.add_argument(
        "--labels", nargs=3, action="append", metavar=("METHOD", "TRAIN", "VAL")
    )
    parser.add_argument("--output", type=external_directory, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Saved-result aggregation belongs in an SHHPC allocation")
    geometry_paths = {m: Path(v[0]) for m, v in keyed(args.geometry).items()}
    physics_paths = {m: Path(v[0]) for m, v in keyed(args.physics).items()}
    geometry = {m: json.loads(p.read_text()) for m, p in geometry_paths.items()}
    physics = {m: json.loads(p.read_text()) for m, p in physics_paths.items()}
    if set(geometry_paths) != set(physics_paths):
        parser.error("Geometry and physics method sets differ")
    for m in geometry:
        if (
            physics[m]["evaluation"]["sha256"]
            != fingerprint(geometry_paths[m])["sha256"]
        ):
            raise ValueError(
                "Physics report belongs to a different geometry evaluation"
            )
    report = screen(geometry, physics, keyed(args.labels) if args.labels else None)
    report.update(
        provenance=provenance(),
        inputs={
            m: {
                "geometry": fingerprint(geometry_paths[m]),
                "physics": fingerprint(physics_paths[m]),
            }
            for m in geometry
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "advance_candidate": report["advance_candidate"],
                "criteria": report["criteria"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
