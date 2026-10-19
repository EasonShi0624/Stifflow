"""Frozen train-only locality protocol and the gate for the two delayed runs."""

import json

import numpy as np

from .preflight import fingerprint

LOCALITY_PROTOCOL = {
    "molecules": 128,
    "noise_draws": 4,
    "quadrature_nodes": 16,
    "starts": [0.0, 0.25, 0.5, 0.75],
    "seed": 42,
    "relative_error_limit": 0.25,
    "energy_floor_kcal_mol": 0.1,
    "force_floor_kcal_mol_angstrom": 0.1,
    "required_valid_signal_fraction": 0.8,
    "bootstrap": 2000,
    "weight": "quadrature_weight * gdot^2 * ||(L-S)z||^2; equal molecule means",
    "anchor": "original target, actual first derivative, transported raw reference Hessian",
    "coordinates": "angstrom; centered, fixed atom order, proper alignment only",
}


def choose_start(candidates):
    """Select once on train; failures and absent directional evidence fail closed."""
    if [c["stiffness_start"] for c in candidates] != LOCALITY_PROTOCOL["starts"]:
        raise ValueError("Locality candidates differ from the frozen grid")
    if any(
        not np.isfinite(c["valid_signal_fraction"])
        or not 0 <= c["valid_signal_fraction"] <= 1
        or not isinstance(c["failed_queries"], int)
        or c["failed_queries"] < 0
        for c in candidates
    ):
        raise ValueError("Invalid locality fraction/failure count")
    required = LOCALITY_PROTOCOL["required_valid_signal_fraction"]
    if candidates[0]["valid_signal_fraction"] >= required:
        return None
    for candidate in candidates[1:]:
        delta = candidate.get("force_prediction_delta")
        if (
            candidate.get("failed_queries") == 0
            and candidate["valid_signal_fraction"] >= required
            and delta is not None
            and delta["molecules"] == LOCALITY_PROTOCOL["molecules"]
            and np.asarray(delta["ci95"]).shape == (2,)
            and np.isfinite(delta["ci95"]).all()
            and delta["ci95"][0] <= delta["ci95"][1]
            and delta["ci95"][1] < 0
        ):
            return candidate["stiffness_start"]
    return None


def load_locality_report(path, train_labels, source_config):
    report = json.loads(path.read_text())
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "stifflow-locality"
        or report.get("split") != "train"
        or report.get("status") != "completed"
        or report.get("protocol") != LOCALITY_PROTOCOL
        or report.get("source_config") != source_config
        or report.get("train_manifest", {}).get("sha256")
        != fingerprint(train_labels)["sha256"]
    ):
        raise ValueError(
            "Expected a completed frozen locality report for these train labels/source"
        )
    records = report.get("records", [])
    ids = [r["id"] for r in records]
    if (
        len(ids) != LOCALITY_PROTOCOL["molecules"]
        or len(set(ids)) != len(ids)
        or any(r["status"] != "ok" for r in records)
    ):
        raise ValueError("Locality gate requires all 128 selected training molecules")
    selected = choose_start(report["candidates"])
    if selected is None or report.get("selected_start") != selected:
        raise ValueError("Locality gate did not pass; do not launch delayed training")
    return float(selected), fingerprint(path)["sha256"]
