"""Basic smoke-test metrics for generated/source coordinates."""

from __future__ import annotations

from typing import Any

from src.utils.geometry import Coordinates, assert_finite_coords, distance, mean, rmsd


def evaluate_smoke_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot evaluate an empty sample list")

    rmsds = []
    bond_deviations = []
    finite_failures = 0

    for sample in samples:
        molecule_id = sample["molecule_id"]
        x0 = sample["x0"]
        x1 = sample["x1"]
        try:
            assert_finite_coords(x0, f"{molecule_id}.x0")
            assert_finite_coords(x1, f"{molecule_id}.x1")
        except ValueError:
            finite_failures += 1
            continue

        rmsds.append(rmsd(x0, x1))
        bond_deviations.extend(_bond_length_deviations(sample["bonds"], x0, x1))

    threshold = 0.5
    cov = sum(1 for value in rmsds if value <= threshold) / len(rmsds) if rmsds else 0.0
    mat = mean(rmsds)

    return {
        "metric_scope": "synthetic_smoke_source_vs_target",
        "sample_count": len(samples),
        "finite_failure_count": finite_failures,
        "cov_mat": {
            "threshold": threshold,
            "cov_r_smoke": cov,
            "mat_r_smoke": mat,
            "note": "Smoke-only RMSD proxy comparing corrupted source x0 to target x1.",
        },
        "validity": {
            "finite_coordinate_rate": (len(samples) - finite_failures) / len(samples),
            "mean_abs_bond_length_delta": mean(abs(value) for value in bond_deviations),
            "note": "Intramolecular smoke checks only; not PoseBusters/MMFF/UFF.",
        },
    }


def _bond_length_deviations(
    bonds: list[list[int]], coords_a: Coordinates, coords_b: Coordinates
) -> list[float]:
    deviations = []
    for bond in bonds:
        i, j = int(bond[0]), int(bond[1])
        deviations.append(distance(coords_a, i, j) - distance(coords_b, i, j))
    return deviations
