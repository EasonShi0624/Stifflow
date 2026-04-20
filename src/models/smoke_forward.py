"""Parameter-free CPU model execution smoke helpers."""

from __future__ import annotations

import math
from typing import Any

from src.utils.geometry import Coordinates, assert_finite_coords, mean


class ParameterFreeVelocitySmokeModel:
    """A tiny deterministic velocity head for forward/loss smoke tests.

    This is not a trainable backbone. It accepts the same basic geometric inputs
    a Cartesian flow-matching model would see and emits a finite velocity field
    so the model-side execution/loss interface can be validated without training.
    """

    name = "parameter_free_velocity_smoke"
    backend = "python_float_lists_cpu"
    trainable_parameter_count = 0

    def forward(self, sample: dict[str, Any]) -> Coordinates:
        xt = [[float(value) for value in xyz] for xyz in sample["xt"]]
        assert_finite_coords(xt, f"{sample['molecule_id']}.xt")
        t = float(sample["t"])
        centroid = _centroid(xt)
        scale = 0.1 * (1.0 - t)
        prediction = [
            [scale * (center - value) for value, center in zip(xyz, centroid)]
            for xyz in xt
        ]
        assert_finite_coords(prediction, f"{sample['molecule_id']}.predicted_velocity")
        return prediction


def evaluate_velocity_forward_loss(
    samples: list[dict[str, Any]],
    model: ParameterFreeVelocitySmokeModel | None = None,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot run model execution smoke on an empty sample list")
    model = model or ParameterFreeVelocitySmokeModel()
    per_sample = []
    squared_errors = []
    absolute_errors = []

    for sample in samples:
        predicted = model.forward(sample)
        target = [[float(value) for value in xyz] for xyz in sample["target_velocity"]]
        assert_finite_coords(target, f"{sample['molecule_id']}.target_velocity")
        if len(predicted) != len(target):
            raise ValueError(f"Velocity atom-count mismatch for {sample['molecule_id']}")

        sample_squared = []
        sample_absolute = []
        for pred_xyz, target_xyz in zip(predicted, target):
            if len(pred_xyz) != len(target_xyz):
                raise ValueError(f"Velocity coordinate mismatch for {sample['molecule_id']}")
            for pred_value, target_value in zip(pred_xyz, target_xyz):
                delta = float(pred_value) - float(target_value)
                sample_squared.append(delta * delta)
                sample_absolute.append(abs(delta))

        squared_errors.extend(sample_squared)
        absolute_errors.extend(sample_absolute)
        per_sample.append(
            {
                "molecule_id": sample["molecule_id"],
                "split": sample.get("split"),
                "atom_count": len(predicted),
                "component_count": len(sample_squared),
                "velocity_mse": mean(sample_squared),
                "velocity_mae": mean(sample_absolute),
                "predicted_velocity_norm": _coordinate_norm(predicted),
                "target_velocity_norm": _coordinate_norm(target),
            }
        )

    return {
        "model_name": model.name,
        "backend": model.backend,
        "trainable_parameter_count": model.trainable_parameter_count,
        "forward_pass_count": 1,
        "sample_count": len(samples),
        "component_count": len(squared_errors),
        "mean_velocity_mse": mean(squared_errors),
        "mean_velocity_mae": mean(absolute_errors),
        "max_velocity_abs_error": max(absolute_errors),
        "per_sample": per_sample,
        "note": "Parameter-free CPU forward/loss smoke metrics only; not training and not benchmark metrics.",
    }


def _centroid(coords: Coordinates) -> list[float]:
    atom_count = len(coords)
    if atom_count == 0:
        raise ValueError("Cannot compute centroid for empty coordinates")
    return [sum(float(xyz[axis]) for xyz in coords) / atom_count for axis in range(3)]


def _coordinate_norm(coords: Coordinates) -> float:
    squared = 0.0
    for xyz in coords:
        for value in xyz:
            squared += float(value) * float(value)
    return math.sqrt(squared)
