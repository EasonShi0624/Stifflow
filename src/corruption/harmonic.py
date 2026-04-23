"""Graph-harmonic Cartesian source corruption.

This is a minimal source-only proxy for a harmonic-prior / ET-Flow-style
baseline. It keeps the same Cartesian flow objective and backbone, but smooths
isotropic atom noise over the molecular bond graph so neighboring atoms tend to
move coherently. It is not a full reproduction of ET-Flow.
"""

from __future__ import annotations

from typing import Any

from src.corruption.base import Corruptor
from src.utils.geometry import Coordinates


class HarmonicPriorCorruptor(Corruptor):
    def __init__(self, config: dict[str, Any], seed: int) -> None:
        super().__init__(config, seed)
        corruption = config["corruption"]
        self.sigma = float(corruption.get("harmonic_sigma", corruption["isotropic_sigma"]))
        self.smoothing_steps = int(corruption.get("harmonic_smoothing_steps", 2))
        self.neighbor_weight = float(corruption.get("harmonic_neighbor_weight", 0.65))
        self.anchor_weight = float(corruption.get("harmonic_anchor_weight", 0.15))
        if self.sigma < 0.0:
            raise ValueError("harmonic_sigma must be non-negative")
        if self.smoothing_steps < 0:
            raise ValueError("harmonic_smoothing_steps must be non-negative")
        for name, value in [
            ("harmonic_neighbor_weight", self.neighbor_weight),
            ("harmonic_anchor_weight", self.anchor_weight),
        ]:
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def corrupt(self, record: dict[str, Any], coords: Coordinates) -> Coordinates:
        atom_count = len(coords)
        adjacency = _adjacency(record.get("bonds", []), atom_count)
        original_noise = [
            [self.rng.gauss(0.0, self.sigma) for _ in xyz]
            for xyz in coords
        ]
        noise = [list(row) for row in original_noise]
        for _ in range(self.smoothing_steps):
            next_noise = []
            for atom_index, row in enumerate(noise):
                neighbors = adjacency[atom_index]
                if not neighbors:
                    next_noise.append(list(row))
                    continue
                neighbor_mean = [
                    sum(noise[neighbor][axis] for neighbor in neighbors) / len(neighbors)
                    for axis in range(3)
                ]
                smoothed = []
                for axis in range(3):
                    local_average = (
                        (1.0 - self.neighbor_weight) * row[axis]
                        + self.neighbor_weight * neighbor_mean[axis]
                    )
                    smoothed.append(
                        self.anchor_weight * original_noise[atom_index][axis]
                        + (1.0 - self.anchor_weight) * local_average
                    )
                next_noise.append(smoothed)
            noise = next_noise
        return [
            [float(value) + noise[atom_index][axis] for axis, value in enumerate(xyz)]
            for atom_index, xyz in enumerate(coords)
        ]


def _adjacency(bonds: list[list[int | float]], atom_count: int) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(atom_count)]
    for bond in bonds:
        if len(bond) < 2:
            continue
        i, j = int(bond[0]), int(bond[1])
        if i < 0 or j < 0 or i >= atom_count or j >= atom_count or i == j:
            continue
        adjacency[i].append(j)
        adjacency[j].append(i)
    return adjacency
