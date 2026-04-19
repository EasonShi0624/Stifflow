"""Isotropic Cartesian source corruption."""

from __future__ import annotations

from typing import Any

from src.corruption.base import Corruptor
from src.utils.geometry import Coordinates


class IsotropicCorruptor(Corruptor):
    def __init__(self, config: dict[str, Any], seed: int) -> None:
        super().__init__(config, seed)
        self.sigma = float(config["corruption"]["isotropic_sigma"])

    def corrupt(self, record: dict[str, Any], coords: Coordinates) -> Coordinates:
        return [
            [float(value) + self.rng.gauss(0.0, self.sigma) for value in xyz]
            for xyz in coords
        ]
