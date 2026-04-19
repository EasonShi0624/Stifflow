"""Shared source-corruption interface."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from src.utils.geometry import Coordinates


class Corruptor(ABC):
    def __init__(self, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.rng = random.Random(seed)

    @abstractmethod
    def corrupt(self, record: dict[str, Any], coords: Coordinates) -> Coordinates:
        """Return a source geometry x0 for target coordinates x1."""


def build_corruptor(config: dict[str, Any], seed: int) -> Corruptor:
    mode = config["corruption"]["mode"]
    if mode == "isotropic":
        from src.corruption.isotropic import IsotropicCorruptor

        return IsotropicCorruptor(config, seed)
    if mode == "stiffness_structured":
        from src.corruption.stiffness_structured import StiffnessStructuredCorruptor

        return StiffnessStructuredCorruptor(config, seed)
    raise ValueError(f"Unsupported corruption mode: {mode}")
