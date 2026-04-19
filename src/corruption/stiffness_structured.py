"""Heuristic stiffness-structured Cartesian source corruption.

This is a smoke-test proxy for the source-construction intervention. It uses
record metadata to perturb protected ring/amide atoms less than flexible atoms,
without claiming to be a complete internal-coordinate corruption implementation.
"""

from __future__ import annotations

from typing import Any

from src.corruption.base import Corruptor
from src.utils.geometry import Coordinates


class StiffnessStructuredCorruptor(Corruptor):
    def __init__(self, config: dict[str, Any], seed: int) -> None:
        super().__init__(config, seed)
        corruption = config["corruption"]
        self.bond_sigma = float(corruption["bond_sigma"])
        self.angle_sigma = float(corruption["angle_sigma"])
        self.torsion_sigma = float(corruption["torsion_sigma"])
        self.protect_rings = bool(corruption["protect_rings"])
        self.protect_amides = bool(corruption["protect_amides"])

    def corrupt(self, record: dict[str, Any], coords: Coordinates) -> Coordinates:
        sigmas = self.atom_sigmas(record, len(coords))
        corrupted: Coordinates = []
        for xyz, sigma in zip(coords, sigmas):
            corrupted.append([float(value) + self.rng.gauss(0.0, sigma) for value in xyz])
        return corrupted

    def atom_sigmas(self, record: dict[str, Any], atom_count: int) -> list[float]:
        ring_atoms = set(record.get("ring_atom_indices", [])) if self.protect_rings else set()
        amide_atoms = set(record.get("amide_atom_indices", [])) if self.protect_amides else set()
        degrees = _atom_degrees(record.get("bonds", []), atom_count)

        sigmas: list[float] = []
        for atom_index in range(atom_count):
            if atom_index in ring_atoms or atom_index in amide_atoms:
                sigmas.append(self.bond_sigma)
            elif degrees[atom_index] > 1:
                sigmas.append(self.angle_sigma)
            else:
                sigmas.append(self.torsion_sigma)
        return sigmas


def _atom_degrees(bonds: list[list[int]], atom_count: int) -> list[int]:
    degrees = [0 for _ in range(atom_count)]
    for bond in bonds:
        i, j = int(bond[0]), int(bond[1])
        degrees[i] += 1
        degrees[j] += 1
    return degrees
