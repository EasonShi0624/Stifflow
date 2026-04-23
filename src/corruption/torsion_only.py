"""Torsion-focused source corruption.

This source-only baseline rotates one side of eligible single bonds while
leaving bond lengths along the rotation axis unchanged. It is intentionally
minimal and Cartesian-closed: no internal-coordinate reconstruction chain is
introduced.
"""

from __future__ import annotations

import math
from typing import Any

from src.corruption.base import Corruptor
from src.utils.geometry import Coordinates


class TorsionOnlyCorruptor(Corruptor):
    def __init__(self, config: dict[str, Any], seed: int) -> None:
        super().__init__(config, seed)
        corruption = config["corruption"]
        self.torsion_sigma = float(corruption["torsion_sigma"])
        self.protect_rings = bool(corruption.get("protect_rings", True))
        self.max_rotatable_bonds = int(corruption.get("torsion_only_max_bonds", 12))
        if self.torsion_sigma < 0.0:
            raise ValueError("torsion_sigma must be non-negative")
        if self.max_rotatable_bonds < 0:
            raise ValueError("torsion_only_max_bonds must be non-negative")

    def corrupt(self, record: dict[str, Any], coords: Coordinates) -> Coordinates:
        corrupted = [[float(value) for value in xyz] for xyz in coords]
        candidates = _rotatable_candidates(
            record=record,
            atom_count=len(corrupted),
            protect_rings=self.protect_rings,
        )
        self.rng.shuffle(candidates)
        if self.max_rotatable_bonds:
            candidates = candidates[: self.max_rotatable_bonds]

        for i, j, moving_atoms in candidates:
            angle = self.rng.gauss(0.0, self.torsion_sigma)
            if abs(angle) <= 1.0e-12:
                continue
            _rotate_atoms_around_axis(corrupted, i, j, moving_atoms, angle)
        return corrupted


def _rotatable_candidates(
    record: dict[str, Any],
    atom_count: int,
    protect_rings: bool,
) -> list[tuple[int, int, list[int]]]:
    bonds = record.get("bonds", [])
    adjacency = _adjacency(bonds, atom_count)
    ring_atoms = set(int(index) for index in record.get("ring_atom_indices", [])) if protect_rings else set()
    candidates: list[tuple[int, int, list[int]]] = []
    for bond in bonds:
        if len(bond) < 2:
            continue
        i, j = int(bond[0]), int(bond[1])
        if i < 0 or j < 0 or i >= atom_count or j >= atom_count or i == j:
            continue
        bond_order = float(bond[2]) if len(bond) >= 3 else 1.0
        if abs(bond_order - 1.0) > 1.0e-6:
            continue
        if i in ring_atoms and j in ring_atoms:
            continue
        side_i = _component_without_edge(adjacency, start=i, blocked_a=i, blocked_b=j)
        side_j = _component_without_edge(adjacency, start=j, blocked_a=i, blocked_b=j)
        moving_i = sorted(atom for atom in side_i if atom not in {i, j})
        moving_j = sorted(atom for atom in side_j if atom not in {i, j})
        moving_atoms = _smaller_nonempty(moving_i, moving_j)
        if moving_atoms:
            candidates.append((i, j, moving_atoms))
    return candidates


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


def _component_without_edge(
    adjacency: list[list[int]],
    start: int,
    blocked_a: int,
    blocked_b: int,
) -> set[int]:
    seen = {start}
    stack = [start]
    while stack:
        atom = stack.pop()
        for neighbor in adjacency[atom]:
            if {atom, neighbor} == {blocked_a, blocked_b}:
                continue
            if neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return seen


def _smaller_nonempty(left: list[int], right: list[int]) -> list[int]:
    if left and right:
        return left if len(left) <= len(right) else right
    return left or right


def _rotate_atoms_around_axis(
    coords: Coordinates,
    axis_start_index: int,
    axis_end_index: int,
    moving_atoms: list[int],
    angle: float,
) -> None:
    origin = coords[axis_start_index]
    axis_end = coords[axis_end_index]
    axis = [axis_end[axis_index] - origin[axis_index] for axis_index in range(3)]
    norm = math.sqrt(sum(value * value for value in axis))
    if norm <= 1.0e-12:
        return
    unit = [value / norm for value in axis]
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    for atom_index in moving_atoms:
        rel = [coords[atom_index][axis] - origin[axis] for axis in range(3)]
        cross = [
            unit[1] * rel[2] - unit[2] * rel[1],
            unit[2] * rel[0] - unit[0] * rel[2],
            unit[0] * rel[1] - unit[1] * rel[0],
        ]
        dot = sum(unit[axis] * rel[axis] for axis in range(3))
        rotated = [
            rel[axis] * cos_angle
            + cross[axis] * sin_angle
            + unit[axis] * dot * (1.0 - cos_angle)
            for axis in range(3)
        ]
        coords[atom_index] = [origin[axis] + rotated[axis] for axis in range(3)]
