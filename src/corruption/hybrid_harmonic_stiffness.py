"""Hybrid harmonic-prior plus local stiffness source corruptions.

These v1 modes preserve the existing graph-smoothed harmonic source as the
dominant construction and add only conservative local stiffness information.
They rely on metadata already present in the processed GEOM records; when richer
chemistry annotations are absent, bond order and atom-degree heuristics are used.
"""

from __future__ import annotations

import math
from typing import Any

from src.corruption.harmonic import HarmonicPriorCorruptor, _apply_noise
from src.utils.geometry import Coordinates


class HybridHarmonicStiffnessV1Corruptor(HarmonicPriorCorruptor):
    """Graph-harmonic source with conservative stiffness-scaled input noise."""

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        super().__init__(config, seed)
        corruption = config["corruption"]
        self.modulation_strength = _bounded_strength(corruption, "modulation_strength")
        self.protect_rings = bool(corruption.get("protect_rings", True))
        self.protect_amides = bool(corruption.get("protect_amides", True))
        self.protect_aromatic_conjugated = bool(corruption.get("protect_aromatic_conjugated", True))
        self.min_noise_scale = float(corruption.get("hybrid_min_noise_scale", 0.50))
        self.max_noise_scale = float(corruption.get("hybrid_max_noise_scale", 1.25))
        if self.min_noise_scale <= 0.0 or self.max_noise_scale < self.min_noise_scale:
            raise ValueError("hybrid noise scale bounds must satisfy 0 < min <= max")

    def corrupt(self, record: dict[str, Any], coords: Coordinates) -> Coordinates:
        scales = self.atom_noise_scales(record, len(coords))
        sigmas = [self.sigma * scale for scale in scales]
        noise = self.sample_harmonic_noise(record, coords, sigmas)
        return _apply_noise(coords, noise)

    def atom_noise_scales(self, record: dict[str, Any], atom_count: int) -> list[float]:
        classes = _atom_stiffness_classes(
            record=record,
            atom_count=atom_count,
            protect_rings=self.protect_rings,
            protect_amides=self.protect_amides,
            protect_aromatic_conjugated=self.protect_aromatic_conjugated,
        )
        class_offsets = {
            "stiff": -0.50,
            "medium": -0.15,
            "soft": 0.20,
        }
        scales = []
        for stiffness_class in classes:
            scale = 1.0 + self.modulation_strength * class_offsets[stiffness_class]
            scales.append(_clamp(scale, self.min_noise_scale, self.max_noise_scale))
        return scales


class HarmonicStiffnessResidualV1Corruptor(HarmonicPriorCorruptor):
    """Graph-harmonic source plus a small torsional residual on soft fragments."""

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        super().__init__(config, seed)
        corruption = config["corruption"]
        self.residual_strength = _bounded_strength(corruption, "residual_strength")
        self.residual_torsion_sigma = float(corruption.get("residual_torsion_sigma", corruption["torsion_sigma"]))
        self.residual_max_rotatable_bonds = int(corruption.get("residual_max_rotatable_bonds", 8))
        self.residual_max_angle = float(corruption.get("residual_max_angle", 0.25))
        self.protect_rings = bool(corruption.get("protect_rings", True))
        self.protect_amides = bool(corruption.get("protect_amides", True))
        self.protect_aromatic_conjugated = bool(corruption.get("protect_aromatic_conjugated", True))
        if self.residual_torsion_sigma < 0.0:
            raise ValueError("residual_torsion_sigma must be non-negative")
        if self.residual_max_rotatable_bonds < 0:
            raise ValueError("residual_max_rotatable_bonds must be non-negative")
        if self.residual_max_angle < 0.0:
            raise ValueError("residual_max_angle must be non-negative")

    def corrupt(self, record: dict[str, Any], coords: Coordinates) -> Coordinates:
        corrupted = super().corrupt(record, coords)
        if self.residual_strength <= 0.0 or self.residual_torsion_sigma <= 0.0:
            return corrupted
        protected_atoms = _protected_atom_indices(
            record=record,
            atom_count=len(corrupted),
            protect_rings=self.protect_rings,
            protect_amides=self.protect_amides,
            protect_aromatic_conjugated=self.protect_aromatic_conjugated,
        )
        candidates = _soft_rotatable_candidates(
            record=record,
            atom_count=len(corrupted),
            protected_atoms=protected_atoms,
        )
        self.rng.shuffle(candidates)
        if self.residual_max_rotatable_bonds:
            candidates = candidates[: self.residual_max_rotatable_bonds]
        sigma = self.residual_strength * self.residual_torsion_sigma
        for i, j, moving_atoms in candidates:
            angle = self.rng.gauss(0.0, sigma)
            if self.residual_max_angle > 0.0:
                angle = _clamp(angle, -self.residual_max_angle, self.residual_max_angle)
            if abs(angle) <= 1.0e-12:
                continue
            _rotate_atoms_around_axis(corrupted, i, j, moving_atoms, angle)
        return corrupted


def _bounded_strength(corruption: dict[str, Any], field: str) -> float:
    value = float(corruption[field])
    upper_bound = 50.0 if field == "modulation_strength" else 1.0
    if value < 0.0 or value > upper_bound:
        raise ValueError(f"{field} must be in [0, {upper_bound:g}]")
    return value


def _atom_stiffness_classes(
    record: dict[str, Any],
    atom_count: int,
    protect_rings: bool,
    protect_amides: bool,
    protect_aromatic_conjugated: bool,
) -> list[str]:
    protected_atoms = _protected_atom_indices(
        record=record,
        atom_count=atom_count,
        protect_rings=protect_rings,
        protect_amides=protect_amides,
        protect_aromatic_conjugated=protect_aromatic_conjugated,
    )
    degrees = _atom_degrees(record.get("bonds", []), atom_count)
    soft_atoms = _soft_flexible_atom_indices(record, atom_count, protected_atoms)
    classes = []
    for atom_index in range(atom_count):
        if atom_index in protected_atoms:
            classes.append("stiff")
        elif atom_index in soft_atoms or degrees[atom_index] <= 1:
            classes.append("soft")
        else:
            classes.append("medium")
    return classes


def _protected_atom_indices(
    record: dict[str, Any],
    atom_count: int,
    protect_rings: bool,
    protect_amides: bool,
    protect_aromatic_conjugated: bool,
) -> set[int]:
    protected: set[int] = set()
    if protect_rings:
        protected.update(_valid_atom_indices(record.get("ring_atom_indices", []), atom_count))
    if protect_amides:
        protected.update(_amide_like_atom_indices(record, atom_count))
    if protect_aromatic_conjugated:
        protected.update(_unsaturated_atom_indices(record, atom_count))
    return protected


def _valid_atom_indices(values: Any, atom_count: int) -> set[int]:
    indices = set()
    if not isinstance(values, list):
        return indices
    for value in values:
        try:
            atom_index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= atom_index < atom_count:
            indices.add(atom_index)
    return indices


def _amide_like_atom_indices(record: dict[str, Any], atom_count: int) -> set[int]:
    atom_symbols = [str(symbol) for symbol in record.get("atom_symbols", [])]
    if len(atom_symbols) != atom_count:
        return set()
    adjacency = _adjacency(record.get("bonds", []), atom_count)
    bond_orders = _bond_order_lookup(record.get("bonds", []), atom_count)
    protected: set[int] = set()
    for atom_index, symbol in enumerate(atom_symbols):
        if symbol != "C":
            continue
        oxygen_neighbors = [
            neighbor
            for neighbor in adjacency[atom_index]
            if atom_symbols[neighbor] == "O" and _bond_order(bond_orders, atom_index, neighbor) >= 1.5
        ]
        nitrogen_neighbors = [
            neighbor
            for neighbor in adjacency[atom_index]
            if atom_symbols[neighbor] == "N" and abs(_bond_order(bond_orders, atom_index, neighbor) - 1.0) <= 1.0e-6
        ]
        if oxygen_neighbors and nitrogen_neighbors:
            protected.add(atom_index)
            protected.update(oxygen_neighbors)
            protected.update(nitrogen_neighbors)
    return protected


def _unsaturated_atom_indices(record: dict[str, Any], atom_count: int) -> set[int]:
    protected: set[int] = set()
    for bond in record.get("bonds", []):
        if len(bond) < 3:
            continue
        i, j = int(bond[0]), int(bond[1])
        if i < 0 or j < 0 or i >= atom_count or j >= atom_count or i == j:
            continue
        order = float(bond[2])
        if abs(order - 1.0) > 1.0e-6:
            protected.add(i)
            protected.add(j)
    return protected


def _soft_flexible_atom_indices(record: dict[str, Any], atom_count: int, protected_atoms: set[int]) -> set[int]:
    soft_atoms = set()
    for _, _, moving_atoms in _soft_rotatable_candidates(record, atom_count, protected_atoms):
        soft_atoms.update(moving_atoms)
    degrees = _atom_degrees(record.get("bonds", []), atom_count)
    for atom_index, degree in enumerate(degrees):
        if degree <= 1 and atom_index not in protected_atoms:
            soft_atoms.add(atom_index)
    return soft_atoms


def _soft_rotatable_candidates(
    record: dict[str, Any],
    atom_count: int,
    protected_atoms: set[int],
) -> list[tuple[int, int, list[int]]]:
    bonds = record.get("bonds", [])
    adjacency = _adjacency(bonds, atom_count)
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
        if i in protected_atoms or j in protected_atoms:
            continue
        side_i = _component_without_edge(adjacency, start=i, blocked_a=i, blocked_b=j)
        side_j = _component_without_edge(adjacency, start=j, blocked_a=i, blocked_b=j)
        moving_i = sorted(atom for atom in side_i if atom not in {i, j})
        moving_j = sorted(atom for atom in side_j if atom not in {i, j})
        moving_atoms = _smaller_unprotected_nonempty(moving_i, moving_j, protected_atoms)
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


def _atom_degrees(bonds: list[list[int | float]], atom_count: int) -> list[int]:
    degrees = [0 for _ in range(atom_count)]
    for bond in bonds:
        if len(bond) < 2:
            continue
        i, j = int(bond[0]), int(bond[1])
        if i < 0 or j < 0 or i >= atom_count or j >= atom_count or i == j:
            continue
        degrees[i] += 1
        degrees[j] += 1
    return degrees


def _bond_order_lookup(bonds: list[list[int | float]], atom_count: int) -> dict[tuple[int, int], float]:
    orders = {}
    for bond in bonds:
        if len(bond) < 2:
            continue
        i, j = int(bond[0]), int(bond[1])
        if i < 0 or j < 0 or i >= atom_count or j >= atom_count or i == j:
            continue
        order = float(bond[2]) if len(bond) >= 3 else 1.0
        orders[(i, j)] = order
        orders[(j, i)] = order
    return orders


def _bond_order(orders: dict[tuple[int, int], float], i: int, j: int) -> float:
    return orders.get((i, j), 1.0)


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


def _smaller_unprotected_nonempty(
    left: list[int],
    right: list[int],
    protected_atoms: set[int],
) -> list[int]:
    candidates = [
        atoms
        for atoms in [left, right]
        if atoms and not any(atom in protected_atoms for atom in atoms)
    ]
    if not candidates:
        return []
    return min(candidates, key=len)


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


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
