"""Small geometry helpers that avoid heavy numerical dependencies."""

from __future__ import annotations

import math
from typing import Iterable


Coordinates = list[list[float]]


def copy_coords(coords: Coordinates) -> Coordinates:
    return [[float(value) for value in xyz] for xyz in coords]


def assert_finite_coords(coords: Coordinates, label: str) -> None:
    for atom_index, xyz in enumerate(coords):
        for axis, value in enumerate(xyz):
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} has non-finite coordinate at atom {atom_index}, axis {axis}")


def rmsd(coords_a: Coordinates, coords_b: Coordinates) -> float:
    if len(coords_a) != len(coords_b):
        raise ValueError("RMSD requires coordinate arrays with the same atom count")
    if not coords_a:
        raise ValueError("RMSD requires at least one atom")

    total = 0.0
    for xyz_a, xyz_b in zip(coords_a, coords_b):
        for a, b in zip(xyz_a, xyz_b):
            diff = float(a) - float(b)
            total += diff * diff
    return math.sqrt(total / len(coords_a))


def distance(coords: Coordinates, i: int, j: int) -> float:
    return math.sqrt(sum((float(coords[i][axis]) - float(coords[j][axis])) ** 2 for axis in range(3)))


def mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)
