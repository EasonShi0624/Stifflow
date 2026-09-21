"""Float64, non-mass-weighted Cartesian geometry (core v2 eq:space/rotspace)."""

import numpy as np


def centered_basis(num_atoms):
    """Deterministic Helmert basis, atom-major Cartesian ordering."""
    if not isinstance(num_atoms, (int, np.integer)) or num_atoms < 2:
        raise ValueError("At least two atoms are required")
    helmert = np.zeros((num_atoms, num_atoms - 1), dtype=np.float64)
    for j in range(num_atoms - 1):
        norm = np.sqrt((j + 1) * (j + 2))
        helmert[: j + 1, j] = 1 / norm
        helmert[j + 1, j] = -(j + 1) / norm
    return np.kron(helmert, np.eye(3))


def centered(positions):
    positions = np.asarray(positions, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or not np.isfinite(positions).all()
    ):
        raise ValueError("Expected finite [N,3] Cartesian coordinates")
    return positions - positions.mean(axis=0)


def rigid_subspaces(positions, Q):
    """Return reduced-space rotation/vibration bases; rank comes from geometry."""
    positions = centered(positions)
    tangents = np.stack(
        [np.cross(axis, positions).ravel() for axis in np.eye(3)], axis=1
    )
    u, singular, _ = np.linalg.svd(Q.T @ tangents, full_matrices=True)
    rank = int(np.sum(singular > (singular[0] * 1e-10))) if singular[0] else 0
    return u[:, :rank], u[:, rank:]


def proper_align(reference, target):
    """Align only a relaxed reference to its target, never source noise."""
    reference, target = centered(reference), centered(target)
    if reference.shape != target.shape:
        raise ValueError("Atom correspondence requires identical coordinate shapes")
    u, _, vt = np.linalg.svd(reference.T @ target)
    correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    return reference @ u @ correction @ vt


def spd_eigh(matrix):
    """Validate SPD; no jitter, absolute eigenvalues, upper clipping or rescaling."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Expected a finite square matrix")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError("Matrix must be symmetric")
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    if len(values) == 0 or values[0] <= 0:
        raise ValueError("Matrix must be strictly positive definite")
    return values, vectors
