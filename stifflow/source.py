"""Graph-only source, core Version 2 eq:source/graphsource; units are Angstrom."""

from dataclasses import dataclass

import numpy as np

from .geometry import centered_basis, spd_eigh


@dataclass(frozen=True)
class Source:
    Q: np.ndarray
    S: np.ndarray
    C0: np.ndarray


def prepare_source(graph, source_config=None):
    """Only atomic_numbers and unweighted chemical bonds enter the prior."""
    config = {"sigma0": 1.0, "rho": 1.0}
    if source_config:
        if set(source_config) - config.keys():
            raise ValueError("Unknown source parameters")
        config.update(source_config)
    sigma, rho = config["sigma0"], config["rho"]
    if not np.isfinite([sigma, rho]).all() or min(sigma, rho) <= 0:
        raise ValueError("sigma0 and rho must be finite and positive")
    atoms = np.asarray(graph["atomic_numbers"])
    if atoms.ndim != 1 or np.any(atoms < 1) or np.any(atoms != atoms.astype(int)):
        raise ValueError("Invalid atomic numbers")
    n = len(atoms)
    Q = centered_basis(n)
    laplacian = np.zeros((n, n), dtype=np.float64)
    bonds = np.asarray(graph["bonds"])
    if bonds.size:
        if bonds.ndim != 2 or bonds.shape[1] != 2 or np.any(bonds != bonds.astype(int)):
            raise ValueError("Expected chemical bond index pairs")
        for i, j in {tuple(sorted(pair)) for pair in bonds.astype(int)}:
            if i < 0 or j >= n or i == j:
                raise ValueError("Invalid bond indices")
            laplacian[i, i] += 1
            laplacian[j, j] += 1
            laplacian[i, j] -= 1
            laplacian[j, i] -= 1
    # Reduce the atom Laplacian first; this preserves isotropy exactly.
    H = Q[::3, ::3]
    values, vectors = spd_eigh(H.T @ (laplacian + rho * np.eye(n)) @ H)
    C0 = np.kron((vectors * (sigma**2 / values)) @ vectors.T, np.eye(3))
    S = np.kron((vectors * (sigma / np.sqrt(values))) @ vectors.T, np.eye(3))
    return Source(Q=Q, S=S, C0=C0)
