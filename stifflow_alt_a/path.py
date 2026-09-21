"""Euclidean Alternative A, eq:T/factor/offlineL/coding; float64 throughout.

Source and mean schedule are shared with StifFlow; only supervision uses K.
"""

import numpy as np
from scipy.linalg import solve
from stifflow.geometry import spd_eigh
from stifflow.precision import sample_from_factors, schedules


def prepare_label(source, K, epsilon_E):
    """Prepare L_K = T_K S, never substitute an arbitrary covariance factor."""
    if not np.isfinite(epsilon_E) or epsilon_E <= 0:
        raise ValueError("epsilon_E must be finite and positive")
    K = np.asarray(K, dtype=np.float64)
    if K.shape != source.S.shape:
        raise ValueError("K must act on the source's translation-free space")
    stiffness, _ = spd_eigh(K)
    spd_eigh(source.S)
    CK = epsilon_E * solve(K, np.eye(len(K)), assume_a="pos")
    values, vectors = spd_eigh(source.S @ CK @ source.S)
    root = (vectors * np.sqrt(values)) @ vectors.T
    # eq:offlineL: left multiplication by S^{-1}; order matters.
    LK = solve(source.S, root, assume_a="pos")
    residual = float(np.linalg.norm(LK @ LK.T - CK) / np.linalg.norm(CK))
    if not np.isfinite(residual) or residual > 1e-8:
        raise FloatingPointError(f"Unreliable A_E factor: residual={residual:.3g}")
    return {
        "L_K": LK,
        "covariance_residual": residual,
        "condition_K": float(stiffness[-1] / stiffness[0]),
        "condition_SCKS": float(values[-1] / values[0]),
    }


def factor(t, source, label):
    """Reduced-space A_t and its analytic derivative, including exact endpoints."""
    _, _, _, s = schedules(t)
    LK = np.asarray(label["L_K"], dtype=np.float64)
    if LK.shape != source.S.shape or not np.isfinite(LK).all():
        raise ValueError("Invalid aligned label factor")
    # eq:coding: factored schedules avoid cancellation near t=1.
    f, g = s * s * (1 + t), t * t * s
    gdot = t * (2 - 3 * t)
    return f * source.S + g * LK, (-1 - gdot) * source.S + gdot * LK


def sample_target(x1, t, z, source, method_label):
    """Return centered [N,3] position/velocity from the same independent z."""
    return sample_from_factors(x1, t, z, source, *factor(t, source, method_label))
