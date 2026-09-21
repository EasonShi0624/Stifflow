"""Core Version 2 (2026-09-15), eq:defaults/factor/adot/trajectory.

All dense matrices and targets are float64. No target alignment or loss metric.
"""

import numpy as np

from .geometry import centered, spd_eigh


def schedules(t):
    if not np.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("Time must be finite and in [0,1]")
    s = 1 - t
    # Both complements computed directly to avoid losing soft modes near t=1.
    return t * t * (3 - 2 * t), s * s * (1 + 2 * t), 6 * t * s, s


def prepare_label(source, K, epsilon_E):
    if not np.isfinite(epsilon_E) or epsilon_E <= 0:
        raise ValueError("epsilon_E must be finite and positive")
    K = np.asarray(K, dtype=np.float64)
    if K.shape != source.S.shape:
        raise ValueError("K must act on the source's translation-free space")
    spd_eigh(K)
    values, vectors = spd_eigh(source.S.T @ K @ source.S / epsilon_E)
    return {"values": values, "vectors": vectors}


def factor(t, source, label):
    h, one_minus_h, hdot, s = schedules(t)
    b, V = label["values"], label["vectors"]
    d = one_minus_h + h * b
    scale = 1 / np.sqrt(d)
    SV = source.S @ V
    A = (SV * (s * scale)) @ V.T
    Adot = (SV * (-scale * (1 + 0.5 * s * hdot * (b - 1) / d))) @ V.T
    return A, Adot


def sample_from_factors(x1, t, z, source, A, Adot):
    alpha, _, alphadot, _ = schedules(t)
    x1 = centered(x1)
    z = np.asarray(z, dtype=np.float64)
    if (
        x1.size != source.Q.shape[0]
        or z.shape != (source.S.shape[0],)
        or not np.isfinite(z).all()
    ):
        raise ValueError("Target/noise dimensions do not match source")
    # The same fixed z must appear in positions and velocities (eq:trajectory).
    x = alpha * x1 + (source.Q @ (A @ z)).reshape(x1.shape)
    u = alphadot * x1 + (source.Q @ (Adot @ z)).reshape(x1.shape)
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise FloatingPointError("Non-finite path or velocity")
    return x, u


def sample_target(x1, t, z, source, method_label):
    return sample_from_factors(x1, t, z, source, *factor(t, source, method_label))


def control_target(x1, t, z, source, method_label=None):
    """Explicit stiffness-free control with the same smoothstep mean."""
    schedules(t)
    return sample_from_factors(x1, t, z, source, (1 - t) * source.S, -source.S)
