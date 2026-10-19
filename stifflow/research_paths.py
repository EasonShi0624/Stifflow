"""Explicit research variants; the three published v1 paths are unchanged."""

import hashlib
import json
from numbers import Real

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import centered, proper_align
from .precision import schedules

BASE_METHODS = ("precision", "alternative_a", "control")
DELAYED_METHODS = ("a_delayed", "a_delayed_rot_null")
RESEARCH_METHODS = (
    "a_rot_null",
    "a_trace_null",
    "linear",
    "aligned_linear",
    *DELAYED_METHODS,
    "precision_clock",
)
METHODS = BASE_METHODS + RESEARCH_METHODS


def label_method(method):
    """Reuse existing factors without relabeling or changing the accepted cohort."""
    if method not in METHODS:
        raise ValueError(f"Unknown path method: {method}")
    return {
        "a_rot_null": "alternative_a",
        "a_trace_null": "alternative_a",
        "linear": "control",
        "aligned_linear": "control",
        "a_delayed": "alternative_a",
        "a_delayed_rot_null": "alternative_a",
        "precision_clock": "precision",
    }.get(method, method)


def validate_stiffness_start(method, value):
    if method not in DELAYED_METHODS:
        if value is not None:
            raise ValueError("Only delayed paths accept stiffness_start")
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or not 0 <= value < 1
    ):
        raise ValueError("Delayed paths require finite 0 <= stiffness_start < 1")
    return float(value)


def curvature_schedule(t, start):
    """StifFlow_A_delayed_curvature.tex eq:gate: fixed control prefix."""
    schedules(t)
    validate_stiffness_start("a_delayed", start)
    if t <= start:
        return 0.0, 0.0
    u = (t - start) / (1 - start)
    return (1 - t) * u * u, u * (2 - 3 * u)


def relative_rotation(rng):
    """Separate deterministic domain; never advance the shared t/z/Haar stream."""
    state = json.dumps(rng.bit_generator.state, sort_keys=True).encode()
    seed = int.from_bytes(
        hashlib.sha256(b"stifflow/rot-null/v1:" + state).digest(), "big"
    )
    return Rotation.random(random_state=np.random.default_rng(seed)).as_matrix()


def rotate_reduced(vector, source, rotation):
    return source.Q.T @ ((source.Q @ vector).reshape(-1, 3) @ rotation.T).ravel()


def sample_target(
    x1, t, z, source, label, method, rotation=None, *, stiffness_start=None
):
    """Analytic sample derivatives with fixed z and fixed relative R along a path."""
    alpha, _, alphadot, s = schedules(t)
    validate_stiffness_start(method, stiffness_start)
    x1, z = centered(x1), np.asarray(z, dtype=np.float64)
    if (
        x1.size != source.Q.shape[0]
        or z.shape != (source.S.shape[0],)
        or not np.isfinite(z).all()
    ):
        raise ValueError("Invalid research target/noise shape or values")
    x0 = (source.Q @ (source.S @ z)).reshape(x1.shape)
    if method in ("linear", "aligned_linear"):
        if method == "aligned_linear":
            # A new coupling: a subsequent independent Haar rotation restores p0.
            x0 = proper_align(x0, x1)
        return s * x0 + t * x1, x1 - x0
    if method not in ("a_rot_null", "a_trace_null", *DELAYED_METHODS):
        raise ValueError(f"Not a null or linear path: {method}")
    L = np.asarray(label["L_K"], dtype=np.float64)
    if L.shape != source.S.shape or not np.isfinite(L).all():
        raise ValueError("Invalid A_E label factor")
    if method == "a_delayed":
        lz = L @ z
    elif method == "a_trace_null":
        scale = np.linalg.norm(L) / np.linalg.norm(source.S)
        lz = scale * (source.S @ z)
    else:
        R = np.asarray(rotation, dtype=np.float64)
        if (
            R.shape != (3, 3)
            or not np.isfinite(R).all()
            or not np.allclose(R.T @ R, np.eye(3), rtol=1e-12, atol=1e-12)
            or not np.isclose(np.linalg.det(R), 1, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("Rotation null requires a fixed proper rotation")
        lz = rotate_reduced(L @ rotate_reduced(z, source, R.T), source, R)
    endpoint_noise = (source.Q @ lz).reshape(x1.shape)
    if method in DELAYED_METHODS:
        g, gdot = curvature_schedule(t, stiffness_start)
        u = max(0.0, (t - stiffness_start) / (1 - stiffness_start))
        # Avoid subtracting source-sized terms when terminal hard-mode noise is tiny.
        coefficient = (
            s * s * (1 + u) / (1 - stiffness_start) if t > stiffness_start else s
        )
        derivative = (
            -s * (1 + 3 * u) / (1 - stiffness_start) if t > stiffness_start else -1.0
        )
        return (
            alpha * x1 + coefficient * x0 + g * endpoint_noise,
            alphadot * x1 + derivative * x0 + gdot * endpoint_noise,
        )
    delta = endpoint_noise - x0
    g, gdot = t * t * s, t * (2 - 3 * t)
    return alpha * x1 + s * x0 + g * delta, alphadot * x1 - x0 + gdot * delta
