"""Shared CPU supervision; hidden labels never enter network inputs."""

import numpy as np
from scipy.spatial.transform import Rotation

from . import precision
from .geometry import centered


def path_module(method):
    if method == "precision":
        return precision
    if method == "alternative_a":
        from stifflow_alt_a import path

        return path
    if method == "control":
        return None
    raise ValueError(f"Unknown path method: {method}")


def precision_target(x1, t, z, source, label):
    """Core Version 2 eq:factor/adot/trajectory, applied without dense factors."""
    alpha, complement, alphadot, s = precision.schedules(t)
    x1, z = centered(x1), np.asarray(z, dtype=np.float64)
    dimension = source.S.shape[0]
    b = np.asarray(label["values"], dtype=np.float64)
    V = np.asarray(label["vectors"], dtype=np.float64)
    if (
        x1.size != source.Q.shape[0]
        or z.shape != (dimension,)
        or b.shape != (dimension,)
        or V.shape != (dimension, dimension)
        or not np.isfinite(z).all()
        or not np.isfinite(b).all()
        or not np.isfinite(V).all()
        or np.any(b <= 0)
    ):
        raise ValueError("Invalid precision target/noise/factor dimensions or values")
    denominator = complement + alpha * b
    scale = 1 / np.sqrt(denominator)
    projected = V.T @ z
    # S stays on the left; no commutation of source and energetic precision.
    az = source.S @ (V @ ((s * scale) * projected))
    adotz = source.S @ (
        V @ ((-scale * (1 + 0.5 * s * alphadot * (b - 1) / denominator)) * projected)
    )
    x = alpha * x1 + (source.Q @ az).reshape(x1.shape)
    u = alphadot * x1 + (source.Q @ adotz).reshape(x1.shape)
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise FloatingPointError("Non-finite path or velocity")
    return x, u


def draw_target(x1, source, label, method, rng):
    """Core eq:joint/samplinglaw: independent z and one Haar proper rotation."""
    # Reject machine endpoints after float32 rounding; no fixed epsilon truncation.
    t = float(np.float32(rng.random()))
    while t == 0.0 or t == 1.0:
        t = float(np.float32(rng.random()))
    z = rng.standard_normal(source.S.shape[0])
    module = path_module(method)
    target = (
        precision_target
        if method == "precision"
        else (precision.control_target if module is None else module.sample_target)
    )
    xt, ut = target(x1, t, z, source, label)
    R = Rotation.random(random_state=rng).as_matrix()
    # Rotate x1 and both factor*z terms jointly, without refactorizing K.
    return xt @ R.T, ut @ R.T, t
