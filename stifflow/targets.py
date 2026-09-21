"""Shared CPU supervision; hidden labels never enter network inputs."""

import numpy as np
from scipy.spatial.transform import Rotation

from . import precision


def path_module(method):
    if method == "precision":
        return precision
    if method == "alternative_a":
        from stifflow_alt_a import path

        return path
    if method == "control":
        return None
    raise ValueError(f"Unknown path method: {method}")


def draw_target(x1, source, label, method, rng):
    """Core eq:joint/samplinglaw: independent z and one Haar proper rotation."""
    # Reject machine endpoints after float32 rounding; no fixed epsilon truncation.
    t = float(np.float32(rng.random()))
    while t == 0.0 or t == 1.0:
        t = float(np.float32(rng.random()))
    z = rng.standard_normal(source.S.shape[0])
    module = path_module(method)
    target = precision.control_target if module is None else module.sample_target
    xt, ut = target(x1, t, z, source, label)
    R = Rotation.random(random_state=rng).as_matrix()
    # Rotate x1 and both factor*z terms jointly, without refactorizing K.
    return xt @ R.T, ut @ R.T, t
