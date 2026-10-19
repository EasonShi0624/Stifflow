"""Float64 boundary, locality-gate and noncommuting delayed-path checks."""

import copy
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from stifflow.geometry import centered
from stifflow.locality import LOCALITY_PROTOCOL, choose_start, load_locality_report
from stifflow.precision import control_target
from stifflow.preflight import fingerprint
from stifflow.research_paths import (
    curvature_schedule,
    sample_target,
    validate_stiffness_start,
)
from stifflow.source import prepare_source
from stifflow_alt_a.path import prepare_label
from stifflow_alt_a.path import sample_target as original


@pytest.fixture
def case():
    rng = np.random.default_rng(42)
    source = prepare_source({"atomic_numbers": [6, 6, 1], "bonds": [[0, 1], [0, 2]]})
    d = len(source.S)
    M = rng.normal(size=(d, d))
    label = prepare_label(source, M @ M.T + 0.1 * np.eye(d), 0.592485)
    assert np.linalg.norm(source.S @ label["L_K"] - label["L_K"] @ source.S) > 1e-3
    return source, label, centered(rng.normal(size=(3, 3))), rng.normal(size=d)


@pytest.mark.parametrize("start", [0.0, 0.25, 0.5, 0.75])
def test_delayed_derivatives_prefix_endpoints_and_normalized_covariance(case, start):
    source, label, x1, z = case

    def path(t):
        return sample_target(
            x1, t, z, source, label, "a_delayed", stiffness_start=start
        )

    for t in [0.0, start / 2, start]:
        assert_allclose(path(t), control_target(x1, t, z, source), atol=1e-13)
    assert_allclose(path(1)[0], x1, atol=1e-13)
    assert_allclose(path(1)[1].ravel(), -source.Q @ label["L_K"] @ z, atol=1e-13)
    for t in [0.03, 0.31, 0.67, 0.93]:
        assert_allclose(
            path(t)[1], (path(t + 1e-6)[0] - path(t - 1e-6)[0]) / 2e-6, atol=1e-8
        )
        if start == 0:
            assert_allclose(path(t), original(x1, t, z, source, label), atol=2e-13)
    if start:
        errors = [
            np.linalg.norm(
                (path(start + h)[0] - path(start - h)[0]) / (2 * h) - path(start)[1]
            )
            for h in [1e-4, 5e-5, 2.5e-5]
        ]
        assert_allclose(np.array(errors[1:]) / errors[:-1], 0.5, atol=0.002)
    D = label["L_K"] - source.S
    t = 1 - 1e-8
    g, _ = curvature_schedule(t, start)
    scaled = source.S + (g / (1 - t)) * D
    assert_allclose(
        scaled @ scaled.T, label["L_K"] @ label["L_K"].T, rtol=2e-5, atol=2e-6
    )
    q, w = np.polynomial.legendre.leggauss(20)
    u, w = (q + 1) / 2, w / 2
    assert_allclose(
        (1 - start) * np.dot(w, ((2 - 6 * u) / (1 - start)) ** 2), 4 / (1 - start)
    )
    implemented = [curvature_schedule(start + (1 - start) * v, start)[1] for v in u]
    assert_allclose(
        (1 - start) * np.dot(w, np.square(implemented)), (2 / 15) * (1 - start)
    )


def test_delayed_rotation_null_matches_all_time_spectra(case):
    source, label, x1, z = case
    R = Rotation.from_rotvec([0.3, -0.8, 0.1]).as_matrix()
    W = source.Q.T @ np.kron(np.eye(len(x1)), R) @ source.Q
    L, S = label["L_K"], source.S
    assert_allclose(W @ S, S @ W, atol=1e-13)
    for t in [0.0, 0.2, 0.5, 0.7, 0.999, 1.0]:
        g, gp = curvature_schedule(t, 0.5)
        A, Ad = (1 - t) * S + g * (L - S), -S + gp * (L - S)
        N, Nd = W @ A @ W.T, W @ Ad @ W.T
        assert_allclose(
            np.linalg.eigvalsh(A @ A.T), np.linalg.eigvalsh(N @ N.T), atol=1e-12
        )
        assert_allclose(np.linalg.norm(Ad), np.linalg.norm(Nd), atol=1e-12)
        x, v = sample_target(
            np.zeros_like(x1),
            t,
            z,
            source,
            label,
            "a_delayed_rot_null",
            R,
            stiffness_start=0.5,
        )
        assert_allclose(x.ravel(), source.Q @ N @ z, atol=1e-12)
        assert_allclose(v.ravel(), source.Q @ Nd @ z, atol=1e-12)


def test_extreme_terminal_hard_mode_avoids_source_cancellation():
    source = prepare_source({"atomic_numbers": [6, 6], "bonds": [[0, 1]]})
    L = np.diag([1e-12, 1.0, 1e-6])
    x1, z = np.zeros((2, 3)), np.array([1.0, 0.0, 0.0])
    kwargs = {"stiffness_start": 0.75}
    x, v = sample_target(x1, 1.0, z, source, {"L_K": L}, "a_delayed", **kwargs)
    assert_allclose(x, 0.0, atol=0.0)
    assert_allclose(v.ravel(), -source.Q @ L @ z, rtol=1e-14, atol=1e-28)
    t = 1.0 - 1e-14
    x, _ = sample_target(x1, t, z, source, {"L_K": L}, "a_delayed", **kwargs)
    s, u = 1 - t, (t - 0.75) / 0.25
    expected = s * s / 0.25 * (1 + u) * source.S[0, 0] + s * u * u * L[0, 0]
    assert_allclose((source.Q.T @ x.ravel())[0], expected, rtol=2e-15, atol=0.0)


@pytest.mark.parametrize("value", [None, -0.1, 1.0, np.nan, np.inf, True, "0.5"])
def test_invalid_start_rejected(value):
    with pytest.raises(ValueError):
        validate_stiffness_start("a_delayed", value)
    with pytest.raises(ValueError):
        validate_stiffness_start("control", 0.5)


def gate_candidates():
    return [
        {
            "stiffness_start": t,
            "valid_signal_fraction": v,
            "failed_queries": 0,
            "force_prediction_delta": {
                "mean": -0.2,
                "ci95": [-0.3, -0.1],
                "molecules": 128,
            },
        }
        for t, v in zip(LOCALITY_PROTOCOL["starts"], [0.5, 0.7, 0.85, 0.95])
    ]


def test_frozen_locality_gate_and_train_identity(tmp_path):
    train = tmp_path / "train.json"
    train.write_text("{}")
    candidates = gate_candidates()
    assert choose_start(candidates) == 0.5
    changed = copy.deepcopy(candidates)
    changed[0]["valid_signal_fraction"] = 0.9
    assert choose_start(changed) is None
    report = {
        "schema_version": 1,
        "kind": "stifflow-locality",
        "split": "train",
        "status": "completed",
        "protocol": LOCALITY_PROTOCOL,
        "source_config": {"rho": 1.0},
        "train_manifest": fingerprint(train),
        "selected_start": 0.5,
        "candidates": candidates,
        "records": [{"id": str(i), "status": "ok"} for i in range(128)],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    assert load_locality_report(path, train, {"rho": 1.0}) == (
        0.5,
        fingerprint(path)["sha256"],
    )
    for patch in [{"split": "val"}, {"selected_start": 0.75}, {"status": "failed"}]:
        path.write_text(json.dumps(report | patch))
        with pytest.raises(ValueError):
            load_locality_report(path, train, {"rho": 1.0})
    path.write_text(json.dumps(report))
    train.write_text('{"different": true}')
    with pytest.raises(ValueError, match="train labels"):
        load_locality_report(path, train, {"rho": 1.0})
