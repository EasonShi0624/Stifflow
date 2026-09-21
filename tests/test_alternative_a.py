"""Independent CPU checks for the declared A_E, not the A_W variant."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from stifflow.geometry import centered, centered_basis
from stifflow.source import Source, prepare_source
from stifflow_alt_a.path import factor, prepare_label, sample_target


def problem():
    rng = np.random.default_rng(20260916)
    source = prepare_source(
        {"atomic_numbers": [6, 6, 8, 1], "bonds": [[0, 1], [1, 2], [2, 3]]}
    )
    matrix = rng.normal(size=source.S.shape)
    K = matrix @ matrix.T + 0.5 * np.eye(len(matrix))
    x1 = centered(rng.normal(size=(4, 3)))
    z = rng.normal(size=len(matrix))
    assert np.linalg.norm(K @ source.C0 - source.C0 @ K) > 0.1
    return source, K, x1, z


def test_aligned_factor_and_sample_derivative():
    source, K, x1, z = problem()
    epsilon = 0.592485
    label = prepare_label(source, K, epsilon)
    LK = label["L_K"]
    CK = epsilon * np.linalg.solve(K, np.eye(len(K)))
    assert LK.dtype == np.float64
    assert label["covariance_residual"] < 1e-10
    assert_allclose(LK @ LK.T, CK, atol=1e-10, rtol=1e-10)
    TK = np.linalg.solve(source.S, LK.T).T
    assert_allclose(TK, TK.T, atol=1e-11)
    assert np.linalg.eigvalsh(TK)[0] > 0
    assert_allclose(TK @ source.C0 @ TK, CK, atol=1e-10)
    # A principal CK root has the right covariance but the wrong coupling.
    values, vectors = np.linalg.eigh(CK)
    arbitrary_root = (vectors * np.sqrt(values)) @ vectors.T
    assert np.linalg.norm(LK - arbitrary_root) > 1e-3
    assert np.linalg.norm(LK - source.S) < np.linalg.norm(arbitrary_root - source.S)

    dt = 1e-5
    for t in (0.02, 0.19, 0.5, 0.81, 0.98):
        A, Adot = factor(t, source, label)
        x, u = sample_target(x1, t, z, source, label)
        plus = sample_target(x1, t + dt, z, source, label)[0]
        minus = sample_target(x1, t - dt, z, source, label)[0]
        assert_allclose((plus - minus) / (2 * dt), u, atol=1e-8, rtol=1e-8)
        assert_allclose(x.mean(axis=0), 0, atol=1e-14)
        assert_allclose(u.mean(axis=0), 0, atol=1e-14)
        # eq:conditional-continuity, finite-difference covariance evolution.
        Ap = factor(t + dt, source, label)[0]
        Am = factor(t - dt, source, label)[0]
        covariance = A @ A.T
        M = np.linalg.solve(A.T, Adot.T).T
        assert_allclose(M, M.T, atol=1e-10)
        assert_allclose(
            (Ap @ Ap.T - Am @ Am.T) / (2 * dt),
            M @ covariance + covariance @ M.T,
            atol=1e-8,
            rtol=1e-8,
        )


def test_endpoints_and_full_error_limit():
    source, K, x1, z = problem()
    label = prepare_label(source, K, 0.592485)
    xi = (source.Q @ source.S @ z).reshape(x1.shape)
    eta = (source.Q @ label["L_K"] @ z).reshape(x1.shape)
    x0, u0 = sample_target(x1, 0, z, source, label)
    xend, uend = sample_target(x1, 1, z, source, label)
    assert_allclose(x0, xi, atol=1e-13)
    assert_allclose(u0, -xi, atol=1e-13)
    assert_allclose(xend, x1, atol=1e-13)
    assert_allclose(uend, -eta, atol=1e-13)
    other = prepare_label(source, 100 * K, 0.592485)
    assert_allclose(sample_target(2 * x1, 0, z, source, other), (x0, u0))

    CK = 0.592485 * np.linalg.solve(K, np.eye(len(K)))
    errors = []
    for s in (1e-2, 1e-3, 1e-4, 1e-5):
        t = 1 - s
        A = factor(t, source, label)[0]
        # eq:full-error includes the nonzero mean at every finite endpoint gap.
        reduced_x = source.Q.T @ x1.ravel()
        mean = -s * (1 + 2 * t) * reduced_x
        error_second_moment = A @ A.T / s**2 + np.outer(mean, mean)
        errors.append(np.linalg.norm(error_second_moment - CK))
        x = sample_target(x1, t, z, source, label)[0]
        exact = -s * (1 + 2 * t) * x1 + (1 - t * t) * xi + t * t * eta
        assert_allclose((x - x1) / s, exact, atol=1e-9)
    assert np.all(np.asarray(errors[1:]) < 0.12 * np.asarray(errors[:-1]))

    neutral = prepare_label(source, np.linalg.inv(source.C0), 1.0)
    for t in (0, 0.17, 0.8, 1):
        A, Adot = factor(t, source, neutral)
        assert_allclose(A, (1 - t) * source.S, atol=1e-12)
        assert_allclose(Adot, -source.S, atol=1e-12)


def test_rotations_permutations_and_translation():
    source, K, x1, z = problem()
    label = prepare_label(source, K, 0.592485)
    rotation = Rotation.from_rotvec([0.3, -0.7, 0.2]).as_matrix()
    permutation = np.eye(4)[[2, 0, 3, 1]]
    for full in (np.kron(np.eye(4), rotation), np.kron(permutation, np.eye(3))):
        O = source.Q.T @ full @ source.Q
        transformed = Source(source.Q, O @ source.S @ O.T, O @ source.C0 @ O.T)
        other = prepare_label(transformed, O @ K @ O.T, 0.592485)
        assert_allclose(other["L_K"], O @ label["L_K"] @ O.T, atol=1e-11)
        x, u = sample_target(x1, 0.39, z, source, label)
        xp, up = sample_target(
            (full @ x1.ravel()).reshape(4, 3), 0.39, O @ z, transformed, other
        )
        assert_allclose(xp.ravel(), full @ x.ravel(), atol=1e-11)
        assert_allclose(up.ravel(), full @ u.ravel(), atol=1e-11)
    assert_allclose(
        sample_target(x1 + [10, 20, -30], 0.39, z, source, label),
        sample_target(x1, 0.39, z, source, label),
        atol=1e-12,
    )


def test_documented_noncommuting_variance_counterexample():
    # eq:order-example/order-failure embedded in a true 3-atom centered space.
    C0 = np.kron(np.diag([4.0, 1.0]), np.eye(3))
    source = Source(centered_basis(3), np.diag(np.sqrt(np.diag(C0))), C0)
    K0 = np.kron(np.diag([0.25, 0.5]), np.eye(3))
    K1 = K0 + np.kron(np.ones((2, 2)), np.eye(3))
    assert np.linalg.eigvalsh(K1 - K0)[0] >= -1e-14
    A0 = factor(0.5, source, prepare_label(source, K0, 1.0))[0]
    A1 = factor(0.5, source, prepare_label(source, K1, 1.0))[0]
    difference = np.linalg.eigvalsh(A0 @ A0.T - A1 @ A1.T)
    assert_allclose(difference, [-0.00260433] * 3 + [0.22255381] * 3, atol=5e-9)


def test_extreme_spectra_and_invalid_inputs():
    source = prepare_source({"atomic_numbers": [1, 1], "bonds": [[0, 1]]})
    K = np.diag([1e-6, 1.0, 1e12])
    label = prepare_label(source, K, 1.0)
    assert label["condition_K"] == 1e18  # Report extremes, never upper-clip.
    for t in (0.0, 1e-12, 0.5, 1 - 1e-12, 1.0):
        A, Adot = factor(t, source, label)
        assert np.isfinite(A).all() and np.isfinite(Adot).all()
    near_repeated = prepare_label(source, np.diag([1, 1 + 1e-12, 2]), 1.0)
    assert near_repeated["covariance_residual"] < 1e-10
    for invalid in (np.diag([-1, 1, 1]), np.zeros((3, 3)), np.eye(2)):
        with pytest.raises(ValueError):
            prepare_label(source, invalid, 1.0)
    with pytest.raises(ValueError):
        prepare_label(source, np.eye(3), 0)
    for t in (-0.01, 1.01, np.nan):
        with pytest.raises(ValueError):
            factor(t, source, label)
