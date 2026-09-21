"""Core v2 float64 identities; no molecular or neural performance claim."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from stifflow.geometry import centered_basis, rigid_subspaces
from stifflow.precision import factor, prepare_label, sample_target, schedules
from stifflow.source import Source, prepare_source


class PathChecks(unittest.TestCase):
    def test_noncommuting_and_limits(self):
        rng = np.random.default_rng(42)
        graph = {"atomic_numbers": [6, 1, 1, 1], "bonds": [[0, 1], [1, 2], [2, 3]]}
        src = prepare_source(graph)
        d = src.S.shape[0]
        M = rng.normal(size=(d, d))
        K = M @ M.T + np.eye(d)
        epsilon = 0.592485
        label = prepare_label(src, K, epsilon)
        assert np.linalg.norm(src.C0 @ K - K @ src.C0) > 1
        Ck = epsilon * np.linalg.inv(K)
        x1, z = rng.normal(size=(4, 3)), rng.normal(size=d)
        x1 -= x1.mean(axis=0)
        x0, u0 = sample_target(x1, 0.0, z, src, label)
        assert_allclose(x0.ravel(), src.Q @ src.S @ z, atol=1e-12)
        assert_allclose(u0, -x0, atol=1e-12)
        assert_allclose(sample_target(x1, 1.0, z, src, label)[0], x1, atol=1e-12)
        for t in (0.01, 0.2, 0.5, 0.9, 0.99):
            h, hc, hd, s = schedules(t)
            A, Ad = factor(t, src, label)
            P = hc * np.linalg.inv(src.C0) + h * K / epsilon
            Pd = hd * (K / epsilon - np.linalg.inv(src.C0))
            cov = s * s * np.linalg.inv(P)
            assert_allclose(A @ A.T, cov, rtol=1e-10, atol=1e-12)
            # eq:u ordering P^{-1} Pdot, never the reversed product.
            drift = -np.eye(d) / s - 0.5 * np.linalg.solve(P, Pd)
            assert_allclose(Ad @ np.linalg.inv(A), drift, atol=1e-10)
            covariance_dot = Ad @ A.T + A @ Ad.T
            assert_allclose(covariance_dot, drift @ cov + cov @ drift.T, atol=1e-10)
            step = 1e-6
            xp = sample_target(x1, t + step, z, src, label)[0]
            xm = sample_target(x1, t - step, z, src, label)[0]
            assert_allclose(
                (xp - xm) / (2 * step),
                sample_target(x1, t, z, src, label)[1],
                rtol=1e-6,
                atol=1e-7,
            )
        errors = []
        for s in (1e-2, 1e-3, 1e-4):
            t = 1 - s
            A, _ = factor(t, src, label)
            _, hc, _, _ = schedules(t)
            mean = -hc / s * (src.Q.T @ x1.ravel())
            full_error_second_moment = A @ A.T / s**2 + np.outer(mean, mean)
            errors.append(np.linalg.norm(full_error_second_moment - Ck))
        assert errors[2] < errors[1] < errors[0]
        assert errors[-1] < 1e-5
        same = prepare_label(src, epsilon * np.linalg.inv(src.C0), epsilon)
        for t in (0.0, 0.4, 1.0):
            A, Ad = factor(t, src, same)
            assert_allclose(A, (1 - t) * src.S, atol=1e-12)
            assert_allclose(Ad, -src.S, atol=1e-12)

    def test_symmetry_geometry_and_spectra(self):
        rng = np.random.default_rng(16)
        graph = {"atomic_numbers": [6, 1, 8], "bonds": [[0, 1], [0, 2]]}
        src = prepare_source(graph)
        Q = src.Q
        assert_allclose(Q.T @ Q, np.eye(6), atol=1e-14)
        assert_allclose(
            Q @ Q.T, np.kron(np.eye(3) - np.ones((3, 3)) / 3, np.eye(3)), atol=1e-14
        )
        for positions, rank in (
            ([[0, 0, 0], [1, 0, 0], [2, 0, 0]], 2),
            ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], 3),
        ):
            rot, vib = rigid_subspaces(positions, Q)
            assert rot.shape[1] == rank
            assert_allclose(rot @ rot.T + vib @ vib.T, np.eye(6), atol=1e-14)
        x1, z = rng.normal(size=(3, 3)), rng.normal(size=6)
        M = rng.normal(size=(6, 6))
        K = M @ M.T + np.eye(6)
        label = prepare_label(src, K, 1.0)
        x, u = sample_target(x1, 0.4, z, src, label)
        assert_allclose(sample_target(x1 + 100, 0.4, z, src, label)[0], x, atol=1e-13)
        R = Rotation.random(random_state=rng).as_matrix()
        perm = np.array([2, 0, 1])
        atom_transform = np.eye(3)[perm]
        transform = np.kron(atom_transform, R)
        reduced = Q.T @ transform @ Q
        inverse = np.argsort(perm)
        perm_graph = {
            "atomic_numbers": np.array(graph["atomic_numbers"])[perm],
            "bonds": inverse[np.array(graph["bonds"])],
        }
        other = prepare_source(perm_graph)
        assert_allclose(other.C0, reduced @ src.C0 @ reduced.T, atol=1e-13)
        other_label = prepare_label(other, reduced @ K @ reduced.T, 1.0)
        xx, uu = sample_target(x1[perm] @ R.T, 0.4, reduced @ z, other, other_label)
        assert_allclose(xx, x[perm] @ R.T, atol=1e-12)
        assert_allclose(uu, u[perm] @ R.T, atol=1e-12)
        stress = Source(centered_basis(3), np.eye(6), np.eye(6))
        for spectrum in (np.geomspace(1e-12, 1e12, 6), 1 + np.arange(6) * 1e-12):
            stress_label = prepare_label(stress, np.diag(spectrum), 1.0)
            for t in (0.0, 1e-8, 0.5, 1 - 1e-8, 1.0):
                A, Ad = factor(t, stress, stress_label)
                assert np.isfinite(A).all() and np.isfinite(Ad).all()
            terminal = factor(1.0, stress, stress_label)[1]
            assert_allclose(terminal @ terminal.T, np.diag(1 / spectrum), rtol=1e-12)
        for bad in (np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), np.zeros((6, 6))):
            with self.assertRaises(ValueError):
                prepare_label(stress, bad, 1.0)
        with self.assertRaises(ValueError):
            factor(-0.1, src, label)


if __name__ == "__main__":
    unittest.main()
