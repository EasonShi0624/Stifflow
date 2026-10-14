import unittest

import numpy as np
from numpy.testing import assert_allclose
from rdkit import Chem

from stifflow.features import stereo_features
from stifflow.precision import prepare_label
from stifflow.source import prepare_source
from stifflow.targets import draw_target


class TargetChecks(unittest.TestCase):
    def test_stereo_and_reproducible_joint_target(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("F[C@](Cl)(Br)I"))
        opposite = Chem.AddHs(Chem.MolFromSmiles("F[C@@](Cl)(Br)I"))
        nodes, bonds, attrs = stereo_features(mol)
        self.assertFalse(np.array_equal(nodes, stereo_features(opposite)[0]))
        order = list(reversed(range(mol.GetNumAtoms())))
        new_nodes, new_bonds, new_attrs = stereo_features(
            Chem.RenumberAtoms(mol, order)
        )
        assert_allclose(new_nodes, nodes[order])
        original_attrs = {tuple(pair): a for pair, a in zip(bonds, attrs)}
        for pair, attr in zip(new_bonds, new_attrs):
            assert_allclose(attr, original_attrs[tuple(np.array(order)[pair])])
        e = Chem.AddHs(Chem.MolFromSmiles("F/C=C/F"))
        z = Chem.AddHs(Chem.MolFromSmiles("F/C=C\\F"))
        self.assertFalse(np.array_equal(stereo_features(e)[2], stereo_features(z)[2]))
        source = prepare_source(
            {"atomic_numbers": [6, 1, 1], "bonds": [[0, 1], [0, 2]]}
        )
        label = prepare_label(source, np.eye(6), 0.592485)
        x1 = np.arange(9, dtype=float).reshape(3, 3)
        first = draw_target(x1, source, label, "precision", np.random.default_rng(42))
        second = draw_target(x1, source, label, "precision", np.random.default_rng(42))
        for a, b in zip(first, second):
            assert_allclose(a, b)
        assert_allclose(first[0].mean(axis=0), 0, atol=1e-14)
        assert_allclose(first[1].mean(axis=0), 0, atol=1e-14)
        self.assertTrue(0 < first[2] < 1)

    def test_operator_actions_match_dense_factors_and_derivatives(self):
        from unittest.mock import patch

        from scipy.spatial.transform import Rotation

        from stifflow.precision import sample_target
        from stifflow.targets import precision_target

        rng = np.random.default_rng(23)
        for atoms in (3, 8, 29):
            source = prepare_source(
                {
                    "atomic_numbers": [6] * atoms,
                    "bonds": [[i, i + 1] for i in range(atoms - 1)],
                }
            )
            dimension = source.S.shape[0]
            V = np.linalg.qr(rng.normal(size=(dimension, dimension)))[0]
            label = {"values": np.geomspace(1e-6, 1e6, dimension), "vectors": V}
            x1, z = rng.normal(size=(atoms, 3)), rng.normal(size=dimension)
            for t in (0, 0.2, 0.5, 1 - 1e-6, 1):
                fast = precision_target(x1, t, z, source, label)
                dense = sample_target(x1, t, z, source, label)
                for actual, expected in zip(fast, dense):
                    assert_allclose(actual, expected, atol=1e-9, rtol=1e-10)
            t, step = 0.37, 1e-6
            x, velocity = precision_target(x1, t, z, source, label)
            finite_difference = (
                precision_target(x1, t + step, z, source, label)[0]
                - precision_target(x1, t - step, z, source, label)[0]
            ) / (2 * step)
            assert_allclose(velocity, finite_difference, atol=1e-7, rtol=1e-6)
            R = Rotation.random(random_state=rng).as_matrix()
            assert_allclose(x.mean(axis=0), 0, atol=1e-12)
            assert_allclose((x @ R.T).mean(axis=0), 0, atol=1e-12)
            with self.assertRaises(ValueError):
                precision_target(x1, t, z[:-1], source, label)
            fast_rng, dense_rng = np.random.default_rng(42), np.random.default_rng(42)
            fast = draw_target(x1, source, label, "precision", fast_rng)
            with patch("stifflow.targets.precision_target", sample_target):
                dense = draw_target(x1, source, label, "precision", dense_rng)
            for actual, expected in zip(fast, dense):
                assert_allclose(actual, expected, atol=1e-9, rtol=1e-10)
            self.assertEqual(
                fast_rng.bit_generator.state, dense_rng.bit_generator.state
            )


if __name__ == "__main__":
    unittest.main()
