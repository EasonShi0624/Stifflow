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


if __name__ == "__main__":
    unittest.main()
