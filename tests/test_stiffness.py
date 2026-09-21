import unittest

import numpy as np
from numpy.testing import assert_allclose
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation
from stifflow.geometry import centered_basis
from stifflow.precision import factor, prepare_label
from stifflow.source import prepare_source
from stifflow.stiffness import estimate_stiffness
from stifflow_alt_a.path import prepare_label as alt_label


class StiffnessChecks(unittest.TestCase):
    def test_stationary_curvature_and_rotation(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
        self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=42), 0)
        original = mol.GetConformer().GetPositions().copy()
        label = estimate_stiffness(mol)
        assert_allclose(mol.GetConformer().GetPositions(), original, atol=0.0)
        assert_allclose(label["target"], original, atol=0.0)
        self.assertEqual(label["audit"]["rotation_rank"], 3)
        self.assertLess(label["audit"]["half_step_relative_error"], 1e-4)
        self.assertGreater(np.linalg.eigvalsh(label["K"])[0], 0)
        ratios = np.array([item["ratio"] for item in label["audit"]["calibration"]])
        self.assertLess(abs(ratios[-1] - 1), abs(ratios[0] - 1))
        self.assertLess(abs(ratios[-1] - 1), 1e-3)
        source = prepare_source(
            {
                "atomic_numbers": [a.GetAtomicNum() for a in mol.GetAtoms()],
                "bonds": [
                    [b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()
                ],
            }
        )
        prec = prepare_label(source, label["K"], label["epsilon_E"])
        alt = alt_label(source, label["K"], label["epsilon_E"])
        self.assertLess(alt["covariance_residual"], 1e-10)
        _, Ad = factor(1.0, source, prec)
        assert_allclose(
            Ad @ Ad.T, label["epsilon_E"] * np.linalg.inv(label["K"]), atol=1e-10
        )
        # A rotated stationary reference should not change physical K; regenerate
        # at the already relaxed minimum to separate alignment from relaxation.
        R = Rotation.from_rotvec([0.3, -0.5, 0.7]).as_matrix()
        reference_mol = Chem.Mol(mol)
        for i, p in enumerate(label["reference"] @ R.T):
            reference_mol.GetConformer().SetAtomPosition(i, p)
        transformed = estimate_stiffness(reference_mol)
        Q = centered_basis(mol.GetNumAtoms())
        O = Q.T @ np.kron(np.eye(mol.GetNumAtoms()), R) @ Q
        assert_allclose(transformed["K"], O @ label["K"] @ O.T, rtol=1e-5, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
