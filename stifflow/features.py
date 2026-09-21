"""Stereo corrections for new StifFlow models; official checkpoint inputs stay intact."""

import numpy as np
from rdkit import Chem


def stereo_features(mol):
    mol = Chem.Mol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    # Absolute CIP survives atom relabeling; neighbor-order CW/CCW does not.
    cip = {"R": 1, "S": 2, "r": 3, "s": 4}
    nodes = np.array(
        [
            (
                cip.get(a.GetProp("_CIPCode"), 5)
                if a.HasProp("_CIPCode")
                else (0 if a.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED else 5)
            )
            for a in mol.GetAtoms()
        ],
        dtype=np.float32,
    )
    kinds = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
    stereo = [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ]
    bonds, attributes = [], []
    for b in mol.GetBonds():
        kind, configuration = str(b.GetBondType()), str(b.GetStereo())
        attr = [
            kinds.index(kind) + 1 if kind in kinds else 5,
            stereo.index(configuration) if configuration in stereo else 6,
            int(b.GetIsConjugated()),
        ]
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonds.extend([[i, j], [j, i]])
        attributes.extend([attr, attr])
    return (
        nodes,
        np.asarray(bonds, dtype=np.int64).reshape(-1, 2),
        np.asarray(attributes, dtype=np.float32).reshape(-1, 3),
    )


def graph_from_mol(mol):
    """Reuse upstream ten atom features, replacing only the broken chirality slot."""
    from etflow.commons.featurization import MoleculeFeaturizer

    stereo, bonds, attributes = stereo_features(mol)
    nodes = MoleculeFeaturizer().get_atom_features_from_mol(mol).numpy()
    nodes[:, 0] = stereo
    return {
        "atomic_numbers": np.array([a.GetAtomicNum() for a in mol.GetAtoms()]),
        "bonds": bonds,
        "node_attr": nodes,
        "bond_attr": attributes,
    }
