import pytest
from rdkit import Chem

from stifflow.identity import SplitMapper, molecule_identity, ordered_signature


@pytest.mark.parametrize(
    "left,right",
    [
        ("[NH4]", "N([H])([H])([H])[H]"),
        ("[CH5]", "C([H])([H])([H])([H])[H]"),
        ("CC", "[CH3:4][CH3:9]"),
    ],
)
def test_identity_normalizes_hydrogens_and_maps(left, right):
    assert molecule_identity(left) == molecule_identity(right)


def test_identity_preserves_stereo_isotopes_and_charge():
    assert molecule_identity("F[C@](Cl)(Br)I") != molecule_identity("F[C@@](Cl)(Br)I")
    assert molecule_identity("[2H]O") != molecule_identity("[H]O")
    assert molecule_identity("[NH4+]") != molecule_identity("[NH4]")


def mapped_pair():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i)
    other = Chem.Mol(mol)
    for atom in other.GetAtoms():
        if atom.GetAtomMapNum() == 2:
            atom.SetAtomMapNum(5)
        elif atom.GetAtomMapNum() == 5:
            atom.SetAtomMapNum(2)
    return mol, other


def test_full_signature_retains_hydrogen_connectivity():
    mol, other = mapped_pair()
    left, right = [Chem.MolToSmiles(m, canonical=False) for m in (mol, other)]
    assert molecule_identity(left) == molecule_identity(right)
    assert ordered_signature(left) != ordered_signature(right)
    assert len(ordered_signature(left)[0]) == mol.GetNumAtoms()


def test_ties_are_allowed_only_inside_one_inventory_group():
    mol, _ = mapped_pair()
    keys = [Chem.MolToSmiles(mol, canonical=False, rootedAtAtom=i) for i in (0, 1)]
    target = Chem.MolToSmiles(mol, canonical=False, rootedAtAtom=2)
    assert target not in keys
    assert SplitMapper(keys, [{0}, {1}]).resolve(target) == (None, "ambiguous_group")
    for order in (keys, keys[::-1]):
        uid, method = SplitMapper(order, [{7}, {7}]).resolve(target)
        assert uid is not None and method == "fingerprint"
