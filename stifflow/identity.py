"""Stereo-preserving identities and indexed graphs for split metadata only."""

from rdkit import Chem


def parse_smiles(smiles):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        params.sanitize = False
        mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError(f"Unparseable molecule: {smiles}")
    mol.UpdatePropertyCache(strict=False)
    return mol


def molecule_identity(smiles):
    mol = parse_smiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    # Remove ordinary explicit H consistently even for sanitize=False aliases;
    # RDKit retains isotopic H and stereo. Never change the stored training graph.
    mol = Chem.RemoveHs(mol, updateExplicitCount=True, sanitize=False)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def ordered_signature(smiles):
    mol = parse_smiles(smiles)
    maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    if sorted(maps) != list(range(mol.GetNumAtoms())):
        return None
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    atoms = sorted(mol.GetAtoms(), key=lambda a: a.GetAtomMapNum())
    atom_signature = tuple(
        (
            a.GetAtomicNum(),
            a.GetFormalCharge(),
            a.GetIsotope(),
            a.GetIsAromatic(),
            a.GetTotalNumHs(includeNeighbors=True),
            a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else "",
        )
        for a in atoms
    )
    bonds = tuple(
        sorted(
            (
                min(b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()),
                max(b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()),
                str(b.GetBondType()),
                str(b.GetStereo()),
            )
            for b in mol.GetBonds()
        )
    )
    return atom_signature, bonds


class SplitMapper:
    """Resolve a list key to exactly one inventory group, never the first tie."""

    def __init__(self, uniques, uid_groups):
        self.uniques = uniques
        self.uid_groups = uid_groups
        self.exact = {str(s): i for i, s in enumerate(uniques)}
        self.identities = None
        self.signatures = {}

    def resolve(self, smiles):
        if smiles in self.exact:
            uid = self.exact[smiles]
            groups = self.uid_groups[uid]
            return (uid, "exact") if len(groups) == 1 else (None, "ambiguous_group")
        if self.identities is None:
            self.identities = {}
            for uid, key in enumerate(self.uniques):
                try:
                    identity = molecule_identity(str(key))
                except (ValueError, RuntimeError):
                    continue
                self.identities.setdefault(identity, []).append(uid)
        try:
            ids = self.identities.get(molecule_identity(smiles), [])
            signature = ordered_signature(smiles)
        except (ValueError, RuntimeError):
            return None, "unparseable"
        if not ids:
            return None, "unmapped"
        if signature is not None:
            matched = []
            for uid in ids:
                if uid not in self.signatures:
                    try:
                        self.signatures[uid] = ordered_signature(str(self.uniques[uid]))
                    except (ValueError, RuntimeError):
                        self.signatures[uid] = None
                if self.signatures[uid] == signature:
                    matched.append(uid)
            ids = matched
            if not ids:
                return None, "fingerprint_miss"
        groups = {g for uid in ids for g in self.uid_groups[uid]}
        if len(groups) != 1:
            return None, "ambiguous_group"
        method = "fingerprint" if signature is not None else "canonical_group"
        return min(ids), method
