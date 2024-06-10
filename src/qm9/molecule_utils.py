from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem


def molecule_from_data(data: any, return_sanitized: bool = False) -> Optional[Chem.Mol]:
    """
    Create an RDKit molecule object from data containing a SMILES string.

    Parameters
    ----------
    data : any
        Data object containing a SMILES string.

    Returns
    -------
    Optional[Chem.Mol]
        An RDKit molecule object or None if conversion fails.
    bool
        Indicates whether the molecule was successfully sanitized.
    """
    try:
        mol = Chem.MolFromSmiles(data.smiles, sanitize=True)
        if mol is None:
            raise ValueError("Molecule could not be created from SMILES string.")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        is_sanitized = True
    except ValueError:
        mol = Chem.MolFromSmiles(data.smiles, sanitize=False)
        is_sanitized = False

    if return_sanitized:
        return mol, is_sanitized

    else:
        return mol
