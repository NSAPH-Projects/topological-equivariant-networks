from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem


def molecule_from_data(data: any) -> Optional[Chem.Mol]:
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
    """
    if hasattr(data, "smiles") and data.smiles:
        try:
            mol = Chem.MolFromSmiles(data.smiles, sanitize=True)
            if mol is None:
                raise ValueError("Molecule could not be created from SMILES string.")
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        except ValueError:
            mol = Chem.MolFromSmiles(data.smiles, sanitize=False)

        return mol
    return None
