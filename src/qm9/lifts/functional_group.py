from collections import namedtuple

from rdkit import Chem
from torch.geometric.data import Data

from .common import Cell


def functional_group_lift(graph: Data) -> set[Cell]:
    """
    Identify functional groups within a molecule and returns them as lists of atom indices.

    This function first checks if the input `graph` contains a SMILES attribute. If present, it
    converts the SMILES string into an RDKit molecule object and then identifies functional groups
    within this molecule. Each functional group is represented as a list of atom indices. If the
    input does not contain a valid SMILES attribute, the function raises an AttributeError. If the
    molecule cannot be processed, it returns an empty list.

    Parameters
    ----------
    graph : torch_geometric.data.Data
        A data structure containing a SMILES representation of the molecule.

    Returns
    -------
    set[Cell]
        A set of frozensets, where each frozenset contains the atom indices of a functional group in
        the molecule.

    Raises
    ------
    AttributeError
        If the input `graph` does not have a valid SMILES attribute or if the SMILES string cannot
        be converted into an RDKit molecule.

    Examples
    --------
    >>> graph = Data(smiles='CC(=O)OC1=CC=CC=C1C(=O)O')
    >>> functional_group_lift(graph)
    [[1, 2, 3], [8, 9, 10, 11, 12, 13, 14]]
    """
    if not hasattr(graph, "smiles"):
        raise AttributeError(
            "The given graph does not have a SMILES attribute! You are either not "
            "using the QM9 dataset or you haven't preprocessed the dataset using rdkit!"
        )
    try:
        molecule = Chem.MolFromSmiles(graph.smiles)
        functional_groups = identify_functional_groups(molecule)
        return {frozenset(fg.atomIds) for fg in functional_groups if len(fg.atomIds) >= 3}
    except AttributeError:
        return set()


# The code below was taken from https://github.com/rdkit/rdkit/blob/master/Contrib/IFG/ifg.py
# and serves to identify functional groups within a molecule.

#  Original authors: Richard Hall and Guillaume Godin
#  This file is part of the RDKit.
#  The contents are covered by the terms of the BSD license
#  which is included in the file license.txt, found at the root
#  of the RDKit source tree.


# Richard hall 2017
# IFG main code
# Guillaume Godin 2017
# refine output function
# astex_ifg: identify functional groups a la Ertl, J. Cheminform (2017) 9:36


def merge(mol, marked, aset):
    bset = set()
    for idx in aset:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            jdx = nbr.GetIdx()
            if jdx in marked:
                marked.remove(jdx)
                bset.add(jdx)
    if not bset:
        return
    merge(mol, marked, bset)
    aset.update(bset)


# atoms connected by non-aromatic double or triple bond to any heteroatom
# c=O should not match (see fig1, box 15).  I think using A instead of * should sort that out?
PATT_DOUBLE_TRIPLE = Chem.MolFromSmarts("A=,#[!#6]")
# atoms in non aromatic carbon-carbon double or triple bonds
PATT_CC_DOUBLE_TRIPLE = Chem.MolFromSmarts("C=,#C")
# acetal carbons, i.e. sp3 carbons connected to tow or more oxygens, nitrogens or sulfurs; these O,
# N or S atoms must have only single bonds
PATT_ACETAL = Chem.MolFromSmarts("[CX4](-[O,N,S])-[O,N,S]")
# all atoms in oxirane, aziridine and thiirane rings
PATT_OXIRANE_ETC = Chem.MolFromSmarts("[O,N,S]1CC1")

PATT_TUPLE = (PATT_DOUBLE_TRIPLE, PATT_CC_DOUBLE_TRIPLE, PATT_ACETAL, PATT_OXIRANE_ETC)


def identify_functional_groups(mol):
    marked = set()
    # mark all heteroatoms in a molecule, including halogens
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (6, 1):  # would we ever have hydrogen?
            marked.add(atom.GetIdx())

    # mark the four specific types of carbon atom
    for patt in PATT_TUPLE:
        for path in mol.GetSubstructMatches(patt):
            for atomindex in path:
                marked.add(atomindex)

    # merge all connected marked atoms to a single FG
    groups = []
    while marked:
        grp = set([marked.pop()])
        merge(mol, marked, grp)
        groups.append(grp)

    # extract also connected unmarked carbon atoms
    ifg = namedtuple("IFG", ["atomIds", "atoms", "type"])
    ifgs = []
    for g in groups:
        uca = set()
        for atomidx in g:
            for n in mol.GetAtomWithIdx(atomidx).GetNeighbors():
                if n.GetAtomicNum() == 6:
                    uca.add(n.GetIdx())
        ifgs.append(
            ifg(
                atomIds=tuple(list(g)),
                atoms=Chem.MolFragmentToSmiles(mol, g, canonical=True),
                type=Chem.MolFragmentToSmiles(mol, g.union(uca), canonical=True),
            )
        )
    return ifgs
