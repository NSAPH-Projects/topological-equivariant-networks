import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Atom
from torch_geometric.data import Data
from torch_geometric.utils import one_hot

from etnn.combinatorial_data import Cell

NUM_FEATURES = 15
DUMMY_NUM_FEATURES = 0


def node_lift(graph: Data) -> set[Cell]:
    """
    Identify nodes in a graph.

    This function returns the nodes of the given graph. Each node is represented as a
    singleton list containing its index.

    Parameters
    ----------
    graph : Data
        The input graph represented as a PyTorch Geometric Data object.

    Returns
    -------
    set[Cell]
        A set of tuples, where each tuple is a singleton set of node indices and a feature vector.

    Raises
    ------
    ValueError
        If the input graph does not contain a feature matrix 'x'.

    Notes
    -----
    The function directly works with the PyTorch Geometric Data object. Nodes are inferred from the
    x attribute.

    Attributes
    ----------
    num_features : int
        The number of features for each node.
    """

    if (not hasattr(graph, "x")) or (graph.x is None):
        raise ValueError("The given graph does not have a feature matrix 'x'!")

    # Create nodes
    dummy_features = tuple(range(DUMMY_NUM_FEATURES))
    nodes = {(frozenset([node]), dummy_features) for node in range(graph.x.size(0))}

    return nodes


node_lift.num_features = DUMMY_NUM_FEATURES


def atom_lift(graph: Data) -> set[Cell]:
    """
    Identify atoms in a graph.

    This function returns the atoms of the given graph. Each atom is represented as a tuple
    containing a singleton set of atom index and a feature vector.

    Parameters
    ----------
    graph : Data
        The input graph represented as a PyTorch Geometric Data object.

    Returns
    -------
    set[Cell]
        A set of tuples, where each tuple is a singleton set of atom index and a feature vector.

    Raises
    ------
    ValueError
        If the input graph does not contain a feature matrix 'x'.

    Notes
    -----
    The function converts the input graph to an RDKit molecule and then iterates over each atom
    to create the atom nodes. Atom features are computed using the `compute_atom_features` function.

    Attributes
    ----------
    num_features : int
        The number of features for each atom.
    """
    # Convert graph to RDKit molecule
    molecule = graph.mol

    # Create atoms
    atoms = set()
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        atoms.add((frozenset([index]), compute_atom_features(atom)))

    return atoms


atom_lift.num_features = NUM_FEATURES

# These are the only elements that appear in QM9
atomic_numbers = {num: torch.tensor([idx]) for idx, num in enumerate([1, 6, 7, 8, 9])}


def compute_atom_features(atom: Atom) -> tuple[float]:
    """
    Compute features for a given atom.

    Parameters
    ----------
    atom : Atom
        The atom object for which features are computed.

    Returns
    -------
    tuple[float]
        A tuple of computed features for the atom.

    Notes
    -----
    This function computes features for a given atom based on its atomic number.
    The computed features include:
    - One-hot encoding of the atomic number
    - One-hot encoding of the atomic number multiplied by the atomic number fraction
    - One-hot encoding of the atomic number multiplied by the atomic number fraction squared

    """
    atomic_number = atom.GetAtomicNum()
    element_enc = atomic_numbers[atomic_number]
    element_oh = one_hot(element_enc, num_classes=len(atomic_numbers)).flatten()  # 1d tensor
    max_atomic_number = max(atomic_numbers.keys())

    atomic_num_fraction = atomic_number / max_atomic_number
    feature_vector = torch.cat(
        (element_oh, atomic_num_fraction * element_oh, (atomic_num_fraction**2) * element_oh)
    )
    return tuple(feature_vector.tolist())
