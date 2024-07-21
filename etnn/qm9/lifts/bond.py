import torch
from rdkit.Chem.rdchem import Bond
from rdkit.Chem.rdchem import BondDir as BD
from rdkit.Chem.rdchem import BondStereo as BS
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data
from torch_geometric.utils import one_hot

from etnn.combinatorial_data import Cell

NUM_FEATURES = 0


def edge_lift(graph: Data) -> set[Cell]:
    """
    Identify edges in a graph.

    This function returns the edges of the given graph. Each edge is represented as a list of two
    node indices and an empty feature vector, since an edge as a mathematical object does not
    possess any inherent features.

    Parameters
    ----------
    graph : Data
        The input graph represented as a PyTorch Geometric Data object.

    Returns
    -------
    set[Cell]
        A set of graph elements, where each element is an edge (frozenset of two node indices) and
        an empty feature vector.

    Raises
    ------
    ValueError
        If the input graph does not contain an edge index 'edge_index'.

    Notes
    -----
    The function directly works with the PyTorch Geometric Data object. Edges are inferred from the
    edge_index attribute.

    Attributes
    ----------
    num_features : int
        The number of features for each edge/bond.
    """

    if (not hasattr(graph, "edge_index")) or (graph.edge_index is None):
        raise ValueError("The given graph does not have an edge index 'edge_index'!")

    # Create edges
    edges = {
        frozenset(edge)
        for edge in graph.edge_index.t().tolist()
        if len(edge) == 2
        if edge[0] != edge[1]
    }
    dummy_features = tuple(range(NUM_FEATURES))
    return {(edge, dummy_features) for edge in edges}


edge_lift.num_features = NUM_FEATURES

# we do not consider all bond types, only the most common ones
bond_types = {
    bond_type: torch.tensor([idx])
    for idx, bond_type in enumerate([BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC])
}
# rdkit version 2023.03.3 returns 6 stereos
bond_stereos = {stereo: torch.tensor([idx]) for idx, stereo in BS.values.items()}

# bond directions
bond_directions = {direction: torch.tensor([idx]) for idx, direction in BD.values.items()}


def bond_lift(graph: Data) -> set[Cell]:
    """
    Compute the bond lifts for a given molecular graph.

    Parameters
    ----------
    graph : Data
        The input molecular graph.

    Returns
    -------
    set[Cell]
        A set of bond lifts, where each bond lift is represented as a tuple
        containing a frozenset of node indices and the computed bond features.

    """
    cells = set()
    mol = graph.mol
    for bond in mol.GetBonds():
        node_idc = frozenset([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        cells.add((node_idc, compute_bond_features(bond)))
    return cells


bond_lift.num_features = len(bond_types) + 2 + len(bond_stereos) + len(bond_directions)


def compute_bond_features(bond: Bond) -> tuple[float]:
    """
    Compute bond features.

    This function computes the bond features for a given bond. The features include the bond type,
    whether the bond is conjugated, whether the bond is in a ring, the bond stereochemistry and the
    bond direction.

    Parameters
    ----------
    bond : Bond
        The input bond.

    Returns
    -------
    tuple[float]
        The bond features, including the bond type, whether the bond is conjugated, whether the bond
        is in a ring, and the bond stereochemistry.

    Notes
    -----
    The bond type is encoded as follows: single=0, double=1, triple=2, aromatic=3. The conjugation
    status is encoded as 1 if the bond is conjugated and 0 otherwise. The ring status is encoded as
    1 if the bond is in a ring and 0 otherwise. The bond stereochemistry is encoded based on the
    RDKit BondStereo values. The bond direction is encoded based on the RDKit BondDir values.

    """
    # Get the bond type and encode it as a one-hot tensor
    bond_type = bond_types[bond.GetBondType()]
    bond_type_oh = one_hot(bond_type, num_classes=len(bond_types)).flatten()  # 1d tensor

    # Get the conjugation status of the bond
    is_conjugated = float(bond.GetIsConjugated())  # float

    # Get the ring status of the bond
    is_in_ring = float(bond.IsInRing())  # float

    # Get the bond stereochemistry and encode it as a one-hot tensor
    bond_stereo = bond_stereos[bond.GetStereo()]
    bond_stereo_oh = one_hot(bond_stereo, num_classes=len(bond_stereos)).flatten()  # 1d tensor

    # Get the bond direction and encode it as a one-hot tensor
    bond_dir = bond_directions[bond.GetBondDir()]
    bond_dir_oh = one_hot(bond_dir, num_classes=len(bond_directions)).flatten()  # 1d tensor

    # Combine all the bond features into a single tuple
    bond_features = bond_type_oh.tolist()
    bond_features.append(is_conjugated)
    bond_features.append(is_in_ring)
    bond_features.extend(bond_stereo_oh.tolist())
    bond_features.extend(bond_dir_oh.tolist())

    return tuple(bond_features)
