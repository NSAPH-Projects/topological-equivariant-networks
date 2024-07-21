import networkx as nx
import torch_geometric.utils as pyg_utils
from rdkit import Chem
from torch_geometric.data import Data

from etnn.combinatorial_data import Cell

NUM_FEATURES = 0


def cycle_lift(graph: Data) -> set[Cell]:
    """
    Identify minimal cycles in a graph.

    This function finds all cycles in a given graph and then filters out those that contain simpler
    cycles within themselves. It returns minimal cycles, which are those that do not encompass any
    smaller cycle. The minimal cycle length is 3.

    Parameters
    ----------
    graph : torch.Tensor
        The input graph represented as a PyTorch tensor.

    Returns
    -------
    set[Cell]
        A set of minimal cycles, each cycle is represented as a frozenset of node indices and a
        corresponding feature vector.

    Raises
    ------
    ValueError
        If the input graph does not contain an edge index 'edge_index'.

    Attributes
    ----------
    num_features : int
        The number of features for each node. It is set to 0, since cycles do not have inherent
        features.
    """

    if (not hasattr(graph, "edge_index")) or (graph.edge_index is None):
        raise ValueError("The given graph does not have an edge index 'edge_index'!")

    # Convert to networkx graph
    G = pyg_utils.to_networkx(graph, to_undirected=True)

    # Compute all cycles (using a set with sorting removes duplicates)
    cycles = {frozenset(cycle) for cycle in nx.simple_cycles(G) if len(cycle) >= 3}

    # Filter out cycles that contain simpler cycles within themselves
    minimal_cycles = set()
    for cycle in cycles:
        if not any(cycle > other_cycle for other_cycle in cycles if cycle != other_cycle):
            minimal_cycles.add(cycle)

    # Add feature vectors
    dummy_features = tuple(range(NUM_FEATURES))
    minimal_cycles = {(cycle, dummy_features) for cycle in minimal_cycles}
    return minimal_cycles


cycle_lift.num_features = NUM_FEATURES


def ring_lift(graph: Data) -> set[Cell]:
    """
    Identify rings in a graph.

    This function identifies rings in a given graph and returns them as a set of cells. Each cell
    represents a ring and consists of a frozenset of node indices and a feature vector.It only
    returns minimal rings, which are those that do not encompass any smaller ring.

    Parameters
    ----------
    graph : torch.Tensor
        The input graph represented as a PyTorch tensor.

    Returns
    -------
    set[Cell]
        A set of cells, each representing a ring in the graph. Each cell consists of a frozenset of
        node indices and a feature vector.

    Raises
    ------
    ValueError
        If the input graph does not contain an edge index 'edge_index'.

    Attributes
    ----------
    num_features : int
        The number of features for each node in the ring. It is set to 4, representing the ring
        size, whether the ring is aromatic, whether the ring has a heteroatom, and whether the ring
        is saturated.

    Notes
    -----
    The function uses the RDKit library to extract ring information from the graph.

    """
    cells = set()
    mol = graph.mol
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        node_idc = frozenset(ring)
        feature_vector = compute_ring_features(node_idc, mol)
        cells.add((node_idc, feature_vector))

    # Remove rings with less than 3 atoms
    cells = {cell for cell in cells if len(cell[0]) >= 3}

    # Filter out rings that contain simpler rings within themselves
    filtered_cells = {
        cell
        for cell in cells
        if not any(cell[0] > other_cell[0] for other_cell in cells if cell != other_cell)
    }

    return filtered_cells


ring_lift.num_features = 4


def compute_ring_features(ring: frozenset[int], molecule: Chem.Mol) -> tuple[float]:
    """
    Compute features for a ring in a molecule.

    This function computes features for a ring in a molecule. The features include the ring size,
    whether the ring is aromatic, whether the ring has a heteroatom, and whether the ring is
    saturated.

    Parameters
    ----------
    ring : frozenset[int]
        A set of atom indices representing the ring.
    molecule : Chem.Mol
        The RDKit molecule object representing the molecule.

    Returns
    -------
    tuple[float]
        A tuple of features for the ring, including the ring size, whether the ring is aromatic,
        whether the ring has a heteroatom, and whether the ring is saturated.

    Notes
    -----
    The function uses the RDKit library to extract ring information from the molecule.

    """
    ring_atoms = [molecule.GetAtomWithIdx(idx) for idx in ring]
    ring_size = float(len(ring))
    is_aromatic = float(all(atom.GetIsAromatic() for atom in ring_atoms))
    has_heteroatom = float(any(atom.GetSymbol() not in ("C", "H") for atom in ring_atoms))
    is_saturated = float(
        all(atom.GetHybridization() == Chem.HybridizationType.SP3 for atom in ring_atoms)
    )
    return (ring_size, is_aromatic, has_heteroatom, is_saturated)
