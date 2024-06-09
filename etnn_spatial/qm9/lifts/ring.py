import networkx as nx
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data

from .common import Cell

NUM_FEATURES = 4


def ring_lift(graph: Data) -> set[Cell]:
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
        The number of features for each node.
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


ring_lift.num_features = NUM_FEATURES
