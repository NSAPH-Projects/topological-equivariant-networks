import networkx as nx
import torch_geometric.utils as pyg_utils
from torch_geometric import Data

from .common import Cell


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
        A set of minimal cycles, each cycle is represented as a frozenset of node indices.

    Raises
    ------
    ValueError
        If the input graph does not contain an edge index 'edge_index'.
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

    return minimal_cycles
