from torch_geometric import Data

from .common import Cell


def supercell_lift(graph: Data) -> set[Cell]:
    """
    Return the entire graph as a single cell.

    This function returns the entire graph as a single cell, represented as a frozenset of all
    node indices. If the graph has less than 2 nodes, it returns an empty set.

    Parameters
    ----------
    graph : Data
        The input graph represented as a PyTorch Geometric Data object.

    Returns
    -------
    set[Cell]
        A singleton set containing a frozenset of all node indices.

    Raises
    ------
    ValueError
        If the input graph does not contain a feature matrix 'x'.
    """

    if (not hasattr(graph, "x")) or (graph.x is None):
        raise ValueError("The given graph does not have a feature matrix 'x'!")
    num_nodes = graph.x.size(0)
    if num_nodes < 2:
        return set()
    else:
        return {frozenset([node for node in range(num_nodes)])}
