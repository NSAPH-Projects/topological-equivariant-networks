from torch.geometric.data import Data

from .common import Cell


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
        A set of graph elements, where each element is a node (singleton frozenset).

    Raises
    ------
    ValueError
        If the input graph does not contain a feature matrix 'x'.

    Notes
    -----
    The function directly works with the PyTorch Geometric Data object. Nodes are inferred from the
    x attribute.
    """

    if (not hasattr(graph, "x")) or (graph.x is None):
        raise ValueError("The given graph does not have a feature matrix 'x'!")

    # Create nodes
    nodes = {frozenset([node]) for node in range(graph.x.size(0))}

    return nodes
