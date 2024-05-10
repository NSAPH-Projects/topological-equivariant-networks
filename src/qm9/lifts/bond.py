from torch_geometric.data import Data

from .common import Cell


def edge_lift(graph: Data) -> set[Cell]:
    """
    Identify edges in a graph.

    This function returns the edges of the given graph. Each edge is represented as a list of two
    node indices.

    Parameters
    ----------
    graph : Data
        The input graph represented as a PyTorch Geometric Data object.

    Returns
    -------
    set[Cell]
        A set of graph elements, where each element is an edge (frozenset of two node indices) and
        its associated feature vector.

    Raises
    ------
    ValueError
        If the input graph does not contain an edge index 'edge_index'.

    Notes
    -----
    The function directly works with the PyTorch Geometric Data object. Edges are inferred from the
    edge_index attribute.
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

    return {(edge, ()) for edge in edges}
