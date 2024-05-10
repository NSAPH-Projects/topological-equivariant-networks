from torch_geometric.data import Data

from .common import Cell

NUM_FEATURES = 15


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
    dummy_features = tuple(range(NUM_FEATURES))
    nodes = {(frozenset([node]), dummy_features) for node in range(graph.x.size(0))}

    return nodes


node_lift.num_features = NUM_FEATURES
