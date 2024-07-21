from torch_geometric.data import Data

from etnn.combinatorial_data import Cell

NUM_FEATURES = 0


def supercell_lift(graph: Data) -> set[Cell]:
    """
    Return the entire graph as a single cell.

    This function returns the entire graph as a single cell, represented as a frozenset of all node
    indices and a corresponding feature vector. If the graph has less than 2 nodes, it returns an
    empty set.

    Parameters
    ----------
    graph : Data
        The input graph represented as a PyTorch Geometric Data object.

    Returns
    -------
    set[Cell]
        A singleton set containing a frozenset of all node indices and a corresponding feature
        vector.

    Raises
    ------
    ValueError
        If the input graph does not contain a feature matrix 'x'.

    Attributes
    ----------
    num_features : int
        The number of features for each node.
    """

    if (not hasattr(graph, "x")) or (graph.x is None):
        raise ValueError("The given graph does not have a feature matrix 'x'!")
    num_nodes = graph.x.size(0)
    if num_nodes < 2:
        return set()
    else:
        dummy_features = tuple(range(NUM_FEATURES))
        return {(frozenset([node for node in range(num_nodes)]), dummy_features)}


supercell_lift.num_features = NUM_FEATURES
