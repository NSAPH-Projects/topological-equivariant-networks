"""Module for constructing topological structures from graphs."""

from itertools import combinations

import gudhi
from torch_geometric.data import Data


def clique_lift(graph: Data) -> list[list[int]]:
    """
    Construct a clique complex from a graph. Not yet implemented.

    Parameters
    ----------
    graph : object
        The graph from which to construct the clique complex.

    Returns
    -------
    list[list[int]]
        Simplices of the clique complex.

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError


def rips_lift(graph: Data, dim: int, dis: float, fc_nodes: bool = True) -> list[list[int]]:
    """
    Construct a Rips complex from a graph and returns its simplices.

    Parameters
    ----------
    graph : object
        A graph object containing vertices 'x' and their positions 'pos'.
    dim : int
        Maximum dimension of simplices in the Rips complex.
    dis : float
        Maximum distance between any two points in a simplex.
    fc_nodes : bool, optional
        If True, force inclusion of all edges as 1-dimensional simplices.
        Default is True.

    Returns
    -------
    list[list[int]]
        A list of lists, where each sublist represents a simplex in the Rips
        complex. Each simplex is a list of vertex indices.

    Notes
    -----
    The function uses the `gudhi` library to construct the Rips complex. It
    first converts the graph positions to a list of points, then generates the
    Rips complex and its simplex tree up to the specified dimension and edge
    length. Optionally, it includes all nodes as 0-dimensional simplices.
    Finally, it extracts and returns the simplices from the simplex tree.
    """
    # create simplicial complex
    x_0, pos = graph.x, graph.pos
    points = [pos[i].tolist() for i in range(pos.shape[0])]
    rips_complex = gudhi.RipsComplex(points=points, max_edge_length=dis)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=dim)

    if fc_nodes:
        nodes = list(range(x_0.shape[0]))
        for edge in combinations(nodes, 2):
            simplex_tree.insert(edge)

    # convert simplicial complex to list of lists
    simplexes = []
    for simplex, _ in simplex_tree.get_simplices():
        simplexes.append(simplex)

    return simplexes
