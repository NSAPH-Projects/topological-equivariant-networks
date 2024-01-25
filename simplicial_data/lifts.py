"""Module for constructing topological structures from graphs."""

from itertools import combinations

import gudhi
import networkx as nx
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data


def clique_lift(graph_data) -> list[list[int]]:
    """
    Construct a clique complex from a graph represented as a torch_geometric.data.Data object.

    Parameters
    ----------
    graph_data : torch_geometric.data.Data
        The graph from which to construct the clique complex, represented as a
        torch_geometric.data.Data object.

    Returns
    -------
    list[list[int]]
        Simplices of the clique complex.
    """
    # Convert torch_geometric.data.Data to networkx graph
    G = pyg_utils.to_networkx(graph_data, to_undirected=True)

    simplices = []

    # Find all maximal cliques in the graph
    maximal_cliques = list(nx.find_cliques(G))

    # Generate all subsets of each maximal clique to include all cliques
    for clique in maximal_cliques:
        for i in range(1, len(clique) + 1):
            for subset in combinations(clique, i):
                simplices.append(list(subset))

    # Remove duplicates by converting each simplex to a tuple (for hashing),
    # making a set (to remove duplicates), and then back to a list
    simplices = list({tuple(sorted(simplex)) for simplex in simplices})
    simplices = [list(simplex) for simplex in simplices]

    return simplices


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
