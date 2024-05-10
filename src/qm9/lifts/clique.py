from itertools import combinations

import networkx as nx
import torch_geometric.utils as pyg_utils

from qm9.lifts.common import Cell


def clique_lift(graph_data) -> set[Cell]:
    """
    Construct a clique complex from a graph represented as a torch_geometric.data.Data object.

    Parameters
    ----------
    graph_data : torch_geometric.data.Data
        The graph from which to construct the clique complex, represented as a
        torch_geometric.data.Data object.

    Returns
    -------
    set[Cell]
        Simplices of the clique complex. A 0-dimensional feature vector is attached to each simplex
        as a clique has no inherent features.
    """
    # Convert torch_geometric.data.Data to networkx graph
    G = pyg_utils.to_networkx(graph_data, to_undirected=True)

    simplices = set()

    # Find all maximal cliques in the graph
    maximal_cliques = nx.find_cliques(G)

    # Generate all subsets of each maximal clique to include all cliques
    for clique in maximal_cliques:
        for i in range(1, len(clique) + 1):
            simplices.update(frozenset(subset) for subset in combinations(clique, i))

    # Add 0-dimensional feature vectors to each simplex
    simplices = {(simplex, ()) for simplex in simplices}
    return simplices
