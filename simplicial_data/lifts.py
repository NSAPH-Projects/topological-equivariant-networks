"""Module for constructing topological structures from graphs."""

from functools import partial
from itertools import combinations

import gudhi
import networkx as nx
import torch_geometric.utils as pyg_utils
from rdkit import Chem
from torch_geometric.data import Data

from simplicial_data.ifg import identify_functional_groups


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


def functional_group_lift(graph: Data) -> list[list[int]]:
    """
    Identify functional groups within a molecule and returns them as lists of atom indices.

    This function first checks if the input `graph` contains a SMILES attribute. If present, it
    converts the SMILES string into an RDKit molecule object and then identifies functional groups
    within this molecule. Each functional group is represented as a list of atom indices. If the
    input does not contain a valid SMILES attribute, the function raises an AttributeError. If the
    molecule cannot be processed, it returns an empty list.

    Parameters
    ----------
    graph : torch_geometric.data.Data
        A data structure containing a SMILES representation of the molecule.

    Returns
    -------
    List[List[int]]
        A list of lists, where each inner list contains the atom indices of a functional group in
        the molecule.

    Raises
    ------
    AttributeError
        If the input `graph` does not have a valid SMILES attribute or if the SMILES string cannot
        be converted into an RDKit molecule.

    Examples
    --------
    >>> graph = Data(smiles='CC(=O)OC1=CC=CC=C1C(=O)O')
    >>> functional_group_lift(graph)
    [[1, 2, 3], [8, 9, 10, 11, 12, 13, 14]]
    """
    if not hasattr(graph, "smiles"):
        raise AttributeError(
            "The given graph does not have a SMILES attribute! You are either not "
            "using the QM9 dataset or you haven't preprocessed the dataset using rdkit!"
        )
    try:
        molecule = Chem.MolFromSmiles(graph.smiles)
        functional_groups = identify_functional_groups(molecule)
        simplexes = [list(fg.atomIds) for fg in functional_groups]
    except AttributeError:
        simplexes = []
    return simplexes


def identity_lift(graph: Data) -> list[list[int]]:
    """
    Identify nodes and edges in a graph.

    This function returns the nodes and edges of the given graph. Each node
    is represented as a singleton list containing its index, and each edge is
    represented as a list of two node indices.

    Parameters
    ----------
    graph : Data
        The input graph represented as a PyTorch Geometric Data object.

    Returns
    -------
    List[List[int]]
        A list of graph elements, where each element is a node (singleton list)
        or an edge (list of two node indices).

    Notes
    -----
    The function directly works with the PyTorch Geometric Data object. Nodes are
    inferred from the edge_index attribute, and edges are directly extracted from
    it.
    """
    # Create nodes
    nodes = [[node] for node in range(graph.x.size(0))]

    # Create edges
    edges = graph.edge_index.t().tolist()

    # Combine nodes and edges
    return nodes + edges


def ring_lift(graph: Data) -> list[list[int]]:
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
    List[List[int]]
        A list of minimal cycles, each cycle is represented as a list of node indices.
    """
    # Convert to networkx graph
    G = pyg_utils.to_networkx(graph)

    # Compute all cycles (using a set with sorting removes duplicates)
    cycles = {tuple(sorted(cycle)) for cycle in nx.simple_cycles(G) if len(cycle) >= 3}

    # Filter out cycles that contain simpler cycles within themselves
    minimal_cycles = []
    for cycle in cycles:
        if not any(set(cycle) > set(other_cycle) for other_cycle in cycles if cycle != other_cycle):
            minimal_cycles.append(list(cycle))

    return minimal_cycles


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


lifter_registry = {
    "clique": clique_lift,
    "functional_group": functional_group_lift,
    "identity": identity_lift,
    "ring": ring_lift,
    "rips": rips_lift,
}


def get_lifters(args) -> list[callable]:
    """
    Construct a list of lifter functions based on provided arguments.

    This function iterates through a list of lifter names specified in the input arguments. For each
    lifter, it either retrieves the corresponding function from a registry or creates a partial
    function with additional arguments for specific lifters like 'rips'.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments. It should contain 'lifters', a list of lifter names, and
        additional arguments like 'dim' and 'dis' for specific lifters.

    Returns
    -------
    List[Callable]
        A list of callable lifter functions, ready to be applied to data.
    """
    lifters = []
    for lifter in args.lifters:
        if lifter == "rips":
            lifters.append(partial(rips_lift, dim=args.dim, dis=args.dis))
        else:
            lifters.append(lifter_registry[lifter])
    return lifters
