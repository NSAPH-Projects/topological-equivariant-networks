"""Module for constructing topological structures from graphs."""

from functools import partial
from itertools import combinations
from collections import deque
import gudhi
import networkx as nx
import torch_geometric.utils as pyg_utils
from rdkit import Chem
from torch_geometric.data import Data
import random
from math import comb

from combinatorial_data.ifg import identify_functional_groups

def random_lift(graph: Data, K_max: int = 6, num_subsets: int = 10) -> set[frozenset[int]]:
    """
    Generate random subsets of nodes from a graph with cardinalities between 3 and K_max.

    This function randomly selects subsets of the entire node set, where each subset's size
    ranges from 3 to K_max. The function limits the output to a specified number of subsets
    to manage performance and memory usage.

    Parameters
    ----------
    graph : torch.Tensor
        The input graph represented as a PyTorch tensor.
    K_max : int
        The maximum cardinality of the node subsets.
    num_subsets : int
        The number of random subsets to return.

    Returns
    -------
    set[frozenset[int]]
        A set of node subsets, each subset is represented as a frozenset of node indices.

    Raises
    ------
    ValueError
        If the input graph does not contain an edge index 'edge_index'.
    """

    if (not hasattr(graph, "edge_index")) or (graph.edge_index is None):
        raise ValueError("The given graph does not have an edge index 'edge_index'!")

    # Convert to networkx graph
    G = pyg_utils.to_networkx(graph, to_undirected=True)

    nodes = list(G.nodes())
    all_subsets = set()
    total_nodes = len(nodes)
    assert total_nodes >= 3

    # Ensure that K_max is not greater than the total number of nodes
    max_subset_size = min(K_max, total_nodes)

    # Ensure that you are not requiring too many subsets
    n_subsets_at_least_3 = lambda n: 2**n - (1 + n + comb(n, 2))
    num_subsets = min(n_subsets_at_least_3(total_nodes),num_subsets)

    while len(all_subsets) < num_subsets:
        subset_size = random.randint(3, max_subset_size)
        sampled_subset = frozenset(random.sample(nodes, subset_size))
        all_subsets.add(sampled_subset)

    return all_subsets

def path_lift(graph: Data, K_max: int = 4) -> set[frozenset[int]]:
    """
    Identify all paths in a graph with lengths up to K_max, optimized for efficiency.

    This function finds all paths of lengths ranging from 2 to K_max in a given graph,
    with optimizations to improve time and memory performance.

    Parameters
    ----------
    graph : torch.Tensor
        The input graph represented as a PyTorch tensor.
    K_max : int
        The maximum length of the paths to be found.

    Returns
    -------
    set[frozenset[int]]
        A set of paths, each path is represented as a frozenset of node indices.

    Raises
    ------
    ValueError
        If the input graph does not contain an edge index 'edge_index'.
    """

    if (not hasattr(graph, "edge_index")) or (graph.edge_index is None):
        raise ValueError("The given graph does not have an edge index 'edge_index'!")

    # Convert to networkx graph
    G = pyg_utils.to_networkx(graph, to_undirected=True)
    paths = set()
    for source_node in G.nodes():
        paths.add(frozenset([source_node]))
        # Use deque for more efficient popping from left
        queue = deque([(source_node, [source_node])])
        while queue:
            current_node, path = queue.popleft()
            if 1 < len(path) <= K_max:
                paths.add(frozenset(path))
            if len(path) < K_max:
                for neighbor in G.neighbors(current_node):
                    if neighbor not in path:
                        queue.append((neighbor, path + [neighbor]))

    return paths

def clique_lift(graph_data) -> set[frozenset[int]]:
    """
    Construct a clique complex from a graph represented as a torch_geometric.data.Data object.

    Parameters
    ----------
    graph_data : torch_geometric.data.Data
        The graph from which to construct the clique complex, represented as a
        torch_geometric.data.Data object.

    Returns
    -------
    set[frozenset[int]]
        Simplices of the clique complex.
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

    return simplices


def edge_lift(graph: Data) -> set[frozenset[int]]:
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
    set[frozenset[int]]
        A set of graph elements, where each element is an edge (frozenset of two node indices).

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

    return edges


def functional_group_lift(graph: Data) -> set[frozenset[int]]:
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
    set[frozenset[int]]
        A set of frozensets, where each frozenset contains the atom indices of a functional group in
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
        return {frozenset(fg.atomIds) for fg in functional_groups if len(fg.atomIds) >= 3}
    except AttributeError:
        return set()


def node_lift(graph: Data) -> set[frozenset[int]]:
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
    set[frozenset[int]]
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


def supercell_lift(graph: Data) -> set[frozenset[int]]:
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
    set[frozenset[int]]
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


def ring_lift(graph: Data) -> set[frozenset[int]]:
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
    set[frozenset[int]]
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


def rips_lift(graph: Data, dim: int, dis: float, fc_nodes: bool = True) -> set[frozenset[int]]:
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
    set[frozenset[int]]
        A set of frozensets, where each frozenset represents a simplex in the Rips
        complex. Each simplex is a frozenset of vertex indices.

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

    # convert simplicial complex to set of frozensets
    simplexes = set()
    simplexes.update(frozenset(simplex) for simplex, _ in simplex_tree.get_simplices())

    return simplexes


lifter_registry = {
    "atom": node_lift,  # atom is an alias for node
    "bond": edge_lift,  # bond is an alias for edge
    "clique": clique_lift,
    "edge": edge_lift,
    "functional_group": functional_group_lift,
    "node": node_lift,
    "ring": ring_lift,
    "rips": rips_lift,
    "supercell": supercell_lift,
    "path": path_lift,
    "random": random_lift,
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
        lifter = lifter.split(":")[0]
        if lifter == "rips":
            lifters.append(partial(rips_lift, dim=args.dim, dis=args.dis))
        else:
            lifters.append(lifter_registry[lifter])
    return lifters
