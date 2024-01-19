import gudhi
import torch
from itertools import combinations
from torch import tensor, Tensor
from torch_geometric.data import Data
from collections import defaultdict
from typing import Tuple, Dict, Set, FrozenSet
from gudhi.simplex_tree import SimplexTree


def rips_lift(graph: Data, dim: int, dis: float, fc_nodes: bool=True) -> Tuple[Dict[int, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Generates simplicial complex based on Rips complex generated from point cloud or geometric graph. Returns a dictionary
    for the simplice and their features (x_dict), a dictionary for the different adjacencies (adj) and a dictionary with
    the different E(n) invariant geometric information as described in the paper.
    """

    # create simplicial complex
    x_0, pos = graph.x, graph.pos
    points = [pos[i].tolist() for i in range(pos.shape[0])]
    rips_complex = gudhi.RipsComplex(points=points, max_edge_length=dis)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=dim)

    if fc_nodes:
        nodes = [i for i in range(x_0.shape[0])]
        for edge in combinations(nodes, 2):
            simplex_tree.insert(edge)

    # generate dictionaries
    index_phonebook, counter = generate_indices(simplex_tree)
    adj, inv = generate_adjacencies_and_invariants(index_phonebook, simplex_tree, pos)
    x_dict = generate_features(index_phonebook, simplex_tree, counter)

    return x_dict, adj, inv


def generate_simplices(simplex_tree: SimplexTree) -> Dict[int, Set[FrozenSet]]:
    """Generates dictionary of simplices."""
    sim = dict()

    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        sim[dim].add(frozenset(simplex))

    return sim


def generate_indices(simplex_tree) -> Dict[int, Dict[FrozenSet, int]]:
    """
    Generates a dictionary which assigns to each simplex a unique index used for reference when finding the different
    adjacency types and invariants.
    """
    index_phonebook = dict()
    counter = defaultdict(int)

    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        index_phonebook[frozenset(simplex)] = counter[dim]
        counter[dim] += 1

    return index_phonebook, counter


def generate_adjacencies_and_invariants(index_phonebook: Dict, simplex_tree: SimplexTree, pos: Tensor) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """todo: add"""
    adj = defaultdict(list)
    inv = defaultdict(list)

    for simplex, _ in simplex_tree.get_simplices():
        # get index
        dim = len(simplex)-1
        simplex_index = index_phonebook[frozenset(simplex)]

        for boundary, _ in simplex_tree.get_boundaries(simplex):
            boundary_index = index_phonebook[frozenset(boundary)]
            # save adjacency
            adj[f'{dim-1}_{dim}'].append(tensor([boundary_index, simplex_index]))

            # calculate the boundary invariants and save
            shared = [vertex for vertex in simplex if vertex in boundary]
            b = [vertex for vertex in simplex if vertex not in shared]

            inv[f'{dim-1}_{dim}'].append(tensor([p for p in shared] + [b[0]]))

        for coface, _ in simplex_tree.get_cofaces(simplex, 1):
            coface_boundaries = simplex_tree.get_boundaries(coface)

            for coface_boundary, _ in coface_boundaries:
                # check if coface is distinct from the simplex
                if frozenset(coface_boundary) != frozenset(simplex):
                    coface_boundary_index = index_phonebook[frozenset(coface_boundary)]
                    # save adjacency
                    adj[f'{dim}_{dim}'].append(tensor([coface_boundary_index, simplex_index]))

                    # calculate the upper adjacent invariants and save
                    shared = [vertex for vertex in simplex if vertex in coface_boundary]
                    a, b = [vertex for vertex in simplex if vertex not in shared], [vertex for vertex in coface_boundary if vertex not in shared]

                    inv[f'{dim}_{dim}'].append(tensor([p for p in shared] + [a[0], b[0]]))

    for k, v in adj.items():
        adj[k] = torch.stack(v, dim=1)

    for k, v in inv.items():
        inv[k] = torch.stack(v, dim=1)

    return adj, inv


def generate_features(index_phonebook, simplex_tree, counter) -> Dict[int, Tensor]:
    x_dict = {dim: torch.zeros(size=(counter[dim], dim+1)) for dim in range(len(counter))}

    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        index = index_phonebook[frozenset(simplex)]
        x_dict[dim][index] = torch.tensor(simplex).long()

    return x_dict
