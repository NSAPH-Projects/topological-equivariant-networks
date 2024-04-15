"""
Verifies consistency of old vs. new Rips lift transformations on graphs.

Utilizes the QM9 dataset for graph inputs and compares outputs from legacy and
current versions of the Rips lift.
"""

import functools

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import QM9

from combinatorial_data.combinatorial_data_utils import (
    CombinatorialComplexTransform as NewTransform,
)
from combinatorial_data.lifts import *
from combinatorial_data.ranker import get_ranker
from legacy.simplicial_data.rips_lift import rips_lift as old_rips_lift
from utils import get_adjacency_types


@pytest.mark.parametrize(
    "edge_index, expected_simplices",
    [
        # Test with a simple graph
        (
            torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
            {
                frozenset([0]),
                frozenset([1]),
                frozenset([2]),
                frozenset([3]),
                frozenset([0, 1]),
                frozenset([1, 2]),
                frozenset([2, 3]),
                frozenset([3, 0]),
            },
        ),
        # Test with an empty graph
        (torch.tensor([[], []], dtype=torch.long), set()),
        # Test with a graph with isolated nodes
        (
            torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
            {
                frozenset([0]),
                frozenset([1]),
                frozenset([2]),
                frozenset([3]),
                frozenset([0, 1]),
                frozenset([2, 3]),
            },
        ),
        # Test with a graph with self-loops
        (
            torch.tensor([[0, 1, 1], [0, 1, 2]], dtype=torch.long),
            {
                frozenset([0]),
                frozenset([1]),
                frozenset([2]),
                frozenset([1, 2]),
            },
        ),
    ],
)
def test_clique_lift(edge_index, expected_simplices):
    # Create a simple graph
    graph_data = Data(edge_index=edge_index)

    # Call the clique_lift function
    simplices = clique_lift(graph_data)

    # Check if the returned simplices match the expected simplices
    assert simplices == expected_simplices


@pytest.mark.parametrize(
    "edge_index, expected",
    [
        # Test with an empty graph
        (torch.tensor([[], []], dtype=torch.long), set()),
        # Test with a simple graph
        (
            torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
            {
                frozenset([0, 1]),
                frozenset([2, 3]),
            },
        ),
        # Test with a graph with self-loops
        (
            torch.tensor([[0, 1, 1], [0, 1, 2]], dtype=torch.long),
            {
                frozenset([1, 2]),
            },
        ),
        # Test with missing edge_index
        (None, ValueError),
    ],
)
def test_edge_lift(edge_index, expected):
    # Call the edge_lift function
    if expected is ValueError:
        with pytest.raises(ValueError):
            graph_data = Data(edge_index=edge_index)
            edge_lift(graph_data)
    else:
        graph_data = Data(edge_index=edge_index)
        output = edge_lift(graph_data)
        assert output == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        # Test with an empty graph
        (torch.tensor([], dtype=torch.float), set()),
        # Test with a normal graph
        (
            torch.tensor([[-1], [-1], [-1], [-1]], dtype=torch.float),
            {
                frozenset([0]),
                frozenset([1]),
                frozenset([2]),
                frozenset([3]),
            },
        ),
        # Test with a graph with no feature matrix
        (
            None,
            ValueError,
        ),
    ],
)
def test_node_lift(x, expected):
    # Call the node_lift function
    if expected is ValueError:
        with pytest.raises(ValueError):
            graph_data = Data(x=x)
            node_lift(graph_data)
    else:
        graph_data = Data(x=x)
        output = node_lift(graph_data)
        assert output == expected


@pytest.mark.parametrize(
    "edge_index, expected",
    [
        # Test with a simple graph with no rings
        (
            torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            set(),
        ),
        # Test with a simple graph with one ring
        (
            torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
            {frozenset([0, 1, 2, 3])},
        ),
        # Test with a composite ring (two directed triangles sharing an edge)
        (
            torch.tensor([[0, 1, 2, 2, 3], [1, 2, 0, 3, 1]], dtype=torch.long),
            {frozenset([0, 1, 2]), frozenset([1, 2, 3])},
        ),
        # Test with a composite ring (two doubly-connected triangles sharing a face)
        (
            torch.tensor(
                [[0, 1, 2, 2, 3, 1, 2, 0, 3, 1], [1, 2, 0, 3, 1, 0, 1, 2, 2, 3]], dtype=torch.long
            ),
            {frozenset([0, 1, 2]), frozenset([1, 2, 3])},
        ),
        # Test with a composite ring (two undirected triangles sharing an edge)
        (
            torch.tensor([[0, 2, 2, 3, 3], [1, 1, 0, 2, 1]], dtype=torch.long),
            {frozenset([0, 1, 2]), frozenset([1, 2, 3])},
        ),
        # Test with missing edge_index
        (
            None,
            ValueError,
        ),
    ],
)
def test_ring_lift(edge_index, expected):
    if expected is ValueError:
        with pytest.raises(ValueError):
            graph_data = Data(edge_index=edge_index)
            ring_lift(graph_data)
    else:
        # Create a simple graph
        graph_data = Data(edge_index=edge_index)

        # Call the ring_lift function
        output = ring_lift(graph_data)

        # Check if the returned output matches the expected output
        assert output == expected


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("dis", [0.5, 1.5, 2.5, 4.0])
def test_rips_transform(dim: int, dis: float):
    """
    Test consistency of simplicial transformations between old and new Rips lift implementations.

    Parameters
    ----------
    dim : int
        Dimensionality of the simplicial complex.
    dis : float
        Maximum edge length for simplices in the Rips complex.

    Assertions
    ----------
    Asserts equivalence of simplicial complexes from both old and new implementations.
    """
    n_test_samples = 3
    data_root = "./datasets/QM9_test"
    dataset = QM9(root=data_root)
    dataset = dataset.shuffle()
    for graph in dataset[:n_test_samples]:
        old_x_dict, old_adj, old_inv = old_rips_lift(graph, dim=dim, dis=dis)

        fixed_rips_lift = functools.partial(rips_lift, dim=dim, dis=dis)
        ranker = get_ranker(["rips"])
        adjacencies = get_adjacency_types(
            max_dim=dim + 1, connectivity="self_and_next", neighbor_types=["+1"], visible_dims=None
        )

        simplicial_transform = NewTransform(
            lifters=fixed_rips_lift,
            ranker=ranker,
            dim=dim + 1,
            adjacencies=adjacencies,
            merge_neighbors=True,
        )
        new_x_dict, _, new_adj, new_inv = simplicial_transform.get_relevant_dicts(graph)

        # Check if x_dict are the same
        for i in range(dim + 1):
            if not (torch.numel(old_x_dict[i]) == 0 and torch.numel(new_x_dict[i]) == 0):
                assert torch.equal(
                    old_x_dict[i], new_x_dict[i]
                ), f"old_x_dict[{i}] != new_x_dict[{i}]"

        # Check if adjs and invs are the same
        for i in range(dim):
            for j in [i, i + 1]:
                sorted_old_adj, idc = sort_tensor_columns_and_get_indices(old_adj[f"{i}_{j}"])
                if not (torch.numel(sorted_old_adj) == 0 and torch.numel(new_adj[f"{i}_{j}"]) == 0):
                    assert torch.equal(
                        sorted_old_adj, new_adj[f"{i}_{j}"]
                    ), f"sorted(old_adj[{i}_{j}]) != new_adj[{i}_{j}]"
                    assert torch.equal(
                        old_inv[f"{i}_{j}"][:, idc], new_inv[f"{i}_{j}"]
                    ), f"sorted(old_inv[{i}_{j}]) != new_inv[{i}_{j}]"


@pytest.mark.parametrize(
    "x, expected",
    [
        (torch.tensor([[-1], [-1], [-1], [-1]], dtype=torch.float), {frozenset([0, 1, 2, 3])}),
        (torch.tensor([[-1]], dtype=torch.float), set()),
        (torch.zeros(0, 1, dtype=torch.float), set()),
        (None, ValueError),
    ],
)
def test_supercell_lift(x, expected):
    # Call the supercell_lift function
    if expected is ValueError:
        with pytest.raises(ValueError):
            graph_data = Data(x=x)
            supercell_lift(graph_data)
    else:
        graph_data = Data(x=x)
        output = supercell_lift(graph_data)
        assert output == expected


def sort_tensor_columns_and_get_indices(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sort a 2D tensor by its second, then first row, returning sorted tensor and indices.

    Parameters
    ----------
    tensor : torch.Tensor
        A 2D tensor to sort.

    Returns
    -------
    torch.Tensor
        The tensor sorted by columns.
    torch.Tensor
        Indices reflecting the final column sorting order.
    """
    # Sort by the second row (preserve original indices)
    _, indices_second_row = torch.sort(tensor[1])
    sorted_tensor = tensor[:, indices_second_row]

    # Then sort by the first row, in a stable manner
    _, indices_first_row = torch.sort(sorted_tensor[0], stable=True)
    final_sorted_tensor = sorted_tensor[:, indices_first_row]

    # Combine the indices to reflect the final sorting order
    final_indices = indices_second_row[indices_first_row]

    return final_sorted_tensor, final_indices
