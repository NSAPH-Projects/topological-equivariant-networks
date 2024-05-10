import pytest
import torch
from torch_geometric.data import Data

from qm9.lifts.clique import clique_lift


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
