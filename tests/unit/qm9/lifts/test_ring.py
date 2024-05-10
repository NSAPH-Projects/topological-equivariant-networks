import pytest
import torch
from torch_geometric.data import Data

from qm9.lifts.ring import ring_lift


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
        output = {t[0] for t in output}

        # Check if the returned output matches the expected output
        assert output == expected
