import pytest
import torch
from torch_geometric.data import Data

from qm9.lifts.bond import edge_lift


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
        output = {t[0] for t in output}
        assert output == expected
