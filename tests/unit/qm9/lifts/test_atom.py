import pytest
import torch
from torch_geometric.data import Data

from qm9.lifts.atom import node_lift


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
