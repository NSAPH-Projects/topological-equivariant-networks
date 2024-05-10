import pytest
import torch
from torch_geometric.data import Data

from qm9.lifts.molecule import supercell_lift


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
