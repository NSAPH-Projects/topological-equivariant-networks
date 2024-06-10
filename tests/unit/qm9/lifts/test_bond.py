import pytest
import torch
from rdkit.Chem.rdchem import BondDir as BD
from rdkit.Chem.rdchem import BondStereo as BS
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data

from qm9.lifts.bond import compute_bond_features, edge_lift


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


# Mock bond data
class MockBond:
    def __init__(self, bond_type, is_conjugated, is_in_ring, stereo, direction):
        self.bond_type = bond_type
        self.is_conjugated = is_conjugated
        self.is_in_ring = is_in_ring
        self.stereo = stereo
        self.direction = direction

    def GetBondType(self):
        return self.bond_type

    def GetIsConjugated(self):
        return self.is_conjugated

    def IsInRing(self):
        return self.is_in_ring

    def GetStereo(self):
        return self.stereo

    def GetBondDir(self):
        return self.direction


# Function to be tested
def test_compute_bond_features():
    bond = MockBond(BT.SINGLE, True, False, BS.STEREOCIS, BD.BEGINWEDGE)
    expected_features = (
        [1, 0, 0, 0]  # bond_type_oh
        + [1]  # is_conjugated
        + [0]  # is_in_ring
        + [0, 0, 0, 0, 1, 0]  # bond_stereo_oh
        + [0, 1, 0, 0, 0, 0, 0]  # bond_dir_oh
    )
    expected_features = tuple(float(i) for i in expected_features)

    result = compute_bond_features(bond)
    assert result == tuple(expected_features)
