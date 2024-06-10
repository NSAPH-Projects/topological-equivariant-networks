import pytest
import torch
from rdkit import Chem
from torch_geometric.data import Data

from qm9.lifts.atom import compute_atom_features, node_lift


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
        output = {t[0] for t in output}
        assert output == expected


# One-hot encoding function as used in compute_atom_features
def one_hot(index, num_classes):
    vec = torch.zeros(num_classes, dtype=torch.float)
    vec[index] = 1.0
    return vec


@pytest.mark.parametrize(
    "atomic_num, expected",
    [
        # Hydrogen
        (
            1,
            tuple(
                one_hot(0, 5).tolist()
                + (1 / 9 * one_hot(0, 5)).tolist()
                + ((1 / 9) ** 2 * one_hot(0, 5)).tolist()
            ),
        ),
        # Carbon
        (
            6,
            tuple(
                one_hot(1, 5).tolist()
                + (6 / 9 * one_hot(1, 5)).tolist()
                + ((6 / 9) ** 2 * one_hot(1, 5)).tolist()
            ),
        ),
        # Nitrogen
        (
            7,
            tuple(
                one_hot(2, 5).tolist()
                + (7 / 9 * one_hot(2, 5)).tolist()
                + ((7 / 9) ** 2 * one_hot(2, 5)).tolist()
            ),
        ),
        # Oxygen
        (
            8,
            tuple(
                one_hot(3, 5).tolist()
                + (8 / 9 * one_hot(3, 5)).tolist()
                + ((8 / 9) ** 2 * one_hot(3, 5)).tolist()
            ),
        ),
        # Fluorine
        (
            9,
            tuple(
                one_hot(4, 5).tolist()
                + (9 / 9 * one_hot(4, 5)).tolist()
                + ((9 / 9) ** 2 * one_hot(4, 5)).tolist()
            ),
        ),
    ],
)
def test_compute_atom_features(atomic_num, expected):
    atom = Chem.Atom(atomic_num)
    features = compute_atom_features(atom)
    assert features == expected, f"Expected {expected}, got {features}"
