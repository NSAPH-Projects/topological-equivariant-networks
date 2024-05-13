import pytest
import torch
from torch_geometric.data import Data

from qm9.lifts.ring import compute_ring_features, cycle_lift
from qm9.molecule_utils import molecule_from_data


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
def test_cycle_lift(edge_index, expected):
    if expected is ValueError:
        with pytest.raises(ValueError):
            graph_data = Data(edge_index=edge_index)
            cycle_lift(graph_data)
    else:
        # Create a simple graph
        graph_data = Data(edge_index=edge_index)

        # Call the cycle_lift function
        output = cycle_lift(graph_data)
        output = {t[0] for t in output}

        # Check if the returned output matches the expected output
        assert output == expected


@pytest.mark.parametrize(
    "smiles, expected",
    [
        # Pyridine
        ("C1=CC=NC=C1", (6.0, 1.0, 1.0, 0.0)),
        # Cyclohexane
        ("C1CCCCC1", (6.0, 0.0, 0.0, 1.0)),
        # Benzene
        ("C1=CC=CC=C1", (6.0, 1.0, 0.0, 0.0)),
    ],
)
def test_compute_ring_features(smiles: str, expected: tuple[float]):
    # Create an RDKit molecule object from the SMILES string
    graph = Data(smiles=smiles)
    mol = molecule_from_data(graph)

    # Get the atom indices of the first ring
    ring = frozenset(mol.GetRingInfo().AtomRings()[0])

    # Compute the ring features
    output = compute_ring_features(ring, mol)

    # Check if the returned output matches the expected output
    assert output == expected
