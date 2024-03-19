# tests/test_utils.py

from unittest.mock import Mock

import numpy as np
import pytest
from toponetx.classes import CombinatorialComplex

from combinatorial_data.utils import CombinatorialComplexTransform, create_combinatorial_complex


@pytest.mark.parametrize(
    "neighbor_type, rank, expected",
    [
        # Test fully-connectedness when using "any_adjacency"
        ("any_adjacency", 0, np.ones((6, 6)) - np.eye(6)),
        ("any_adjacency", 1, np.ones((7, 7)) - np.eye(7)),
        ("any_adjacency", 2, np.ones((3, 3)) - np.eye(3)),
        ("any_adjacency", 3, np.ones((1, 1)) - np.eye(1)),
        # Test "adjacency"
        (
            "adjacency",
            0,
            np.array(
                [
                    [0, 1, 1, 0, 0, 0],
                    [1, 0, 1, 1, 0, 0],
                    [1, 1, 0, 1, 0, 0],
                    [0, 1, 1, 0, 1, 1],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                ]
            ),
        ),
        (
            "adjacency",
            1,
            np.array(
                [
                    [0, 1, 1, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 0],
                    [1, 1, 0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 0],
                ]
            ),
        ),
        (
            "adjacency",
            2,
            np.array(
                [
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                ]
            ),
        ),
        (
            "adjacency",
            3,
            np.array([[0]]),
        ),
        # Test "coadjacency"
        (
            "coadjacency",
            0,
            np.zeros((6, 6)),
        ),
        (
            "coadjacency",
            1,
            np.array(
                [
                    [0, 1, 1, 1, 0, 0, 0],
                    [1, 0, 1, 0, 1, 0, 0],
                    [1, 1, 0, 1, 1, 0, 0],
                    [1, 0, 1, 0, 1, 1, 1],
                    [0, 1, 1, 1, 0, 1, 1],
                    [0, 0, 0, 1, 1, 0, 1],
                    [0, 0, 0, 1, 1, 1, 0],
                ]
            ),
        ),
        (
            "coadjacency",
            2,
            np.array(
                [
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                ]
            ),
        ),
        (
            "coadjacency",
            3,
            np.array([[0]]),
        ),
        # Test "direct"
        (
            "direct",
            0,
            np.array(
                [
                    [0, 1, 1, 0, 0, 0],
                    [1, 0, 1, 1, 0, 0],
                    [1, 1, 0, 1, 0, 0],
                    [0, 1, 1, 0, 1, 1],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                ]
            ),
        ),
        (
            "direct",
            1,
            np.array(
                [
                    [0, 1, 1, 1, 0, 0, 0],
                    [1, 0, 1, 0, 1, 0, 0],
                    [1, 1, 0, 1, 1, 0, 0],
                    [1, 0, 1, 0, 1, 1, 1],
                    [0, 1, 1, 1, 0, 1, 1],
                    [0, 0, 0, 1, 1, 0, 1],
                    [0, 0, 0, 1, 1, 1, 0],
                ]
            ),
        ),
        (
            "direct",
            2,
            np.array(
                [
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                ]
            ),
        ),
        (
            "direct",
            3,
            np.array([[0]]),
        ),
        # Test "any_coadjacency"
        (
            "any_coadjacency",
            2,
            np.array(
                [
                    [0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0],
                ]
            ),
        ),
    ],
)
def test_extended_adjacency_matrix(neighbor_type, rank, expected):
    # Create an instance of the class
    instance = Mock()

    # Set the needed attributes
    instance.neighbor_type = neighbor_type
    instance.dim = 3

    # Define the inputs
    cell_dict = {
        # Atoms
        0: [
            frozenset([0]),
            frozenset([1]),
            frozenset([2]),
            frozenset([3]),
            frozenset([4]),
            frozenset([5]),
        ],
        # Bonds
        1: [
            frozenset([0, 1]),
            frozenset([0, 2]),
            frozenset([1, 2]),
            frozenset([1, 3]),
            frozenset([2, 3]),
            frozenset([3, 4]),
            frozenset([3, 5]),
        ],
        # Rings/FGs
        2: [frozenset([0, 1, 2]), frozenset([1, 2, 3]), frozenset([3, 4, 5])],
        # Molecule
        3: [frozenset([0, 1, 2, 3, 4, 5])],
    }
    cc = create_combinatorial_complex(cell_dict)

    index, matrix = CombinatorialComplexTransform.extended_adjacency_matrix(instance, cc, rank)

    assert index == cc.skeleton(rank)
    assert np.array_equal(matrix.todense(), expected)


@pytest.mark.parametrize(
    "cell_dict, expected",
    [
        # Simple CC with all rank-0 cells present
        (
            {
                0: [frozenset([0]), frozenset([1]), frozenset([2])],
                1: [frozenset([0, 1]), frozenset([1, 2])],
                2: [frozenset([0, 1, 2])],
            },
            None,
        ),
        # Simple CC using dictionaries with random values as iterables
        (
            {
                0: {frozenset([0]): "a", frozenset([1]): "b", frozenset([2]): "c"},
                1: [frozenset([0, 1]), frozenset([1, 2])],
                2: {frozenset([0, 1, 2]): "d"},
            },
            None,
        ),
        # CC with missing rank-0 cells
        (
            {
                1: [frozenset([0, 1]), frozenset([1, 2])],
                2: [frozenset([0, 1, 2])],
            },
            None,
        ),
        # cell_dict is not a dictionary
        (42, TypeError),
        # Values of cell_dict are not iterables
        ({0: 42}, TypeError),
        # Values within the iterables are not frozensets
        (
            {
                1: [[0, 1], [1, 2]],
                2: [[0, 1, 2]],
            },
            TypeError,
        ),
    ],
)
def test_create_combinatorial_complex(cell_dict, expected):

    if expected is TypeError:
        with pytest.raises(TypeError):
            create_combinatorial_complex(cell_dict)

    else:
        cc = create_combinatorial_complex(cell_dict)

        max_rank = max(cell_dict.keys())
        for rank in range(max_rank):
            expected = [cell for cell in cell_dict[rank]] if rank in cell_dict.keys() else []
            assert cc.skeleton(rank) == expected
