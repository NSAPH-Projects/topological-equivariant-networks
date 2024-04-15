# tests/test_utils.py

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from toponetx.classes import CombinatorialComplex

from combinatorial_data.combinatorial_data_utils import (
    CombinatorialComplexTransform,
    create_combinatorial_complex,
    merge_neighbors,
)


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


@pytest.fixture
def adjacency_dict():
    return {
        "0_0_2": torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
        "0_1": torch.tensor([[0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        "1_1_0": torch.tensor([[0, 1, 0], [1, 0, 0], [0, 1, 0]]),
        "1_1_2": torch.tensor([[0, 1, 0], [0, 0, 1], [0, 1, 0]]),
        "1_2": torch.tensor([[0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        "2_2_1": torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
    }


def test_merge_neighbors(adjacency_dict):
    merged_adj, adj_types = merge_neighbors(adjacency_dict)

    assert set(adj_types) == set(["0_0", "0_1", "1_1", "1_2", "2_2"])

    assert merged_adj["0_0"].equal(adjacency_dict["0_0_2"])
    assert merged_adj["1_1"].equal(torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
    assert merged_adj["2_2"].equal(adjacency_dict["2_2_1"])


def test_merge_neighbors_with_unmerged_adjacencies():
    adjacency_dict = {
        "0_0": torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
    }

    with pytest.raises(AssertionError):
        merge_neighbors(adjacency_dict)
