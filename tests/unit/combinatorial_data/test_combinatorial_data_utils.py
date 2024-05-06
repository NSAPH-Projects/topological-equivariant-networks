# tests/test_utils.py

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from scipy.sparse import csc_matrix
from toponetx.classes import CombinatorialComplex

from combinatorial_data.combinatorial_data_utils import (
    CombinatorialComplexTransform,
    adjacency_matrix,
    create_combinatorial_complex,
    extract_cell_and_membership_data,
    incidence_matrix,
    merge_neighbors,
    sparse_to_dense,
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


@pytest.fixture
def combinatorial_complex():
    cc = CombinatorialComplex()
    cells = {
        0: [[0], [1], [2], [3]],
        1: [[0, 1], [0, 2], [1, 2]],
        3: [[0, 1, 2], [1, 2, 3]],
        4: [[0, 1, 2, 3]],
    }
    # First add higher-order cells
    for rank, cells in cells.items():
        cc.add_cells_from(cells, ranks=rank)

    return cc


@pytest.mark.parametrize(
    "rank, via_rank, expected_matrix",
    [
        (1, 0, csc_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))),
        (1, 2, csc_matrix((3, 3), dtype=np.float64)),
        (2, 3, csc_matrix((0, 0), dtype=np.float64)),
        (2, 1, csc_matrix((0, 0), dtype=np.float64)),
        (1, 1, None),
        (-1, 0, None),
        (0, -1, None),
    ],
)
def test_adjacency_matrix(combinatorial_complex, rank, via_rank, expected_matrix):
    cc = combinatorial_complex

    if expected_matrix is None:
        with pytest.raises(ValueError):
            adjacency_matrix(cc, rank, via_rank)
    else:
        assert torch.equal(
            sparse_to_dense(adjacency_matrix(cc, rank, via_rank)), sparse_to_dense(expected_matrix)
        )


@pytest.mark.parametrize(
    "rank, to_rank, expected_matrix",
    [
        (1, 0, csc_matrix(np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0]]))),
        (1, 2, csc_matrix((3, 0), dtype=np.float64)),
        (2, 3, csc_matrix((0, 2), dtype=np.float64)),
        (2, 1, csc_matrix((0, 3), dtype=np.float64)),
        (1, 1, None),
        (-1, 0, None),
        (0, -1, None),
    ],
)
def test_incidence_matrix(combinatorial_complex, rank, to_rank, expected_matrix):
    cc = combinatorial_complex

    if expected_matrix is None:
        with pytest.raises(ValueError):
            incidence_matrix(cc, rank, to_rank)
    else:
        assert torch.equal(
            sparse_to_dense(incidence_matrix(cc, rank, to_rank)), sparse_to_dense(expected_matrix)
        )


def _dense_to_sparse(dense):
    rows, cols = np.indices(dense.shape)
    sparse = csc_matrix((dense.flatten(), (rows.flatten(), cols.flatten())), shape=dense.shape)
    return sparse


@pytest.mark.parametrize(
    "sparse, expected",
    [
        # case 1: regular sparse matrix
        (
            csc_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])),
            torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]),
        ),
        # case 2: inefficient sparse matrix (also stores zeros)
        (
            _dense_to_sparse(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])),
            torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]),
        ),
    ],
)
def test_sparse_to_dense(sparse, expected):

    assert torch.equal(sparse_to_dense(sparse), expected)


def test_extract_cell_and_membership_data():
    input_dict = {
        0: {frozenset([0]): [True], frozenset([1]): [False], frozenset([2]): [True]},
        1: {frozenset([0, 1]): [True, False], frozenset([1, 2]): [False, True]},
        2: {},
    }

    x_dict, mem_dict = extract_cell_and_membership_data(input_dict)

    assert x_dict == {0: [[0], [1], [2]], 1: [[0, 1], [1, 2]], 2: []}

    assert mem_dict == {0: [[True], [False], [True]], 1: [[True, False], [False, True]], 2: []}
