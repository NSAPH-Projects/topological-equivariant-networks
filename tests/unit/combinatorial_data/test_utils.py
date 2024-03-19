# tests/test_utils.py


import pytest

from combinatorial_data.utils import create_combinatorial_complex


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
