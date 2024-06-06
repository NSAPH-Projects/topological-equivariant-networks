import pytest

from combinatorial_data.ranker import get_ranker


@pytest.mark.parametrize(
    "lifter_args, cell, memberships, expected",
    [
        # simple 1
        (["identity:c", "ring:2"], ["a", "b", "c"], [True, False], 2),
        # simple 2
        (["identity:c", "ring:2"], ["a", "b"], [False, True], 2),
        # simple 3
        (["identity:c", "ring:2"], ["a", "b"], [True, True], 1),
        # no lifters
        (["identity:c", "ring:2"], ["a", "b", "c"], [False, False], ValueError),
        # length mismatch
        (["identity:c", "ring:2"], ["a", "b"], [True], ValueError),
        # negative rank
        (["identity:-1"], ["a", "b"], [True], ValueError),
        # unspecified rank
        (["identity"], ["a", "b"], [True], 1),
        # unknown/illegal rank
        (["identity:abc"], ["a", "b"], [True], ValueError),
    ],
)
def test_get_ranker(
    lifter_args: list[str], cell: frozenset[int], memberships: list[bool], expected: int
):
    if expected is ValueError:
        with pytest.raises(ValueError):
            ranker = get_ranker(lifter_args)
            ranker(cell, memberships)
    else:
        ranker = get_ranker(lifter_args)
        assert ranker(cell, memberships) == expected
