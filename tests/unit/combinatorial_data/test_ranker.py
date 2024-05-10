from argparse import Namespace

import pytest

from combinatorial_data.lifter import Lifter
from qm9.lifts.registry import lifter_registry


@pytest.mark.parametrize(
    "lifter_args, cell, memberships, expected",
    [
        # simple 1
        (["clique:c", "ring:2"], ["a", "b", "c"], [True, False], 2),
        # simple 2
        (["clique:c", "ring:2"], ["a", "b"], [False, True], 2),
        # simple 3
        (["clique:c", "ring:2"], ["a", "b"], [True, True], 1),
        # no lifters
        (["clique:c", "ring:2"], ["a", "b", "c"], [False, False], ValueError),
        # length mismatch
        (["clique:c", "ring:2"], ["a", "b"], [True], ValueError),
        # negative rank
        (["clique:-1"], ["a", "b"], [True], ValueError),
        # unspecified rank
        (["clique"], ["a", "b"], [True], 1),
        # unknown/illegal rank
        (["clique:abc"], ["a", "b"], [True], ValueError),
    ],
)
def test_ranker(
    lifter_args: list[str], cell: frozenset[int], memberships: list[bool], expected: int
):

    # Create the lifter
    args = Namespace(lifters=lifter_args, dim=3)

    # Test the ranker
    if expected is ValueError:
        with pytest.raises(ValueError):
            lifter = Lifter(args, lifter_registry)
            lifter.ranker(cell, memberships)
    else:
        lifter = Lifter(args, lifter_registry)
        assert lifter.ranker(cell, memberships) == expected
