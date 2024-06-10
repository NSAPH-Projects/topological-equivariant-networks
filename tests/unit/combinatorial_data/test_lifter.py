from argparse import Namespace
from collections import defaultdict
from typing import DefaultDict

import pytest

from combinatorial_data.lifter import Lifter, get_num_features_dict
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


@pytest.mark.parametrize(
    "lifters, expected",
    [
        # empty list
        ([], defaultdict(int)),
        # single lifter with integer ranking logic
        (
            [(lifter_registry["atom"], 0)],
            defaultdict(int, {0: lifter_registry["atom"].num_features}),
        ),
        # single lifter with string ranking logic
        ([(lifter_registry["clique"], "c")], defaultdict(int)),
        # multiple lifters with integer and string ranking logic
        (
            [
                (lifter_registry["atom"], 0),
                (lifter_registry["bond"], 1),
                (lifter_registry["ring"], 2),
                (lifter_registry["functional_group"], 2),
                (lifter_registry["supercell"], 3),
            ],
            defaultdict(
                int,
                {
                    0: lifter_registry["atom"].num_features,
                    1: lifter_registry["bond"].num_features,
                    2: lifter_registry["ring"].num_features
                    + lifter_registry["functional_group"].num_features,
                },
            ),
        ),
    ],
)
def test_get_num_features_dict(lifters, expected):
    output = get_num_features_dict(lifters)
    assert compare_defaultdicts(output, expected)


def compare_defaultdicts(d1: DefaultDict, d2: DefaultDict) -> bool:
    all_keys = set(d1.keys()).union(d2.keys())
    for key in all_keys:
        if d1[key] != d2[key]:
            return False
    return True
