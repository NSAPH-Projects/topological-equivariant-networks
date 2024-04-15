import pytest

from utils import get_adjacency_types


@pytest.mark.parametrize(
    "max_dim, connectivity, neighbor_types, visible_dims, expected",
    [
        (2, "self_and_next", ["+1"], None, ["0_0_1", "0_1", "1_1_2", "1_2"]),
        (2, "self_and_next", ["+1"], [0, 1], ["0_0_1", "0_1", "1_1_2"]),
        (2, "self_and_higher", ["-1"], None, ["0_1", "0_2", "1_1_0", "1_2", "2_2_1"]),
        (2, "self_and_higher", ["-1"], [0], []),
        (
            2,
            "all_to_all",
            ["-1", "+1", "max", "min"],
            None,
            [
                "0_0_1",
                "0_0_2",
                "0_1",
                "0_2",
                "1_0",
                "1_1_0",
                "1_1_2",
                "1_2",
                "2_0",
                "2_1",
                "2_2_1",
                "2_2_0",
            ],
        ),
        (
            2,
            "all_to_all",
            ["-1", "+1", "max", "min"],
            [0, 1, 2],
            [
                "0_0_1",
                "0_0_2",
                "0_1",
                "0_2",
                "1_0",
                "1_1_0",
                "1_1_2",
                "1_2",
                "2_0",
                "2_1",
                "2_2_1",
                "2_2_0",
            ],
        ),
    ],
)
def test_get_adjacency_types(max_dim, connectivity, neighbor_types, visible_dims, expected):
    result = get_adjacency_types(max_dim, connectivity, neighbor_types, visible_dims)
    assert set(result) == set(expected)


def test_invalid_connectivity():
    with pytest.raises(ValueError) as e:
        get_adjacency_types(2, "invalid_connectivity", ["+1"], None)
    assert str(e.value) == "invalid_connectivity is not a known connectivity pattern!"
