import pytest
from utilities import one_hot

from qm9.lifts.functional_group import get_pattern_features


# Test cases
@pytest.mark.parametrize(
    "pattern_name, expected_output",
    [
        (
            "carboxyl",
            (
                0.0,
                1.0,  # conjugation
                0.0,
                1.0,
                0.0,
                0.0,  # acidity
                1.0,
                0.0,
                0.0,  # hydrophobicity
                0.0,
                1.0,
                0.0,  # electrophilicity
                0.0,
                1.0,
                0.0,  # nucleophilicity
                0.0,
                0.0,
                1.0,  # polarity
            ),
        ),
        (
            "nitro",
            (
                1.0,
                0.0,  # conjugation
                1.0,
                0.0,
                0.0,
                0.0,  # acidity
                0.0,
                1.0,
                0.0,  # hydrophobicity
                0.0,
                0.0,
                1.0,  # electrophilicity
                1.0,
                0.0,
                0.0,  # nucleophilicity
                0.0,
                0.0,
                1.0,  # polarity
            ),
        ),
        (
            "ketone",
            (
                0.0,
                1.0,  # conjugation
                1.0,
                0.0,
                0.0,
                0.0,  # acidity
                0.0,
                1.0,
                0.0,  # hydrophobicity
                0.0,
                0.0,
                1.0,  # electrophilicity
                1.0,
                0.0,
                0.0,  # nucleophilicity
                0.0,
                0.0,
                1.0,  # polarity
            ),
        ),
        (
            "ester",
            (
                0.0,
                1.0,  # conjugation
                1.0,
                0.0,
                0.0,
                0.0,  # acidity
                0.0,
                1.0,
                0.0,  # hydrophobicity
                0.0,
                1.0,
                0.0,  # electrophilicity
                0.0,
                1.0,
                0.0,  # nucleophilicity
                0.0,
                0.0,
                1.0,  # polarity
            ),
        ),
        (
            "ether",
            (
                1.0,
                0.0,  # conjugation
                1.0,
                0.0,
                0.0,
                0.0,  # acidity
                0.0,
                1.0,
                0.0,  # hydrophobicity
                1.0,
                0.0,
                0.0,  # electrophilicity
                1.0,
                0.0,
                0.0,  # nucleophilicity
                0.0,
                1.0,
                0.0,  # polarity
            ),
        ),
        (
            "amide",
            (
                0.0,
                1.0,  # conjugation
                1.0,
                0.0,
                0.0,
                0.0,  # acidity
                0.0,
                1.0,
                0.0,  # hydrophobicity
                0.0,
                1.0,
                0.0,  # electrophilicity
                0.0,
                1.0,
                0.0,  # nucleophilicity
                0.0,
                0.0,
                1.0,  # polarity
            ),
        ),
        (
            "benzene",
            (
                0.0,
                1.0,  # conjugation
                1.0,
                0.0,
                0.0,
                0.0,  # acidity
                0.0,
                1.0,
                0.0,  # hydrophobicity
                1.0,
                0.0,
                0.0,  # electrophilicity
                1.0,
                0.0,
                0.0,  # nucleophilicity
                0.0,
                1.0,
                0.0,  # polarity
            ),
        ),
        (
            "aniline",
            (
                0.0,
                1.0,  # conjugation
                0.0,
                0.0,
                1.0,
                0.0,  # acidity
                0.0,
                1.0,
                0.0,  # hydrophobicity
                1.0,
                0.0,
                0.0,  # electrophilicity
                0.0,
                0.0,
                1.0,  # nucleophilicity
                0.0,
                1.0,
                0.0,  # polarity
            ),
        ),
        (
            "phenol",
            (
                0.0,
                1.0,  # conjugation
                0.0,
                0.0,
                0.0,
                1.0,  # acidity
                0.0,
                1.0,
                0.0,  # hydrophobicity
                1.0,
                0.0,
                0.0,  # electrophilicity
                0.0,
                0.0,
                1.0,  # nucleophilicity
                0.0,
                1.0,
                0.0,  # polarity
            ),
        ),
        (
            "carbamate",
            (
                1.0,
                0.0,  # conjugation
                1.0,
                0.0,
                0.0,
                0.0,  # acidity
                0.0,
                1.0,
                0.0,  # hydrophobicity
                0.0,
                1.0,
                0.0,  # electrophilicity
                0.0,
                1.0,
                0.0,  # nucleophilicity
                0.0,
                0.0,
                1.0,  # polarity
            ),
        ),
    ],
)
def test_get_pattern_features(pattern_name, expected_output):
    assert get_pattern_features(pattern_name) == expected_output
