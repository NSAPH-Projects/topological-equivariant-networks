from collections import namedtuple

import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils import one_hot

from etnn.combinatorial_data import Cell

functional_group_patterns = {
    "carboxyl": "C(=O)O",
    "nitro": "[N+](=O)[O-]",
    "ketone": "[CX3](=O)[C]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "ether": "[OD2]([#6])[#6]",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "benzene": "c1ccccc1",
    "aniline": "Nc1ccccc1",
    "phenol": "Oc1ccccc1",
    "carbamate": "[NX3][CX3](=[OX1])[OX2H0]",
}

# Functional group features
feature_spaces = {
    "conjugation": [False, True],
    "acidity": ["neutral", "high", "basic", "weakly acidic"],
    "hydrophobicity": ["hydrophilic", "moderate", "hydrophobic"],
    "electrophilicity": ["low", "moderate", "high"],
    "nucleophilicity": ["low", "moderate", "high"],
    "polarity": ["low", "moderate", "high"],
}

functional_group_features = {
    "carboxyl": {
        "conjugation": feature_spaces["conjugation"].index(True),
        "acidity": feature_spaces["acidity"].index("high"),
        "hydrophobicity": feature_spaces["hydrophobicity"].index("hydrophilic"),
        "electrophilicity": feature_spaces["electrophilicity"].index("moderate"),
        "nucleophilicity": feature_spaces["nucleophilicity"].index("moderate"),
        "polarity": feature_spaces["polarity"].index("high"),
    },
    "nitro": {
        "conjugation": feature_spaces["conjugation"].index(False),
        "acidity": feature_spaces["acidity"].index("neutral"),
        "hydrophobicity": feature_spaces["hydrophobicity"].index("moderate"),
        "electrophilicity": feature_spaces["electrophilicity"].index("high"),
        "nucleophilicity": feature_spaces["nucleophilicity"].index("low"),
        "polarity": feature_spaces["polarity"].index("high"),
    },
    "ketone": {
        "conjugation": feature_spaces["conjugation"].index(True),
        "acidity": feature_spaces["acidity"].index("neutral"),
        "hydrophobicity": feature_spaces["hydrophobicity"].index("moderate"),
        "electrophilicity": feature_spaces["electrophilicity"].index("high"),
        "nucleophilicity": feature_spaces["nucleophilicity"].index("low"),
        "polarity": feature_spaces["polarity"].index("high"),
    },
    "ester": {
        "conjugation": feature_spaces["conjugation"].index(True),
        "acidity": feature_spaces["acidity"].index("neutral"),
        "hydrophobicity": feature_spaces["hydrophobicity"].index("moderate"),
        "electrophilicity": feature_spaces["electrophilicity"].index("moderate"),
        "nucleophilicity": feature_spaces["nucleophilicity"].index("moderate"),
        "polarity": feature_spaces["polarity"].index("high"),
    },
    "ether": {
        "conjugation": feature_spaces["conjugation"].index(False),
        "acidity": feature_spaces["acidity"].index("neutral"),
        "hydrophobicity": feature_spaces["hydrophobicity"].index("moderate"),
        "electrophilicity": feature_spaces["electrophilicity"].index("low"),
        "nucleophilicity": feature_spaces["nucleophilicity"].index("low"),
        "polarity": feature_spaces["polarity"].index("moderate"),
    },
    "amide": {
        "conjugation": feature_spaces["conjugation"].index(True),
        "acidity": feature_spaces["acidity"].index("neutral"),
        "hydrophobicity": feature_spaces["hydrophobicity"].index("moderate"),
        "electrophilicity": feature_spaces["electrophilicity"].index("moderate"),
        "nucleophilicity": feature_spaces["nucleophilicity"].index("moderate"),
        "polarity": feature_spaces["polarity"].index("high"),
    },
    "benzene": {
        "conjugation": feature_spaces["conjugation"].index(True),
        "acidity": feature_spaces["acidity"].index("neutral"),
        "hydrophobicity": feature_spaces["hydrophobicity"].index("moderate"),
        "electrophilicity": feature_spaces["electrophilicity"].index("low"),
        "nucleophilicity": feature_spaces["nucleophilicity"].index("low"),
        "polarity": feature_spaces["polarity"].index("moderate"),
    },
    "aniline": {
        "conjugation": feature_spaces["conjugation"].index(True),
        "acidity": feature_spaces["acidity"].index("basic"),
        "hydrophobicity": feature_spaces["hydrophobicity"].index("moderate"),
        "electrophilicity": feature_spaces["electrophilicity"].index("low"),
        "nucleophilicity": feature_spaces["nucleophilicity"].index("high"),
        "polarity": feature_spaces["polarity"].index("moderate"),
    },
    "phenol": {
        "conjugation": feature_spaces["conjugation"].index(True),
        "acidity": feature_spaces["acidity"].index("weakly acidic"),
        "hydrophobicity": feature_spaces["hydrophobicity"].index("moderate"),
        "electrophilicity": feature_spaces["electrophilicity"].index("low"),
        "nucleophilicity": feature_spaces["nucleophilicity"].index("high"),
        "polarity": feature_spaces["polarity"].index("moderate"),
    },
    "carbamate": {
        "conjugation": feature_spaces["conjugation"].index(False),
        "acidity": feature_spaces["acidity"].index("neutral"),
        "hydrophobicity": feature_spaces["hydrophobicity"].index("moderate"),
        "electrophilicity": feature_spaces["electrophilicity"].index("moderate"),
        "nucleophilicity": feature_spaces["nucleophilicity"].index("moderate"),
        "polarity": feature_spaces["polarity"].index("high"),
    },
}

# Cast all indices to tensors
for pattern_name, feature_dict in functional_group_features.items():
    for feature_name, feature_value in feature_dict.items():
        feature_dict[feature_name] = torch.tensor([feature_value])


def functional_group_lift(graph: Data) -> set[Cell]:
    """
    Identify and return the functional groups present in a given molecule.

    Parameters
    ----------
    graph : Data
        The input molecule graph.

    Returns
    -------
    set[Cell]
        A set of tuples representing the identified functional groups.
        Each tuple contains the node indices of the atoms in the functional group
        and the corresponding feature vector.
    """
    mol = graph.mol
    cells = set()
    for pattern_name, smart in functional_group_patterns.items():
        pattern = Chem.MolFromSmarts(smart)
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            # We only consider functional groups with at least 3 atoms
            if len(match) < 3:
                continue
            node_idc = frozenset(match)
            feature_vector = get_pattern_features(pattern_name)
            cells.add((node_idc, feature_vector))
    return cells


def get_pattern_features(pattern_name: str) -> tuple[float]:
    """
    Get the feature vector for a given functional group pattern.

    Parameters
    ----------
    pattern_name : str
        The name of the functional group pattern.

    Returns
    -------
    tuple[float]
        The feature vector for the functional group pattern.
    """
    feature_dict = functional_group_features[pattern_name]
    vector_list = []
    for feature_name, feature_value in feature_dict.items():
        feature_vector = one_hot(
            feature_value, num_classes=len(feature_spaces[feature_name])
        ).flatten()
        vector_list.append(feature_vector)
    vector_tensor = torch.cat(vector_list, dim=0)
    vector_tuple = tuple(vector_tensor.tolist())
    return vector_tuple


functional_group_lift.num_features = sum(
    len(feature_space) for feature_space in feature_spaces.values()
)
