import os
import random
from argparse import Namespace
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from qm9.utils import generate_loaders_qm9
from synthetic.utils import generate_loaders_synthetic_chains


def accuracy(output, target):
    """Compute accuracy."""
    return (output == target).sum().item() / target.size(0)


task_settings = {
    "regression": {
        "criterion": torch.nn.L1Loss(reduction="mean"),
        "metric": {
            "name": "MAE",
            "worst_value": float("inf"),
            "greater_is_better": False,
            "fct": torch.nn.L1Loss(reduction="mean"),
        },
    },
    "classification": {
        "criterion": torch.nn.CrossEntropyLoss(reduction="mean"),
        "metric": {
            "name": "Accuracy",
            "worst_value": 0,
            "greater_is_better": True,
            "fct": accuracy,
        },
    },
}


def get_adjacency_types(
    max_dim: int, connectivity: str, neighbor_types: list[str], visible_dims: list[int] | None
) -> list[str]:
    """
    Generate a list of adjacency type strings based on the specified connectivity pattern.

    Parameters
    ----------
    max_dim : int
        The maximum dimension (inclusive) for which to generate adjacency types. Represents the
        highest rank of cells in the connectivity pattern.
    connectivity : str
        The connectivity pattern to use. Must be one of the options defined below:
        - "self_and_next" generates adjacencies where each rank is connected to itself and the next
        (higher) rank.
        - "self_and_higher" generates adjacencies where each rank is connected to itself and all
        higher ranks.
        - "self_and_previous" generates adjacencies where each rank is connected to itself and the
        previous (lower) rank.
        - "self_and_lower" generates adjacencies where each rank is connected to itself and all
        lower ranks.
        - "self_and_neighbors" generates adjacencies where each rank is connected to itself, the
        next (higher) rank and the previous (lower) rank.
        - "all_to_all" generates adjacencies where each rank is connected to every other rank,
        including itself.
        - "legacy" ignores the max_dim parameter and returns ['0_0', '0_1', '1_1', '1_2'].
    neighbor_types : list[str]
        The types of adjacency between cells of the same rank. Must be one of the following:
        +1: two cells of same rank i are neighbors if they are both neighbors of a cell of rank i+1
        -1: two cells of same rank i are neighbors if they are both neighbors of a cell of rank i-1
        max: two cells of same rank i are neighbors if they are both neighbors of a cell of max rank
        min: two cells of same rank i are neighbors if they are both neighbors of a cell of min rank
    visible_dims: list[int] | None
        A list of ranks to explicitly represent as nodes. If None, all ranks are represented.

    Returns
    -------
    list[str]
        A list of strings representing the adjacency types for the specified connectivity pattern.
        Each string is in the format "i_j" where "i" and "j" are ranks indicating an adjacency
        from rank "i" to rank "j".

    Raises
    ------
    ValueError
        If `connectivity` is not one of the known connectivity patterns.

    Examples
    --------
    >>> get_adjacency_types(2, "self_and_next", ["+1"])
    ['0_0_1', '0_1', '1_1_2', '1_2']

    >>> get_adjacency_types(2, "self_and_higher", ["-1"])
    ['0_1', '0_2', '1_1_0', '1_2', '2_2_1']

    >>> get_adjacency_types(2, "all_to_all", ["-1", "+1", "max", "min"])
    ['0_0_1', '0_0_2','0_1', '0_2', '1_0', '1_1_0', '1_1_2', '1_2', '2_0', '2_1', '2_2_1', '2_2_0']
    """
    adj_types = []
    if connectivity not in [
        "self",
        "self_and_next",
        "self_and_higher",
        "self_and_previous",
        "self_and_lower",
        "self_and_neighbors",
        "all_to_all",
        "legacy",
    ]:
        raise ValueError(f"{connectivity} is not a known connectivity pattern!")

    if connectivity == "self":
        for i in range(max_dim + 1):
            adj_types.append(f"{i}_{i}")

    elif connectivity == "self_and_next":
        for i in range(max_dim + 1):
            adj_types.append(f"{i}_{i}")
            if i < max_dim:
                adj_types.append(f"{i}_{i+1}")

    elif connectivity == "self_and_higher":
        for i in range(max_dim + 1):
            for j in range(i, max_dim + 1):
                adj_types.append(f"{i}_{j}")

    elif connectivity == "self_and_previous":
        for i in range(max_dim + 1):
            adj_types.append(f"{i}_{i}")
            if i > 0:
                adj_types.append(f"{i}_{i-1}")

    elif connectivity == "self_and_lower":
        for i in range(max_dim + 1):
            for j in range(0, i):
                adj_types.append(f"{i}_{j}")

    elif connectivity == "self_and_neighbors":
        for i in range(max_dim + 1):
            adj_types.append(f"{i}_{i}")
            if i > 0:
                adj_types.append(f"{i}_{i-1}")
            if i < max_dim:
                adj_types.append(f"{i}_{i+1}")

    elif connectivity == "all_to_all":
        for i in range(max_dim + 1):
            for j in range(max_dim + 1):
                adj_types.append(f"{i}_{j}")

    else:
        adj_types = ["0_0", "0_1", "1_1", "1_2"]

    # Add one adjacency type for each neighbor type
    new_adj_types = []
    for adj_type in adj_types:
        i, j = map(int, adj_type.split("_"))
        if i == j:
            for neighbor_type in neighbor_types:
                if neighbor_type == "+1":
                    if i < max_dim:
                        new_adj_types.append(f"{i}_{i}_{i+1}")
                elif neighbor_type == "-1":
                    if i > 0:
                        new_adj_types.append(f"{i}_{i}_{i-1}")
                elif neighbor_type == "max":
                    if i < max_dim:
                        new_adj_types.append(f"{i}_{i}_{max_dim}")
                elif neighbor_type == "min":
                    if i > 0:
                        new_adj_types.append(f"{i}_{i}_0")
        else:
            new_adj_types.append(adj_type)
    new_adj_types = list(set(new_adj_types))
    adj_types = new_adj_types

    # Filter adjacencies with invisible ranks
    if visible_dims is not None:
        adj_types = [
            adj_type
            for adj_type in adj_types
            if all(int(dim) in visible_dims for dim in adj_type.split("_")[:2])
        ]

    return adj_types


def merge_adjacencies(adjacencies: list[str]) -> list[str]:
    """
    Merge all adjacency types i_i_j into a single i_i.

    We merge adjacencies of the form i_i_j into a single adjacency i_i. This is useful when we want
    to represent all rank i neighbors of a cell of rank i as a single adjacency matrix.

    Parameters
    ----------
    adjacencies : list[str]
        A list of adjacency types.

    Returns
    -------
    list[str]
        A list of merged adjacency types.

    """
    return list(set(["_".join(adj_type.split("_")[:2]) for adj_type in adjacencies]))


def get_model(args: Namespace) -> nn.Module:
    """Return model based on name."""
    if args.dataset == "qm9":
        num_node_features = 15
        num_lifters = len(args.lifters)
        if args.initial_features == "node":
            num_input = num_node_features
        elif args.initial_features == "mem":
            num_input = num_lifters
        elif args.initial_features == "both":
            num_input = num_node_features + num_lifters
        num_out = 1
    elif args.dataset == "synthetic_chains":
        num_input = 1
        num_out = 1
    else:
        raise ValueError(f"Do not recognize dataset {args.dataset}.")

    if args.model_name == "egnn":
        from models.egnn import EGNN

        model = EGNN(
            num_input=num_input,
            num_hidden=args.num_hidden,
            num_out=num_out,
            num_layers=args.num_layers,
        )
    elif args.model_name == "empsn":
        from models.empsn import EMPSN

        model = EMPSN(
            num_input=num_input,
            num_hidden=args.num_hidden,
            num_out=num_out,
            num_layers=args.num_layers,
            max_com=args.max_com,
            initial_features=args.initial_features,
        )

    elif args.model_name == "ten":
        from models.ten import TEN

        model = TEN(
            num_input=num_input,
            num_hidden=args.num_hidden,
            num_out=num_out,
            num_layers=args.num_layers,
            max_dim=args.dim,
            adjacencies=args.processed_adjacencies,
            initial_features=args.initial_features,
            visible_dims=args.visible_dims,
            task_type=args.task_type,
        )
    else:
        raise ValueError(f"Model type {args.model_name} not recognized.")

    return model


def get_loaders(args: Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return dataloaders based on dataset."""
    if args.dataset == "qm9":
        train_loader, val_loader, test_loader = generate_loaders_qm9(args)
    elif args.dataset == "synthetic_chains":
        train_loader, val_loader, test_loader = generate_loaders_synthetic_chains(args)
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized.")

    return train_loader, val_loader, test_loader


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
