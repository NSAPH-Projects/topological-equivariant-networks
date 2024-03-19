import os
import random
from argparse import Namespace
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader


def get_adjacency_types(max_dim: int, connectivity: str) -> list[str]:
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
    >>> get_adjacency_types(2, "self_and_next")
    ['0_0', '0_1', '1_1', '1_2', '2_2']

    >>> get_adjacency_types(2, "self_and_higher")
    ['0_0', '0_1', '0_2', '1_1', '1_2', '2_2']

    >>> get_adjacency_types(2, "all_to_all")
    ['0_0', '0_1', '0_2', '1_0', '1_1', '1_2', '2_0', '2_1', '2_2']
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

    return adj_types


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
            adjacencies=args.adjacencies,
            initial_features=args.initial_features,
            post_pool_filter=args.post_pool_filter,
        )
    else:
        raise ValueError(f"Model type {args.model_name} not recognized.")

    return model


def get_loaders(args: Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return dataloaders based on dataset."""
    if args.dataset == "qm9":
        from qm9.utils import generate_loaders_qm9

        train_loader, val_loader, test_loader = generate_loaders_qm9(args)
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
