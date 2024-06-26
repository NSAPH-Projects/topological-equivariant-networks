import os
import random
from argparse import Namespace
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from models.ten import TEN

def get_model(args: Namespace) -> nn.Module:
    """Return model based on name."""
    if args.dataset == "qm9":
        num_features_per_rank = {dim: 0 for dim in args.visible_dims}
        if "node" in args.initial_features:
            num_node_features = 15
            num_features_per_rank = {
                k: v + num_node_features for k, v in num_features_per_rank.items()
            }
        if "mem" in args.initial_features:
            num_lifters = len(args.lifter.lifters)
            num_features_per_rank = {k: v + num_lifters for k, v in num_features_per_rank.items()}
        if "hetero" in args.initial_features:
            num_hetero_features = args.lifter.num_features_dict
            num_features_per_rank = {
                k: v + num_hetero_features[k] for k, v in num_features_per_rank.items()
            }
        if set(args.initial_features).difference(set(["node", "mem", "hetero"])):
            raise ValueError(f"Do not recognize initial features {args.initial_features}.")
        num_out = 1
    else:
        raise ValueError(f"Do not recognize dataset {args.dataset}.")

    model = TEN(
        num_features_per_rank=num_features_per_rank,
        num_hidden=args.num_hidden,
        num_out=num_out,
        num_layers=args.num_layers,
        max_dim=args.dim,
        adjacencies=args.processed_adjacencies,
        initial_features=args.initial_features,
        normalize_invariants=args.normalize_invariants,
        visible_dims=args.visible_dims,
        batch_norm=args.batch_norm,
        lean=args.lean,
    )
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
