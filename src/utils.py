import os
import random
from argparse import Namespace
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch_geometric.loader import DataLoader

from models.ten import TEN
from qm9.qm9cc import get_adjacency_types, merge_adjacencies
# from qm9.lifts.registry import LIFTER_REGISTRY
# from combinatorial_data.lifter import Lifter
from torch_geometric.data import Dataset


def get_model(cfg: DictConfig, dataset: Dataset) -> nn.Module:
    """Return model based on name."""
    num_features_per_rank = dataset[0].num_features_per_rank
    dim = max(num_features_per_rank.keys())

    if cfg.model.visible_dims is None:
        visible_dims = list(sorted(num_features_per_rank.keys()))
    else:
        visible_dims = cfg.model.visible_dims

    if cfg.task_name == "QM9":
        # num_features_per_rank = {dim: 0 for dim in cfg.model.visible_dims}
        # num_features_per_rank = dataset[0].num_features_per_rank
        #     if "node" in cfg.lifter.initial_features:
        #         num_node_features = 15
        #         num_features_per_rank = {
        #             k: v + num_node_features for k, v in num_features_per_rank.items()
        #         }
        if "mem" in cfg.model.initial_features:
            num_lifters = len(cfg.lifter.lifters)
            num_features_per_rank = {
                k: v + num_lifters for k, v in num_features_per_rank.items()
            }
    #     if "hetero" in cfg.lifter.initial_features:
    #         # num_hetero_features = lifter.num_features_dict
    #         num_features_per_rank = {
    #             k: v + num_hetero_features[k] for k, v in num_features_per_rank.items()
    #         }
    #     if set(cfg.lifter.initial_features).difference(set(["node", "mem", "hetero"])):
    #         raise ValueError(
    #             f"Do not recognize initial features {cfg.lifter.initial_features}."
    #         )
    #     num_out = 1
    # else:
    #     raise ValueError(f"Do not recognize dataset {cfg.dataset}.")
    num_out = 1  # currently only one-dim output is supported

    adjacencies = get_adjacency_types(
        dim,
        cfg.dataset.connectivity,
        cfg.dataset.neighbor_types,
        # visible_dims,
    )

    # if cfg.model.merge_neighbors:
    #     processed_adjacencies = merge_adjacencies(adjacencies)
    # else:
    #     processed_adjacencies = adjacencies

    model = TEN(
        num_features_per_rank=num_features_per_rank,
        num_hidden=cfg.model.num_hidden,
        num_out=num_out,  # currently only one-dim output is supported
        num_layers=cfg.model.num_layers,
        # max_dim=lifter.dim,
        # adjacencies=processed_adjacencies,
        adjacencies=adjacencies,
        initial_features=cfg.model.initial_features,
        normalize_invariants=cfg.model.normalize_invariants,
        visible_dims=visible_dims,
        batch_norm=cfg.model.batch_norm,
        lean=cfg.model.lean,
    )
    return model


def get_loaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return dataloaders based on dataset."""
    if cfg.dataset == "qm9":
        from qm9.utils import generate_loaders_qm9

        train_loader, val_loader, test_loader = generate_loaders_qm9(cfg)
    else:
        raise ValueError(f"Dataset {cfg.dataset} not recognized.")

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
