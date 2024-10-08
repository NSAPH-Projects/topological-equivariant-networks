import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from etnn.lifter import get_adjacency_types
from etnn.model import ETNN


def load_checkpoint(checkpoint_path, model, opt, sched, force_restart):
    best_model = copy.deepcopy(model)
    device = next(model.parameters()).device
    if not force_restart and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.to("cpu")
        best_model.to("cpu")
        model.load_state_dict(checkpoint["model"])
        best_model.load_state_dict(checkpoint["best_model"])
        best_loss = checkpoint["best_loss"]
        opt.load_state_dict(checkpoint["optimizer"])
        sched.load_state_dict(checkpoint["scheduler"])
        model.to(device)
        best_model.to(device)
        return checkpoint["epoch"], checkpoint["run_id"], best_model, best_loss
    else:
        return 0, None, best_model, float("inf")


def save_checkpoint(path, model, best_model, best_loss, opt, sched, epoch, run_id):
    device = next(model.parameters()).device
    model.to("cpu")
    best_model.to("cpu")
    state = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "best_model": best_model.state_dict(),
        "best_loss": best_loss,
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "run_id": run_id,
    }
    model.to(device)
    best_model.to(device)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def calc_mean_mad(loader: DataLoader) -> tuple[Tensor, Tensor]:
    """Return mean and mean average deviation of target in loader."""
    values = [graph.y for graph in loader.dataset]
    mean = sum(values) / len(values)
    mad = sum([abs(v - mean) for v in values]) / len(values)
    return mean, mad


def get_model(cfg: DictConfig, dataset: Dataset) -> nn.Module:
    """Return model based on name."""
    num_features_per_rank = dataset[0].num_features_per_rank
    dim = max(num_features_per_rank.keys())

    if cfg.model.visible_dims is None:
        visible_dims = list(sorted(num_features_per_rank.keys()))
    else:
        visible_dims = cfg.model.visible_dims

    if cfg.task_name == "QM9":
        if "mem" in cfg.model.initial_features:
            num_lifters = len(cfg.lifter.lifters)
            num_features_per_rank = {
                k: v + num_lifters for k, v in num_features_per_rank.items()
            }
        global_pool = True
        sparse_invariant_computation = False
        pos_update = False

        adjacencies = get_adjacency_types(
            dim,
            cfg.dataset.connectivity,
            cfg.dataset.neighbor_types,
        )

    elif cfg.task_name == "geospatial":
        global_pool = False
        sparse_invariant_computation = True
        adjacencies = ["0_0", "0_1", "1_0", "1_1", "1_2", "2_1", "2_2"]
        pos_update = True

    model = ETNN(
        num_features_per_rank=num_features_per_rank,
        num_hidden=cfg.model.num_hidden,
        num_out=1,  # currently only one-dim output is supported
        num_layers=cfg.model.num_layers,
        adjacencies=adjacencies,
        initial_features=cfg.model.initial_features,
        normalize_invariants=cfg.model.normalize_invariants,
        visible_dims=visible_dims,
        batch_norm=cfg.model.batch_norm,
        lean=cfg.model.lean,
        global_pool=global_pool,
        sparse_invariant_computation=sparse_invariant_computation,
        pos_update=pos_update,
    )
    return model


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set as {seed}")
