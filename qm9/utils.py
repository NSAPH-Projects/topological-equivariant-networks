import functools
import random
from argparse import Namespace
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import simplicial_data.lifts as lifts
from simplicial_data.utils import SimplicialTransform


def calc_mean_mad(loader: DataLoader) -> Tuple[Tensor, Tensor]:
    """Return mean and mean average deviation of target in loader."""
    values = [graph.y for graph in loader.dataset]
    mean = sum(values) / len(values)
    mad = sum([abs(v - mean) for v in values]) / len(values)
    return mean, mad


def prepare_data(graph: Data, index: int, target_name: str, qm9_to_ev: Dict[str, float]) -> Data:
    graph.y = graph.y[0, index]
    one_hot = graph.x[:, :5]  # only get one_hot for cormorant
    # change unit of targets
    if target_name in qm9_to_ev:
        graph.y *= qm9_to_ev[target_name]

    Z_max = 9
    Z = graph.x[:, 5]
    Z_tilde = (Z / Z_max).unsqueeze(1).repeat(1, 5)

    graph.x = torch.cat((one_hot, Z_tilde * one_hot, Z_tilde * Z_tilde * one_hot), dim=1)

    return graph


def generate_loaders_qm9(args: Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    num_samples_suffix = "" if args.num_samples is None else f"_num_samples_{args.num_samples}"
    data_root = f"./datasets/QM9_delta_{args.dis}_dim_{args.dim}{num_samples_suffix}"
    rips_lift = functools.partial(lifts.rips_lift, dim=args.dim, dis=args.dis)
    transform = SimplicialTransform(lifter_fct=rips_lift, dim=args.dim)
    dataset = QM9(root=data_root)
    dataset = dataset.shuffle()
    dataset = dataset[: args.num_samples]
    dataset = [transform(sample) for sample in tqdm(dataset)]

    # filter relevant index and update units to eV
    qm9_to_ev = {
        "U0": 27.2114,
        "U": 27.2114,
        "G": 27.2114,
        "H": 27.2114,
        "zpve": 27211.4,
        "gap": 27.2114,
        "homo": 27.2114,
        "lumo": 27.2114,
    }
    targets = [
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "U0",
        "U",
        "H",
        "G",
        "Cv",
        "U0_atom",
        "U_atom",
        "H_atom",
        "G_atom",
        "A",
        "B",
        "C",
    ]
    index = targets.index(args.target_name)
    dataset = [
        prepare_data(graph, index, args.target_name, qm9_to_ev)
        for graph in tqdm(dataset, desc="Preparing data")
    ]

    # train/val/test split
    if args.num_samples is None:
        n_train, n_test = 100000, 110000
    else:
        n_train = int(len(dataset) * 0.75)
        n_test = n_train + int(len(dataset) * 0.075)
    train_dataset = dataset[:n_train]
    test_dataset = dataset[n_train:n_test]
    val_dataset = dataset[n_test:]

    # dataloaders
    follow = [f"x_{i}" for i in range(args.dim + 1)] + ["x"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        follow_batch=follow,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        follow_batch=follow,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        follow_batch=follow,
    )

    return train_loader, val_loader, test_loader
