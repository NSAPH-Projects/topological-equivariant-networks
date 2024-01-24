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


def get_subsampler(fraction: float) -> Optional[callable]:
    """
    Get a subsampling function for PyTorch datasets.

    This function returns a subsampler lambda function that can be used as the
    pre_filter argument in PyTorch datasets. It randomly retains a specified fraction
    of the input dataset. If the fraction is set to 1, it returns None, indicating
    that no subsampling is to be applied.

    Parameters
    ----------
    fraction : float
        The fraction of the dataset to retain. This should be a float between 0 and 1.
        If set to 1, the function returns None.

    Returns
    -------
    Optional[callable]
        A lambda function that takes a single argument and returns a boolean value.
        This function can be used as a pre_filter in PyTorch datasets to subsample
        the data. If `fraction` is 1, returns None.

    Examples
    --------
    >>> subsampler = get_subsampler(0.5)
    >>> isinstance(subsampler, callable)
    True
    >>> get_subsampler(1) is None
    True

    Notes
    -----
    The subsampling is done randomly based on the specified fraction. Each element
    in the dataset has an independent chance of `fraction` to be retained. This
    function does not guarantee that exactly `fraction` of the dataset is retained
    due to the random nature of the process.

    """
    if fraction == 1:
        return None
    else:
        return lambda _: bool(random.random() < fraction)


def generate_loaders_qm9(args: Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_root = f"./datasets/QM9_delta_{args.dis}_dim_{args.dim}_subsample_{args.subsample}_debug_2"
    subsampler = get_subsampler(args.subsample)
    rips_lift = functools.partial(lifts.rips_lift, dim=args.dim, dis=args.dis)
    transform = SimplicialTransform(lifter_fct=rips_lift, dim=args.dim)
    dataset = QM9(root=data_root, pre_filter=subsampler, pre_transform=transform)
    dataset = dataset.shuffle()

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
    if args.subsample == 1:
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
