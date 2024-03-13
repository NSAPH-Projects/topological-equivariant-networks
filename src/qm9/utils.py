import hashlib
import json
from argparse import Namespace

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from tqdm import tqdm

from combinatorial_data.lifts import get_lifters
from combinatorial_data.ranker import get_ranker
from combinatorial_data.utils import CombinatorialComplexTransform, CustomCollater


def calc_mean_mad(loader: DataLoader) -> tuple[Tensor, Tensor]:
    """Return mean and mean average deviation of target in loader."""
    values = [graph.y for graph in loader.dataset]
    mean = sum(values) / len(values)
    mad = sum([abs(v - mean) for v in values]) / len(values)
    return mean, mad


def prepare_data(graph: Data, index: int, target_name: str, qm9_to_ev: dict[str, float]) -> Data:
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


def generate_loaders_qm9(args: Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    # define data_root
    data_root = "./datasets/QM9_"
    data_root += generate_dataset_dir_name(
        args, ["num_samples", "target_name", "lifters", "dim", "dis"]
    )

    # load, subsample and transform the dataset
    lifters = get_lifters(args)
    ranker = get_ranker(args.lifters)
    transform = CombinatorialComplexTransform(
        lifters=lifters, ranker=ranker, dim=args.dim, adjacencies=args.adjacencies
    )
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
    follow_batch = [f"x_{i}" for i in range(args.dim + 1)] + ["x"]
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": True,
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=CustomCollater(train_dataset, follow_batch=follow_batch),
        **dataloader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=CustomCollater(val_dataset, follow_batch=follow_batch),
        **dataloader_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=CustomCollater(test_dataset, follow_batch=follow_batch),
        **dataloader_kwargs,
    )

    return train_loader, val_loader, test_loader


def generate_dataset_dir_name(args, relevant_args) -> str:
    """
    Generate a directory name based on a subset of script arguments.

    Parameters:
    args (dict): A dictionary of all script arguments.
    relevant_args (list): A list of argument names that are relevant to dataset generation.

    Returns:
    str: A hash-based directory name representing the relevant arguments.
    """
    # Convert Namespace to a dictionary
    args_dict = vars(args)

    # Filter the arguments, keeping only the relevant ones
    filtered_args = {key: args_dict[key] for key in relevant_args if key in args_dict}

    # Convert relevant arguments to a JSON string for consistent ordering
    args_str = json.dumps(filtered_args, sort_keys=True)

    # Create a hash of the relevant arguments string
    hash_obj = hashlib.sha256(args_str.encode())
    hash_hex = hash_obj.hexdigest()

    # Optional: truncate the hash for a shorter name
    short_hash = hash_hex[:16]  # First 16 characters

    return short_hash