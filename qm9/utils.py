import hashlib
import json
from argparse import Namespace

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torchvision.transforms import Compose
from tqdm import tqdm

from combinatorial_data.lifts import get_lifters
from combinatorial_data.ranker import get_ranker
from combinatorial_data.transforms import CombinatorialComplexTransform, FeatureEngineeringTransform
from combinatorial_data.utils import CustomCollater


def calc_mean_mad(loader: DataLoader) -> tuple[Tensor, Tensor]:
    """
    Calculate the mean and mean absolute deviation (MAD) of targets from a DataLoader.

    Parameters
    ----------
    loader : DataLoader
        DataLoader containing the dataset with targets.

    Returns
    -------
    tuple[Tensor, Tensor]
        A tuple containing the mean and MAD of the dataset's targets.

    """
    targets = loader.dataset.y
    mean = targets.mean()
    mad = (targets - mean).abs().mean()

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


def transform_targets(raw_targets: torch.FloatTensor, target_name: str) -> torch.FloatTensor:
    """
    Select and transform raw target values to a specific unit (e.g., eV) based on the target name.

    Parameters
    ----------
    raw_targets : torch.FloatTensor
        A 2D tensor of shape (n_samples, n_targets) containing the raw target values.
    target_name : str
        The name of the target to be transformed. Must be one of the predefined target names.

    Returns
    -------
    torch.FloatTensor
        A 1D tensor of shape (n_samples,) containing the transformed target values for the specified
        target name.

    Raises
    ------
    ValueError
        If `target_name` is not among the predefined targets.

    Notes
    -----
    The transformation applies a conversion factor to the specified target values based on a
    predefined dictionary (`QM9_TO_EV`). If the target name does not require conversion (not found
    in the dictionary), the original values are returned without modification.
    """
    QM9_TO_EV = {
        "U0": 27.2114,
        "U": 27.2114,
        "G": 27.2114,
        "H": 27.2114,
        "zpve": 27211.4,
        "gap": 27.2114,
        "homo": 27.2114,
        "lumo": 27.2114,
    }
    TARGETS = [
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
    conversion_factor = QM9_TO_EV.get(target_name, 1)
    index = TARGETS.index(target_name)
    transformed_targets = conversion_factor * raw_targets[:, index]
    return transformed_targets


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
    transform = Compose(
        [
            CombinatorialComplexTransform(
                lifters=lifters, ranker=ranker, dim=args.dim, adjacencies=args.adjacencies
            ),
            FeatureEngineeringTransform(),
        ]
    )
    dataset = QM9(root=data_root, transform=transform)
    dataset._data.y = transform_targets(dataset.y, args.target_name)
    dataset = dataset.shuffle()
    dataset = dataset[: args.num_samples]

    # dataset = [transform(sample) for sample in tqdm(dataset)]

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
