import hashlib
import json
import os
import pickle
import random
from argparse import Namespace
from typing import Dict, List, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from tqdm import tqdm

from combinatorial_data.combinatorial_data_utils import (
    CombinatorialComplexData,
    CombinatorialComplexTransform,
    CustomCollater,
)
from combinatorial_data.lifter import Lifter
from qm9.lifts.registry import lifter_registry


def calc_mean_mad(loader: DataLoader) -> tuple[Tensor, Tensor]:
    """Return mean and mean average deviation of target in loader."""
    values = [graph.y for graph in loader.dataset]
    mean = sum(values) / len(values)
    mad = sum([abs(v - mean) for v in values]) / len(values)
    return mean, mad


def prepare_data(graph: Data, index: int, target_name: str) -> Data:
    """
    Preprocesses the input graph data.

    Two main modifications are made:
    1. The target value is extracted and stored in the 'y' attribute. Since QM9 targets are
    graph-level, we throw away the vector of 'y' values and only keep the target value of the
    first node in the graph, at the given index.
    2. If the target name is 'zpve', the target value is multiplied by 1e3. This is consistent with
    EGNN.
    3. The feature vector of each node  is computed as a concatenation of the one-hot encoding of
    the atomic number, the atomic number scaled by 1/9, and the atomic number scaled by 1/9 squared.

    Parameters
    ----------
    graph : Data
        The input graph data. It should be an instance of the torch_geometric.data.Data class.
    index : int
        The index of the target value to extract. It should be a non-negative integer.
    target_name: str
        The name of the target.

    Returns
    -------
    Data
        The preprocessed graph data. It is an instance of the Data class with modified features.
    """
    graph.y = graph.y[0, index]
    one_hot = graph.x[:, :5]  # only get one_hot for cormorant
    if target_name == "zpve":
        graph.y *= 1e3
    Z_max = 9
    Z = graph.x[:, 5]
    Z_tilde = (Z / Z_max).unsqueeze(1).repeat(1, 5)

    graph.x = torch.cat((one_hot, Z_tilde * one_hot, Z_tilde * Z_tilde * one_hot), dim=1)

    return graph


def lift_qm9_to_cc(args: Namespace) -> list[dict]:
    """
    Lift QM9 dataset to CombinatorialComplexData format.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    Returns
    -------
    list[dict]
        List of Combinatorial Complex representations of QM9 molecules.

    Notes
    -----
    The QM9 dataset is loaded and each sample is transformed into a dictionary representation of
    the CombinatorialComplexData class. We transform to dictionary format to allow for storage as
    JSON files.
    """
    qm9 = QM9("./datasets/QM9")
    # Create the transform
    lifter = Lifter(args, lifter_registry)
    transform = CombinatorialComplexTransform(
        lifter=lifter,
        dim=args.dim,
        adjacencies=args.adjacencies,
        processed_adjacencies=args.processed_adjacencies,
        merge_neighbors=args.merge_neighbors,
    )
    qm9_cc = []
    for graph in tqdm(qm9, desc="Lifting QM9 samples"):
        qm9_cc.append(transform.graph_to_ccdict(graph))
    return qm9_cc


def save_lifted_qm9(storage_path: str, samples: list[dict]) -> None:
    """
    Save the lifted QM9 samples to individual JSON files.

    Parameters
    ----------
    storage_path : str
        The path to the directory where the JSON files will be saved.
    samples : list[dict]
        The list of lifted QM9 samples.

    Returns
    -------
    None
    """

    if os.path.exists(storage_path):
        raise FileExistsError(f"Path '{storage_path}' already exists.")
    os.makedirs(storage_path, exist_ok=True)

    for idx, sample in tqdm(enumerate(samples), desc="Saving lifted QM9 samples"):
        file_name = f"{idx}.json"
        file_path = f"{storage_path}/{file_name}"
        with open(file_path, "w") as f:
            json.dump(sample, f)


def generate_loaders_qm9(args: Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:

    # Load the QM9 dataset just to get the number of samples
    data_root = "./datasets/QM9"
    num_samples = len(QM9(root=data_root))

    # Compute split indices
    with open("misc/egnn_splits.pkl", "rb") as f:
        egnn_splits = pickle.load(f)

    if args.splits == "egnn":
        split_indices = egnn_splits
        for split in egnn_splits.keys():
            random.shuffle(egnn_splits[split])
    elif args.splits == "random":
        indices = list(range(num_samples))
        random.shuffle(indices)
        train_end_idx = len(egnn_splits["train"])
        val_end_idx = train_end_idx + len(egnn_splits["valid"])
        test_end_idx = val_end_idx + len(egnn_splits["test"])
        split_indices = {
            "train": indices[:train_end_idx],
            "valid": indices[train_end_idx:val_end_idx],
            "test": indices[val_end_idx:test_end_idx],
        }
    else:
        raise ValueError(f"Unknown split type: {args.splits}")

    # Subsample if requested
    for split in split_indices.keys():
        n_split = len(split_indices[split])
        if args.num_samples is not None:
            n_split = min(args.num_samples, n_split)
            split_indices[split] = split_indices[split][:n_split]

    # Compute the target index
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
    target_map = {target: target for target in targets}
    for key in ["U0", "U", "H", "G"]:
        target_map[key] = f"{key}_atom"
    assert target_map["U0"] == "U0_atom"
    index = targets.index(target_map[args.target_name])

    # Create DataLoader kwargs
    follow_batch = [f"cell_{i}" for i in range(args.dim + 1)] + ["x"]
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": True,
    }

    # Compute the data path
    relevant_args = [
        "lifters",
        "neighbor_types",
        "connectivity",
        "visible_dims",
        "merge_neighbors",
        "dim",
        "dis",
    ]
    data_path = "./datasets/QM9_" + generate_dataset_dir_name(args, relevant_args)

    # Check if data path already exists
    if not os.path.exists(data_path):
        qm9_cc = lift_qm9_to_cc(args)
        save_lifted_qm9(data_path, qm9_cc)

    # Process data splits
    loaders = {}
    data_files = sorted(os.listdir(data_path))
    for split in ["train", "valid", "test"]:

        # Filter out the relevant data files
        split_files = [data_files[i] for i in split_indices[split]]

        # Load the ccdicts from the data files
        split_ccdicts = []
        for file in tqdm(split_files, desc="Reading lifted QM9 samples"):
            with open(f"{data_path}/{file}", "r") as f:
                split_ccdicts.append(json.load(f))

        # Convert the dictionaries to CombinatorialComplexData objects
        split_dataset = []
        for ccdict in tqdm(
            split_ccdicts, desc="Converting ccdicts to CombinatorialComplexData objects"
        ):
            ccdata = CombinatorialComplexData().from_json(ccdict)
            split_dataset.append(ccdata)

        # Preprocess data
        processed_split_dataset = []
        for cc in tqdm(split_dataset, desc="Preparing data"):
            preprocessed_graph = prepare_data(cc, index, args.target_name)
            processed_split_dataset.append(preprocessed_graph)

        # Create DataLoader
        loaders[split] = torch.utils.data.DataLoader(
            processed_split_dataset,
            collate_fn=CustomCollater(processed_split_dataset, follow_batch=follow_batch),
            **dataloader_kwargs,
        )

    return tuple(loaders.values())


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
