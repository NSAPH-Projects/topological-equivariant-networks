import hashlib
import json
import os
import pickle
import random
from argparse import Namespace

import torch
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
from qm9.qm9_cc import QM9_CC
from utils import get_adjacency_types, merge_adjacencies

dataset_args = [
    "lifters",
    "neighbor_types",
    "connectivity",
    "visible_dims",
    "merge_neighbors",
    "initial_features",
    "dim",
    "dis",
]


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

def process_qm9_dataset(lifter_names, neighbor_types, connectivity, visible_dims, initial_features, dim, dis, merge_neighbors):
    """
    Process the QM9 dataset.

    Parameters
    ----------
    lifter_names : list[str]
    neighbor_types : list[str]
    connectivity : str
    visible_dims : list[int]
    initial_features : str
    dim : int
    dis : bool
    merge_neighbors : bool

    Returns
    -------
    None

    Notes
    -----
    This function computes the data path based on the relevant arguments provided in `args`. It then
    lifts samples of the QM9 dataset to combinatorial complexes using the `lift_qm9_to_cc` function.
    Finally, it saves the lifted QM9 dataset to the specified data path using the `save_lifted_qm9`
    function.
    """

    # Compute the data path
    data_path = (
        "data/qm9_cc/QM9_CC_" + 
        generate_dataset_dir_name(lifter_names, neighbor_types, connectivity, visible_dims, merge_neighbors, initial_features, dim, dis) + 
        ".jsonl"
    )

    if os.path.exists(data_path):
        print(f"File '{data_path}' already exists.")
        return

    # Lift the QM9 dataset to CombinatorialComplexData format
    qm9_cc = lift_qm9_to_cc(lifter_names, neighbor_types, connectivity, visible_dims, initial_features, dim, dis, merge_neighbors)

    # Save the lifted QM9 dataset to the specified data path
    save_lifted_qm9(data_path, qm9_cc)

def lift_qm9_to_cc(lifter_names, neighbor_types, connectivity, visible_dims, initial_features, dim, dis, merge_neighbors) -> list[dict]:
    """
    Lift QM9 dataset to CombinatorialComplexData format.

    Parameters
    ----------
    lifter_names : list[str]
        The names of the lifters to apply.
    neighbor_types : list[str]
        The types of neighbors to consider. Defines adjacency between cells of the same rank.
    connectivity : str
        The connectivity pattern between ranks.
    visible_dims : list[int]
        Specifies which ranks to explicitly represent as nodes.
    initial_features : list[str]
        The initial features to use.
    dim : int
        The ASC dimension.
    dis : bool
        Radius for Rips complex
    merge_neighbors : bool
        Whether to merge neighbors.

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

    #dim : int
    #neighbor_types : list[str]
    #connectivity : str
    #visible_dims : list[int]
    adjacencies = get_adjacency_types(
        dim,
        connectivity,
        neighbor_types,
        visible_dims,
    )
    # If merge_neighbors is True, the adjacency types we feed to the model will be the merged ones
    if merge_neighbors:
        processed_adjacencies = merge_adjacencies(adjacencies)
    else:
        processed_adjacencies = adjacencies

    initial_features = sorted(initial_features)
    #lifter_names : list[str]
    #initial_features : str
    #dim : int
    #dis : bool
    lifter = Lifter(lifter_names, initial_features, dim, dis, lifter_registry)

    # Create the transform lifter, dim, adjacencies, processed_adjacencies, merge_neighbors
    #dim : int
    #merge_neighbors : bool
    transform = CombinatorialComplexTransform(
        lifter=lifter,
        dim=dim,
        adjacencies=adjacencies,
        processed_adjacencies=processed_adjacencies,
        merge_neighbors=merge_neighbors,
    )

    qm9_cc = QM9_CC("data/qm9_cc", pre_transform=transform.graph_to_ccdict) 
    # the QM9_CC class in an InMemoryDataset, so we can pass the pre_transform argument to the constructor
    # the self.root is the root path that determines self.raw_dir and self.processed_dir
    # by default is self.raw_dir=<self.root>/raw self.processed_dir=<self.root>/processed
    return qm9_cc


def save_lifted_qm9(storage_path: str, lifted_qm9: QM9_CC) -> None:
    """
    Save the lifted QM9 samples to a single JSON Lines (.jsonl) file.

    Parameters
    ----------
    storage_path : str
        The path to the .jsonl file where the data will be saved.
    samples : list[dict]
        The list of lifted QM9 samples.

    Returns
    -------
    None
    """

    samples = lifted_qm9.data_list

    if os.path.exists(storage_path):
        raise FileExistsError(f"File '{storage_path}' already exists.")

    with open(storage_path, "w") as f:
        for sample in tqdm(samples, desc="Saving lifted QM9 samples"):
            json.dump(sample, f)
            f.write("\n")


def generate_loaders_qm9(args: Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:

    # Load the QM9 dataset

    # Compute the data path
    filtered_args = {key: value for key, value in vars(args).items() if key in dataset_args}
    data_path = "datasets/QM9_CC_" + generate_dataset_dir_name(filtered_args) + ".jsonl"

    # Check if data path already exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File '{data_path}' does not exist.")

    # Load the data
    json_list = []
    with open(data_path, "r") as f:
        for line in tqdm(f):
            json_list.append(json.loads(line))
    num_samples = len(json_list)

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

    # Process data splits
    loaders = {}
    for split in ["train", "valid", "test"]:

        # Filter out the relevant data files
        split_ccdicts = [json_list[i] for i in split_indices[split]]

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


def generate_dataset_dir_name(lifters, neighbor_types, connectivity, visible_dims, merge_neighbors, initial_features, dim, dis) -> str:
    """
    Generate a directory name based on molecule characteristics.

    Parameters
    ----------
    lifters : list[str]
    neighbor_types : list[str]
    connectivity : str
    visible_dims : list[int]
    initial_features : str
    dim : int
    dis : bool
    merge_neighbors : bool

    Returns
    -------
    str: A hash-based directory name representing the relevant arguments.
    """
    dataset_args = {
        "lifters": lifters,
        "neighbor_types": neighbor_types,
        "connectivity": connectivity,
        "visible_dims": visible_dims,
        "merge_neighbors": merge_neighbors,
        "initial_features": initial_features,
        "dim": dim,
        "dis": dis,
    }

    # Convert relevant arguments to a JSON string for consistent ordering
    args_str = json.dumps(dataset_args, sort_keys=True)

    # Create a hash of the relevant arguments string
    hash_obj = hashlib.sha256(args_str.encode())
    hash_hex = hash_obj.hexdigest()

    # Optional: truncate the hash for a shorter name
    short_hash = hash_hex[:16]  # First 16 characters

    return short_hash
