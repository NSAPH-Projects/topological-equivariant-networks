import hashlib
import json
import pickle
import random
from argparse import Namespace

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from tqdm import tqdm

from combinatorial_data.lifts import get_lifters
from combinatorial_data.ranker import get_ranker
from combinatorial_data.utils import CombinatorialComplexTransform, CustomCollater

def generate_loaders_chain(args: Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:

    # Create the transform
    lifters = get_lifters(args)
    ranker = get_ranker(args.lifters)
    transform = CombinatorialComplexTransform(
        lifters=lifters,
        ranker=ranker,
        dim=args.dim,
        adjacencies=args.adjacencies,
        neighbor_type=args.neighbor_type,
    )

    # Create DataLoader kwargs
    follow_batch = [f"x_{i}" for i in range(args.dim + 1)] + ["x"]
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": True,
    }

    # Generate k-chains
    k = args.k_chain 
    assert k >= 2
    
    dataset = []
    
    # Graph 0
    x = torch.zeros(len(atoms),0)
    atoms = torch.LongTensor( [0] + [0] + [0]*(k-1) + [0] )
    edge_index = torch.LongTensor( [ [i for i in range((k+2) - 1)], [i for i in range(1, k+2)] ] )
    pos = torch.FloatTensor(
        [[-4, -3, 0]] + 
        [[0, 5*i , 0] for i in range(k)] + 
        [[4, 5*(k-1) + 3, 0]]
    )
    center_of_mass = torch.mean(pos, dim=0)
    pos = pos - center_of_mass
    y = torch.tensor(0)  # Label 0
    graph1 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y, x=x)
    graph1.edge_index = to_undirected(graph1.edge_index)
    dataset.append(graph1)
    
    # Graph 1
    x = torch.zeros(len(atoms),0)
    atoms = torch.LongTensor( [0] + [0] + [0]*(k-1) + [0] )
    edge_index = torch.LongTensor( [ [i for i in range((k+2) - 1)], [i for i in range(1, k+2)] ] )
    pos = torch.FloatTensor(
        [[4, -3, 0]] + 
        [[0, 5*i , 0] for i in range(k)] + 
        [[4, 5*(k-1) + 3, 0]]
    )
    center_of_mass = torch.mean(pos, dim=0)
    pos = pos - center_of_mass
    y = torch.tensor(1)  # Label 1
    graph2 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y, x=x)
    graph2.edge_index = to_undirected(graph2.edge_index)
    dataset.append(graph2)

    # Process data splits
    loaders = {}
    for split in ["train", "valid", "test"]:

        # Transform and preprocess data
        processed_split_dataset = []
        for graph in tqdm(dataset, desc="Preparing data"):
            transformed_graph = transform(graph)
            processed_split_dataset.append(transformed_graph)

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
