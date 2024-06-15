import argparse
import os

from combinatorial_data.lifter import Lifter
from qm9.lifts.registry import lifter_registry
from qm9.utils import dataset_args, generate_dataset_dir_name
from utils import get_adjacency_types, merge_adjacencies


def add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--lifters",
        nargs="+",
        help="List of lifters to apply and their ranking logic.",
        default=["identity:c", "functional_group:2", "ring:2"],
        required=True,
    )
    parser.add_argument(
        "--neighbor_types",
        nargs="+",
        type=str,
        default=["+1"],
        help="""Defines adjacency between cells of the same rank. Default is '+1'. Two cells of rank
                i are connected if they are both connected to the same cell of rank i+1.""",
    )
    parser.add_argument(
        "--connectivity",
        type=str,
        default="self_and_next",
        help="Connectivity pattern between ranks.",
    )
    parser.add_argument(
        "--visible_dims",
        nargs="+",
        type=int,
        default=None,
        help="Specifies which ranks to explicitly represent as nodes. Default is None.",
    )
    parser.add_argument(
        "--merge_neighbors",
        action="store_true",
        default=False,
        help="""If set, all the neighbors of different types will be represented as a single
             adjacency matrix.""",
    )
    parser.add_argument(
        "--initial_features",
        nargs="+",
        type=str,
        default=["node"],
        help="Features to use.",
    )
    parser.add_argument(
        "--triangles_only",
        action="store_true",
        default=False,
        help="""If set, only triangles will be considered as rings.""",
    )
    parser.add_argument("--dim", type=int, default=2, help="ASC dimension.")
    parser.add_argument("--dis", type=float, default=4.0, help="Radius for Rips complex.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    parser.add_argument(
        "--storage_path",
        type=str,
        default=os.getenv("STORAGE_PATH"),
        help="Path to store data, model checkpoints, etc.",
    )

    return parser


def add_common_derived_arguments(parsed_args):
    parsed_args.adjacencies = get_adjacency_types(
        parsed_args.dim,
        parsed_args.connectivity,
        parsed_args.neighbor_types,
        parsed_args.visible_dims,
    )
    # If merge_neighbors is True, the adjacency types we feed to the model will be the merged ones
    if parsed_args.merge_neighbors:
        parsed_args.processed_adjacencies = merge_adjacencies(parsed_args.adjacencies)
    else:
        parsed_args.processed_adjacencies = parsed_args.adjacencies

    parsed_args.initial_features = sorted(parsed_args.initial_features)
    parsed_args.lifter = Lifter(parsed_args, lifter_registry)
    filtered_args = {key: value for key, value in vars(parsed_args).items() if key in dataset_args}
    parsed_args.data_path = os.path.join(
        parsed_args.storage_path,
        "datasets",
        "QM9_CC_" + generate_dataset_dir_name(filtered_args) + ".jsonl",
    )

    return parsed_args
