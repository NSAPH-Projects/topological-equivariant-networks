import argparse
from qm9.utils import process_qm9_dataset
from utils import set_seed

def main(args):
    set_seed(args.seed)
    process_qm9_dataset(
        args.lifter_names, 
        args.neighbor_types, 
        args.connectivity, 
        args.visible_dims,  
        args.initial_features, 
        args.dim, 
        args.dis,
        args.merge_neighbors,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # lifter_names : list[str]
    # neighbor_types : list[str]
    # connectivity : str
    # visible_dims : list[int]
    # initial_features : str
    # dim : int
    # dis : bool
    # merge_neighbors : bool
    
    parser.add_argument(
        "--lifter_names",
        nargs="+",
        type=str,
        help="List of lifters to apply and their ranking logic.",
        default=["atom:0", "bond:1", "supercell:2"],
    )
    parser.add_argument(
        "--neighbor_types",
        nargs="+",
        type=str,
        help="""
        Defines adjacency between cells of the same rank. 
        For '+1', two cells of rank i are connected if they are both connected to the same cell of rank i+1.
        """,
        default=["+1", "max"],
    )
    parser.add_argument(
        "--connectivity",
        type=str,
        help="Connectivity pattern between ranks.",
        default="self",
    )
    parser.add_argument(
        "--visible_dims",
        nargs="+",
        type=int,
        help="Specifies which ranks to explicitly represent as nodes.",
        default=[0, 1],
    )
    parser.add_argument(
        "--initial_features",
        nargs="+",
        type=str,
        help="List of features to use.",
        default=["node", "hetero"],
    )
    parser.add_argument(
        "--dim",
        type=int,
        help="ASC dimension.",
        default=2,
    )
    parser.add_argument(
        "--dis",
        type=float,
        help="Radius for Rips complex.",
        default=4.0,
    )
    parser.add_argument(
        "--merge_neighbors",
        #action="store_true",
        help="""If set, all the neighbors of different types will be represented as a single
             adjacency matrix.""",
        default=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
        default=42,
    )
    args = parser.parse_args()

    main(args)