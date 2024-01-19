import torch.nn as nn
import random
import numpy as np
import os
import torch
from torch import Tensor
from argparse import Namespace
from torch_geometric.loader import DataLoader
from typing import Tuple


def get_model(args: Namespace) -> nn.Module:
    """Return model based on name."""
    if args.dataset == 'qm9':
        num_input = 15
        num_out = 1
    else:
        raise ValueError(f'Do not recognize dataset {args.dataset}.')

    if args.model_name == 'egnn':
        from models.egnn import EGNN
        model = EGNN(
            num_input=num_input,
            num_hidden=args.num_hidden,
            num_out=num_out,
            num_layers=args.num_layers
        )
    elif args.model_name == 'empsn':
        from models.empsn import EMPSN
        model = EMPSN(
            num_input=num_input,
            num_hidden=args.num_hidden,
            num_out=num_out,
            num_layers=args.num_layers,
            max_com=args.max_com
        )
    else:
        raise ValueError(f'Model type {args.model_name} not recognized.')

    return model


def get_loaders(args: Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return dataloaders based on dataset."""
    if args.dataset == 'qm9':
        from qm9.utils import generate_loaders_qm9
        train_loader, val_loader, test_loader = generate_loaders_qm9(args)
    else:
        raise ValueError(f'Dataset {args.dataset} not recognized.')

    return train_loader, val_loader, test_loader


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
