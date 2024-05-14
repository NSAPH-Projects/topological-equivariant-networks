import argparse
import copy
import time

import torch
from tqdm import tqdm

import wandb

from utils import (
    get_adjacency_types,
    get_loaders,
    get_model,
    merge_adjacencies,
    set_seed,
)


def main(cfg):
    # # Generate model
    model = get_model(cfg).to(cfg.device)
    if cfg.compile:
        model = torch.compile(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    print(model)

    # Setup wandb
    wandb.init(entity="ten-harvard", project=f"QM9-{cfg.target_name}")
    wandb.config.update(vars(cfg))

    # # Get loaders
    train_loader, val_loader, test_loader = get_loaders(cfg)
    mean, mad = calc_mean_mad(train_loader)
    mean, mad = mean.to(cfg.device), mad.to(cfg.device)

    # Get optimization objects
    criterion = torch.nn.L1Loss(reduction="mean")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    T_max = cfg.epochs // cfg.num_lr_cycles
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)
    best_val_mae, best_model = float("inf"), None

    for _ in tqdm(range(cfg.epochs)):
        epoch_start_time, epoch_mae_train, epoch_mae_val = time.time(), 0, 0

        model.train()
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(cfg.device)

            pred = model(batch)
            loss = criterion(pred, (batch.y - mean) / mad)
            mae = criterion(pred * mad + mean, batch.y)
            loss.backward()

            if cfg.clip_gradient:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_amount)

            optimizer.step()
            epoch_mae_train += mae.item()

        scheduler.step()
        model.eval()
        for _, batch in enumerate(val_loader):
            batch = batch.to(cfg.device)
            pred = model(batch)
            mae = criterion(pred * mad + mean, batch.y)

            epoch_mae_val += mae.item()

        epoch_mae_train /= len(train_loader)
        epoch_mae_val /= len(val_loader)

        if epoch_mae_val < best_val_mae:
            best_val_mae = epoch_mae_val
            best_model = copy.deepcopy(model)

        epoch_end_time = time.time()  # End timing the epoch
        epoch_duration = epoch_end_time - epoch_start_time  # Calculate the duration

        wandb.log(
            {
                "Train MAE": epoch_mae_train,
                "Validation MAE": epoch_mae_val,
                "Epoch Duration": epoch_duration,
                "Learning Rate": scheduler.get_last_lr()[0],
            }
        )

    test_mae = 0
    best_model.eval()
    for _, batch in enumerate(test_loader):
        batch = batch.to(cfg.device)
        pred = best_model(batch)
        mae = criterion(pred * mad + mean, batch.y)
        test_mae += mae.item()

    test_mae /= len(test_loader)
    print(f"Test MAE: {test_mae}")

    wandb.log(
        {
            "Test MAE": test_mae,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=96, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="num workers")

    # Model parameters
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="if the model should be compiled",
    )
    parser.add_argument("--model_name", type=str, default="empsn", help="model")
    parser.add_argument(
        "--max_com", type=str, default="1_2", help="model type"
    )  # e.g. 1_2
    parser.add_argument("--num_hidden", type=int, default=77, help="hidden features")
    parser.add_argument("--num_layers", type=int, default=7, help="number of layers")
    parser.add_argument(
        "--act_fn", type=str, default="silu", help="activation function"
    )
    parser.add_argument(
        "--lifters",
        nargs="+",
        help="list of lifters to apply and their ranking logic",
        default="identity:c functional_group:2 ring:2",
        required=True,
    )
    parser.add_argument(
        "--initial_features", type=str, default="node", help="features to use"
    )
    parser.add_argument(
        "--connectivity",
        type=str,
        default="self_and_next",
        help="connectivity pattern between ranks",
    )
    parser.add_argument(
        "--neighbor_types",
        nargs="+",
        type=str,
        default=["+1"],
        help="""How adjacency between cells of same rank is defined. Default is +1, meaning that
                two cells of rank i are connected if they are both connected to the same cell of 
                rank i+1. See src.utils.py::get_adjacencies for a list of possible values.""",
    )
    parser.add_argument(
        "--merge_neighbors",
        action="store_true",
        default=False,
        help="""if all the neighbors of different types should be represented as a single adjacency
             matrix""",
    )
    parser.add_argument(
        "--visible_dims",
        nargs="+",
        type=int,
        default=None,
        help="specifies which ranks to explicitly represent as nodes",
    )
    parser.add_argument(
        "--normalize_invariants",
        action="store_true",
        default=False,
        help="if the invariant features should be normalized (via batch normalization)",
    )
    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-16, help="learning rate"
    )
    parser.add_argument(
        "--clip_gradient", action="store_true", default=False, help="gradient clipping"
    )
    parser.add_argument(
        "--clip_amount", type=float, default=1.0, help="gradient clipping amount"
    )
    parser.add_argument(
        "--num_lr_cycles", type=int, default=3, help="number of learning rate cycles"
    )

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="qm9", help="dataset")
    parser.add_argument("--target_name", type=str, default="H", help="regression task")
    parser.add_argument("--dim", type=int, default=2, help="ASC dimension")
    parser.add_argument("--dis", type=float, default=4.0, help="radius Rips complex")
    parser.add_argument(
        "--num_samples", type=int, default=None, help="num samples to to train on"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--splits", type=str, default="egnn", help="split type")

    cfg = parser.parse_args()
    cfg.adjacencies = get_adjacency_types(
        cfg.dim,
        cfg.connectivity,
        cfg.neighbor_types,
        cfg.visible_dims,
    )
    # If merge_neighbors is True, the adjacency types we feed to the model will be the merged ones
    if cfg.merge_neighbors:
        cfg.processed_adjacencies = merge_adjacencies(cfg.adjacencies)
    else:
        cfg.processed_adjacencies = cfg.adjacencies

    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed)
    main(cfg)