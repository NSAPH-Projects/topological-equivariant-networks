import argparse
import copy
import time

import torch
from tqdm import tqdm

# import wandb
from qm9.utils import calc_mean_mad
from utils import (
    get_adjacency_types,
    get_loaders,
    get_model,
    merge_adjacencies,
    set_seed,
    task_settings,
)
torch.autograd.set_detect_anomaly(True)

def main(args):
    # # Generate model
    model = get_model(args).to(args.device)
    if args.compile:
        model = torch.compile(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    print(model)

    # Setup wandb
    # wandb.init(entity="ten-harvard", project=f"{args.dataset.upper()}-{args.target_name}")
    # wandb.config.update(vars(args))

    # # Get loaders
    train_loader, val_loader, test_loader = get_loaders(args)
    if args.dataset == "qm9":
        mean, mad = calc_mean_mad(train_loader)
        mean, mad = mean.to(args.device), mad.to(args.device)

    # Get optimization objects
    criterion = task_settings[args.task_type]["criterion"]
    metric = task_settings[args.task_type]["metric"]
    # criterion = torch.nn.L1Loss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    T_max = args.epochs // args.num_lr_cycles
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)
    best_val_metric, best_model = metric["worst_value"], None

    # torch.autograd.set_detect_anomaly(True)
    for _ in tqdm(range(args.epochs)):
        epoch_start_time, epoch_metric_train, epoch_metric_val = time.time(), 0, 0

        model.train()
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()

            batch = batch.to(args.device)
            pred = model(batch)

            transformed_batch = (batch.y - mean) / mad if args.dataset == "qm9" else batch.y
            transformed_pred = pred * mad + mean if args.dataset == "qm9" else pred

            loss = criterion(pred, transformed_batch)
            metric_value = metric["fct"](transformed_pred, batch.y)
            loss.backward()

            if args.clip_gradient:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_amount)

            optimizer.step()
            epoch_metric_train += metric_value.item()

        scheduler.step()
        model.eval()
        for _, batch in enumerate(val_loader):
            batch = batch.to(args.device)
            pred = model(batch)
            transformed_pred = pred * mad + mean if args.dataset == "qm9" else pred
            metric_value = metric["fct"](transformed_pred, batch.y)
            pos_idx = 0 if batch.y[0] > batch.y[1] else 1
            epoch_metric_val += metric_value.item()

        epoch_metric_train /= len(train_loader)
        epoch_metric_val /= len(val_loader)

        if metric["greater_is_better"] == (epoch_metric_val > best_val_metric):
            best_val_metric = epoch_metric_val
            best_model = copy.deepcopy(model)

        epoch_end_time = time.time()  # End timing the epoch
        epoch_duration = epoch_end_time - epoch_start_time  # Calculate the duration

        # wandb.log(
        #     {
        #         f"Train {metric['name']}": epoch_metric_train,
        #         f"Validation {metric['name']}": epoch_metric_val,
        #         "Epoch Duration": epoch_duration,
        #         "Learning Rate": scheduler.get_last_lr()[0],
        #         "Logit_0": pred[1 - pos_idx].item(),
        #         "Logit_1": pred[pos_idx].item(),
        #     }
        # )

    test_metric = 0
    best_model.eval()
    for _, batch in enumerate(test_loader):
        batch = batch.to(args.device)
        pred = best_model(batch)
        transformed_pred = pred * mad + mean if args.dataset == "qm9" else pred
        metric_value = metric["fct"](transformed_pred, batch.y)
        test_metric += metric_value.item()

    test_metric /= len(test_loader)
    print(f"Test {metric['name']}: {test_metric}")

    # wandb.log(
    #     {
    #         f"Test {metric['name']}": test_metric,
    #     }
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=96, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="num workers")

    # Model parameters
    parser.add_argument(
        "--compile", action="store_true", default=False, help="if the model should be compiled"
    )
    parser.add_argument("--model_name", type=str, default="empsn", help="model")
    parser.add_argument("--max_com", type=str, default="1_2", help="model type")  # e.g. 1_2
    parser.add_argument("--num_hidden", type=int, default=77, help="hidden features")
    parser.add_argument("--num_layers", type=int, default=7, help="number of layers")
    parser.add_argument("--act_fn", type=str, default="silu", help="activation function")
    parser.add_argument(
        "--lifters",
        nargs="+",
        help="list of lifters to apply and their ranking logic",
        default="identity:c functional_group:2 ring:2",
        required=True,
    )
    parser.add_argument("--initial_features", type=str, default="node", help="features to use")
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
    "--equivariant",
    action="store_true",
    default=False,
    help="""specifies whether to enable the equivariant mode of the model""",
    )
    parser.add_argument(
        "--visible_dims",
        nargs="+",
        type=int,
        default=None,
        help="specifies which ranks to explicitly represent as nodes",
    )
    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-16, help="learning rate")
    parser.add_argument(
        "--clip_gradient", action="store_true", default=False, help="gradient clipping"
    )
    parser.add_argument("--clip_amount", type=float, default=1.0, help="gradient clipping amount")
    parser.add_argument(
        "--num_lr_cycles", type=int, default=3, help="number of learning rate cycles"
    )

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="qm9", help="dataset")
    parser.add_argument("--task_type", type=str, default="regression", help="task type")
    parser.add_argument(
        "--chain_length", type=int, default=4, help="chain length for the synthetic task"
    )
    parser.add_argument("--target_name", type=str, default="H", help="regression task")
    parser.add_argument("--dim", type=int, default=2, help="ASC dimension")
    parser.add_argument("--dis", type=float, default=4.0, help="radius Rips complex")
    parser.add_argument(
        "--max_path_length",
        type=int,
        default=3,
        help="maximum path length considered for path lifter",
    )
    parser.add_argument("--num_samples", type=int, default=None, help="num samples to to train on")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--splits", type=str, default="egnn", help="split type")

    # Other arguments
    parser.add_argument(
        "--enable_indexing_bug",
        action="store_true",
        help="""If the buggy legacy implementation should be used for the combinatorial complex
             transform. Needed to reproduce EMSPN.""",
    )

    parsed_args = parser.parse_args()
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

    parsed_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(parsed_args.seed)
    main(parsed_args)
