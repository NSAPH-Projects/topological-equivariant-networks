import argparse
import copy
import hashlib
import json
import os
import time

import torch
from tqdm import tqdm

import parser_utils
import wandb
from qm9.utils import calc_mean_mad
from utils import get_loaders, get_model, set_seed

torch.set_float32_matmul_precision("high")
os.environ["WANDB__SERVICE_WAIT"] = "600"


def save_checkpoint(state, checkpoint_path):
    # Create the checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, checkpoint_path)


def load_checkpoint(checkpoint_path, model, best_model, optimizer, scheduler):
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"]
        best_val_mae = checkpoint["best_val_mae"]
        model.load_state_dict(checkpoint["state_dict"])
        best_model.load_state_dict(checkpoint["best_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        run_id = checkpoint["run_id"]
        run_name = checkpoint["run_name"]
        return start_epoch, best_val_mae, model, best_model, optimizer, scheduler, run_id, run_name
    else:
        print(f"=> no checkpoint found at '{checkpoint_path}'")
    return 0, float("inf"), model, best_model, optimizer, scheduler, None, None


def args_to_hash(args):
    def is_json_serializable(value):
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False

    def serialize_value(value):
        if isinstance(value, list):
            return sorted(value)
        if not is_json_serializable(value):
            return str(value)
        return value

    # Convert and sort all arguments, excluding the key "lifters"
    args_dict = {
        k: serialize_value(v)
        for k, v in vars(args).items()
        if k != "lifter" and serialize_value(v) is not None
    }

    # Sort the dictionary by keys to ensure consistent order
    sorted_args_dict = dict(sorted(args_dict.items()))

    # Convert to a JSON string
    args_str = json.dumps(sorted_args_dict, sort_keys=True)

    # Compute the hash
    args_hash = hashlib.md5(args_str.encode()).hexdigest()

    return args_hash


def main(args):
    # Generate model
    model = get_model(args).to(args.device)
    if args.compile:
        model = torch.compile(model)
    best_model = copy.deepcopy(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    print(model)

    # Get loaders
    train_loader, val_loader, test_loader = get_loaders(args)
    mean, mad = calc_mean_mad(train_loader)
    mean, mad = mean.to(args.device), mad.to(args.device)

    # Get optimization objects
    criterion = torch.nn.L1Loss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    T_max = args.epochs // args.num_lr_cycles
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=args.min_lr)
    best_val_mae = float("inf")

    # Create checkpoint filename based on args hash
    args_hash = args_to_hash(args)
    checkpoint_filename = f"checkpoint_{args_hash}.pth.tar"
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_filename)

    # Load checkpoint if exists
    start_epoch, best_val_mae, model, best_model, optimizer, scheduler, run_id, run_name = (
        load_checkpoint(checkpoint_path, model, best_model, optimizer, scheduler)
    )

    # Setup wandb
    if run_id and run_name:
        wandb.init(
            project="QM9-clean-experiments",
            id=run_id,
            name=run_name,
            resume="must",
        )
    else:
        run_name = args.run_name
        wandb.init(project="QM9-clean-experiments", name=run_name)
        run_id = wandb.run.id

    for epoch in tqdm(range(start_epoch, args.epochs)):
        epoch_start_time, epoch_mae_train, epoch_mae_val = time.time(), 0, 0

        model.train()
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(args.device)

            pred = model(batch)
            loss = criterion(pred, (batch.y - mean) / mad)
            mae = criterion(pred * mad + mean, batch.y)
            loss.backward()

            if args.clip_gradient:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_amount)

            optimizer.step()
            epoch_mae_train += mae.item()

        scheduler.step()
        model.eval()
        for _, batch in enumerate(val_loader):
            batch = batch.to(args.device)
            pred = model(batch)
            mae = criterion(pred * mad + mean, batch.y)

            epoch_mae_val += mae.item()

        epoch_mae_train /= len(train_loader)
        epoch_mae_val /= len(val_loader)

        if epoch_mae_val < best_val_mae:
            best_val_mae = epoch_mae_val
            best_model = copy.deepcopy(model)

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_model_state_dict": best_model.state_dict(),
                "best_val_mae": best_val_mae,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "run_id": run_id,
                "run_name": run_name,
            },
            checkpoint_path=checkpoint_path,
        )

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        wandb.log(
            {
                "Train MAE": epoch_mae_train,
                "Validation MAE": epoch_mae_val,
                "Epoch Duration": epoch_duration,
                "Learning Rate": scheduler.get_last_lr()[0],
            }
        )

        # Compute and log test error every test_interval epochs
        if (epoch + 1) % args.test_interval == 0:
            test_mae = 0
            best_model.eval()
            for _, batch in enumerate(test_loader):
                batch = batch.to(args.device)
                pred = best_model(batch)
                mae = criterion(pred * mad + mean, batch.y)
                test_mae += mae.item()

            test_mae /= len(test_loader)
            print(f"Epoch {epoch + 1} Test MAE: {test_mae}")

            wandb.log(
                {
                    "Interval Test MAE": test_mae,
                    "Epoch": epoch + 1,
                }
            )

    test_mae = 0
    best_model.eval()
    for _, batch in enumerate(test_loader):
        batch = batch.to(args.device)
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
    parser = parser_utils.add_common_arguments(parser)

    # General parameters
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=96, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="num workers")

    # Checkpoint parameters
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/",
        help="directory to save/load checkpoints",
    )

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
        "--normalize_invariants",
        action="store_true",
        default=False,
        help="if the invariant features should be normalized (via batch normalization)",
    )
    parser.add_argument(
        "--batch_norm",
        action="store_true",
        default=False,
        help="""if batch normalization should be used in the model. If True, batch normalization
             is applied after many layers""",
    )
    parser.add_argument(
        "--lean",
        action="store_true",
        default=False,
        help="""if a lean architecture should be used. drops up to half of the layers depending on
             the number of message passing layers""",
    )
    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--min_lr", type=float, default=0, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-16, help="learning rate")
    parser.add_argument(
        "--clip_gradient", action="store_true", default=False, help="gradient clipping"
    )
    parser.add_argument("--clip_amount", type=float, default=1.0, help="gradient clipping amount")
    parser.add_argument(
        "--num_lr_cycles", type=int, default=3, help="number of learning rate cycles"
    )
    parser.add_argument(
        "--test_interval", type=int, default=10, help="interval to test the model during training"
    )

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="qm9", help="dataset")
    parser.add_argument("--target_name", type=str, default="H", help="regression task")
    parser.add_argument("--num_samples", type=int, default=None, help="num samples to to train on")
    parser.add_argument("--splits", type=str, default="egnn", help="split type")

    # wandb arguments
    parser.add_argument("--run_name", type=str, default=None, help="run name")

    parsed_args = parser.parse_args()
    parsed_args = parser_utils.add_common_derived_arguments(parsed_args)
    parsed_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(parsed_args.seed)
    main(parsed_args)
