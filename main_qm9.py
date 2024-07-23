import copy
import json
import logging
import os
import random
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import utils
import wandb
from etnn.qm9.qm9cc import QM9CC

# torch.set_float32_matmul_precision("high")  # Use high precision for matmul
os.environ["WANDB__SERVICE_WAIT"] = "600"


logger = logging.getLogger(__name__)


@hydra.main(config_path="conf/conf_qm9", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # ==== Initial setup =====
    utils.set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==== Get dataset and loader ======
    def prepare_targets_transform(data):
        col_ix = QM9CC.targets.index(cfg.target)
        data.y = data.y[:, col_ix]
        return data

    dataset = QM9CC(
        f"data/qm9cc_{cfg.dataset_name}",
        lifters=list(cfg.dataset.lifters),
        neighbor_types=list(cfg.dataset.neighbor_types),
        connectivity=cfg.dataset.connectivity,
        supercell=cfg.dataset.supercell,
        force_reload=False,
        transform=prepare_targets_transform,
    )

    # ==== Get model =====
    model = utils.get_model(cfg, dataset)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params:}")
    logger.info(model)

    # Get train/test splits using the original egnn splits for reference
    with open("data/input/egnn_splits.json", "r") as io:
        egnn_splits = json.load(io)
    if cfg.train_test_splits == "egnn":
        split_indices = egnn_splits

    elif cfg.train_test_splits == "random":
        num_samples = len(dataset)
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
        raise ValueError(f"Unknown split type: {cfg.train_test_splits}")

    # Get dataloaders
    loaders = {
        key: DataLoader(
            dataset[indices],
            batch_size=cfg.training.batch_size,
            shuffle=True,
        )
        for key, indices in split_indices.items()
    }

    # Precompute average deviation of target in loader
    mean, mad = utils.calc_mean_mad(loaders["train"])
    mean, mad = mean.to(device), mad.to(device)

    # ==== Get optimization objects =====
    crit = torch.nn.L1Loss(reduction="mean")
    opt_kwargs = dict(lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    opt = torch.optim.Adam(model.parameters(), **opt_kwargs)
    T_max = cfg.training.epochs // cfg.training.num_lr_cycles
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max, eta_min=cfg.training.min_lr
    )
    best_loss = float("inf")

    # === Configure checkpoint and wandb logging ===
    ckpt_filename = f"{cfg.experiment_name}__{cfg.target}__{cfg.seed}.pth"
    if cfg.ckpt_prefix is not None:
        ckpt_filename = f"{cfg.ckpt_prefix}_{ckpt_filename}"
    checkpoint_path = f"{cfg.ckpt_dir}/{ckpt_filename}"

    start_epoch, run_id, best_model, best_loss = utils.load_checkpoint(
        checkpoint_path, model, opt, sched, cfg.force_restart
    )

    if start_epoch >= cfg.training.epochs:
        logger.info("Training already completed. Exiting.")
        return

    # init wandb logger
    if run_id is None:
        run_id = ckpt_filename.split(".")[0] + "__" + wandb.util.generate_id()
        if cfg.ckpt_prefix is not None:
            run_id = "__".join([cfg.ckpt_prefix, run_id])

    # create wandb config and add number of parameters
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb_config["num_params"] = num_params

    wandb.init(
        project=f"{cfg.task_name}-experiments",
        entity=os.environ.get("WANDB_ENTITY"),
        config=wandb_config,
        id=run_id,
        resume="allow",
    )

    # === Training loop ===
    for epoch in tqdm(range(start_epoch, cfg.training.epochs)):
        epoch_start_time, epoch_mae_train, epoch_mae_val = time.time(), 0, 0

        model.train()
        for _, batch in enumerate(loaders["train"]):
            opt.zero_grad()
            batch = batch.to(device)

            pred = model(batch)
            loss = crit(pred, (batch.y - mean) / mad)
            mae = crit(pred * mad + mean, batch.y)
            loss.backward()

            if cfg.training.clip_gradients:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.clip_amount
                )

            opt.step()
            epoch_mae_train += mae.item()

        sched.step()
        model.eval()
        for _, batch in enumerate(loaders["valid"]):
            batch = batch.to(device)
            pred = model(batch)
            mae = crit(pred * mad + mean, batch.y)

            epoch_mae_val += mae.item()

        epoch_mae_train /= len(loaders["train"])
        epoch_mae_val /= len(loaders["valid"])

        if epoch_mae_val < best_loss:
            best_loss = epoch_mae_val
            best_model = copy.deepcopy(model)

        # Save checkpoint
        utils.save_checkpoint(
            path=checkpoint_path,
            model=model,
            best_model=best_model,
            best_loss=best_loss,
            opt=opt,
            sched=sched,
            epoch=epoch,
            run_id=run_id,
        )

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        wandb.log(
            {
                "Train MAE": epoch_mae_train,
                "Validation MAE": epoch_mae_val,
                "Epoch Duration": epoch_duration,
                "Learning Rate": sched.get_last_lr()[0],
            },
            step=epoch,
        )

        # Compute and log test error every test_interval epochs
        if (epoch + 1) % cfg.training.test_interval == 0:
            test_mae = 0
            best_model.eval()
            for _, batch in enumerate(loaders["test"]):
                batch = batch.to(device)
                pred = best_model(batch)
                mae = crit(pred * mad + mean, batch.y)
                test_mae += mae.item()

            test_mae /= len(loaders["test"])
            print(f"Epoch {epoch + 1} Test MAE: {test_mae}")

            wandb.log(
                {
                    "Interval Test MAE": test_mae,
                    "Epoch": epoch + 1,
                },
                step=epoch,
            )

    test_mae = 0
    best_model.eval()
    for _, batch in enumerate(loaders["test"]):
        batch = batch.to(device)
        pred = best_model(batch)
        mae = crit(pred * mad + mean, batch.y)
        test_mae += mae.item()

    test_mae /= len(loaders["test"])
    print(f"Test MAE: {test_mae}")

    wandb.log({"Test MAE": test_mae})


if __name__ == "__main__":
    main()
