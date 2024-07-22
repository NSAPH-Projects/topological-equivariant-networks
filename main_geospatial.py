from collections import defaultdict
import copy
import logging
import os
import time
from functools import partial

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm

import utils
import wandb
from etnn.geospatial import pm25cc, transforms

# torch.set_float32_matmul_precision("high")  # Use high precision for matmul
os.environ["WANDB__SERVICE_WAIT"] = "600"


logger = logging.getLogger(__name__)


@hydra.main(config_path="conf/conf_geospatial", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # ==== Initial setup =====
    utils.set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==== Get dataset and loader ======
    pre_transform = []
    if cfg.dataset.standardize:
        pre_transform.append(transforms.standardize_cc)
    if cfg.dataset.randomize_x0:
        pre_transform.append(partial(transforms.randomize, keys=["x_0"]))
    if cfg.dataset.virtual_node:
        pre_transform.append(transforms.add_virtual_node)
    if cfg.dataset.squash_to_graph:
        pre_transform.append(transforms.squash_cc)
    if cfg.dataset.add_positions:
        pre_transform.append(transforms.add_pos_to_cc)
    pre_transform = Compose(pre_transform)

    # mask a fraction of node during prediction, needed for node-level tasks
    masking_transform = partial(
        transforms.create_mask, seed=cfg.seed, rate=cfg.mask_rate
    )

    dataset = pm25cc.PM25CC(
        f"data/geospatialcc_{cfg.dataset_name}",
        pre_transform=pre_transform,
        force_reload=cfg.force_reload,
        transform=masking_transform,
    )
    logger.info(
        f"Created GeospatialCC dataset generated and stored in '{dataset.root}'."
    )
    # ==== Get model =====
    model = utils.get_model(cfg, dataset)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params:}")
    logger.info(model)

    # Get dataloaders
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ==== Get optimization objects =====
    crit = torch.nn.MSELoss(reduction="mean")
    opt_kwargs = dict(lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    opt = torch.optim.Adam(model.parameters(), **opt_kwargs)
    T_max = cfg.training.epochs // cfg.training.num_lr_cycles
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max, eta_min=cfg.training.min_lr
    )
    best_loss = float("inf")

    # === Configure checkpoint and wandb logging ===
    ckpt_filename = f"{cfg.experiment_name}__{cfg.seed}.pth"
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
        epoch_start_time = time.time()

        epoch_metrics = defaultdict(list)

        model.train()
        for _, batch in enumerate(loader):
            opt.zero_grad()
            batch = batch.to(device)

            # == training step ==
            opt.zero_grad()
            outputs = model(batch)
            training_mask = batch.training_mask  # 1-0 mask of nodes used for training
            pred_train = outputs["0"].squeeze()[training_mask]
            target_train = batch.y.squeeze()[training_mask]
            loss_terms = crit(pred_train, target_train)
            train_loss = loss_terms.mean()
            train_loss.backward()  # backpropagate
            if cfg.training.clip_gradients:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), cfg.training.clip_amount
                )
            opt.step()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # == eval step ===
            model.eval()
            if model.dropout > 0:
                with torch.no_grad():
                    pred_eval = model(batch)["0"].squeeze()
            else:
                pred_eval = outputs["0"].squeeze().detach()
            pred_test = pred_eval[batch.test_mask]
            pred_val = pred_eval[batch.validation_mask]
            target_test = batch.y.squeeze()[batch.test_mask]
            target_val = batch.y.squeeze()[batch.validation_mask]

            val_r2 = 1 - (pred_val - target_val).var() / target_val.var()
            test_r2 = 1 - (pred_test - target_test).var() / target_test.var()
            test_mse = (pred_test - target_test).pow(2).mean()
            val_mse = (pred_val - target_val).pow(2).mean()

            epoch_metrics["train_loss"].append(train_loss.item())
            epoch_metrics["val_r2"].append(val_r2.item())
            epoch_metrics["val_mse"].append(val_mse.item())
            epoch_metrics["test_r2"].append(test_r2.item())
            epoch_metrics["test_mse"].append(test_mse.item())
            epoch_metrics["lr"].append(opt.param_groups[0]["lr"])

        sched.step()

        epoch_loss_val = np.mean(epoch_metrics["val_mse"])

        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
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

        epoch_metrics = {k: np.nanmean(v) for k, v in epoch_metrics.items()}

        wandb.log(
            {
                "Epoch Duration": epoch_duration,
                "Learning Rate": sched.get_last_lr()[0],
                "Epoch": epoch + 1,
                **epoch_metrics,
            },
            step=epoch,
        )


if __name__ == "__main__":
    main()
