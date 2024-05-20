import os
import json
import hydra
import numpy as np
import wandb
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict

from torch import nn
from tqdm.auto import tqdm, trange
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from etnn.utils import set_seed


def save_checkpoint(epoch, model, optimizer, scheduler, run_id, filepath):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.cpu().state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "run_id": run_id,
        },
        filepath,
    )
    


def load_checkpoint(filepath, model, optimizer, scheduler):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["epoch"] + 1, checkpoint["run_id"]
    else:
        return 0, None


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # seed everything
    set_seed(cfg.seed)

    # get device
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # == instantiate dataset, loader, model, optimizer, scheduler ==
    dataset: Dataset = instantiate(cfg.dataset)
    collate_fn: callable = instantiate(cfg.collate_fn, dataset)
    loader: DataLoader = instantiate(cfg.loader, dataset, collate_fn=collate_fn)

    # determine number of features per rank
    batch = next(iter(loader))   # get the first element to compute number of features
    # to create the model
    num_feats = {
        k.split("_")[1]: v.shape[1] for k, v in batch.items() if k.startswith("x_")
    }
    adjacencies = [k[4:] for k in batch.keys() if k.startswith("adj_")]

    # instantiate model, optimizer, learning rate scheduler, loss fn
    model: nn.Module = instantiate(
        cfg.model, num_features_per_rank=num_feats, adjacencies=adjacencies
    )
    opt: Optimizer = instantiate(cfg.optimizer, model.parameters())
    sched: LRScheduler = instantiate(cfg.lr_scheduler, opt)
    loss_fn: callable = instantiate(cfg.loss_fn, reduction="none")

    # == checkpointing ==
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_filename = f"{cfg.baseline_name}_{cfg.seed}.pth"
    if cfg.ckpt_prefix is not None:
        checkpoint_filename = f"{cfg.ckpt_prefix}_{checkpoint_filename}"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    # Check for force restart flag
    if not cfg.force_restart:
        start_epoch, run_id = load_checkpoint(checkpoint_path, model, opt, sched)
    else:
        start_epoch, run_id = 0, None

    # == init wandb logger ==
    if run_id is None:
        run_id = "_".join([cfg.baseline_name, str(cfg.seed), wandb.util.generate_id()])
        if cfg.ckpt_prefix is not None:
            run_id = "_".join([cfg.ckpt_prefix, run_id])
        resume = False
    else:
        resume = True

    # create wandb config and add number of parameters
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb_config["num_params"] = num_params

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=wandb_config,
        id=run_id,
        resume=resume,
    )

    # == training loop ==
    model = model.to(dev)
    model.train()

    # since only one batch is used, move it to the device
    batch = batch.to(dev)

    # get training params from config
    pbar = trange(start_epoch, cfg.training.max_epochs, desc="", leave=True)
    for epoch in pbar:
        epoch_metrics = defaultdict(list)
        # for batch in loader:
        # batch = batch.to(dev)

        # == training step ==
        opt.zero_grad()
        outputs = model(batch)
        mask = batch.mask  # 1-0 mask of nodes used for training
        loss_terms = loss_fn(outputs["0"].squeeze(-1), batch.y.squeeze(-1))
        train_loss = (loss_terms * mask).sum() / mask.sum()
        eval_loss = (loss_terms * (1 - mask)).sum() / (1 - mask).sum()
        train_loss.backward()  # backpropagate
        if cfg.training.clip is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(), cfg.training.clip)
        opt.step()

        if dev == "cuda":
            # not really helping, but this should free up unused GPU memory
            torch.cuda.empty_cache()

        # == end training step ==

        epoch_metrics["train_loss"].append(train_loss.item())
        epoch_metrics["eval_loss"].append(eval_loss.item())
        epoch_metrics["lr"].append(opt.param_groups[0]["lr"])

        # update lr scheduler
        sched.step()

        # save checkpoint
        save_checkpoint(epoch, model, opt, sched, run_id, checkpoint_path)

        # update progress bar
        msg = f"Train loss: {train_loss.item():.4f}, Eval loss: {eval_loss.item():.4f}"
        pbar.set_description(msg)
        pbar.refresh()

        # log metrics to wandb and to a file
        mean_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        wandb.log(mean_metrics, step=epoch)

        logline = json.dumps({"epoch": epoch, **mean_metrics})
        with open(f"checkpoints/{cfg.baseline_name}_{cfg.seed}.jsonl", "a") as f:
            f.write(logline + "\n")


if __name__ == "__main__":
    main()
