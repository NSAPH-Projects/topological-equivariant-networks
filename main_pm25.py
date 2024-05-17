from typing import Literal

import hydra
import numpy as np
import wandb
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict


from torch import nn
from tqdm.auto import tqdm, trange
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # seed everything
    pl.seed_everything(cfg.seed)

    # init wandb
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.resolve(cfg),
    )

    # load data
    dataset: Dataset = instantiate(cfg.dataset)
    collate_fn: callable = instantiate(cfg.collate_fn, dataset)
    loader: DataLoader = instantiate(cfg.loader, dataset, collate_fn=collate_fn)

    # determine number of features per rank
    D = next(iter(loader))
    num_feats = {k.split("_")[1]: v.shape[1] for k, v in D.items() if "x_" in k}

    # instantiate model, optimizer, scheduler
    model: nn.Module = instantiate(cfg.model, num_features_per_rank=num_feats)
    opt: Optimizer = instantiate(cfg.optimizer, model.parameters())
    sched: LRScheduler = instantiate(cfg.lr_scheduler, opt)
    loss_fn: callable = instantiate(cfg.loss_fn, reduction="none")

    # get device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(dev)
    model.train()

    # get training params from config
    pbar = trange(cfg.trainer.max_epochs, desc="", leave=True)
    for epoch in pbar:
        epoch_metrics = defaultdict(list)
        for batch in loader:
            # training step, use mask for eval
            opt.zero_grad()
            batch = batch.to(dev)
            outputs = model(batch)
            mask = getattr(batch, f"mask")
            loss_terms = loss_fn(outputs["0"], batch.y)
            train_loss = (loss_terms * mask).sum() / mask.sum()
            eval_loss = (loss_terms * (1 - mask)).sum() / (1 - mask).sum()
            train_loss.backward()
            if cfg.trainer.clip is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), cfg.trainer.clip)
            opt.step()
            # end training step

            epoch_metrics["train_loss"].append(train_loss.item())
            epoch_metrics["eval_loss"].append(eval_loss.item())

        # update schedule
        sched.step()

        # update progress bax
        msg = f"Train loss: {train_loss.item():.4f}, Eval loss: {eval_loss.item():.4f}"
        pbar.set_description(msg)
        pbar.refresh()

        # log metrics
        mean_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        wandb.log(mean_metrics, step=epoch)


if __name__ == "__main__":
    main()
