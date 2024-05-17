from typing import Literal

import hydra
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from etnn.models import ETNN


class TrainingWrapper(pl.LightningModule):

    def __init__(
        self,
        model: ETNN,
        criterion: Literal["mse", "mae"],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def training_step(self, batch, _):
        # currently only loss at dim 0, but could use all dimensions with masks
        outputs = self.model(batch)
        mask = getattr(batch, f"mask")
        error = (batch.y - outputs[0]) * (1 - mask)

        if self.criterion == "mse":
            loss = error.pow(2).sum() / mask.sum()
        elif self.criterion == "mae":
            loss = error.abs().sum() / mask.sum()

        # log the loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, _):
        # currently only loss at dim 0, but could use all dimensions with masks
        outputs = self.model(batch)
        mask = getattr(batch, f"mask")
        error = (batch.y - outputs[0]) * (1 - mask)

        if self.criterion == "mse":
            loss = error.pow(2).sum() / mask.sum()
        elif self.criterion == "mae":
            loss = error.abs().sum() / mask.sum()

        # log the loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # seed everything
    pl.seed_everything(cfg.seed)

    # load data
    dataset = instantiate(cfg.dataset)
    collate_fn = instantiate(cfg.collate_fn, dataset)
    loader = instantiate(cfg.loader, dataset, collate_fn=collate_fn)

    # determine number of features per rank
    D = next(iter(loader))
    num_feats = {k.split("_")[1]: v.shape[1] for k, v in D.items() if "x_" in k}

    # instantiate all objects from config
    model = instantiate(cfg.model, num_features_per_rank=num_feats)
    optimizer = instantiate(cfg.optimizer, model.parameters())
    lr_scheduler = instantiate(cfg.lr_scheduler, optimizer)

    # make lightning module and save hyper params for logging
    lightning_module = TrainingWrapper(model, cfg.criterion, optimizer, lr_scheduler)
    lightning_module.save_hyperparameters(OmegaConf.resolve(cfg))

    # make logger and trainer
    logger = instantiate(cfg.logger)
    trainer = instantiate(cfg.trainer, logger=logger)

    # trainer
    trainer.fit(lightning_module, train_dataloaders=loader, val_dataloaders=loader)


if __name__ == "__main__":
    main()
