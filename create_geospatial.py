import logging
from functools import partial
import hydra
from omegaconf import DictConfig
from torch_geometric.transforms import Compose

from src.pm25 import geospatialcc as pm25cc

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf/conf_geospatial", config_name="config", version_base=None)
def main(cfg: DictConfig):

    pre_transform = []
    if cfg.dataset.standardize:
        pre_transform.append(pm25cc.standardize_cc)
    if cfg.dataset.randomize_x0:
        pre_transform.append(partial(pm25cc.randomize, keys=["x_0"]))
    if cfg.dataset.virtual_node:
        pre_transform.append(pm25cc.add_virtual_node)
    if cfg.dataset.squash:
        pre_transform.append(pm25cc.squash_cc)
    if cfg.dataset.add_positions:
        pre_transform.append(pm25cc.add_pos_to_cc)
    pre_transform = Compose(pre_transform)

    dataset = pm25cc.GeospatialCC(
        f"data/geospatialcc_{cfg.dataset_name}",
        pre_transform=pre_transform,
        force_reload=cfg.force_reload,
    )
    logger.info(
        f"Lifted GeospatialCC dataset generated and stored in '{dataset.root}'."
    )


if __name__ == "__main__":
    main()
