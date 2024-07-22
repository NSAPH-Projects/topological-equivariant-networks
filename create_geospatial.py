import logging
from functools import partial
import hydra
from omegaconf import DictConfig, OmegaConf
from torch_geometric.transforms import Compose

from etnn.geospatial import pm25cc, transforms
import utils

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf/conf_geospatial", config_name="config", version_base=None)
def main(cfg: DictConfig):

    hash = utils.args_to_hash(OmegaConf.to_container(cfg.dataset, resolve=True))

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

    dataset = pm25cc.PM25CC(
        f"data/geospatialcc_{hash}",
        pre_transform=pre_transform,
        force_reload=cfg.force_reload,
    )
    logger.info(
        f"Created GeospatialCC dataset generated and stored in '{dataset.root}'."
    )


if __name__ == "__main__":
    main()
