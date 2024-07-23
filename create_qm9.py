import logging

import hydra
from omegaconf import DictConfig

from etnn.qm9.qm9cc import QM9CC

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf/conf_qm9", config_name="config", version_base=None)
def main(cfg: DictConfig):
    dataset = QM9CC(
        f"data/qm9cc_{cfg.dataset_name}",
        lifters=list(cfg.dataset.lifters),
        neighbor_types=list(cfg.dataset.neighbor_types),
        connectivity=cfg.dataset.connectivity,
        supercell=cfg.dataset.supercell,
        force_reload=False,
    )
    logger.info(f"Lifted QM9 dataset generated and stored in '{dataset.root}'.")


if __name__ == "__main__":
    main()
