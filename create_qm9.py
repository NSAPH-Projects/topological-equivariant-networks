import hydra
from etnn.qm9.qm9cc import QM9CC
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf/conf_qm9", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Lift the QM9 dataset to CombinatorialComplexData format
    dataset = QM9CC(
        f"data/qm9cc_{hash}",
        lifters=list(cfg.dataset.lifters),
        neighbor_types=list(cfg.dataset.neighbor_types),
        connectivity=cfg.dataset.connectivity,
        # cfg.lifter.dim,
        # list(cfg.lifter.initial_features),
        # merge_neighbors=cfg.model.merge_neighbors,
        supercell=cfg.dataset.supercell,
        force_reload=False,
    )
    logger.info(f"Lifted QM9 dataset generated and stored in '{dataset.root}'.")


if __name__ == "__main__":
    main()
