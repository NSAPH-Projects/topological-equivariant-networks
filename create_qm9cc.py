import hydra
from src.qm9.qm9cc import QM9CC
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="..conf/conf_qm9", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Lift the QM9 dataset to CombinatorialComplexData format
    dataset = QM9CC(
        f"data/qm9cc_{cfg.lifter_name}",
        list(cfg.lifter.lifter_names),
        list(cfg.lifter.neighbor_types),
        cfg.lifter.connectivity,
        list(cfg.lifter.visible_dims),
        list(cfg.lifter.initial_features),
        cfg.lifter.dim,
        cfg.lifter.dis,
        cfg.lifter.merge_neighbors,
    )

    logger.info(f"Lifted QM9 dataset generated and stored in '{dataset.root}'.")


if __name__ == "__main__":
    main()
