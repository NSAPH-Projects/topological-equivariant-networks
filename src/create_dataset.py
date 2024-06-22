import hydra
from qm9.qm9_cc import QM9_CC
from omegaconf import DictConfig
#from utils import set_seed

@hydra.main(config_path="../conf/conf_qm9", config_name="config", version_base=None)
def main(cfg: DictConfig):
    #set_seed(args.seed)

    # Lift the QM9 dataset to CombinatorialComplexData format
    qm9_cc = QM9_CC(
        f"data/qm9_cc_{cfg.lifter_name}",
        list(cfg.lifter.lifter_names), 
        list(cfg.lifter.neighbor_types), 
        cfg.lifter.connectivity, 
        list(cfg.lifter.visible_dims), 
        list(cfg.lifter.initial_features), 
        cfg.lifter.dim, 
        cfg.lifter.dis,
        cfg.lifter.merge_neighbors,
    )

    print(f"Lifted QM9 dataset generated and stored in '{qm9_cc.root}'.")

if __name__ == "__main__":
    main()
