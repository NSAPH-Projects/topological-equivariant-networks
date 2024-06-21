import hydra
from qm9.utils import process_qm9_dataset
from utils import set_seed

@hydra.main(config_path="conf_qm9", config_name="config", version_base=None)
def main(cfg: DictConfig):
    #set_seed(args.seed)
    process_qm9_dataset(
        cfg.lifter.lifter_names, 
        cfg.lifter.neighbor_types, 
        cfg.lifter.connectivity, 
        cfg.lifter.visible_dims,  
        cfg.lifter.initial_features, 
        cfg.lifter.dim, 
        cfg.lifter.dis,
        cfg.lifter.merge_neighbors,
    )

if __name__ == "__main__":
    main()
