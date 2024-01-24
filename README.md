# topological-equivariant-networks
Topological Equivariant Networks (TEN)

## Setup instructions
```
git clone git@github.com:NSAPH-Projects/topological-equivariant-networks.git
cd topological-equivariant-networks
chmod +x env_builder.sh
source env_builder.sh
```
This will create and activate a new conda environment called "ten" that contains all the packages necessary to run the code in this repository. To check if things work correctly, try running the following commands from the root directory:
```
pytest
python main_qm9.py --dim 3 --dis 3 --num_layers 4 --target_name alpha --epochs 5
```

