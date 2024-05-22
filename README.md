# topological-equivariant-networks
Topological Equivariant Networks (TEN)

## Setup instructions
```
git clone git@github.com:NSAPH-Projects/topological-equivariant-networks.git
cd topological-equivariant-networks
chmod +x env_builder.sh
source env_builder.sh
```
This will create and activate a new conda environment called "ten" that contains all the packages necessary to run the code in this repository.

To reproduce EGNN within the ETNN framework, run the following command:

```
TARGET_NAME=alpha LR=5e-4 source train_egnn_like.sh
```

To reproduce other target, set the corresponding `TARGET_NAME`.

The scripts to reproduce the reported QM9 experiments can be found under `scripts/experiments`. The experiments are named in the order of their appereance in the results table. For example, to reproduce the first row, run the following:

```
source scripts/experiments/1.sh 0 # creates the preprocessed dataset
source scripts/experiments/1.sh 1 # reproduces the results for the first 6 targets
source scripts/experiments/1.sh 2 # reproduces the results for the last 6 targets
```

The code to reproduce the spatial task is in the anonymized branch `https://anonymous.4open.science/r/topological-equivariant-networks-625B`. The code to reproduce the synthetic tasks is in the anonymized branch `https://anonymous.4open.science/r/topological-equivariant-networks-06C9`. We plan to shortly merge these together under a single branch.
