"""
This script saves the QM9 dataset splits as a dictionary of PyTorch tensors and the split indices as
a pickle file.

Outputs:
- `datasets.pth` (dict[torch.Tensor]): A dictionary of torch.Tensor objects, where each tensor is
the data field of some ProcessedDataset object. These ProcessedDataset objects are used in EGNN and
each of them corresponds to a different data split.
- `egnn_splits.pkl` (dict[list[int]]): A dictionary of python lists of integers, where each list
contains the PyTorch-Geometric-QM9 indices of an EGNN split.

This script is intended to run in the EGNN repository and appears here for completeness.
"""

import pickle

import torch

from qm9.args import init_argparse
from qm9.data.utils import initialize_datasets

# Initialize dataloader
args = init_argparse("qm9")
args, datasets, num_species, charge_scale = initialize_datasets(
    args, args.datadir, "qm9", subtract_thermo=args.subtract_thermo, force_download=True
)
# Save EGNN datasets
datas = {k: v.data for k, v in datasets.items()}
torch.save(datas, "datasets.pth")

# Retrieve EGNN split indices
split_indices = {k: v["index"].tolist() for k, v in datas.items()}

# Map the indices from (0, 132K) (raw QM9 indices) to (0, 130K) (PyTorch-Geometric QM9 indices)
all_idc = []
for idc in split_indices.values():
    all_idc += idc
all_idc = sorted(all_idc)
for split, idc in split_indices.items():
    split_indices[split] = [all_idc.index(i) for i in idc]

# Save the splits
with open("egnn_splits.pkl", "wb") as f:
    pickle.dump(split_indices, f)
