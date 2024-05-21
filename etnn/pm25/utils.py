import numpy as np
import pandas as pd
import torch

from torch_geometric.data import InMemoryDataset
from etnn.combinatorial_complexes import (
    CombinatorialComplexData,
    CombinatorialComplexCollater,
)


class SpatialCC(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["spatial_cc.json", "point_to_cell.csv"]

    @property
    def processed_file_names(self):
        return ["spatial_cc.pt"]

    def process(self):
        # Read data into huge `Data` list.
        path = f"{self.raw_dir}/{self.raw_file_names[0]}"
        data = CombinatorialComplexData.from_json(path)
        data_list = [data]

        # Add road, tract indictor in the data
        tract_indicator = pd.read_csv(f"{self.raw_dir}/{self.raw_file_names[1]}")
        data.index_1 = torch.tensor(tract_indicator.road.values).to(data.pos.device)
        data.index_2 = torch.tensor(tract_indicator.tract.values).to(data.pos.device)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


def standardize_cc(data: CombinatorialComplexData) -> CombinatorialComplexData:
    for key, tensor in data.items():
        if key.startswith("x_"):
            # loop per column
            for i in range(tensor.shape[1]):
                # if dummy variable, skip
                if tensor[:, i].unique().shape[0] == 2:
                    continue
                else:
                    tensor[:, i] = (tensor[:, i] - tensor[:, i].mean()) / tensor[
                        :, i
                    ].std()
        if key.startswith("y"):
            data[key] = (tensor - tensor.mean()) / tensor.std()
        if "pos" == key:
            # normalize to 0-1 range per columns
            data[key] = (tensor - tensor.amin(0)) / (tensor.amax(0) - tensor.amin(0))
    return data


def add_pos_to_cc(data: CombinatorialComplexData) -> CombinatorialComplexData:
    data.x_0 = torch.cat([data.x_0, data.pos], dim=1)
    return data


def squash_cc(
    data: CombinatorialComplexData, soft: bool = False
) -> CombinatorialComplexData:
    x_0 = data.x_0
    for key, tensor in data.items():
        if key.startswith("x_") and key != "x_0":
            # extract i from key
            i = key.split("_")[1]
            x_0 = torch.cat((x_0, tensor[getattr(data, "index_" + i)]), dim=1)
            # remove the original tensor
        if not soft:
            if key.startswith("x_") and key != "x_0":
                delattr(data, key)  # inplace
            elif key.startswith("adj_") and key != "adj_0_0":
                delattr(data, key)
            elif key.startswith("cell_") and key != "cell_0":
                delattr(data, key)
            elif key.startswith("lengths_") and key != "lengths_0":
                delattr(data, key)
    data.x_0 = x_0
    return data


def squash_cc(data: CombinatorialComplexData) -> CombinatorialComplexData:
    x_0 = data.x_0
    for key, tensor in data.items():
        if key.startswith("x_") and key != "x_0":
            # extract i from key
            i = key.split("_")[1]
            x_0 = torch.cat((x_0, tensor[getattr(data, "index_" + i)]), dim=1)
            # remove the original tensor
            delattr(data, key)  # inplace
        elif key.startswith("adj_") and key != "adj_0_0":
            delattr(data, key)
        elif key.startswith("cell_") and key != "cell_0":
            delattr(data, key)
        elif key.startswith("lengths_") and key != "lengths_0":
            delattr(data, key)
    data.x_0 = x_0
    return data


def create_mask(
    data: CombinatorialComplexData, rate: float = 0.1, seed: int | None = None
) -> CombinatorialComplexData:
    cell_2 = data.cell_2
    lengths_2 = data.lengths_2
    cell_ind_2 = torch.split(cell_2, lengths_2.tolist())
    n = len(lengths_2)
    m = int(rate * n)
    rng = np.random.default_rng(seed)
    mask_vals = rng.choice(range(n), m, replace=False)
    to_mask = []
    for i in mask_vals:
        to_mask.extend(cell_ind_2[i].tolist())
    to_mask = np.array(to_mask)
    masked = np.ones(len(data.lengths_0))
    masked[np.array(list(to_mask))] = 0
    data.mask = torch.tensor(masked).float().to(data.pos.device)

    return data


def add_virtual_node(data: CombinatorialComplexData) -> CombinatorialComplexData:
    # add a rank 3 tensor to x_0 with a single one dimension feature vector 0.0
    data.x_3 = torch.tensor([[0.0]]).to(data.pos.device)
    # add the 0-cell with 2 atoms
    data.cell_3 = torch.tensor([0]).to(data.pos.device)
    data.lengths_3 = torch.tensor([1]).to(data.pos.device)

    # connect every two cell
    data.adj_3_2 = torch.tensor([[0, i] for i in range(len(data.lengths_2))]).T.to(
        data.pos.device
    )

    return data


def randomize(data: CombinatorialComplexData, keys=["x_0"]) -> CombinatorialComplexData:
    # permute the x_0
    for key, val in data.items():
        if key in keys:
            perm = torch.randperm(val.shape[0]).to(val.device)
            setattr(data, key, val[perm])
    return data


def x1_labels(data: CombinatorialComplexData) -> CombinatorialComplexData:
    # add a label to x_1
    data.y = data.x_1[data.index_1][:, :1]
    return data


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from etnn.combinatorial_complexes import CombinatorialComplexCollater
    from torch_geometric.transforms import Compose

    # quick test
    dataset = SpatialCC(
        root="data",
        transform=create_mask,
        pre_transform=Compose([standardize_cc, squash_cc]),
        force_reload=True,
    )
    follow_batch = ["cell_0", "cell_1", "cell_2"]
    collate_fn = CombinatorialComplexCollater(dataset, follow_batch=follow_batch)
    loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=1)
    for batch in loader:
        pass
