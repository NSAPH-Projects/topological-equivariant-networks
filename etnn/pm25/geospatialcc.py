import json
import numpy as np
import pandas as pd
import requests
import torch
from etnn.combinatorial_data import CombinatorialComplexData
from torch_geometric.data import InMemoryDataset

class GeospatialCC(InMemoryDataset):
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
        return ["geospatialcc.json"]

    @property
    def processed_file_names(self):
        return [f"geospatialcc.pt"]

    def download(self):
        url = "https://raw.githubusercontent.com/NSAPH-Projects/topological-equivariant-networks/save_pt/data/input/geospatialcc.json"
        path = f"{self.raw_dir}/{self.raw_file_names[0]}"
        response = requests.get(url)
        with open(path, "wb") as f:
            f.write(response.content)

    def process(self):
        # Read data into huge `Data` list.
        path = f"{self.raw_dir}/{self.raw_file_names[0]}"
        with open(path, "r") as f:
            ccdict = json.load(f)
        data = CombinatorialComplexData.from_ccdict(ccdict)
        data_list = [data]

        # Add road, tract indictor in the data
        node_cell_indicator = pd.DataFrame(ccdict["points_to_cells"])
        data.index_1 = node_cell_indicator.road
        data.index_2 = node_cell_indicator.tract

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


def squash_cc(
    data: CombinatorialComplexData, soft: bool = False
) -> CombinatorialComplexData:
    x_0 = data.x_0
    for key, tensor in data.items():
        if key.startswith("x_") and key != "x_0":
            # extract i from key
            i = key.split("_")[1]
            # x_0 = torch.cat((x_0, tensor[getattr(data, "index_" + i)]), dim=1)
            index_i = getattr(data, "index_" + i)
            agg_feats = torch.stack([tensor[ix].mean(0) for ix in index_i])
            x_0 = torch.cat((x_0, agg_feats), dim=1)
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


def create_mask(
    data: CombinatorialComplexData, rate: float = 0.3, seed: int | None = None
) -> CombinatorialComplexData:
    cell_2 = data.cell_2
    lengths_2 = data.lengths_2
    cell_ind_2 = torch.split(cell_2, lengths_2.tolist())
    n = len(lengths_2)
    m = int(rate * n)
    rng = np.random.default_rng(seed)
    train_mask_cells = rng.choice(range(n), m, replace=False)
    remaining_cells = list(set(range(n)) - set(train_mask_cells))
    remaining_cells = np.random.permutation(remaining_cells)
    val_mask_cells = remaining_cells[: ((n - m) // 2)]
    test_mask_cells = remaining_cells[((n - m) // 2) :]

    # create the mask
    test = []
    val = []
    train = []
    for i in range(n):
        if i in train_mask_cells:
            train.extend(cell_ind_2[i].tolist())
        elif i in test_mask_cells:
            test.extend(cell_ind_2[i].tolist())
        elif i in val_mask_cells:
            val.extend(cell_ind_2[i].tolist())

    # create the mask
    k = len(data.lengths_0)
    dev = data.pos.device
    data.training_mask = torch.zeros(k, dtype=torch.bool, device=dev)
    data.validation_mask = torch.zeros(k, dtype=torch.bool, device=dev)
    data.test_mask = torch.zeros(k, dtype=torch.bool, device=dev)

    data.training_mask[train] = 1
    data.validation_mask[val] = 1
    data.test_mask[test] = 1

    return data


def add_virtual_node(data: CombinatorialComplexData) -> CombinatorialComplexData:
    # add a rank 3 tensor to x_0 with a single one dimension feature vector 0.0
    data.x_3 = torch.tensor([[0.0]]).to(data.pos.device)
    # add the 0-cell with 2 atoms
    # data.cell_3 = torch.tensor([0]).to(data.pos.device)
    data.cell_3 = torch.cat(list(data.cell_0)).to(data.pos.device)

    # data.lengths_3 = torch.tensor([1]).to(data.pos.device)

    # connect every two cell
    # num_cells_2 = len(data.lengths_2)
    num_cells_2 = len(data.cell_2)
    adj_3_2 = torch.tensor([[0, i] for i in range(num_cells_2)])
    data.adj_3_2 = adj_3_2.T.to(data.pos.device)
    data.adj_2_3 = data.adj_3_2.flip(0)    

    return data


def randomize(data: CombinatorialComplexData, keys=["x_0"]) -> CombinatorialComplexData:
    # permute the x_0
    for key, val in data.items():
        if key in keys:
            new_val = torch.randn(val.shape).to(val.device) * 0.001
            setattr(data, key, new_val)
            # perm = torch.randperm(val.shape[0]).to(val.device)
            # setattr(data, key, val[perm])
    return data


def x1_labels(data: CombinatorialComplexData) -> CombinatorialComplexData:
    # add a label to x_1
    data.y = data.x_1[data.index_1][:, :1]
    return data


# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     from etnn_spatial.combinatorial_complexes import CombinatorialComplexCollater
#     from torch_geometric.transforms import Compose

#     # quick test
#     dataset = GeospatialCC(
#         root="data",
#         transform=create_mask,
#         pre_transform=Compose([standardize_cc, squash_cc]),
#         force_reload=True,
#     )
#     follow_batch = ["cell_0", "cell_1", "cell_2"]
#     collate_fn = CombinatorialComplexCollater(dataset, follow_batch=follow_batch)
#     loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=1)
#     for batch in loader:
#         pass
