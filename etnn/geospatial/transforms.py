import numpy as np
import torch

from etnn.combinatorial_data import CombinatorialComplexData


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
            elif key.startswith("slices_") and key != "slices_0":
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
            elif key.startswith("slices_") and key != "slices_0":
                delattr(data, key)
    data.x_0 = x_0
    return data


def create_mask(
    data: CombinatorialComplexData, rate: float = 0.3, seed: int | None = None
) -> CombinatorialComplexData:
    cell_ind_2 = data.cell_list(rank=2)
    n = len(cell_ind_2)
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
    k = len(data.slices_0)
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

    # add the cell 3 representation consisting of 
    data.cell_3 = torch.cat(list(data.cell_0)).to(data.pos.device)
    data.slices_3 = torch.tensor([len(data.cell_3)]).to(data.pos.device)

    # connect every two cell
    # num_cells_2 = len(data.slices_2)
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
