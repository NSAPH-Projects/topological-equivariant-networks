import functools
import warnings

import pytest
import torch
from torch_geometric.datasets import QM9

from legacy.simplicial_data.rips_lift import rips_lift as old_rips_lift
from qm9.utils import get_subsampler
from simplicial_data.lifts import rips_lift as new_rips_lift
from simplicial_data.utils import SimplicialTransform as NewTransform


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("dis", [0.5, 1.5, 2.5, 4.0])
def test_transform(dim: int, dis: float):
    n_test_samples = 20
    data_root = f"./datasets/QM9"
    dataset = QM9(root=data_root)
    dataset = dataset.shuffle()
    for graph in dataset[:n_test_samples]:
        old_x_dict, old_adj, old_inv = old_rips_lift(graph, dim=dim, dis=dis)

        fixed_rips_lift = functools.partial(new_rips_lift, dim=dim, dis=dis)
        simplicial_transform = NewTransform(fixed_rips_lift, dim)
        new_x_dict, new_adj, new_inv = simplicial_transform.get_relevant_dicts(graph)

        # Check if x_dict are the same
        for i in range(dim + 1):
            assert torch.equal(old_x_dict[i], new_x_dict[i]), f"old_x_dict[{i}] != new_x_dict[{i}]"

        # Check if adjs and invs are the same
        for i in range(dim):
            for j in [i, i + 1]:
                sorted_old_adj, idc = sort_tensor_columns_and_get_indices(old_adj[f"{i}_{j}"])
                assert torch.equal(
                    sorted_old_adj, new_adj[f"{i}_{j}"]
                ), f"sorted(old_adj[{i}_{j}]) != new_adj[{i}_{j}]"
                assert torch.equal(
                    old_inv[f"{i}_{j}"][:, idc], new_inv[f"{i}_{j}"]
                ), f"sorted(old_inv[{i}_{j}]) != new_inv[{i}_{j}]"


def sort_tensor_columns_and_get_indices(tensor):
    # Sort by the second row (preserve original indices)
    _, indices_second_row = torch.sort(tensor[1])
    sorted_tensor = tensor[:, indices_second_row]

    # Then sort by the first row, in a stable manner
    _, indices_first_row = torch.sort(sorted_tensor[0], stable=True)
    final_sorted_tensor = sorted_tensor[:, indices_first_row]

    # Combine the indices to reflect the final sorting order
    final_indices = indices_second_row[indices_first_row]

    return final_sorted_tensor, final_indices
