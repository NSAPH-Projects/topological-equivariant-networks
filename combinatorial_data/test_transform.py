"""
Verifies consistency of old vs. new Rips lift transformations on graphs.

Utilizes the QM9 dataset for graph inputs and compares outputs from legacy and
current versions of the Rips lift.
"""

import functools

import pytest
import torch
from torch_geometric.datasets import QM9

from combinatorial_data.lifts import rips_lift as new_rips_lift
from combinatorial_data.ranker import get_ranker
from combinatorial_data.utils import CombinatorialComplexTransform as NewTransform
from legacy.simplicial_data.rips_lift import rips_lift as old_rips_lift


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("dis", [0.5, 1.5, 2.5, 4.0])
def test_rips_transform(dim: int, dis: float):
    """
    Test consistency of simplicial transformations between old and new Rips lift implementations.

    Parameters
    ----------
    dim : int
        Dimensionality of the simplicial complex.
    dis : float
        Maximum edge length for simplices in the Rips complex.

    Assertions
    ----------
    Asserts equivalence of simplicial complexes from both old and new implementations.
    """
    n_test_samples = 3
    data_root = "./datasets/QM9_test"
    dataset = QM9(root=data_root)
    dataset = dataset.shuffle()
    for graph in dataset[:n_test_samples]:
        old_x_dict, old_adj, old_inv = old_rips_lift(graph, dim=dim, dis=dis)

        fixed_rips_lift = functools.partial(new_rips_lift, dim=dim, dis=dis)
        ranker = get_ranker("rips")
        simplicial_transform = NewTransform(fixed_rips_lift, ranker, dim)
        new_x_dict, _, new_adj, new_inv = simplicial_transform.get_relevant_dicts(graph)

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


def sort_tensor_columns_and_get_indices(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sort a 2D tensor by its second, then first row, returning sorted tensor and indices.

    Parameters
    ----------
    tensor : torch.Tensor
        A 2D tensor to sort.

    Returns
    -------
    torch.Tensor
        The tensor sorted by columns.
    torch.Tensor
        Indices reflecting the final column sorting order.
    """
    # Sort by the second row (preserve original indices)
    _, indices_second_row = torch.sort(tensor[1])
    sorted_tensor = tensor[:, indices_second_row]

    # Then sort by the first row, in a stable manner
    _, indices_first_row = torch.sort(sorted_tensor[0], stable=True)
    final_sorted_tensor = sorted_tensor[:, indices_first_row]

    # Combine the indices to reflect the final sorting order
    final_indices = indices_second_row[indices_first_row]

    return final_sorted_tensor, final_indices
