import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import QM9

from combinatorial_data.combinatorial_data_utils import (
    CombinatorialComplexTransform as NewTransform,
)
from combinatorial_data.ranker import get_ranker
from legacy.simplicial_data.rips_lift import rips_lift as old_rips_lift
from qm9.lifts.rips_vietoris_complex import rips_lift
from utils import get_adjacency_types, merge_adjacencies


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

        def fixed_rips_lift(graph: Data) -> set[frozenset[int]]:
            output = rips_lift(graph, dim=dim, dis=dis)
            return {t[0] for t in output}

        ranker = get_ranker(["rips"])
        adjacencies = get_adjacency_types(
            max_dim=dim + 1, connectivity="self_and_next", neighbor_types=["+1"], visible_dims=None
        )
        processed_adjacencies = merge_adjacencies(adjacencies)

        simplicial_transform = NewTransform(
            lifters=fixed_rips_lift,
            ranker=ranker,
            dim=dim + 1,
            adjacencies=adjacencies,
            processed_adjacencies=processed_adjacencies,
            merge_neighbors=True,
        )
        cc = simplicial_transform(graph)

        # Check if x_dict are the same
        for i in range(dim + 1):
            if not (torch.numel(old_x_dict[i]) == 0 and torch.numel(cc[f"x_{i}"]) == 0):
                assert torch.equal(
                    old_x_dict[i], cc[f"x_{i}"]
                ), f"old_x_dict[{i}] != new_x_dict[{i}]"

        # Check if adjs and invs are the same
        for i in range(dim):
            for j in [i, i + 1]:
                sorted_old_adj, idc = sort_tensor_columns_and_get_indices(old_adj[f"{i}_{j}"])
                if not (torch.numel(sorted_old_adj) == 0 and torch.numel(cc[f"adj_{i}_{j}"]) == 0):
                    assert torch.equal(
                        sorted_old_adj, cc[f"adj_{i}_{j}"]
                    ), f"sorted(old_adj[{i}_{j}]) != new_adj[{i}_{j}]"
                    assert torch.equal(
                        old_inv[f"{i}_{j}"][:, idc], cc[f"inv_{i}_{j}"]
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
