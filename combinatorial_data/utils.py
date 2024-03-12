import re
from typing import Union

import numpy as np
import torch
from scipy.sparse import csc_matrix
from toponetx.classes import CombinatorialComplex
from torch_geometric.data import Batch, Data
from torch_geometric.loader.dataloader import Collater
from torch_geometric.transforms import BaseTransform


class CustomCollater(Collater):

    def __call__(self, batch: list[Data]) -> Batch:
        """
        Perform custom collation by pre-collating to pad tensors and using the superclass collation
        logic.

        Parameters
        ----------
        batch : list[Data]
            A list of Data objects to be collated.

        Returns
        -------
        Batch
            The collated list of Data objects, after pre-collation.
        """
        # Apply pre-collate logic
        batch = self.precollate(batch)

        # Use the original collation logic
        collated_batch = super().__call__(batch)

        return collated_batch

    def precollate(self, batch: list[Data]) -> list[Data]:
        """
        Pre-collate logic to pad tensors within each sample based on naming patterns.

        In particular, tensors that are named f'x_{i}' for some integer i are padded to have the
        same number of columns, and tensors that are named f'inv_{i}_{j}' for some integers i, j are
        padded to have the same number of rows. The variable number of columns/rows can arise if
        cells with the same rank may consist of a variable number of nodes. The padding value is -1
        which results in lookups to the last row of the feature matrix, which is overridden in
        postcollate() to be all zeros.

        Parameters
        ----------
        batch : list[Data]
            A list of Data objects to be pre-collated.

        Returns
        -------
        list[Data]
            The batch with tensors padded according to their naming patterns.
        """
        max_cols = {}
        max_rows = {}

        for data in batch:
            for attr_name in data.keys():
                if re.match(r"x_\d+", attr_name):
                    tensor = getattr(data, attr_name)
                    max_cols[attr_name] = max(max_cols.get(attr_name, 0), tensor.size(1))
                elif re.match(r"inv_\d+_\d+", attr_name):
                    tensor = getattr(data, attr_name)
                    max_rows[attr_name] = max(max_rows.get(attr_name, 0), tensor.size(0))

        for data in batch:
            for attr_name in data.keys():
                if attr_name in max_cols:
                    tensor = getattr(data, attr_name)
                    setattr(data, attr_name, pad_tensor(tensor, max_cols[attr_name], 1))
                elif attr_name in max_rows:
                    tensor = getattr(data, attr_name)
                    setattr(data, attr_name, pad_tensor(tensor, max_rows[attr_name], 0))

        return batch


class CombinatorialComplexData(Data):
    """
    Abstract combinatorial complex class that generalises the pytorch geometric graph (Data).
    Adjacency tensors are stacked in the same fashion as the standard edge_index.
    """

    def __inc__(self, key: str, value: any, *args, **kwargs) -> any:
        num_nodes = getattr(self, "x_0").size(0)
        if "adj" in key:
            i, j = key[4], key[6]
            return torch.tensor(
                [[getattr(self, f"x_{i}").size(0)], [getattr(self, f"x_{j}").size(0)]]
            )
        elif "inv" in key:
            return num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: any, *args, **kwargs) -> any:
        if "adj" in key or "inv" in key:
            return 1
        else:
            return 0


class CombinatorialComplexTransform(BaseTransform):
    """Todo: add
    The adjacency types (adj) are saved as properties, e.g. object.adj_1_2 gives the edge index from
    1-complexes to 2-complexes."""

    def __init__(
        self,
        lifters: Union[list[callable], callable],
        ranker: callable,
        dim: int,
        adjacencies: list[str],
    ):
        if isinstance(lifters, list):
            self.lifters = lifters
        else:
            self.lifters = [lifters]
        self.rank = ranker
        self.dim = dim
        self.adjacencies = adjacencies

    def __call__(self, graph: Data) -> CombinatorialComplexData:
        if not torch.is_tensor(graph.x):
            graph.x = torch.nn.functional.one_hot(graph.z, torch.max(graph.z) + 1)

        assert torch.is_tensor(graph.pos) and torch.is_tensor(graph.x)

        # get relevant dictionaries using the Rips complex based on the geometric graph/point cloud
        x_dict, mem_dict, adj_dict, inv_dict = self.get_relevant_dicts(graph)

        com_com_data = CombinatorialComplexData()
        com_com_data = com_com_data.from_dict(graph.to_dict())

        for k, v in x_dict.items():
            com_com_data[f"x_{k}"] = v

        for k, v in mem_dict.items():
            com_com_data[f"mem_{k}"] = v

        for k, v in adj_dict.items():
            com_com_data[f"adj_{k}"] = v

        for k, v in inv_dict.items():
            com_com_data[f"inv_{k}"] = v

        for att in ["edge_attr", "edge_index"]:
            if hasattr(com_com_data, att):
                com_com_data.pop(att)

        return com_com_data

    def get_relevant_dicts(self, graph):
        # compute cells
        cells = self.lift(graph)

        # compute ranks for each cell
        cell_dict = {rank: {} for rank in range(self.dim + 1)}
        for cell, memberships in cells.items():
            cell_rank = self.rank(cell, memberships)
            if cell_rank <= self.dim:
                cell_dict[cell_rank][cell] = memberships

        # create x_dict
        x_dict, mem_dict = map_to_tensors(cell_dict, len(self.lifters))

        # create the combinatorial complex
        # first add higher-order cells
        cc = CombinatorialComplex()
        for rank, cells in cell_dict.items():
            if rank > 0:
                cc.add_cells_from(cells, ranks=rank)

        # then remove the artificially created 0-rank cells
        zero_rank_cells = [cell for cell in cc.cells if len(cell) == 1]
        cc.remove_cells(zero_rank_cells)

        # finally add the organic 0-rank cells
        cc.add_cells_from(cell_dict[0], ranks=0)

        # compute adjancencies and incidences
        adj, idx_to_cell = dict(), dict()
        for adj_type in self.adjacencies:
            i, j = [int(rank) for rank in adj_type.split("_")]
            if i != j:
                if i < j:
                    matrix = cc.incidence_matrix(rank=i, to_rank=j)
                else:
                    matrix = cc.incidence_matrix(rank=j, to_rank=i).T

            else:
                index, matrix = cc.adjacency_matrix(rank=i, via_rank=i + 1, index=True)
                idx_to_cell[i] = {v: sorted(k) for k, v in index.items()}

            if np.array_equal(matrix, np.zeros(1)):
                matrix = torch.zeros(2, 0).long()
            else:
                matrix = sparse_to_dense(matrix)

            adj[adj_type] = matrix

        # for each adjacency/incidence, store the nodes to be used for computing geometric features
        inv = dict()
        for adj_type in self.adjacencies:
            inv[adj_type] = []
            neighbors = adj[adj_type]
            i, j = [int(rank) for rank in adj_type.split("_")]

            for connection in neighbors.t():
                idx_a, idx_b = connection[0], connection[1]
                cell_a = idx_to_cell[i][idx_a.item()]
                cell_b = idx_to_cell[j][idx_b.item()]
                shared = [node for node in cell_a if node in cell_b]
                only_in_a = [node for node in cell_a if node not in shared]
                only_in_b = [node for node in cell_b if node not in shared]
                inv_nodes = shared + only_in_b + only_in_a
                inv[adj_type].append(inv_nodes)

        for k, v in inv.items():
            if len(v) == 0:
                inv[k] = torch.zeros(0, 0, dtype=torch.float32)
            else:
                inv[k] = torch.tensor(pad_lists_to_same_length(v), dtype=torch.float32).t()

        return x_dict, mem_dict, adj, inv

    def lift(self, graph: Data) -> dict[frozenset[int], list[bool]]:
        """
        Apply lifters to a data point, process their outputs, and track contributions.

        This method applies each registered lifter to the given graph, processes the output to
        ensure uniqueness, and tracks which lifters contributed to each resulting cell. A cell is
        considered contributed by a lifter if it appears in the lifter's processed output.

        Parameters
        ----------
        graph : Data
            The input graph to which the lifters are applied. It should be an instance of a Data
            class or similar structure containing graph data.

        Returns
        -------
        dict[frozenset[int], list[bool]]
            A dictionary mapping each unique cell (a list of integers representing a cell) to a list
            of booleans. Each boolean value indicates whether the corresponding lifter (by index)
            contributed to the creation of that cell. The length of the boolean list for each cell
            equals the total number of lifters, with True indicating contribution and False
            indicating no contribution. The keys of the dictionary are ordered according to the
            sorting order they would have if they were cast to lists and internally sorted.

        Examples
        --------
        Assuming `self.lifters` contains two lifters and the graph is such that both lifters
        contribute to the cell [1, 2], and only the first lifter contributes to the cell [2, 3], the
        method might return: {
            [1, 2]: [True, True], [2, 3]: [True, False]
        }

        Notes
        -----
        The keys of the dictionary are reordered according to the sorting order they would have if
        they were cast to lists and internally sorted. This sorting must happen here as the keys of
        this dictionary determine the ordering of rows/columns in all of x_dict, mem_dict, adj and
        inv in get_relevant_dicts(), which is the only caller of this function.
        """
        cell_lifter_map = {}
        for lifter_idx, lifter in enumerate(self.lifters):
            lifter_output = lifter(graph)
            for cell in lifter_output:
                if cell not in cell_lifter_map:
                    cell_lifter_map[cell] = [False] * len(self.lifters)
                cell_lifter_map[cell][lifter_idx] = True

        # Reorder the dictionary keys numerically
        sorted_cells = sorted(sorted(list(cell)) for cell in cell_lifter_map.keys())
        cell_lifter_map = {
            frozenset(cell): cell_lifter_map[frozenset(cell)] for cell in sorted_cells
        }

        return cell_lifter_map

    def extended_adjacency_matrix(self, cc, rank):
        """
        Compute the extended adjacency matrix for a given combinatorial complex and rank.

        Parameters
        ----------
        cc : CombinatorialComplex
            The combinatorial complex.
        rank : int
            The rank of the complex.

        Returns
        -------
        tuple[list[frozenset], np.ndarray]
            A tuple containing the index and the extended adjacency matrix.

        Notes
        -----
        - The extended adjacency matrix represents the connectivity between cells of a combinatorial complex.
        - The matrix is computed based on the neighbor type specified in the class.
        - The index represents the cells of the complex.
        """
        index = cc.skeleton(rank=rank)
        num_cells = len(index)
        matrix = csc_matrix((num_cells, num_cells), dtype=int)
        if self.neighbor_type == "adjacency":
            via_ranks = [rank + 1]
        elif self.neighbor_type == "any_adjacency":
            via_ranks = list(range(rank + 1, self.dim + 1))
        elif self.neighbor_type == "coadjacency":
            via_ranks = [rank - 1]
        elif self.neighbor_type == "any_coadjacency":
            via_ranks = list(range(rank - 1, -1, -1))
        elif self.neighbor_type == "direct":
            via_ranks = [rank - 1, rank + 1]
        elif self.neighbor_type == "all":
            via_ranks = list(range(self.dim + 1))
        via_ranks = [r for r in via_ranks if r >= 0 and r <= self.dim and r != rank]
        for r in via_ranks:
            kwargs = dict(rank=rank, via_rank=r, index=False)
            if r < rank:
                matrix += cc.coadjacency_matrix(**kwargs)
            else:
                matrix += cc.adjacency_matrix(**kwargs)
        return index, matrix


def map_to_tensors(
    input_dict: dict[int, dict[frozenset[int], list[bool]]], num_lifters: int
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    x_dict, mem_dict = {}, {}
    for rank, cell_lifter_map in input_dict.items():
        if cell_lifter_map:
            sorted_cells = [sorted(cell) for cell in cell_lifter_map.keys()]
            padded_cells = pad_lists_to_same_length(sorted_cells)
            x = torch.tensor(padded_cells, dtype=torch.float32)
            mem = torch.tensor(list(cell_lifter_map.values()), dtype=torch.bool)
        else:
            # For empty lists, create empty tensors
            x = torch.empty((0, 0), dtype=torch.float32)
            mem = torch.empty((0, num_lifters), dtype=torch.bool)
        x_dict[rank] = x
        mem_dict[rank] = mem
    return x_dict, mem_dict


def pad_lists_to_same_length(
    list_of_lists: list[list[int]], pad_value: float = torch.nan
) -> list[list[int]]:
    """
    Pad a list of lists of integers to the same length with a specified value.

    Parameters
    ----------
    list_of_lists : list[list[int]]
        List of lists of integers to be padded.
    pad_value : float, optional
        Value to use for padding, default is torch.nan.

    Returns
    -------
    list[list[int]]
        List of lists of integers padded to the same length.
    """
    # Find the maximum length among all lists
    max_length = max(len(inner_list) for inner_list in list_of_lists)

    # Pad each list to match the maximum length
    padded_list = [
        inner_list + [pad_value] * (max_length - len(inner_list)) for inner_list in list_of_lists
    ]

    return padded_list


def pad_tensor(
    tensor: torch.Tensor, max_size: int, dim: int, pad_value: float = torch.nan
) -> torch.Tensor:
    """
    Pad a 2D tensor to the specified size along a given dimension with a pad value.

    Empty tensors are handled as a special case.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to be padded.
    max_size : int
        The maximum size to pad the tensor to.
    dim : int
        The dimension along which to pad.
    pad_value : float, optional
        The value to use for padding, by default torch.nan.

    Returns
    -------
    torch.Tensor
        The padded tensor.

    Raises
    ------
    ValueError
        If the input tensor is not 2D.
    """

    # Assert that the input tensor is 2D
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D.")

    # Check if the tensor is empty
    if torch.numel(tensor) == 0:
        # Return an empty tensor of shape (max_size, 0) or (0, max_size)
        # depending on the padding dimension
        if dim == 0:
            return torch.empty((max_size, 0), dtype=tensor.dtype)
        else:  # dim == 1
            return torch.empty((0, max_size), dtype=tensor.dtype)

    pad_sizes = [0] * (2 * len(tensor.size()))  # Pad size must be even, for each dim.
    pad_sizes[(1 - dim) * 2 + 1] = max_size - tensor.size(dim)  # Only pad at the end
    return torch.nn.functional.pad(tensor, pad=pad_sizes, value=pad_value)


def sparse_to_dense(sparse_matrix):
    # Extract row and column indices of non-zero elements
    rows, cols = sparse_matrix.nonzero()

    # Convert to a 2D NumPy array
    dense_array = np.array([rows, cols])

    # Convert the NumPy array to a PyTorch tensor
    return torch.from_numpy(dense_array).type(torch.int64)


if __name__ == "__main__":
    import functools
    import random

    from torch_geometric.datasets import QM9

    from combinatorial_data.lifts import rips_lift

    data = QM9("../datasets/QM9")
    dim = 2
    transform_2 = CombinatorialComplexTransform(
        functools.partial(rips_lift, dim=dim, dis=2.0), dim=dim
    )
    transform_3 = CombinatorialComplexTransform(
        functools.partial(rips_lift, dim=dim, dis=3.0), dim=dim
    )
    transform_4 = CombinatorialComplexTransform(
        functools.partial(rips_lift, dim=dim, dis=4.0), dim=dim
    )

    random_graph = random.randint(0, len(data) - 1)

    # transform data[0] to combinatorial complex
    com_2 = transform_2(data[random_graph])
    com_3 = transform_3(data[random_graph])
    com_4 = transform_4(data[random_graph])

    print(f"Original Graph: {data[random_graph]}")
    print(f"Combinatorial Complex (delta = 2.0): {com_2}")
    print(f"Combinatorial Complex (delta = 3.0): {com_3}")
    print(f"Combinatorial Complex (delta = 4.0): {com_4}")
