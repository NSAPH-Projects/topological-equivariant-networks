import re
from collections.abc import Iterable
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


class LegacyCombinatorialComplexData(Data):
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


class CombinatorialComplexData(Data):
    """
    A subclass of PyTorch Geometric's Data class designed for representing combinatorial complexes.

    This class extends standard graph representation to handle higher-dimensional structures found
    in complex networks. It supports adjacency tensors for cells of different ranks (vertices,
    edges, faces, etc.) and their relationships, following a specific naming convention for dynamic
    attribute recognition and processing.

    Attributes
    ----------
    x_i : torch.FloatTensor
        Node indices associated with cells of rank i, where i is a non-negative integer.
    mem_i : torch.BoolTensor
        Lifters associated with cells of rank i, where i is a non-negative integer.
    adj_i_j : torch.LongTensor
        Adjacency tensors representing the relationships (edges) between cells of rank i and j,
        where i and j are non-negative integers.
    inv_i_j : torch.FloatTensor
        Node indices that can be used to compute legacy geometric features for each cell pair.
    """

    def __inc__(self, key: str, value: any, *args, **kwargs) -> any:
        """
        Specify how to increment indices for batch processing of data, based on the attribute key.

        Parameters
        ----------
        key : str
            The attribute name to be incremented.
        value : Any
            The value associated with the attribute `key`.
        *args : Additional positional arguments. **kwargs : Additional keyword arguments.

        Returns
        -------
        any
            The increment value for the attribute `key`. Returns a tensor for `adj_i_j` attributes,
            the number of nodes for `inv_i_j` and `x_i` attributes, or calls the superclass's
            `__inc__` method for other attributes.
        """
        num_nodes = getattr(self, "x_0").size(0)
        # The adj_i_j attribute holds cell indices, increment each dim by the number of cells of
        # corresponding rank
        if re.match(r"adj_(\d+_\d+|\d+_\d+_\d+)", key):
            i, j = key.split("_")[1:3]
            return torch.tensor(
                [[getattr(self, f"x_{i}").size(0)], [getattr(self, f"x_{j}").size(0)]]
            )
        # The inv_i_j and x_i attributes hold node indices, they should be incremented
        elif re.match(r"inv_(\d+_\d+|\d+_\d+_\d+)", key) or re.match(r"x_\d+", key):
            return num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: any, *args, **kwargs) -> int:
        """
        Specify the dimension over which to concatenate tensors for batch processing, based on the
        attribute key.

        Parameters
        ----------
        key : str
            The attribute name for which the concatenation dimension is specified.
        value : Any
            The value associated with the attribute `key`.
        *args : Additional positional arguments. **kwargs : Additional keyword arguments.

        Returns
        -------
        int
            The dimension over which to concatenate the attribute `key`. Returns 1 for `adj_i_j` and
            `inv_i_j` attributes, and 0 otherwise.
        """
        if re.match(r"(adj|inv)_\d+_\d+", key):
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
        processed_adjacencies: list[str],
        merge_neighbors: bool,
        enable_indexing_bug: bool = False,
    ):
        if isinstance(lifters, list):
            self.lifters = lifters
        else:
            self.lifters = [lifters]
        self.rank = ranker
        self.dim = dim
        self.adjacencies = adjacencies
        self.processed_adjacencies = processed_adjacencies
        self.merge_neighbors = merge_neighbors
        self.enable_indexing_bug = enable_indexing_bug

    def __call__(self, graph: Data) -> CombinatorialComplexData:
        if not torch.is_tensor(graph.x):
            graph.x = torch.nn.functional.one_hot(graph.z, torch.max(graph.z) + 1)

        assert torch.is_tensor(graph.pos) and torch.is_tensor(graph.x)

        # get relevant dictionaries using the Rips complex based on the geometric graph/point cloud
        x_dict, mem_dict, adj_dict, inv_dict = self.get_relevant_dicts(graph)

        if self.enable_indexing_bug:
            com_com_data = LegacyCombinatorialComplexData()
        else:
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
        cc = create_combinatorial_complex(cell_dict)

        # compute adjancencies and incidences
        adj = dict()
        for adj_type in self.adjacencies:
            ranks = [int(rank) for rank in adj_type.split("_")]
            i, j = ranks[:2]
            if i != j:
                matrix = incidence_matrix(cc, i, j)
            else:
                # if i == j, we must have a third rank specifying via_rank
                assert len(ranks) == 3
                matrix = adjacency_matrix(cc, i, ranks[2])

            adj[adj_type] = matrix

        # merge matching adjacencies
        if self.merge_neighbors:
            adj, processed_adjacencies = merge_neighbors(adj)
            assert set(processed_adjacencies) == set(self.processed_adjacencies)

        # convert from sparse numpy matrices to dense torch tensors
        for adj_type, matrix in adj.items():
            adj[adj_type] = sparse_to_dense(matrix)

        # pre-compute idx_to_cell mappings
        idx_to_cell = {
            rank: [sorted(cell) for cell in cc.skeleton(rank=rank)] for rank in range(self.dim + 1)
        }

        # for each adjacency/incidence, store the nodes to be used for computing geometric features
        inv = dict()
        for adj_type in self.processed_adjacencies:
            inv[adj_type] = []
            neighbors = adj[adj_type]
            ranks = [int(rank) for rank in adj_type.split("_")]
            i, j = ranks[:2]

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


def adjacency_matrix(cc: CombinatorialComplex, rank: int, via_rank: int) -> csc_matrix:
    """
    Compute the adjacency matrix for a given combinatorial complex, rank and via_rank.

    The adjacency matrix is computed based on the ranks of the cells in the complex. If the via_rank
    is lower than the rank, the coadjacency matrix is computed instead.

    Parameters
    ----------
    cc : CombinatorialComplex
        The combinatorial complex.
    rank : int
        The rank for which we want the adjacency matrix.
    via_rank : int
        The rank to compute the adjacency matrix via.

    Returns
    -------
    scipy.sparse.csc_matrix
        The adjacency matrix.

    Raises
    ------
    ValueError
        If rank and via_rank are the same, or if any of the input values are negative.

    Notes
    -----
    The toponetx adjacency_matrix() and coadjacency_matrix() methods have a bug if rank < via_rank
    and there are no cells with that rank. This function is a workaround for that bug as well as a
    convenience wrapper for the toponetx methods.

    """
    # check for invalid input
    if rank == via_rank:
        raise ValueError("rank and via_rank must be different.")
    if rank < 0:
        raise ValueError(f"rank must be a non-negative integer, but was {rank}.")
    if via_rank < 0:
        raise ValueError(f"via_rank must be a non-negative integer, but was {via_rank}.")

    # compute the adjacency matrix
    kwargs = dict(rank=rank, via_rank=via_rank, index=False)
    if via_rank < rank:
        matrix = cc.coadjacency_matrix(**kwargs)
    else:
        matrix = cc.adjacency_matrix(**kwargs)

    # triggered if rank < via_rank, i.e. adj_type = i_j, i != j and i is an empty rank but j is not
    num_cells = len(cc.skeleton(rank=rank))
    if matrix.shape != (num_cells, num_cells):
        matrix = csc_matrix((num_cells, num_cells), dtype=np.float64)

    return matrix


def incidence_matrix(cc, rank, to_rank):
    """
    Compute the incidence matrix between two ranks in a combinatorial complex.

    Parameters
    ----------
    cc : CombinatorialComplex
        The combinatorial complex.
    rank : int
        The starting rank.
    to_rank : int
        The ending rank.

    Returns
    -------
    csc_matrix
        The incidence matrix between the two ranks.

    Raises
    ------
    ValueError
        If rank and to_rank are the same.
    ValueError
        If rank or to_rank is a negative integer.

    """
    # check for invalid input
    if rank == to_rank:
        raise ValueError("rank and to_rank must be different.")

    def error_msg(arg_name, arg_value):
        return f"{arg_name} must be a non-negative integer, but was {arg_value}."

    if rank < 0:
        raise ValueError(error_msg("rank", rank))
    if to_rank < 0:
        raise ValueError(error_msg("to_rank", to_rank))

    # compute the incidence matrix
    if rank < to_rank:
        matrix = cc.incidence_matrix(rank=rank, to_rank=to_rank)
    else:
        matrix = cc.incidence_matrix(rank=to_rank, to_rank=rank).T

    # triggered if adj_type = i_j, i != j and i is an empty rank
    num_cells_i, num_cells_j = len(cc.skeleton(rank=rank)), len(cc.skeleton(rank=to_rank))
    if matrix.shape != (num_cells_i, num_cells_j):
        matrix = csc_matrix((num_cells_i, num_cells_j), dtype=np.float64)

    return matrix


def create_combinatorial_complex(
    cell_dict: dict[int, Iterable[frozenset[int]]]
) -> CombinatorialComplex:
    """
    Create a combinatorial complex from a dictionary of cells.

    Parameters
    ----------
    cell_dict : dict[int, Iterable[frozenset[int]]]
        A dictionary of cells, where the keys are the ranks of the cells and the values are
        iterables of cell indices.

    Returns
    -------
    CombinatorialComplex
        The combinatorial complex created from the input cells.

    Raises
    ------
    TypeError
        If the input cell_dict is not a dictionary or if any of its values are not iterables of
        frozensets.

    Notes
    -----
    When a high-rank cell (rank > 0) is added to the CombinatorialComplex, TopoNetX automatically
    adds the nodes which constitute the cell as 0-rank cells. As such under-the-hood behavior is
    prone to introduce hidden bugs, this custom wrapper function was created. This function first
    adds higher-order cells, then removes the artificially created 0-rank cells, and finally adds
    the organic 0-rank cells.

    The type of dictionary values is chosen to be Iterable[frozenset[int]] to allow for flexibility.
    In particular, this choice permits the dictionary values to be themselves dictionaries, in which
    case only their keys are used to create the complex. This is useful for creating a complex from
    the keys of a dictionary of cells and their lifter contributions, for example.
    """

    if not isinstance(cell_dict, dict):
        raise TypeError("Input cell_dict must be a dictionary.")

    for rank, cells in cell_dict.items():
        if (not isinstance(cells, Iterable)) or any(
            not isinstance(cell, frozenset) for cell in cells
        ):
            raise TypeError(f"Cells of rank {rank} must be Iterable[frozenset[int]].")

    # Create an instance of the combinatorial complex
    cc = CombinatorialComplex()

    # First add higher-order cells
    for rank, cells in cell_dict.items():
        if rank > 0:
            cc.add_cells_from(cells, ranks=rank)

    # Then remove the artificially created 0-rank cells
    zero_rank_cells = [cell for cell in cc.cells if len(cell) == 1]
    cc.remove_cells(zero_rank_cells)

    # Finally add the organic 0-rank cells
    if 0 in cell_dict.keys():
        cc.add_cells_from(cell_dict[0], ranks=0)

    return cc


def extract_cell_and_membership_data(
    input_dict: dict[int, dict[frozenset[int], list[bool]]]
) -> tuple[dict[int, list[list[int]]], dict[int, list[list[bool]]]]:
    """
    Extract cell and membership data from the input dictionary.

    Parameters
    ----------
    input_dict : dict[int, dict[frozenset[int], list[bool]]]
        The input dictionary containing cell and membership data.

    Returns
    -------
    x_dict : dict[int, list[list[int]]]
        A dictionary mapping ranks to a list of sorted cells.
    mem_dict : dict[int, list[list[bool]]]
        A dictionary mapping ranks to a list of membership values.

    """
    x_dict, mem_dict = {}, {}
    for rank, cell_lifter_map in input_dict.items():
        x_dict[rank] = [sorted(cell) for cell in cell_lifter_map.keys()]
        mem_dict[rank] = list(cell_lifter_map.values())
    return x_dict, mem_dict


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


def merge_neighbors(adj: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    """
    Merge matching adjacency relationships in the adjacency dictionary.

    Specifically, merge all adjacency relationships between cells of the same rank into a single
    adjacency relationship. This is useful when the adjacency relationships are computed for
    different types of neighbors, such as direct neighbors, coadjacent neighbors, etc.

    Parameters
    ----------
    adj : dict[str, torch.Tensor]
        The adjacency dictionary containing the neighboring cells.

    Returns
    -------
    dict[str, torch.Tensor]
        The merged adjacency dictionary.
    list[str]
        New list of adjacency types.
    Raises
    ------
    AssertionError
        If any unmerged adjacencies remain in the merged adjacency dictionary.
    """
    new_adj = {}
    for key, value in adj.items():
        i, j = key.split("_")[:2]
        # Copy over incidences
        if i != j:
            new_adj[key] = value
        # Merge same-rank adjacencies
        else:
            assert len(key.split("_")) == 3
            merged_key = f"{i}_{j}"
            if merged_key not in new_adj:
                new_adj[merged_key] = value
            else:
                # new_adj[merged_key] = new_adj[merged_key] | value
                new_adj[merged_key] += value
                new_adj[merged_key][new_adj[merged_key] > 0] = 1
    adj = new_adj
    adj_types = list(adj.keys())
    # Check that no unmerged adjacencies remain
    for adj_type in adj_types:
        assert len(adj_type.split("_")) == 2
    return adj, adj_types


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


def sparse_to_dense(sparse_matrix: csc_matrix) -> torch.Tensor:
    """
    Convert a sparse (n, m) matrix to a dense (2, k) PyTorch tensor in an adjacency list format.

    Parameters
    ----------
    sparse_matrix : csc_matrix
        The sparse matrix to convert.

    Returns
    -------
    torch.Tensor
        The dense PyTorch tensor representation of the sparse matrix.
    """
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
