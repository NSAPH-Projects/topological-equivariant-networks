import itertools
import re
from collections.abc import Iterable
from types import MappingProxyType

import numpy as np
import torch
from scipy.sparse import csc_matrix
from toponetx.classes import CombinatorialComplex
from torch import Tensor
from torch_geometric.transforms import BaseTransform

from etnn.lifter import Lifter
from qm9.lifts.common import Cell

from typing import Iterable, Literal, Union


class CombinatorialComplexData(Data):
    """
    A subclass of PyTorch Geometric's Data class designed for representing combinatorial complexes.

    This class extends standard graph representation to handle higher-dimensional structures found
    in complex networks. It supports adjacency tensors for cells of different ranks (vertices,
    edges, faces, etc.) and their relationships, following a specific naming convention for dynamic
    attribute recognition and processing.

    Attributes
    ----------
    x : torch.FloatTensor
        Features of rank 0 cells (atoms).
    y : torch.FloatTensor
        Target values, assumed to be at the graph level. This tensor should have shape (1,
        num_targets).
    pos : torch.FloatTensor
        Positions of rank 0 cells (atoms). Expected to have shape (num_atoms, 3).
    x_i : torch.FloatTensor
        Features of cells of rank i, where i is a non-negative integer.
    adj_i_j : torch.LongTensor
        Adjacency tensors representing the relationships (edges) between cells of rank i and j,
        where i and j are non-negative integers.
    cells_i : torch.FloatTensor
        Concatenated list of node indices associated with cells of rank i. Note: Use the cell_list
        method to split this tensor into individual cells.
    slices_i : torch.LongTensor
        Slices to split the concatenated cell_i tensor into individual cells. Note: Use the cell_list
        method to split this tensor into individual cells.
    mem_i : torch.BoolTensor
        Optional. Lifters associated with cells of rank i, where i is a non-negative integer.
    # inv_i_j : torch.FloatTensor
    #     Optional. Node indices that can be used to compute legacy geometric features for each cell
    #     pair.
    """

    attr_dtype = MappingProxyType(
        {
            "x_": torch.float32,
            # "cell_": torch.float64,
            "cells_": torch.long,
            "slices_": torch.long,
            "mem_": torch.bool,
            "adj_": torch.int64,
            # "inv_": torch.float64,
        }
    )

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
            the number of nodes for `inv_i_j` and `cell_i` attributes, or calls the superclass's
            `__inc__` method for other attributes.
        """
        # The adj_i_j attribute holds cell indices, increment each dim by the number of cells of
        # corresponding rank
        if re.match(r"adj_(\d+_\d+|\d+_\d+_\d+)", key):
            i, j = key.split("_")[1:3]
            return torch.tensor([[self.num_cells(rank=i)], [self.num_cells(rank=j)]])

        # The inv_i_j and cell_i attributes hold node indices, they should be incremented
        # elif re.match(r"inv_(\d+_\d+|\d+_\d+_\d+)", key) or re.match(r"cell_\d+", key):
        #     return num_nodes
        elif re.match(r"cells_\d+", key):
            return self.num_cells(rank=0)

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
        # if re.match(r"(adj|inv)_\d+_\d+", key):
        if re.match(r"adj_\d+_\d+", key):
            return 1
        else:
            return 0

    def cell_list(
        self,
        rank: int,
        format: Literal["list", "padded"] = "list",
        pad_value: float = torch.nan,
    ) -> Union[list[Tensor], Tensor]:
        """Return the list of cells of a given rank.

        Parameters
        ----------
        rank : int
            The rank of the cells to return.
        as_padded_tensor : bool, optional
            Whether to return the cells as a padded tensor, by default False.

        Returns
        -------
        Union[list[Tensor], Tensor]
            The list of cells of the given rank. Each cell is a tensor of node indices.
        """
        concatenated_cells = getattr(self, f"cells_{rank}")
        slices = getattr(self, f"slices_{rank}")
        cell_list = list(torch.split(concatenated_cells, slices.tolist()))

        if format == "padded":
            cell_list = torch.nested.as_nested_tensor(cell_list, dtype=torch.float)
            return cell_list.to_padded_tensor(padding=pad_value)
        elif format == "list":
            return cell_list
        else:
            raise ValueError(f"Unknown format: {format}")

    def num_cells(self, rank: int) -> int:
        """Return the number of cells of a given rank."""
        return len(getattr(self, f"slices_{rank}"))

    @property
    def num_features_per_rank(self) -> dict[int, int]:
        """
        Return the number of features for each rank in the combinatorial complex.

        Returns
        -------
        dict[int, int]
            The number of features for each rank in the combinatorial complex.
        """
        D = {}
        for key in self.keys():
            if key == "x":
                # this case is for backward compatibility
                D[0] = getattr(self, key).size(1)
            elif key.startswith("x_"):
                rank = int(key.split("_")[1])
                D[int(rank)] = getattr(self, key).size(1)
        return D

    @classmethod
    def from_ccdict(cls, data: dict[str, any]) -> "CombinatorialComplexData":
        """
        Convert a dictionary of data to a CombinatorialComplexData object.

        Parameters
        ----------
        data : dict[str, any]
            The dictionary of data to be converted.

        Returns
        -------
        CombinatorialComplexData
            The CombinatorialComplexData object created from the input dictionary.

        Notes
        -----
        In addition to the attributes listed under CombinatorialComplexData, this method also
        assumes that the input dictionary contains a dictionary under the key "num_features_dict".
        This dictionary should hold the number of heterogeneous features for each rank, and aids
        to initialize tensors with the correct size.

        """
        attr = {}

        for key, value in data.items():
            if any(key.startswith(s) for s in ["pos", "y"]):
                attr[key] = torch.tensor(value)

            # cast the x_i
            if "x_" in key:
                if len(value) == 0:
                    rank = key.split("_")[1]
                    num_features = data["num_features_dict"][rank]
                    attr_value = torch.empty(
                        (0, num_features), dtype=cls.attr_dtype["x_"]
                    )
                else:
                    attr_value = torch.tensor(value, dtype=cls.attr_dtype["x_"])
                attr[key] = attr_value

            # cast the cell_i
            elif "cell_" in key:
                # each cell is recorded as a concatenated list of nodes and slices
                # use the cell_list() method to split the cells into a list of tensors
                if len(value) == 0:
                    slices_val = torch.empty((0,), dtype=cls.attr_dtype["slices_"])
                    cell_val = torch.empty((0,), dtype=cls.attr_dtype["cells_"])
                else:
                    lens = [len(cell) for cell in value]
                    vals = list(itertools.chain.from_iterable(value))
                    cell_val = torch.tensor(vals, dtype=cls.attr_dtype["cells_"])
                    slices_val = torch.tensor(lens, dtype=cls.attr_dtype["slices_"])

                rank = key.split("_")[1]
                attr[f"cells_{rank}"] = cell_val
                attr[f"slices_{rank}"] = slices_val

            # cast the mem_i
            elif "mem_" in key:
                num_lifters = len(data["mem_0"][0])
                if len(value) == 0:
                    attr_value = torch.empty(
                        (0, num_lifters), dtype=cls.attr_dtype["mem_"]
                    )
                else:
                    attr_value = torch.tensor(value, dtype=cls.attr_dtype["mem_"])
                attr[key] = attr_value

            # cast the adj_i_j[_foo]
            elif "adj_" in key:
                attr[key] = torch.tensor(value, dtype=cls.attr_dtype["adj_"])

            # # cast the inv_i_j[_foo]
            # elif "inv_" in key:
            #     # if len(value) == 0:
            #     #     attr_value = torch.empty((0, 0), dtype=cls.attribute_dtype["inv_"])
            #     # else:
            #     #     attr_value = torch.tensor(
            #     #         pad_lists_to_same_length(value),
            #     #         dtype=cls.attribute_dtype["inv_"],
            #     #     ).t()
            #     attr[key] = attr_value

        return cls.from_dict(attr)


class CombinatorialComplexTransform(BaseTransform):
    """Todo: add
    The adjacency types (adj) are saved as properties, e.g. object.adj_1_2 gives the edge index from
    1-complexes to 2-complexes."""

    def __init__(
        self,
        lifter: Lifter,
        # dim: int,
        adjacencies: list[str],
        # processed_adjacencies: list[str],
        # merge_neighbors: bool,
    ):
        self.lifter = lifter
        # self.dim = dim
        self.adjacencies = adjacencies
        # self.processed_adjacencies = processed_adjacencies
        # self.merge_neighbors = merge_neighbors

    def __call__(self, graph: Data) -> CombinatorialComplexData:
        """
        Convert a graph to a CombinatorialComplexData object.

        Parameters
        ----------
        graph : Data
            The input graph to convert.

        Returns
        -------
        CombinatorialComplexData
            The converted combinatorial complex data object.
        """
        ccdict = self.graph_to_ccdict(graph)
        return CombinatorialComplexData.from_ccdict(ccdict)

    def graph_to_ccdict(self, graph: Data) -> dict[str, list]:
        """
        Convert a graph to a combinatorial complex dictionary.

        TODO: refactor into simpler functions.

        Parameters
        ----------
        graph : Data
            The input graph.

        Returns
        -------
        dict[str, list]
            The combinatorial complex dictionary.

        Raises
        ------
        AssertionError
            If the processed adjacencies do not match the expected adjacencies.

        Notes
        -----
        The combinatorial complex dictionary is created by performing the following steps:

        1. Compute the cells of the graph using the lifter.
        memberships.
        2. Extract the cell indices and memberships from the cell dictionary.
        3. Create the combinatorial complex using the `create_combinatorial_complex` method.
        4. Compute the adjacencies and incidences of the combinatorial complex.
        5. Merge matching adjacencies if the `merge_neighbors` flag is set to True.
        6. Convert the sparse numpy matrices to dense torch tensors.
        7. Store the nodes for computing geometric features in the inv_dict.
        8. Convert the graph and other data to a dictionary format.
        9. Convert tensors in the dictionary to lists.

        The resulting combinatorial complex dictionary contains the following keys:
        - 'x': features of rank 0 cells (atoms)
        - 'pos': positions of rank 0 cells (atoms)
        - 'y': target values
        - 'x_i': feature vectors for each rank i
        - 'cell_i': cell indices for each rank i
        - 'mem_i': cell memberships for each rank i
        - 'adj_adj_type': adjacency matrices for each adjacency type
        - 'inv_adj_type': nodes used for computing geometric features for each adjacency type

        The combinatorial complex dictionary can be stored as a .json file.
        """

        # compute cells
        cc_dict = self.lifter.lift(graph)

        # compute cell indices and memberships
        cell_dict, x_dict, mem_dict = extract_cell_and_membership_data(cc_dict)

        # create the combinatorial complex
        cc = create_combinatorial_complex(cc_dict)

        # compute adjancencies and incidences
        adj_dict = dict()
        for adj_type in self.adjacencies:
            ranks = [int(rank) for rank in adj_type.split("_")]
            i, j = ranks[:2]
            if i != j:
                matrix = incidence_matrix(cc, i, j)
            else:
                # if i == j, we must have a third rank specifying via_rank
                assert len(ranks) == 3
                matrix = adjacency_matrix(cc, i, ranks[2])

            adj_dict[adj_type] = matrix

        # merge matching adjacencies
        # if self.merge_neighbors:
        #     adj_dict, processed_adjacencies = merge_neighbors(adj_dict)
        #     assert set(processed_adjacencies) == set(self.processed_adjacencies)

        # convert from sparse numpy matrices list of index lists
        for adj_type, matrix in adj_dict.items():
            adj_dict[adj_type] = sparse_to_dense(matrix)

        # for each adjacency/incidence, store the nodes to be used for computing geometric features
        # inv_dict = dict()
        # for adj_type in processed_adjacencies:
        #     inv_dict[adj_type] = []
        #     neighbors = adj_dict[adj_type]
        #     ranks = [int(rank) for rank in adj_type.split("_")]
        #     i, j = ranks[:2]

        #     num_edges = len(neighbors[0])
        #     for edge_idx in range(num_edges):
        #         idx_a, idx_b = neighbors[0][edge_idx], neighbors[1][edge_idx]
        #         cell_a = cell_dict[i][idx_a]
        #         cell_b = cell_dict[j][idx_b]
        #         shared = [node for node in cell_a if node in cell_b]
        #         only_in_a = [node for node in cell_a if node not in shared]
        #         only_in_b = [node for node in cell_b if node not in shared]
        #         inv_nodes = shared + only_in_b + only_in_a
        #         inv_dict[adj_type].append(inv_nodes)

        cc_dict = graph.to_dict()

        for k, v in cell_dict.items():
            cc_dict[f"cell_{k}"] = v

        for k, v in x_dict.items():
            cc_dict[f"x_{k}"] = v

        for k, v in mem_dict.items():
            cc_dict[f"mem_{k}"] = v

        for k, v in adj_dict.items():
            cc_dict[f"adj_{k}"] = v

        # for k, v in inv_dict.items():
        #     cc_dict[f"inv_{k}"] = v

        # store the number of features for each rank for tensor reconstruction
        cc_dict["num_features_dict"] = {}
        for rank in range(self.lifter.dim + 1):
            cc_dict["num_features_dict"][rank] = self.lifter.num_features_dict[rank]

        for att in ["edge_attr", "edge_index"]:
            if att in cc_dict.keys():
                cc_dict.pop(att)

        # convert tensors to lists
        for k, v in cc_dict.items():
            if torch.is_tensor(v):
                cc_dict[k] = v.tolist()

        # remove the molecule from the dictionary
        cc_dict.pop("mol")

        return cc_dict


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
        raise ValueError(
            f"via_rank must be a non-negative integer, but was {via_rank}."
        )

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
    num_cells_i, num_cells_j = len(cc.skeleton(rank=rank)), len(
        cc.skeleton(rank=to_rank)
    )
    if matrix.shape != (num_cells_i, num_cells_j):
        matrix = csc_matrix((num_cells_i, num_cells_j), dtype=np.float64)

    return matrix


def create_combinatorial_complex(
    cell_dict: dict[int, Iterable[Cell]]
) -> CombinatorialComplex:
    """
    Create a combinatorial complex from a dictionary of cells.

    Parameters
    ----------
    cell_dict : dict[int, Iterable[Cell]]
        A dictionary of cells, where the keys are the ranks of the cells and the values are
        iterables of tuples of cell indices and corresponding feature vectors.

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

    The type of dictionary values is chosen to be Iterable[Cell] to allow for flexibility.
    In particular, this choice permits the dictionary values to be themselves dictionaries, in which
    case only their keys are used to create the complex. This is useful for creating a complex from
    the keys of a dictionary of cells and their lifter contributions, for example.
    """

    if not isinstance(cell_dict, dict):
        raise TypeError("Input cell_dict must be a dictionary.")

    # Create an instance of the combinatorial complex
    cc = CombinatorialComplex()

    # First add higher-order cells
    for rank, cells in cell_dict.items():
        if rank > 0:
            node_idc = [cell[0] for cell in cells.keys()]
            cc.add_cells_from(node_idc, ranks=rank)

    # Then remove the artificially created 0-rank cells
    zero_rank_cells = [cell for cell in cc.cells if len(cell) == 1]
    cc.remove_cells(zero_rank_cells)

    # Finally add the organic 0-rank cells
    if 0 in cell_dict.keys():
        node_idc = [cell[0] for cell in cell_dict[0].keys()]
        cc.add_cells_from(node_idc, ranks=0)

    return cc


def extract_cell_and_membership_data(
    input_dict: dict[int, dict[Cell, list[bool]]]
) -> tuple[dict[int, list[list[int]]], dict[int, list[list[bool]]]]:
    """
    Extract cell and membership data from the input dictionary.

    Parameters
    ----------
    input_dict : dict[int, dict[Cell, list[bool]]]
        The input dictionary containing cell and membership data.

    Returns
    -------
    cell_dict : dict[int, list[list[int]]]
        A dictionary mapping ranks to a list of sorted cells.
    x_dict : dict[int, list[list[float]]]
        A dictionary mapping ranks to a list of feature vectors.
    mem_dict : dict[int, list[list[bool]]]
        A dictionary mapping ranks to a list of membership values.
    """
    cell_dict, x_dict, mem_dict = {}, {}, {}
    for rank, cell_lifter_map in input_dict.items():
        cell_dict[rank] = [sorted(cell[0]) for cell in cell_lifter_map.keys()]
        x_dict[rank] = [cell[1] for cell in cell_lifter_map.keys()]
        mem_dict[rank] = list(cell_lifter_map.values())
    return cell_dict, x_dict, mem_dict


def merge_neighbors(
    adj: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], list[str]]:
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
    if len(list_of_lists) == 0:
        max_length = 0
    else:
        max_length = max(len(inner_list) for inner_list in list_of_lists)

    # Pad each list to match the maximum length
    padded_list = [
        inner_list + [pad_value] * (max_length - len(inner_list))
        for inner_list in list_of_lists
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
