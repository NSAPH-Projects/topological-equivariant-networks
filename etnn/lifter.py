from collections import defaultdict
from functools import partial
from typing import DefaultDict, Iterable

import numpy as np
import torch
from scipy.sparse import csc_matrix
from toponetx.classes import CombinatorialComplex
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from etnn.combinatorial_data import Cell, CombinatorialComplexData


class Lifter:

    def __init__(
        self,
        lifter_names,
        lifter_registry: dict[str, callable],
        lifter_dim: int | None = None,
        **lifter_kwargs,
    ) -> "Lifter":
        """
        Initialize the Lifter object.

        Parameters
        ----------
        lifter_names : list[str]
            A list of lifter names and their ranking logic to be applied to the input data.
        dim : int
            The dimension of the ASC.
        lifter_registry : dict[str, callable]
            A dictionary of lifter names and corresponding functions.

        Returns
        -------
        Lifter
            The Lifter object.
        """

        # TODO: check inputs: a lift with hetero features may not be used with cardinality

        self.lifters = get_lifters(lifter_names, lifter_registry, **lifter_kwargs)
        self.num_features_dict = get_num_features_dict(self.lifters)
        if lifter_dim is None:
            # choose dim as max rank
            lifter_dim = max(int(x.split(":")[1]) for x in lifter_names)
        self.dim = lifter_dim

    def lift(self, graph: Data) -> dict[int, dict[Cell, list[bool]]]:
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
        dict[frozenset[int], list[list[float] | None]]
            A dictionary mapping each unique cell (a list of integers representing a cell) to a list
            of feature vectors. Each feature vector contains the features contributed by the
            corresponding lifter, or is None if the corresponding lifter did not generate this cell.
            The keys of the dictionary are ordered according to the sorting order they would have if
            they were cast to lists and internally sorted.

        Notes
        -----
        The keys of the dictionary are reordered according to the sorting order they would have if
        they were cast to lists and internally sorted. This sorting must happen here as the keys of
        this dictionary determine the ordering of rows/columns in all of the downstream data
        structures.
        """
        cell_lifter_map = {}
        for lifter_idx, (lifter, _) in enumerate(self.lifters):
            lifter_output = lifter(graph)
            for cell in lifter_output:
                node_idc, feature_vec = cell
                if node_idc not in cell_lifter_map:
                    cell_lifter_map[node_idc] = [None] * len(self.lifters)
                cell_lifter_map[node_idc][lifter_idx] = feature_vec

        # Reorder the dictionary keys numerically
        sorted_cells = sorted(sorted(list(cell)) for cell in cell_lifter_map.keys())
        cell_lifter_map = {
            frozenset(cell): cell_lifter_map[frozenset(cell)] for cell in sorted_cells
        }

        # compute ranks and final feature vector for each cell
        cell_dict = {rank: {} for rank in range(self.dim + 1)}
        for node_idc, feature_vectors in cell_lifter_map.items():
            cell, rank = self.aggregate_lifter_outputs(node_idc, feature_vectors)
            memberships = [(ft_vec is not None) for ft_vec in feature_vectors]
            if rank <= self.dim:
                cell_dict[rank][cell] = memberships

        return cell_dict

    def aggregate_lifter_outputs(
        self, node_idc: frozenset[int], feature_vectors: list[list[float] | None]
    ) -> tuple[Cell, int]:
        """
        Aggregate the outputs of the lifters based on the given node IDs and feature vectors.

        Parameters
        ----------
        node_idc : frozenset[int]
            The set of node indices.
        feature_vectors : list[list[float] | None]
            The list of feature vectors for each lifter.

        Returns
        -------
        tuple[Cell, int]
            A tuple containing the aggregated cell and the rank.

        Raises
        ------
        AssertionError
            If the length of feature_vectors is not equal to the number of lifters.

        Notes
        -----
        This method aggregates the outputs of the lifters for the given cell, which is represented
        by its node indices and feature vectors. The rank is the minimal rank assigned by any of the
        generating lifters, and the final feature vector is a concetanation of feature vectors of
        lifters that constantly assign that rank which is minimum. The aggregated cell and the rank
        are returned as a tuple.

        """
        assert len(feature_vectors) == len(self.lifters)

        memberships = [(ft_vec is not None) for ft_vec in feature_vectors]
        rank = self.ranker(node_idc, memberships)

        combined_feature_vec = []
        for i, (lifter, lifter_rank) in enumerate(self.lifters):
            if lifter_rank == rank:
                if feature_vectors[i] is None:
                    lifter_fts = [0.0] * lifter.num_features
                else:
                    lifter_fts = feature_vectors[i]
                combined_feature_vec.extend(lifter_fts)
        combined_feature_vec = tuple(combined_feature_vec)
        cell = (node_idc, combined_feature_vec)
        return cell, rank

    def ranker(self, node_idc: frozenset[int], memberships: list[bool]) -> int:
        """
        Determine the rank of a cell based on its memberships and predefined logics.

        If a cell is a member of multiple lifters, then its final rank will be the minimum among the
        ranks assigned to it by each lifter.

        Parameters
        ----------
        cell : frozenset[int]
            The cell to be ranked, assumed to be a collection or group of items.
        memberships : list[bool]
            A list indicating membership of the cell in various groups, corresponding to the
            `lifter_args` with which the ranker was created.

        Returns
        -------
        int
            The rank of the cell, determined as the minimum rank among the groups to which the cell
            belongs, with 'c' indicating rank based on cardinality.

        Raises
        ------
        ValueError
            If the length of `memberships` does not match the number of lifters.

        Notes
        -----
        The minimum is returned to avoid a situation where a singleton cell gets a nonzero rank,
        which is not allowed in TopoNetX.
        """
        if len(memberships) != len(self.lifters):
            raise ValueError(
                "The length of `memberships` does not match the number of lifters."
            )

        ranks = []
        for idx, is_member in enumerate(memberships):
            if is_member:
                lifter_rank = self.lifters[idx][1]
                if lifter_rank == "c":
                    ranks.append(len(node_idc) - 1)
                else:
                    ranks.append(lifter_rank)
        return min(ranks)


def get_lifters(
    lifter_names, lifter_registry: dict[str, callable], **kwargs
) -> list[tuple[callable, int | str]]:
    """
    Construct a list of lifter functions based on provided arguments.

    This function iterates through a list of lifter names specified in the input arguments. For each
    lifter, it either retrieves the corresponding function from a registry or creates a partial
    function with additional arguments for specific lifters like 'rips'.

    Parameters
    ----------
    lifter_names : list[str]
        A list of lifter names and their ranking logic to be applied to the input data.
    dim : int
        The dimension of the ASC.
    dis : float
        The radius for the Rips complex.

    lifter_registry : dict[str, callable]
        A dictionary of known lifter names and corresponding functions.

    Returns
    -------
    list[tuple[callable, int | str]]
        A list of tuples, where each tuple contains a callable lifter function and its ranking
        logic.
    """
    lifters = []
    for lifter_str in lifter_names:
        # Create the callable
        parts = lifter_str.split(":")
        method_str = parts[0]
        if method_str == "rips":
            lifter = partial(lifter_registry[method_str], **kwargs)
            lifter.num_features = lifter_registry[method_str].num_features
        else:
            lifter = lifter_registry[method_str]

        # Parse the ranking logic
        ranking_logic = parse_ranking_logic(lifter_str)

        lifters.append((lifter, ranking_logic))

    for lifter, lifter_rank in lifters:
        if lifter_rank == "c" and lifter.num_features > 0:
            raise ValueError(
                "cardinality-based rank cannot be combined with heterogeneous features!"
            )
    return lifters


def get_num_features_dict(
    lifters: list[tuple[callable, int | str]]
) -> DefaultDict[int, int]:
    """
    Calculate the number of features for each rank in the given list of lifters.

    Parameters
    ----------
    lifters : list[tuple[callable, int | str]]
        A list of lifters, where each lifter is represented as a tuple containing a callable lifter
        function and a ranking logic (either an integer or a string).

    Returns
    -------
    DefaultDict[int, int]
        A dictionary where the keys represent the ranking logic and the values represent the
        corresponding number of features. The default return value is 0.

    """
    num_features_dict = defaultdict(int)
    for lifter_fct, ranking_logic in lifters:
        if isinstance(ranking_logic, int):
            num_features_dict[ranking_logic] += lifter_fct.num_features
    return num_features_dict


def parse_ranking_logic(lifter_str: str) -> str | int:
    """
    Parse the ranking logic from the given lifter string.

    Parameters
    ----------
    lifter_str : str
        The lifter string containing the rank and logic in the format 'name:rank'.

    Returns
    -------
    str | int
        The parsed ranking logic. If no rank is specified, the string 'c' is returned.
        If the ranking logic is 'c', it returns the string 'c'. Otherwise, it returns the parsed
        rank as an integer.

    Raises
    ------
    ValueError
        If an invalid rank is specified or if a negative rank is requested.

    Examples
    --------
    >>> parse_ranking_logic('item:1')
    1
    >>> parse_ranking_logic('item:c')
    'c'
    >>> parse_ranking_logic('item')
    'c'
    """
    parts = lifter_str.split(":")

    # Default to cardinality if no rank is specified
    if len(parts) == 1:
        return "c"

    rank_str = parts[1]

    # Return 'c' if cardinality is requested
    if rank_str == "c":
        return rank_str

    # Try casting to int
    try:
        rank = int(rank_str)
    except ValueError:
        raise ValueError(
            f"Invalid rank '{rank_str}' specified for lifter '{parts[0]}'."
        )

    # Negative ranks are not allowed
    if rank < 0:
        raise ValueError(
            f"Negative cell ranks are not allowed, but '{lifter_str}' was requested."
        )

    return rank


def get_adjacency_types(
    max_dim: int,
    connectivity: str,
    neighbor_types: list[str],
    # visible_dims: list[int] | None
) -> list[str]:
    """
    Generate a list of adjacency type strings based on the specified connectivity pattern.

    Parameters
    ----------
    max_dim : int
        The maximum dimension (inclusive) for which to generate adjacency types. Represents the
        highest rank of cells in the connectivity pattern.
    connectivity : str
        The connectivity pattern to use. Must be one of the options defined below:
        - "self_and_next" generates adjacencies where each rank is connected to itself and the next
        (higher) rank.
        - "self_and_higher" generates adjacencies where each rank is connected to itself and all
        higher ranks.
        - "self_and_previous" generates adjacencies where each rank is connected to itself and the
        previous (lower) rank.
        - "self_and_lower" generates adjacencies where each rank is connected to itself and all
        lower ranks.
        - "self_and_neighbors" generates adjacencies where each rank is connected to itself, the
        next (higher) rank and the previous (lower) rank.
        - "all_to_all" generates adjacencies where each rank is connected to every other rank,
        including itself.
        - "legacy" ignores the max_dim parameter and returns ['0_0', '0_1', '1_1', '1_2'].
    neighbor_types : list[str]
        The types of adjacency between cells of the same rank. Must be one of the following:
        +1: two cells of same rank i are neighbors if they are both neighbors of a cell of rank i+1
        -1: two cells of same rank i are neighbors if they are both neighbors of a cell of rank i-1
        max: two cells of same rank i are neighbors if they are both neighbors of a cell of max rank
        min: two cells of same rank i are neighbors if they are both neighbors of a cell of min rank

    Returns
    -------
    list[str]
        A list of strings representing the adjacency types for the specified connectivity pattern.
        Each string is in the format "i_j" where "i" and "j" are ranks indicating an adjacency
        from rank "i" to rank "j".

    Raises
    ------
    ValueError
        If `connectivity` is not one of the known connectivity patterns.

    Examples
    --------
    >>> get_adjacency_types(2, "self_and_next", ["+1"])
    ['0_0_1', '0_1', '1_1_2', '1_2']

    >>> get_adjacency_types(2, "self_and_higher", ["-1"])
    ['0_1', '0_2', '1_1_0', '1_2', '2_2_1']

    >>> get_adjacency_types(2, "all_to_all", ["-1", "+1", "max", "min"])
    ['0_0_1', '0_0_2','0_1', '0_2', '1_0', '1_1_0', '1_1_2', '1_2', '2_0', '2_1', '2_2_1', '2_2_0']
    """
    adj_types = []
    if connectivity not in [
        "self",
        "self_and_next",
        "self_and_higher",
        "self_and_previous",
        "self_and_lower",
        "self_and_neighbors",
        "all_to_all",
        "legacy",
    ]:
        raise ValueError(f"{connectivity} is not a known connectivity pattern!")

    if connectivity == "self":
        for i in range(max_dim + 1):
            adj_types.append(f"{i}_{i}")

    elif connectivity == "self_and_next":
        for i in range(max_dim + 1):
            adj_types.append(f"{i}_{i}")
            if i < max_dim:
                adj_types.append(f"{i}_{i+1}")

    elif connectivity == "self_and_higher":
        for i in range(max_dim + 1):
            for j in range(i, max_dim + 1):
                adj_types.append(f"{i}_{j}")

    elif connectivity == "self_and_previous":
        for i in range(max_dim + 1):
            adj_types.append(f"{i}_{i}")
            if i > 0:
                adj_types.append(f"{i}_{i-1}")

    elif connectivity == "self_and_lower":
        for i in range(max_dim + 1):
            for j in range(0, i):
                adj_types.append(f"{i}_{j}")

    elif connectivity == "self_and_neighbors":
        for i in range(max_dim + 1):
            adj_types.append(f"{i}_{i}")
            if i > 0:
                adj_types.append(f"{i}_{i-1}")
            if i < max_dim:
                adj_types.append(f"{i}_{i+1}")

    elif connectivity == "all_to_all":
        for i in range(max_dim + 1):
            for j in range(max_dim + 1):
                adj_types.append(f"{i}_{j}")

    else:
        raise ValueError(f"{connectivity} is not a known connectivity pattern!")
        # adj_types = ["0_0", "0_1", "1_1", "1_2"]

    # Add one adjacency type for each neighbor type
    new_adj_types = []
    for adj_type in adj_types:
        i, j = map(int, adj_type.split("_"))
        if i == j:
            for neighbor_type in neighbor_types:
                if neighbor_type == "+1":
                    if i < max_dim:
                        new_adj_types.append(f"{i}_{i}_{i+1}")
                elif neighbor_type == "-1":
                    if i > 0:
                        new_adj_types.append(f"{i}_{i}_{i-1}")
                elif neighbor_type == "max":
                    if i < max_dim:
                        new_adj_types.append(f"{i}_{i}_{max_dim}")
                elif neighbor_type == "min":
                    if i > 0:
                        new_adj_types.append(f"{i}_{i}_0")
        else:
            new_adj_types.append(adj_type)
    new_adj_types = list(set(new_adj_types))
    adj_types = new_adj_types

    # # Filter adjacencies with invisible ranks
    # if visible_dims is not None:
    #     adj_types = [
    #         adj_type
    #         for adj_type in adj_types
    #         if all(int(dim) in visible_dims for dim in adj_type.split("_")[:2])
    #     ]

    return adj_types


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
