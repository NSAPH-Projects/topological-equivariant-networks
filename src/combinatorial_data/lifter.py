from argparse import Namespace
from collections import defaultdict
from functools import partial
from typing import DefaultDict

from torch_geometric.data import Data

from qm9.lifts.common import Cell


class Lifter:

    def __init__(self, args: Namespace, lifter_registry: dict[str, callable]) -> "Lifter":
        """
        Initialize the Lifter object.

        Parameters
        ----------
        args : argparse.Namespace
            The parsed command line arguments. It should contain 'lifters', a list of lifter names,
            and additional arguments like 'dim' and 'dis' for specific lifters.
        lifter_registry : dict[str, callable]
            A dictionary of lifter names and corresponding functions.

        Returns
        -------
        Lifter
            The Lifter object.
        """

        # TODO: check inputs: a lift with hetero features may not be used with cardinality

        self.lifters = get_lifters(args, lifter_registry)
        self.num_features_dict = get_num_features_dict(self.lifters)
        self.dim = args.dim

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
                    lifter_fts = [0] * lifter.num_features
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
            raise ValueError("The length of `memberships` does not match the number of lifters.")

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
    args: Namespace, lifter_registry: dict[str, callable]
) -> list[tuple[callable, int | str]]:
    """
    Construct a list of lifter functions based on provided arguments.

    This function iterates through a list of lifter names specified in the input arguments. For each
    lifter, it either retrieves the corresponding function from a registry or creates a partial
    function with additional arguments for specific lifters like 'rips'.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments. It should contain 'lifters', a list of lifter names, and
        additional arguments like 'dim' and 'dis' for specific lifters.

    lifter_registry : dict[str, callable]
        A dictionary of known lifter names and corresponding functions.

    Returns
    -------
    list[tuple[callable, int | str]]
        A list of tuples, where each tuple contains a callable lifter function and its ranking
        logic.
    """
    lifters = []
    for lifter_str in args.lifters:
        # Create the callable
        parts = lifter_str.split(":")
        method_str = parts[0]
        if method_str == "rips":
            lifter = partial(lifter_registry[method_str], dim=args.dim, dis=args.dis)
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


def get_num_features_dict(lifters: list[tuple[callable, int | str]]) -> DefaultDict[int, int]:
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
        raise ValueError(f"Invalid rank '{rank_str}' specified for lifter '{parts[0]}'.")

    # Negative ranks are not allowed
    if rank < 0:
        raise ValueError(f"Negative cell ranks are not allowed, but '{lifter_str}' was requested.")

    return rank
