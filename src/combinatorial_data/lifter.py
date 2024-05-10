from argparse import Namespace
from functools import partial

from torch_geometric.data import Data

from combinatorial_data.ranker import get_ranker


class Lifter:

    def __init__(self, args: Namespace, lifter_registry: dict[str, callable]) -> "Lifter":
        """
        Initialize the Lifter object.

        This function constructs a list of lifter functions based on the provided arguments.

        Parameters
        ----------
        args : argparse.Namespace
            The parsed command line arguments. It should contain 'lifters', a list of lifter names,
            and additional arguments like 'dim' and 'dis' for specific lifters.
        lifter_registry : Dict[str, Callable]
            A dictionary of lifter names and corresponding functions.

        Returns
        -------
        Lifter
            The Lifter object.
        """

        # TODO: check inputs: a lift with hetero features may not be used with cardinality

        self.lifters = get_lifters(args, lifter_registry)
        self.ranker = get_ranker(args.lifters)
        self.dim = args.dim

    def lift(self, graph: Data) -> dict[frozenset[int], list[list[float] | None]]:
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
                node_idc, feature_vec = cell
                if node_idc not in cell_lifter_map:
                    cell_lifter_map[node_idc] = [None] * len(self.lifters)
                cell_lifter_map[node_idc][lifter_idx] = feature_vec

        # Reorder the dictionary keys numerically
        sorted_cells = sorted(sorted(list(cell)) for cell in cell_lifter_map.keys())
        cell_lifter_map = {
            frozenset(cell): cell_lifter_map[frozenset(cell)] for cell in sorted_cells
        }

        # compute ranks for each cell
        # TODO: here make cells not frozenset[int], but Cell where the tuple contains the combined
        # feature vector. will probably need to modify ranker
        cell_dict = {rank: {} for rank in range(self.dim + 1)}
        for cell, feature_vectors in cell_lifter_map.items():
            memberships = [(ft_vec is not None) for ft_vec in feature_vectors]
            cell_rank = self.ranker(cell, memberships)
            if cell_rank <= self.dim:
                cell_dict[cell_rank][cell] = memberships

        return cell_dict


def get_lifters(args: Namespace, lifter_registry: dict[str, callable]) -> list[callable]:
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

    Returns
    -------
    List[Callable]
        A list of callable lifter functions, ready to be applied to data.
    """
    lifters = []
    for lifter in args.lifters:
        lifter = lifter.split(":")[0]
        if lifter == "rips":
            lifters.append(partial(lifter_registry[lifter], dim=args.dim, dis=args.dis))
        else:
            lifters.append(lifter_registry[lifter])
    return lifters
