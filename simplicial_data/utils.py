from typing import Union

import numpy as np
import torch
from toponetx.classes import CombinatorialComplex
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


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
        elif key == "inv_0_0":
            return torch.tensor([[num_nodes], [num_nodes]])
        elif key == "inv_0_1":
            return torch.tensor([[num_nodes], [num_nodes]])
        elif key == "inv_1_1":
            return torch.tensor(
                [
                    [num_nodes],
                    [num_nodes],
                    [num_nodes],
                ]
            )
        elif key == "inv_1_2":
            return torch.tensor(
                [
                    [num_nodes],
                    [num_nodes],
                    [num_nodes],
                ]
            )
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

    def __init__(self, lifters: Union[list[callable], callable], dim: int = 2):
        if isinstance(lifters, list):
            self.lifters = lifters
        else:
            self.lifters = [lifters]
        self.dim = dim

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
            # cellrank(cell, memberships)
            if len(cell) <= self.dim + 1:
                cell_dict[len(cell) - 1][cell] = memberships

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
        for i in range(self.dim + 1):
            for j in range(i, self.dim + 1):
                if i != j:
                    matrix = cc.incidence_matrix(rank=i, to_rank=j)

                else:
                    index, matrix = cc.adjacency_matrix(rank=i, via_rank=i + 1, index=True)
                    idx_to_cell[i] = {v: sorted(k) for k, v in index.items()}

                if np.array_equal(matrix, np.zeros(1)):
                    matrix = torch.zeros(2, 0).long()
                else:
                    matrix = sparse_to_dense(matrix)

                adj[f"{i}_{j}"] = matrix

        # for each adjacency/incidence, store the nodes to be used for computing invariant geometric features
        inv = dict()
        for i in range(self.dim):
            inv[f"{i}_{i}"] = []
            inv[f"{i}_{i+1}"] = []

        for i in range(self.dim):
            for j in [i, i + 1]:
                neighbors = adj[f"{i}_{j}"]
                for connection in neighbors.t():
                    idx_a, idx_b = connection[0], connection[1]
                    cell_a = idx_to_cell[i][idx_a.item()]
                    cell_b = idx_to_cell[j][idx_b.item()]
                    shared = [node for node in cell_a if node in cell_b]
                    only_in_a = [node for node in cell_a if node not in shared]
                    only_in_b = [node for node in cell_b if node not in shared]
                    inv_nodes = shared + only_in_b + only_in_a
                    inv[f"{i}_{j}"].append(torch.tensor(inv_nodes))

        for k, v in inv.items():
            if len(v) == 0:
                i, j = k.split("_")
                num_nodes = min(int(i), int(j)) + 2
                inv[k] = torch.zeros(num_nodes, 0).long()
            else:
                inv[k] = torch.stack(v, dim=1)

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


def map_to_tensors(
    input_dict: dict[int, dict[frozenset[int], list[bool]]], num_lifters: int
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    x_dict, mem_dict = {}, {}
    for rank, cell_lifter_map in input_dict.items():
        if cell_lifter_map:
            x = torch.tensor([sorted(cell) for cell in cell_lifter_map.keys()], dtype=torch.float32)
            mem = torch.tensor(list(cell_lifter_map.values()), dtype=torch.bool)
        else:
            # For empty lists, create tensors with the specified size
            # The size is (0, rank + 1) based on your example
            x = torch.empty((0, rank + 1), dtype=torch.float32)
            mem = torch.empty((0, num_lifters), dtype=torch.bool)
        x_dict[rank] = x
        mem_dict[rank] = mem
    return x_dict, mem_dict


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

    from simplicial_data.lifts import rips_lift

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
