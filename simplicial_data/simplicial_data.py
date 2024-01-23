from typing import Any

import numpy as np
import torch
from toponetx.classes import CombinatorialComplex
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from simplicial_data.utils import map_to_tensors, sparse_to_dense


class SimplicialComplexData(Data):
    """
    Abstract simplicial complex class that generalises the pytorch geometric graph (Data).
    Adjacency tensors are stacked in the same fashion as the standard edge_index.
    """

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if "adj" in key:
            i, j = key[4], key[6]
            return torch.tensor(
                [[getattr(self, f"x_{i}").size(0)], [getattr(self, f"x_{j}").size(0)]]
            )
        elif key == "inv_0_0":
            return torch.tensor([[getattr(self, "x_0").size(0)], [getattr(self, "x_0").size(0)]])
        elif key == "inv_0_1":
            return torch.tensor([[getattr(self, "x_0").size(0)], [getattr(self, "x_0").size(0)]])
        elif key == "inv_1_1":
            return torch.tensor(
                [
                    [getattr(self, "x_0").size(0)],
                    [getattr(self, "x_0").size(0)],
                    [getattr(self, "x_0").size(0)],
                ]
            )
        elif key == "inv_1_2":
            return torch.tensor(
                [
                    [getattr(self, "x_0").size(0)],
                    [getattr(self, "x_0").size(0)],
                    [getattr(self, "x_0").size(0)],
                ]
            )
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if "adj" in key or "inv" in key:
            return 1
        else:
            return 0


class SimplicialTransform(BaseTransform):
    """Todo: add
    The adjacency types (adj) are saved as properties, e.g. object.adj_1_2 gives the edge index from
    1-simplices to 2-simplices."""

    def __init__(self, lifter_fct: callable, dim: int = 2):
        self.lift = lifter_fct
        self.dim = dim

    def __call__(self, graph: Data) -> SimplicialComplexData:
        if not torch.is_tensor(graph.x):
            graph.x = torch.nn.functional.one_hot(graph.z, torch.max(graph.z) + 1)

        assert torch.is_tensor(graph.pos) and torch.is_tensor(graph.x)

        # get relevant dictionaries using the Rips complex based on the geometric graph/point cloud
        x_dict, adj_dict, inv_dict = self.get_relevant_dicts(graph)

        sim_com_data = SimplicialComplexData()
        sim_com_data = sim_com_data.from_dict(graph.to_dict())

        for k, v in x_dict.items():
            sim_com_data[f"x_{k}"] = v

        for k, v in adj_dict.items():
            sim_com_data[f"adj_{k}"] = v

        for k, v in inv_dict.items():
            sim_com_data[f"inv_{k}"] = v

        for att in ["edge_attr", "edge_index"]:
            if hasattr(sim_com_data, att):
                sim_com_data.pop(att)

        return sim_com_data

    def get_relevant_dicts(self, graph):
        # compute simplexes
        simplexes = self.lift(graph)

        # compute ranks for each simplex
        simplex_dict = {rank: [] for rank in range(self.dim + 1)}
        for simplex in simplexes:
            simplex_dict[len(simplex) - 1].append(simplex)

        # create x_dict
        x_dict = map_to_tensors(simplex_dict)

        # create the combinatorial complex
        cc = CombinatorialComplex()
        for rank, simplexes in simplex_dict.items():
            for simplex in simplexes:
                cc.add_cell(simplex, rank=rank)

        # compute adjancencies and incidences
        adj = dict()
        for i in range(self.dim):
            for j in range(i, self.dim + 1):
                if i != j:
                    matrix = cc.incidence_matrix(rank=i, to_rank=j)
                else:
                    matrix = cc.adjacency_matrix(rank=i, via_rank=i + 1)

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
                    simplex_a = simplex_dict[i][idx_a]
                    simplex_b = simplex_dict[j][idx_b]
                    shared = [node for node in simplex_a if node in simplex_b]
                    only_in_a = [node for node in simplex_a if node not in shared]
                    only_in_b = [node for node in simplex_b if node not in shared]
                    inv_nodes = shared + only_in_b + only_in_a
                    inv[f"{i}_{j}"].append(torch.tensor(inv_nodes))

        for k, v in inv.items():
            if len(v) == 0:
                i, j = k.split("_")
                num_nodes = min(int(i), int(j)) + 2
                inv[k] = torch.zeros(num_nodes, 0).long()
            else:
                inv[k] = torch.stack(v, dim=1)

        return x_dict, adj, inv


if __name__ == "__main__":
    import functools
    import random

    from torch_geometric.datasets import QM9

    from simplicial_data.lifts import rips_lift

    data = QM9("../datasets/QM9")
    dim = 2
    transform_2 = SimplicialTransform(functools.partial(rips_lift, dim=dim, dis=2.0), dim=dim)
    transform_3 = SimplicialTransform(functools.partial(rips_lift, dim=dim, dis=3.0), dim=dim)
    transform_4 = SimplicialTransform(functools.partial(rips_lift, dim=dim, dis=4.0), dim=dim)

    random_graph = random.randint(0, len(data) - 1)

    # transform data[0] to simplicial complex
    sim_2 = transform_2(data[random_graph])
    sim_3 = transform_3(data[random_graph])
    sim_4 = transform_4(data[random_graph])

    print(f"Original Graph: {data[random_graph]}")
    print(f"Simplicial Complex (delta = 2.0): {sim_2}")
    print(f"Simplicial Complex (delta = 3.0): {sim_3}")
    print(f"Simplicial Complex (delta = 4.0): {sim_4}")
