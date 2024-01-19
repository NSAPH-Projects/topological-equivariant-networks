from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from simplicial_data.rips_lift import rips_lift


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

    def __init__(self, dim=2, dis=3):
        self.dim = dim
        self.dis = dis

    def __call__(self, graph: Data) -> SimplicialComplexData:
        if not torch.is_tensor(graph.x):
            graph.x = torch.nn.functional.one_hot(graph.z, torch.max(graph.z) + 1)

        assert torch.is_tensor(graph.pos) and torch.is_tensor(graph.x)

        # get relevant dictionaries using the Rips complex based on the geometric graph/point cloud
        x_dict, adj_dict, inv_dict = rips_lift(graph, self.dim, self.dis)

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


if __name__ == "__main__":
    import random

    from torch_geometric.datasets import QM9

    data = QM9("../datasets/QM9")

    transform_2 = SimplicialTransform(dim=2, dis=2.0)
    transform_3 = SimplicialTransform(dim=2, dis=3.0)
    transform_4 = SimplicialTransform(dim=2, dis=4.0)

    random_graph = random.randint(0, len(data) - 1)

    # transform data[0] to simplicial complex
    sim_2 = transform_2(data[random_graph])
    sim_3 = transform_3(data[random_graph])
    sim_4 = transform_4(data[random_graph])

    print(f"Original Graph: {data[random_graph]}")
    print(f"Simplicial Complex (delta = 2.0): {sim_2}")
    print(f"Simplicial Complex (delta = 3.0): {sim_3}")
    print(f"Simplicial Complex (delta = 4.0): {sim_4}")
