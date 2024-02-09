import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool

from models.empsn import EMPSNLayer
from models.utils import compute_centroids, compute_invariants


class TEN(nn.Module):
    """
    Topological E(n) Equivariant Networks (TEN)
    """

    def __init__(
        self,
        num_input: int,
        num_hidden: int,
        num_out: int,
        num_layers: int,
        max_com: str,
        initial_features: str,
        compute_invariants: callable = compute_invariants,
    ) -> None:
        super().__init__()

        self.initial_features = initial_features
        self.compute_invariants = compute_invariants
        self.num_inv_fts_map = self.compute_invariants.num_features_map
        # compute adjacencies
        adjacencies = []
        max_dim = int(max_com[2])  # max_com = 1_2 --> max_dim = 2
        self.max_dim = max_dim
        inc_final = max_com[0] == max_com[2]

        for dim in range(max_dim + 1):
            if dim < max_dim or inc_final:
                adjacencies.append(f"{dim}_{dim}")

            if dim > 0:
                adjacencies.append(f"{dim-1}_{dim}")

        self.adjacencies = adjacencies

        # layers
        self.feature_embedding = nn.Linear(num_input, num_hidden)

        self.layers = nn.ModuleList(
            [
                EMPSNLayer(adjacencies, self.max_dim, num_hidden, self.num_inv_fts_map)
                for _ in range(num_layers)
            ]
        )

        self.pre_pool = nn.ModuleDict()
        for dim in range(self.max_dim + 1):
            self.pre_pool[str(dim)] = nn.Sequential(
                nn.Linear(num_hidden, num_hidden), nn.SiLU(), nn.Linear(num_hidden, num_hidden)
            )
        self.post_pool = nn.Sequential(
            nn.Sequential(
                nn.Linear((max_dim + 1) * num_hidden, num_hidden),
                nn.SiLU(),
                nn.Linear(num_hidden, num_out),
            )
        )

    def forward(self, graph: Data) -> Tensor:
        device = graph.pos.device
        x_ind = {str(i): getattr(graph, f"x_{i}") for i in range(self.max_dim + 1)}

        mem = {i: getattr(graph, f"mem_{i}") for i in range(self.max_dim + 1)}

        adj = {
            adj_type: getattr(graph, f"adj_{adj_type}")
            for adj_type in self.adjacencies
            if hasattr(graph, f"adj_{adj_type}")
        }

        inv_ind = {
            adj_type: getattr(graph, f"inv_{adj_type}")
            for adj_type in self.adjacencies
            if hasattr(graph, f"inv_{adj_type}")
        }

        # compute initial features
        node_features = {}
        for i in range(self.max_dim + 1):
            node_features[str(i)] = compute_centroids(x_ind[str(i)], graph.x)

        mem_features = {str(i): mem[i].float() for i in range(self.max_dim + 1)}

        if self.initial_features == "node":
            x = node_features
        elif self.initial_features == "mem":
            x = mem_features
        elif self.initial_features == "both":
            # concatenate
            x = {
                str(i): torch.cat([node_features[str(i)], mem_features[str(i)]], dim=1)
                for i in range(self.max_dim + 1)
            }

        x_batch = {str(i): getattr(graph, f"x_{i}_batch") for i in range(self.max_dim + 1)}

        # embed features and E(n) invariant information
        x = {dim: self.feature_embedding(feature) for dim, feature in x.items()}
        inv = self.compute_invariants(x_ind, graph.pos, adj, inv_ind, device)

        # message passing
        for layer in self.layers:
            x = layer(x, adj, inv)

        # read out
        x = {dim: self.pre_pool[dim](feature) for dim, feature in x.items()}

        # create one dummy node with all features equal to zero for each graph and each rank
        batch_size = graph.y.shape[0]
        x = {
            dim: torch.cat((feature, torch.zeros(batch_size, feature.shape[1]).to(device)), dim=0)
            for dim, feature in x.items()
        }
        x_batch = {
            dim: torch.cat((indices, torch.tensor(range(batch_size)).to(device)))
            for dim, indices in x_batch.items()
        }

        x = {dim: global_add_pool(x[dim], x_batch[dim]) for dim, feature in x.items()}
        state = torch.cat(tuple([feature for dim, feature in x.items()]), dim=1)
        out = self.post_pool(state)
        out = torch.squeeze(out)

        return out

    def __str__(self):
        return f"TEN ({self.type})"
