from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter_add

from models.utils import compute_centroids, compute_invariants


class EMPSN(nn.Module):
    """
    E(n) Equivariant Message Passing Simplicial Networks (EMPSN)
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
        return f"EMPSN ({self.type})"


class EMPSNLayer(nn.Module):
    """
    Layer of E(n) Equivariant Message Passing Simplicial Network.

    A message passing layer is added for each type of adjacency to the message_passing dict. For
    each simplex, a state is found by concatening the messages sent to that simplex, e.g. we update
    an edge by concatenating the messages from nodes, edges, and triangles. The simplex is update by
    passing this state through an MLP as found in the update dict.
    """

    def __init__(
        self,
        adjacencies: List[str],
        max_dim: int,
        num_hidden: int,
        num_features_map: dict[str, int],
    ) -> None:
        super().__init__()
        self.adjacencies = adjacencies
        self.num_features_map = num_features_map

        # messages
        self.message_passing = nn.ModuleDict(
            {
                adj: SimplicialEGNNLayer(num_hidden, self.num_features_map[adj])
                for adj in adjacencies
            }
        )

        # updates
        self.update = nn.ModuleDict()
        for dim in range(max_dim + 1):
            factor = 1 + sum([adj_type[2] == str(dim) for adj_type in adjacencies])
            self.update[str(dim)] = nn.Sequential(
                nn.Linear(factor * num_hidden, num_hidden),
                nn.SiLU(),
                nn.Linear(num_hidden, num_hidden),
            )

    def forward(
        self, x: Dict[str, Tensor], adj: Dict[str, Tensor], inv: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        # pass the different messages of all adjacency types
        mes = {
            adj_type: self.message_passing[adj_type](
                x=(x[adj_type[0]], x[adj_type[2]]), index=index, edge_attr=inv[adj_type]
            )
            for adj_type, index in adj.items()
        }

        # find update states through concatenation, update and add residual connection
        h = {
            dim: torch.cat(
                [feature] + [adj_mes for adj_type, adj_mes in mes.items() if adj_type[2] == dim],
                dim=1,
            )
            for dim, feature in x.items()
        }
        h = {dim: self.update[dim](feature) for dim, feature in h.items()}
        x = {dim: feature + h[dim] for dim, feature in x.items()}

        return x


class SimplicialEGNNLayer(nn.Module):
    def __init__(self, num_hidden, num_inv):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * num_hidden + num_inv, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU(),
        )
        self.edge_inf_mlp = nn.Sequential(nn.Linear(num_hidden, 1), nn.Sigmoid())

    def forward(self, x, index, edge_attr):
        index_send, index_rec = index
        x_send, x_rec = x
        sim_send, sim_rec = x_send[index_send], x_rec[index_rec]
        state = torch.cat((sim_send, sim_rec, edge_attr), dim=1)

        messages = self.message_mlp(state)
        edge_weights = self.edge_inf_mlp(messages)
        messages_aggr = scatter_add(
            messages * edge_weights, index_rec, dim=0, dim_size=x_rec.shape[0]
        )

        return messages_aggr
