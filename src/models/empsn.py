from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_add


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
        visible_dims: list[int],
        num_hidden: int,
        num_features_map: dict[str, int],
        lean: bool = True,
    ) -> None:
        super().__init__()
        self.adjacencies = adjacencies
        self.num_features_map = num_features_map
        self.visible_dims = visible_dims
        self.lean = lean

        # messages
        self.message_passing = nn.ModuleDict(
            {
                adj: SimplicialEGNNLayer(num_hidden, self.num_features_map[adj], lean=lean)
                for adj in adjacencies
            }
        )

        # updates
        self.update = nn.ModuleDict()
        for dim in self.visible_dims:
            factor = 1 + sum([adj_type[2] == str(dim) for adj_type in adjacencies])
            update_layers = [nn.Linear(factor * num_hidden, num_hidden), nn.BatchNorm1d(num_hidden)]
            if not self.lean:
                update_layers.extend(
                    [nn.SiLU(), nn.Linear(num_hidden, num_hidden)], nn.BatchNorm1d(num_hidden)
                )
            self.update[str(dim)] = nn.Sequential(*update_layers)

    def forward(
        self, x: Dict[str, Tensor], adj: Dict[str, Tensor], inv: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        # pass the different messages of all adjacency types
        mes = {
            adj_type: self.message_passing[adj_type](
                x=(x[adj_type[0]], x[adj_type[2]]), index=adj[adj_type], edge_attr=inv[adj_type]
            )
            for adj_type in self.adjacencies
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
    def __init__(self, num_hidden, num_inv, lean: bool = True):
        super().__init__()
        self.lean = lean
        message_mlp_layers = [
            nn.Linear(2 * num_hidden + num_inv, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.SiLU(),
        ]
        if not self.lean:
            message_mlp_layers.extend(
                [nn.Linear(num_hidden, num_hidden), nn.BatchNorm1d(num_hidden), nn.SiLU()]
            )
        self.message_mlp = nn.Sequential(*message_mlp_layers)
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
