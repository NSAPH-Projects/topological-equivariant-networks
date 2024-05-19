from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from etnn.utils import scatter_add

# from torch_scatter import scatter_add


def etnn_block(
    dim_in: int,
    dim_hidden: int,
    dim_out: int,
    num_layers=1,
    act=nn.SiLU,
    last_act: nn.Module | None = None,
    batchnorm: bool = True,
    batchnorm_always: bool = True,
):
    if act is None:
        act = nn.Identity
    layers = []
    if batchnorm:
        layers.append(nn.BatchNorm1d(dim_in))
    if last_act is None:
        last_act = act
    dim_prev = dim_in
    for i in range(num_layers):
        dim_next = dim_out if i == num_layers - 1 else dim_hidden
        act_next = act if i != num_layers - 1 else last_act
        layers.extend([nn.Linear(dim_prev, dim_next), act_next()])
        if batchnorm and batchnorm_always and i != num_layers - 1:
            layers.append(nn.BatchNorm1d(dim_next))
        dim_prev = dim_next
    return nn.Sequential(*layers)


class ETNNLayer(nn.Module):
    """
    Layer of E(n) Equivariant Message Passing Layer for CC

    A message passing layer is added for each type of adjacency to the message_passing dict. For
    each simplex, a state is found by concatening the messages sent to that simplex, e.g. we update
    an edge by concatenating the messages from nodes, edges, and triangles. The simplex is update by
    passing this state through an MLP as found in the update dict.
    """

    def __init__(
        self,
        adjacencies: list[str],
        visible_dims: list[int],
        num_hidden: int,
        num_features_map: dict[str, int],
        num_layers: int = 1,
        equivariant: bool = True,
    ) -> None:
        super().__init__()
        self.adjacencies = adjacencies
        self.num_features_map = num_features_map
        self.visible_dims = visible_dims
        self.equivariant = equivariant

        # messages
        self.message_passing = nn.ModuleDict(
            {
                adj: ETNNMessagerLayer(num_hidden, self.num_features_map[adj])
                for adj in adjacencies
            }
        )

        # updates
        self.update = nn.ModuleDict()
        self.factor = {}
        for dim in self.visible_dims:
            f = 1 + sum([adj_type[2] == str(dim) for adj_type in adjacencies])
            self.factor[str(dim)] = f
            self.update[str(dim)] = etnn_block(
                f * num_hidden, num_hidden, num_hidden, num_layers, last_act=nn.Identity
            )
        if self.equivariant:
            self.pos_update = nn.Linear(num_hidden, 1)
            nn.init.trunc_normal_(self.pos_update.weight, std=0.02)
            nn.init.constant_(self.pos_update.bias, 0)

    def get_pos_delta(
        self, pos: Tensor, mes: dict[str, Tensor], adj: dict[str, Tensor]
    ) -> Tensor:
        sender = adj["0_0"][0]
        receiver = adj["0_0"][1]
        pos_diff = pos[sender] - pos[receiver]
        pos_upd = self.pos_update(mes["0_0"][receiver])
        pos_delta = pos_diff * pos_upd
        # collect the pos_delta for each node: going from [num_edges, num_hidden] to [num_nodes, num_hidden]
        new_pos_delta = scatter_add(pos_delta, sender, dim=0, dim_size=pos.size(0))
        return new_pos_delta

    def forward(
        self,
        x: dict[str, Tensor],
        adj: dict[str, Tensor],
        pos: Tensor,
        inv: dict[str, Optional[Tensor]],
    ) -> tuple[dict[str, Tensor], Tensor]:
        # pass the different messages of all adjacency types
        mes = {}
        for adj_type, layer in self.message_passing.items():
            send_key, rec_key = adj_type.split("_")[:2]
            mes[adj_type] = layer(
                x[send_key], x[rec_key], index=adj[adj_type], edge_attr=inv[adj_type]
            )

        # find update states through concatenation, update and add residual connection
        h = x.copy()
        for adj_type, adj_mes in mes.items():
            send_key, rec_key = adj_type.split("_")[:2]
            h[rec_key] = torch.cat((h[rec_key], adj_mes), dim=1)
        h = {dim: layer_(h[dim]) for dim, layer_ in self.update.items()}
        x = {dim: feature + h[dim] for dim, feature in x.items()}

        if self.equivariant:
            pos_delta = self.get_pos_delta(pos, mes, adj)
            C = 0.1
            pos = pos + C * pos_delta

        return x, pos


class ETNNMessagerLayer(nn.Module):
    def __init__(self, num_hidden: int, num_inv: int, num_layers=1):
        super().__init__()
        self.message_mlp = etnn_block(
            2 * num_hidden + num_inv, num_hidden, num_hidden, num_layers
        )
        self.edge_inf_mlp = nn.Sequential(nn.Linear(num_hidden, 1), nn.Sigmoid())

    def forward(
        self, x_send: Tensor, x_rec: Tensor, index: Tensor, edge_attr: Optional[Tensor]
    ):
        index_send = index[0]
        index_rec = index[1]
        sim_send, sim_rec = x_send[index_send], x_rec[index_rec]
        state = torch.cat((sim_send, sim_rec), dim=1)
        if edge_attr is not None:
            state = torch.cat((state, edge_attr), dim=1)

        messages = self.message_mlp(state)
        edge_weights = self.edge_inf_mlp(messages)
        messages_aggr = scatter_add(
            messages * edge_weights, index_rec, dim=0, dim_size=x_rec.size(0)
        )

        return messages_aggr
