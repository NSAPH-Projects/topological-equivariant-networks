from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from etnn import utils


class ETNNLayer(nn.Module):
    def __init__(
        self,
        adjacencies: List[str],
        visible_dims: list[int],
        num_hidden: int,
        num_features_map: dict[str, int],
        batch_norm: bool = False,
        lean: bool = True,
        pos_update: bool = False,
    ) -> None:
        super().__init__()
        self.adjacencies = adjacencies
        self.num_features_map = num_features_map
        self.visible_dims = visible_dims
        self.batch_norm = batch_norm
        self.lean = lean
        self.pos_update = pos_update

        # messages
        self.message_passing = nn.ModuleDict(
            {
                adj: BaseMessagePassingLayer(
                    num_hidden,
                    self.num_features_map[adj],
                    batch_norm=batch_norm,
                    lean=lean,
                )
                for adj in adjacencies
            }
        )

        # state update
        self.update = nn.ModuleDict()
        for dim in self.visible_dims:
            factor = 1 + sum([adj_type[2] == str(dim) for adj_type in adjacencies])
            update_layers = [nn.Linear(factor * num_hidden, num_hidden)]
            if self.batch_norm:
                update_layers.append(nn.BatchNorm1d(num_hidden))
            if not self.lean:
                extra_layers = [nn.SiLU(), nn.Linear(num_hidden, num_hidden)]
                if self.batch_norm:
                    extra_layers.append(nn.BatchNorm1d(num_hidden))
                update_layers.extend(extra_layers)
            self.update[str(dim)] = nn.Sequential(*update_layers)

        # position update
        if pos_update:
            self.pos_update_wts = nn.Linear(num_hidden, 1, bias=False)
            nn.init.trunc_normal_(self.pos_update_wts.weight, std=0.02)

    def radial_pos_update(
        self, pos: Tensor, mes: dict[str, Tensor], adj: dict[str, Tensor]
    ) -> Tensor:
        # find the key corresponding to the 0_0_x adjacency
        key = [k for k in adj if k[0] == "0" and k[2] == "0"][0]
        send, recv = adj[key]
        wts = self.pos_update_wts(mes[key][recv])

        # collect the pos_delta for each node: going from
        # [num_edges, num_hidden] to [num_nodes, num_hidden]
        delta = utils.scatter_add(
            (pos[send] - pos[recv]) * wts, send, dim=0, dim_size=pos.size(0)
        )
        return pos + 0.1 * delta

    def forward(
        self,
        x: Dict[str, Tensor],
        adj: Dict[str, Tensor],
        inv: Dict[str, Tensor],
        pos: Tensor,
    ) -> Dict[str, Tensor]:
        # pass the different messages of all adjacency types
        mes = {
            adj_type: self.message_passing[adj_type](
                x=(x[adj_type[0]], x[adj_type[2]]),
                index=adj[adj_type],
                edge_attr=inv[adj_type],
            )
            for adj_type in self.adjacencies
        }

        # find update states through concatenation, update and add residual connection
        h = {
            dim: torch.cat(
                [feature]
                + [adj_mes for adj_type, adj_mes in mes.items() if adj_type[2] == dim],
                dim=1,
            )
            for dim, feature in x.items()
        }
        h = {dim: self.update[dim](feature) for dim, feature in h.items()}
        x = {dim: feature + h[dim] for dim, feature in x.items()}

        if self.pos_update:
            pos = self.radial_pos_update(pos, mes, adj)

        return x, pos


class BaseMessagePassingLayer(nn.Module):
    def __init__(
        self, num_hidden, num_inv, batch_norm: bool = False, lean: bool = True
    ):
        super().__init__()
        self.batch_norm = batch_norm
        self.lean = lean
        message_mlp_layers = [
            nn.Linear(2 * num_hidden + num_inv, num_hidden),
            nn.SiLU(),
        ]
        if self.batch_norm:
            message_mlp_layers.insert(1, nn.BatchNorm1d(num_hidden))

        if not self.lean:
            extra_layers = [
                nn.Linear(num_hidden, num_hidden),
                nn.SiLU(),
            ]
            if self.batch_norm:
                extra_layers.insert(1, nn.BatchNorm1d(num_hidden))
            message_mlp_layers.extend(extra_layers)
        self.message_mlp = nn.Sequential(*message_mlp_layers)
        self.edge_inf_mlp = nn.Sequential(nn.Linear(num_hidden, 1), nn.Sigmoid())

    def forward(self, x, index, edge_attr):
        index_send, index_rec = index
        x_send, x_rec = x
        sim_send, sim_rec = x_send[index_send], x_rec[index_rec]
        state = torch.cat((sim_send, sim_rec, edge_attr), dim=1)

        messages = self.message_mlp(state)
        edge_weights = self.edge_inf_mlp(messages)
        messages_aggr = utils.scatter_add(
            messages * edge_weights, index_rec, dim=0, dim_size=x_rec.shape[0]
        )

        return messages_aggr
