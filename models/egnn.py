import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch import Tensor
from torch_geometric.nn import global_add_pool
from models.utils import MessageLayer, UpdateLayer


class EGNN(nn.Module):
    def __init__(self, num_input: int, num_hidden: int, num_out: int, num_layers: int) -> None:
        super().__init__()

        # layers
        self.feature_embedding = nn.Linear(num_input, num_hidden)
        self.layers = nn.ModuleList([EGNNLayer(num_hidden, 1) for _ in range(num_layers)])
        self.pre_pool = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden)
        )
        self.post_pool = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_out)
        )

    def forward(self, graph: Data) -> Tensor:
        feat_ind, adj, inv_ind, x_batch = graph.x_0, graph.adj_0_0, graph.inv_0_0, graph.x_0_batch

        x = graph.x[feat_ind.long()].squeeze()
        adj = adj.long()

        inv = torch.linalg.norm(graph.pos[inv_ind[0]] - graph.pos[inv_ind[1]], dim=1).unsqueeze(1)

        x = self.feature_embedding(x)
        # message passing
        for layer in self.layers:
            x = layer(x, adj, inv)

        x = self.pre_pool(x)
        x = global_add_pool(x, x_batch)
        out = self.post_pool(x)
        out = torch.squeeze(out)

        return out

    def __str__(self):
        return f"EGNN"


class EGNNLayer(nn.Module):
    def __init__(self, num_hidden: int, num_inv: int) -> None:
        super().__init__()
        self.message_layer = MessageLayer(num_hidden, num_inv)
        self.update_layer = UpdateLayer(num_hidden, 1)

    def forward(self, x, adj, inv):
        message = self.message_layer((x, x), adj, inv)
        x_up = self.update_layer(x, None, message)
        return x + x_up
