import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool

from etnn.layers import ETNNLayer
from etnn.utils import compute_centroids, compute_invariants


class ETNN(nn.Module):
    """
    Topological E(n) Equivariant Networks (TEN)
    """

    def __init__(
        self,
        num_features_per_rank: dict[int, int],
        num_hidden: int,
        num_out: int,
        num_layers: int,
        # max_dim: int,
        adjacencies: list[str],
        initial_features: str,
        visible_dims: list[int] | None,
        normalize_invariants: bool,
        compute_invariants: callable = compute_invariants,
        batch_norm: bool = False,
        lean: bool = True,
    ) -> None:
        super().__init__()

        self.initial_features = initial_features
        self.compute_invariants = compute_invariants
        self.num_inv_fts_map = self.compute_invariants.num_features_map
        # self.max_dim = max_dim
        self.adjacencies = adjacencies
        self.normalize_invariants = normalize_invariants
        self.batch_norm = batch_norm
        self.lean = lean
        max_dim = max(num_features_per_rank.keys())

        if visible_dims is not None:
            self.visible_dims = visible_dims
            # keep only adjacencies that are compatible with visible_dims
            self.adjacencies = []
            for adj in adjacencies:
                max_rank = max(int(rank) for rank in adj.split("_")[:2])
                if max_rank in visible_dims:
                    self.adjacencies.append(adj)
        else:
            self.visible_dims = list(range(max_dim + 1))
            self.adjacencies = adjacencies

        # layers
        if self.normalize_invariants:
            self.inv_normalizer = nn.ModuleDict(
                {
                    adj: nn.BatchNorm1d(self.num_inv_fts_map[adj], affine=False)
                    for adj in self.adjacencies
                }
            )

        embedders = {}
        for dim in self.visible_dims:
            embedder_layers = [nn.Linear(num_features_per_rank[dim], num_hidden)]
            if self.batch_norm:
                embedder_layers.append(nn.BatchNorm1d(num_hidden))
            embedders[str(dim)] = nn.Sequential(*embedder_layers)
        self.feature_embedding = nn.ModuleDict(embedders)

        self.layers = nn.ModuleList(
            [
                ETNNLayer(
                    self.adjacencies,
                    self.visible_dims,
                    num_hidden,
                    self.num_inv_fts_map,
                    self.batch_norm,
                    self.lean,
                )
                for _ in range(num_layers)
            ]
        )

        self.pre_pool = nn.ModuleDict()

        for dim in visible_dims:
            pre_pool_layers = [nn.Linear(num_hidden, num_hidden), nn.SiLU()]
            if not self.lean:
                pre_pool_layers.append(nn.Linear(num_hidden, num_hidden))
            self.pre_pool[str(dim)] = nn.Sequential(*pre_pool_layers)

        self.post_pool = nn.Sequential(
            nn.Linear(len(self.visible_dims) * num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_out),
        )

    def forward(self, graph: Data) -> Tensor:
        device = graph.pos.device
        # cell_ind = {str(i): getattr(graph, f"cell_{i}") for i in self.visible_dims}
        cell_ind = {
            str(i): graph.cell_list(i, format="padded") for i in self.visible_dims
        }

        mem = {i: getattr(graph, f"mem_{i}") for i in self.visible_dims}

        adj = {
            adj_type: getattr(graph, f"adj_{adj_type}")
            for adj_type in self.adjacencies
            if hasattr(graph, f"adj_{adj_type}")
        }

        # inv_ind = {
        #     adj_type: getattr(graph, f"inv_{adj_type}")
        #     for adj_type in self.adjacencies
        #     if hasattr(graph, f"inv_{adj_type}")
        # }

        # compute initial features
        features = {}
        for feature_type in self.initial_features:
            features[feature_type] = {}
            for i in self.visible_dims:
                if feature_type == "node":
                    features[feature_type][str(i)] = compute_centroids(
                        cell_ind[str(i)], graph.x
                    )
                elif feature_type == "mem":
                    features[feature_type][str(i)] = mem[i].float()
                elif feature_type == "hetero":
                    features[feature_type][str(i)] = getattr(graph, f"x_{i}")

        x = {
            str(i): torch.cat(
                [
                    features[feature_type][str(i)]
                    for feature_type in self.initial_features
                ],
                dim=1,
            )
            for i in self.visible_dims
        }

        # embed features and E(n) invariant information
        x = {dim: self.feature_embedding[dim](feature) for dim, feature in x.items()}
        inv = self.compute_invariants(cell_ind, graph.pos, adj, device)
        if self.normalize_invariants:
            inv = {
                adj: self.inv_normalizer[adj](feature) for adj, feature in inv.items()
            }
        # message passing
        for layer in self.layers:
            x = layer(x, adj, inv)

        # read out
        x = {dim: self.pre_pool[dim](feature) for dim, feature in x.items()}

        # create one dummy node with all features equal to zero for each graph and each rank
        cell_batch = {
            str(i): _slices_to_batch(graph._slice_dict[f"slices_{i}"])
            for i in self.visible_dims
        }
        x = {
            dim: global_add_pool(x[dim], cell_batch[dim]) for dim, feature in x.items()
        }
        state = torch.cat(
            tuple([feature for dim, feature in x.items()]),
            dim=1,
        )
        out = self.post_pool(state)
        out = torch.squeeze(out, -1)

        return out

    def __str__(self):
        return f"ETNN ({self.type})"


def _slices_to_batch(slices: Tensor) -> Tensor:
    """This auxiliary function converts the the a slices object, which
    is a property of the torch_geometric.Data object, to a batch index,
    which is a tensor of the same length as the number of nodes/cells where
    each element is the index of the batch to which the node/cell belongs.
    For example, if the slices object is torch.tensor([0, 3, 5, 7]),
    then the output of this function is torch.tensor([0, 0, 0, 1, 1, 2, 2]).
    """
    n = slices.size(0) - 1
    return torch.arange(n, device=slices.device).repeat_interleave(
        (slices[1:] - slices[:-1]).long()
    )
