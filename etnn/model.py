from functools import partial
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool

from etnn.layers import ETNNLayer
from etnn import utils, invariants
from etnn.invariants import compute_centroids, compute_invariants


class ETNN(nn.Module):
    """
    The E(n)-Equivariant Topological Neural Network (ETNN) model.
    """

    def __init__(
        self,
        num_features_per_rank: dict[int, int],
        num_hidden: int,
        num_out: int,
        num_layers: int,
        adjacencies: list[str],
        initial_features: str,
        visible_dims: list[int] | None,
        normalize_invariants: bool,
        sparse: bool = False,  # invariant sparse computation
        hausdorff: bool = False,
        batch_norm: bool = False,
        lean: bool = True,
        global_pool: bool = False,  # whether or not to use global pooling
    ) -> None:
        super().__init__()

        self.initial_features = initial_features

        # make inv_fts_map for backward compatibility
        self.num_invariants = 5 if hausdorff else 3
        self.num_inv_fts_map = {k: self.num_invariants for k in adjacencies}
        self.adjacencies = adjacencies
        self.normalize_invariants = normalize_invariants
        self.batch_norm = batch_norm
        self.lean = lean
        self.sparse = sparse
        self.global_pool = global_pool
        max_dim = max(num_features_per_rank.keys())

        if sparse:
            fun = invariants.compute_invariants_sparse
        else:
            fun = invariants.compute_invariants

        self.compute_invariants = partial(fun, hausdorff=hausdorff)

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
            if self.global_pool:
                if not self.lean:
                    self.pre_pool[str(dim)] = nn.Sequential(
                        nn.Linear(num_hidden, num_hidden),
                        nn.SiLU(),
                        nn.Linear(num_hidden, num_hidden),
                    )
                else:
                    self.pre_pool[str(dim)] = nn.Linear(num_hidden, num_hidden)
            else:
                if not self.lean:
                    self.pre_pool[str(dim)] = nn.Sequential(
                        nn.Linear(num_hidden, num_hidden),
                        nn.SiLU(),
                        nn.Linear(num_hidden, num_out),
                    )
                else:
                    self.pre_pool[str(dim)] = nn.Linear(num_hidden, num_out)

        if self.global_pool:
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

        # compute initial features
        features = {}
        for feature_type in self.initial_features:
            features[feature_type] = {}
            for i in self.visible_dims:
                if feature_type == "node":
                    features[feature_type][str(i)] = invariants.compute_centroids(
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
        out = {dim: self.pre_pool[dim](feature) for dim, feature in x.items()}

        if self.global_pool:
            # create one dummy node with all features equal to zero for each graph and each rank
            cell_batch = {
                str(i): utils.slices_to_pointer(graph._slice_dict[f"slices_{i}"])
                for i in self.visible_dims
            }
            out = {
                dim: global_add_pool(out[dim], cell_batch[dim])
                for dim, feature in out.items()
            }
            state = torch.cat(
                tuple([feature for dim, feature in out.items()]),
                dim=1,
            )
            out = self.post_pool(state)
            out = torch.squeeze(out, -1)

        return out

    def __str__(self):
        return f"ETNN ({self.type})"
