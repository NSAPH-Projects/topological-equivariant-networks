import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool

from etnn.layers import ETNNLayer
from etnn.utils import compute_centroids, compute_invariants


class ETNN(nn.Module):
    """
    E(n) Equivariant Topological Neural Networks
    """

    def __init__(
        self,
        num_features_per_rank: dict[int, int],
        num_hidden: int,
        num_out: int,
        num_layers: int,
        max_dim: int,
        adjacencies: list[str],
        initial_features: str,
        visible_dims: list[int] | None,
        normalize_invariants: bool,
        compute_invariants: callable = compute_invariants,
        global_pool: bool = True,
        equivariant_update: bool = False,
        num_pre_pool_layers: int = 2,
        num_post_pool_layer: int = 2,
    ) -> None:
        super().__init__()

        self.initial_features = initial_features
        self.compute_invariants = compute_invariants
        self.num_inv_fts_map = self.compute_invariants.num_features_map
        self.max_dim = max_dim
        self.adjacencies = adjacencies
        self.normalize_invariants = normalize_invariants
        self.global_pool = global_pool

        if visible_dims is not None:
            self.visible_dims = visible_dims
        else:
            self.visible_dims = list(range(max_dim + 1))

        # layers
        if self.normalize_invariants:
            self.inv_normalizer = nn.ModuleDict(
                {
                    adj: nn.BatchNorm1d(self.num_inv_fts_map[adj])
                    for adj in self.adjacencies
                }
            )

        self.feature_embedding = nn.ModuleDict(
            {
                str(dim): nn.Linear(num_features_per_rank[dim], num_hidden)
                for dim in self.visible_dims
            }
        )
        self.layers = nn.ModuleList(
            [
                ETNNLayer(
                    self.adjacencies,
                    self.visible_dims,
                    num_hidden,
                    self.num_inv_fts_map,
                )
                for _ in range(num_layers)
            ]
        )

        self.pre_pool = nn.ModuleDict()

        for dim in visible_dims:
            out_dim = num_hidden if not global_pool else num_out
            self.pre_pool[str(dim)] = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Linear(num_hidden, num_hidden),
                        nn.SiLU(),
                    )
                    for _ in range(num_pre_pool_layers - 1)
                ],
                nn.Linear(num_hidden, out_dim),
            )

        if global_pool:
            self.post_pool = nn.Sequential(
                nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Linear(len(self.visible_dims) * num_hidden, num_hidden),
                            nn.SiLU(),
                        )
                        for _ in range(num_post_pool_layer - 1)
                    ],
                    nn.Linear(num_hidden, num_out),
                )
            )

    def forward(self, graph: Data) -> Tensor:
        device = graph.pos.device

        # make nested tensor for cell_ind
        # cell_ind = {str(i): getattr(graph, f"cell_{i}") for i in self.visible_dims}
        cell_ind = {}
        for r in self.visible_dims:
            lengths = getattr(graph, f"lengths_{r}")
            cat_cells = getattr(graph, f"cell_{r}")
            cell_ind[str(r)] = torch.split(cat_cells, lengths.tolist())

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
        features = {}
        for feature_type in self.initial_features:
            features[feature_type] = {}
            for i in self.visible_dims:
                if feature_type == "node":
                    # features[feature_type][str(i)] = compute_centroids(
                    #     cell_ind[str(i)], graph.x
                    # )
                    raise NotImplementedError
                elif feature_type == "mem":
                    mem_i = getattr(graph, f"mem_{i}")
                    features[feature_type][str(i)] = mem_i.float()
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

        cell_batch = {
            str(i): getattr(graph, f"cell_{i}_batch") for i in self.visible_dims
        }

        # embed features and E(n) invariant information
        x = {dim: self.feature_embedding[dim](feature) for dim, feature in x.items()}
        inv = self.compute_invariants(cell_ind, graph.pos, adj, inv_ind, device)
        if self.normalize_invariants:
            inv = {
                adj: self.inv_normalizer[adj](feature) for adj, feature in inv.items()
            }
        # message passing
        for layer in self.layers:
            x = layer(x, adj, inv)

        # read out
        x = {dim: self.pre_pool[dim](feature) for dim, feature in x.items()}

        if self.global_pool:
            # create one dummy node with all features equal to zero for each graph and each rank
            batch_size = graph.y.shape[0]
            x = {
                dim: torch.cat(
                    (feature, torch.zeros(batch_size, feature.shape[1]).to(device)),
                    dim=0,
                )
                for dim, feature in x.items()
            }
            cell_batch = {
                dim: torch.cat((indices, torch.tensor(range(batch_size)).to(device)))
                for dim, indices in cell_batch.items()
            }
            x = {
                dim: global_add_pool(x[dim], cell_batch[dim])
                for dim, feature in x.items()
            }
            state = torch.cat(
                tuple([feature for dim, feature in x.items()]),
                dim=1,
            )
            out = self.post_pool(state)
            out = torch.squeeze(out)
        else:
            out = x

        return out

    def __str__(self):
        return f"ETNN ({self.type})"


if __name__ == "__main__":
    from etnn.pm25 import SpatialCC
    from etnn.combinatorial_complexes import CombinatorialComplexCollater
    from torch.utils.data import DataLoader
    import time

    dataset = SpatialCC(root="data", force_reload=True)
    follow_batch = ["cell_0", "cell_1", "cell_2"]
    collate_fn = CombinatorialComplexCollater(dataset, follow_batch=follow_batch)
    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        num_workers=0,
        persistent_workers=True,
        batch_size=1,
    )

    # get a sample data point to infer the complex properties
    # TODO: this metadata should be stored in the SpatialCC object
    data = next(iter(loader))
    num_features_per_rank = {
        int(k.split("_")[1]): v.shape[1] for k, v in data.items() if "x_" in k
    }
    max_dim = max(num_features_per_rank.keys())

    model = ETNN(
        num_features_per_rank=num_features_per_rank,
        num_hidden=8,
        num_out=1,
        num_layers=4,
        num_post_pool_layer=1,
        max_dim=max_dim,
        adjacencies=["0_0", "0_1", "1_1", "1_2", "2_2"],
        initial_features=["hetero"],
        visible_dims=[0, 1, 2],
        normalize_invariants=True,
        global_pool=False,
    )

    for batch in loader:
        start = time.time()
        out = model(batch)
        print("Time taken precompile: ", time.time() - start)
