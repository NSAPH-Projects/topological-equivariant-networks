import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data


from etnn.layers import ETNNLayer, etnn_block
# from etnn.invariants import compute_invariants
from etnn.utils import compute_invariants


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
        adjacencies: list[str],
        depth_etnn_layers = 1,
        equivariant: bool = False,
        num_readout_layers: int = 2,
        haussdorf: bool = True,
        invariants: bool = True,
    ) -> None:
        super().__init__()
        self.adjacencies = adjacencies
        self.equivariant = equivariant
        self.haussdorf = haussdorf
        self.invariants = invariants
        self.num_invariants = (5 if haussdorf else 3) * invariants
        self.visible_dims = list(num_features_per_rank.keys())
        self.num_inv_fts_map = {k: self.num_invariants for k in adjacencies}
        self.max_dim = max(self.visible_dims)

        self.feature_embedding = nn.ModuleDict()
        for dim in self.visible_dims:
            self.feature_embedding[str(dim)] = nn.Sequential(
                nn.Linear(num_features_per_rank[dim], num_hidden), nn.SiLU()
            )

        self.layers = nn.ModuleList(
            [
                ETNNLayer(
                    self.adjacencies,
                    self.visible_dims,
                    num_hidden,
                    self.num_inv_fts_map,
                    equivariant=self.equivariant,
                    num_layers=depth_etnn_layers,
                )
                for _ in range(num_layers)
            ]
        )

        self.readout = nn.ModuleDict()
        for dim in self.visible_dims:
            self.readout[str(dim)] = etnn_block(
                num_hidden,
                num_hidden,
                num_out,
                num_readout_layers,
                batchnorm=False,
                last_act=nn.Identity,
            )

        # initialize all layers
        for layer in self.layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.2)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, graph: Data) -> Tensor:
        # nest cell indices from cat format
        cell_ind = {}
        for r in self.visible_dims:
            lengths = getattr(graph, f"lengths_{r}")
            cat_cells = getattr(graph, f"cell_{r}")
            cell_ind[str(r)] = torch.split(cat_cells, lengths.tolist())

        adj = {
            adj_type: getattr(graph, f"adj_{adj_type}")
            for adj_type in self.adjacencies
        }

        # embed features and E(n) invariant information
        x = {str(i): getattr(graph, f"x_{i}") for i in self.visible_dims}
        x = {dim: self.feature_embedding[dim](feature) for dim, feature in x.items()}

        # message passing
        pos = graph.pos
        if self.invariants:
            inv = compute_invariants(cell_ind, pos, adj, self.haussdorf, max_cell_size=100)
        else:
            inv = {adj_type: None for adj_type in self.adjacencies}

        for _, layer in enumerate(self.layers):
            if not self.equivariant:
                x, _ = layer(x, adj, pos, inv)
            else:
                x, pos = layer(x, adj, pos, inv)
                inv = compute_invariants(cell_ind, pos, adj, self.haussdorf, max_cell_size=100)

        # read out
        out = {dim: self.readout[dim](feature) for dim, feature in x.items()}

        return out

    def __str__(self):
        return f"ETNN ({self.type})"


if __name__ == "__main__":
    from etnn.pm25.utils import SpatialCC
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
        batch_size=1,
    )

    # get a sample data point to infer the complex properties
    # TODO: this metadata should be stored in the SpatialCC object
    data = next(iter(loader))
    num_features_per_rank = {
        int(k.split("_")[1]): v.shape[1] for k, v in data.items() if k.startswith("x_")
    }
    max_dim = max(num_features_per_rank.keys())

    model = ETNN(
        num_features_per_rank=num_features_per_rank,
        num_hidden=4,
        num_out=1,
        num_layers=4,
        num_readout_layers=1,
        adjacencies=["0_0", "0_1", "1_1", "1_2", "2_2"],
        equivariant=True,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    for batch in loader:
        start = time.time()
        out = model(batch)
        print("Time taken: ", time.time() - start)
        loss = 0
        for key, val in out.items():
            loss = loss + val.pow(2).mean()
    loss.backward()
    opt.step()

    print("Done")