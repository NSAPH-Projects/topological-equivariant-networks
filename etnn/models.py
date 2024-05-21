from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from copy import deepcopy

from etnn.layers import ETNNLayer, etnn_block

# from etnn.invariants import compute_invariants
from etnn.utils import (
    compute_invariants,
    compute_invariants2,
    fast_agg_indices,
)


def prepare_agg_indices(cell_ind, adj, max_cell_size=100):
    agg_indices = {}

    # first take sub sample of the cells if larger than max size
    cell_ind = deepcopy(cell_ind)
    for rank, cells in cell_ind.items():
        for j, c in enumerate(cells):
            if len(c) > max_cell_size:
                subsample = np.random.choice(c, max_cell_size, replace=False)
                cell_ind[rank][j] = subsample.tolist()

        # create agg indices for rank
        # transform on the format of atoms/sizes needed by the helper functin
        atoms = np.concatenate(cells)
        sizes = np.array([len(c) for c in cells])
        indices = fast_agg_indices(atoms, sizes, atoms, sizes)
        agg_indices[rank] = indices
        # [x.tolist() for x in indices]

    # for each aggregation indices for edges
    for adj_type, edges in adj.items():
        # get the indices of the cells
        send_rank, recv_rank = adj_type.split("_")[:2]

        # get the cells of sender and receiver
        cells_send = [cell_ind[send_rank][i] for i in edges[0]]
        cells_recv = [cell_ind[recv_rank][i] for i in edges[1]]

        # transform on the format of atoms/sizes needed by the helper functin
        atoms_send = np.concatenate(cells_send)
        sizes_send = np.array([len(c) for c in cells_send])
        atoms_recv = np.concatenate(cells_recv)
        sizes_recv = np.array([len(c) for c in cells_recv])

        # get the indices of the cells
        indices = fast_agg_indices(atoms_send, sizes_send, atoms_recv, sizes_recv)
        agg_indices[adj_type] = indices
        # [x.tolist() for x in indices]

    return cell_ind, agg_indices


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
        depth_etnn_layers=1,
        equivariant: bool = False,
        num_readout_layers: int = 2,
        hausdorff: bool = True,
        invariants: bool = True,
        diff_high_order: bool = False,
        pos_in_readout: bool = False,
        pos_dim: int = 2,
        has_virtual_node: bool = False,  # if so, removes Batch norm from top rank
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.adjacencies = adjacencies
        self.equivariant = equivariant
        # self.hausdorff = hausdorff
        self.pos_in_readout = pos_in_readout
        self.invariants = invariants
        self.num_invariants = (5 if hausdorff else 3) * invariants
        self.visible_dims = list(num_features_per_rank.keys())
        self.num_inv_fts_map = {k: self.num_invariants for k in adjacencies}
        self.max_dim = max(self.visible_dims)
        self.finv = partial(
            compute_invariants2, hausdorff=hausdorff, diff_high_order=diff_high_order
        )
        self.dropout = dropout

        self.feature_embedding = nn.ModuleDict()
        out_dim = num_hidden if num_readout_layers > 0 else num_out
        out_act = nn.Identity if num_readout_layers > 0 else nn.Sigmoid
        for dim in self.visible_dims:
            self.feature_embedding[str(dim)] = nn.Sequential(
                nn.Linear(num_features_per_rank[dim], out_dim), out_act()
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
                    has_virtual_node=has_virtual_node,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # readout layers
        self.readout = nn.ModuleDict()
        self.num_readout_layers = num_readout_layers

        for dim in self.visible_dims:
            dim_in = num_hidden
            if dim == "0" and pos_in_readout:
                dim_in += pos_dim
            self.readout[str(dim)] = etnn_block(
                dim_in,
                num_hidden,
                num_out,
                num_readout_layers,
                batchnorm=False,
                dropout=dropout,
                last_act=nn.Identity,
            )

        # initialize all layers
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.trunc_normal_(param, std=0.2)
            if "bias" in name:
                nn.init.constant_(param, 0)
        # for layer in self.layers:
        #     for m in layer.modules():
        #         if isinstance(m, nn.Linear):
        #             nn.init.trunc_normal_(m.weight, std=0.2)
        #             if m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)

    def forward(self, graph: Data) -> Tensor:
        # nest cell indices from cat format
        cell_ind = {}
        for r in self.visible_dims:
            lengths = getattr(graph, f"lengths_{r}")
            cat_cells = getattr(graph, f"cell_{r}")
            ind = torch.split(cat_cells, lengths.tolist())
            cell_ind[str(r)] = [c.cpu().numpy().tolist() for c in ind]

        adj = {
            adj_type: getattr(graph, f"adj_{adj_type}") for adj_type in self.adjacencies
        }

        # embed features and E(n) invariant information
        x = {str(i): getattr(graph, f"x_{i}") for i in self.visible_dims}
        x = {dim: self.feature_embedding[dim](feature) for dim, feature in x.items()}

        # message passing
        pos = graph.pos
        if self.invariants:
            # inv = compute_invariants(
            #     cell_ind, pos, adj, self.hausdorff, max_cell_size=100
            # )
            # get aggregation indices
            cell_ind_inv, agg_indices = prepare_agg_indices(
                cell_ind, adj, max_cell_size=100
            )
            inv = self.finv(cell_ind_inv, pos, adj, agg_indices)
        else:
            inv = {adj_type: None for adj_type in self.adjacencies}

        for _, layer in enumerate(self.layers):
            if not self.equivariant:
                x, _ = layer(x, adj, pos, inv)
            else:
                x, pos = layer(x, adj, pos, inv)
                # inv = compute_invariants(
                #     cell_ind, pos, adj, self.hausdorff, max_cell_size=100
                # )
                if self.invariants:
                    inv = self.finv(cell_ind_inv, pos, adj, agg_indices)

        # if dropout
        if self.dropout > 0:
            x = {dim: F.dropout(feat, p=self.dropout) for dim, feat in x.items()}

        # readout
        if self.pos_in_readout:
            x["0"] = torch.cat([x["0"], pos], dim=1)

        if self.num_readout_layers > 0:
            x = {dim: self.readout[dim](feat) for dim, feat in x.items()}

        return x

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
