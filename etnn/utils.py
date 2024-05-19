import os

# import numba
import random
from argparse import Namespace
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.loader import DataLoader


def scatter_add(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
):
    src_shape = list(src.shape)
    src_shape[dim] = index.max().item() + 1 if dim_size is None else dim_size
    aux = src.new_zeros(src_shape)
    return aux.index_add(dim, index, src)


# def scatter_min(
#     src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None
# ):
#     src_shape = list(src.shape)
#     src_shape[dim] = index.max().item() + 1 if dim_size is None else dim_size
#     aux = src.new_zeros(src_shape)
#     return aux.index_reduce(dim, index, src, reduce="amin", include_self=False)


# def scatter_max(
#     src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None
# ):
#     src_shape = list(src.shape)
#     src_shape[dim] = index.max().item() + 1 if dim_size is None else dim_size
#     aux = src.new_zeros(src_shape)
#     return aux.index_reduce(dim, index, src, reduce="amax", include_self=False)


# @numba.jit(nopython=True)
# def fast_agg_indices(
#     atoms_left: np.ndarray,
#     lengths_left: np.ndarray,
#     atoms_right: np.ndarray,
#     lengths_right: np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     # inputs must be equal size
#     splits_left = np.split(atoms_left, np.cumsum(lengths_left)[:-1])
#     splits_right = np.split(atoms_right, np.cumsum(lengths_right)[:-1])

#     # must return a tuple of 6 arrays
#     n = len(lengths_left)  # must be the same as len(splits_right)
#     m = sum(lengths_left * lengths_right)
#     content_left = np.empty(m, dtype=np.int64)
#     content_right = np.empty(m, dtype=np.int64)
#     index_left = np.empty(m, dtype=np.int64)
#     index_right = np.empty(m, dtype=np.int64)

#     offset_index_left = 0
#     offset_index_right = 0
#     step = 0
#     for i in range(n):
#         n_left = len(splits_left[i])
#         n_right = len(splits_right[i])
#         for j, sj in enumerate(splits_left[i]):
#             for k, sk in enumerate(splits_right[i]):
#                 ind = step + j * n_right + k
#                 content_left[ind] = sj
#                 content_right[ind] = sk
#                 index_left[ind] = offset_index_left + j
#                 index_right[ind] = offset_index_right + k
#         step += n_left * n_right
#         offset_index_left += n_left
#         offset_index_right += n_right

#     step = 0
#     subindex_left = np.empty(sum(lengths_left), dtype=np.int64)
#     for j, sj in enumerate(splits_left):
#         subindex_left[step : step + len(sj)] = j
#         step += len(sj)

#     step = 0
#     subindex_right = np.empty(sum(lengths_right), dtype=np.int64)
#     for j, sj in enumerate(splits_right):
#         subindex_right[step : step + len(sj)] = j
#         step += len(sj)

#     return (
#         content_left,
#         content_right,
#         index_left,
#         index_right,
#         subindex_left,
#         subindex_right,
#     )


# def haussdorff(
#     list_left: np.ndarray,
#     list_right: np.ndarray,
#     pos: torch.FloatTensor | np.ndarray,
# ) -> tuple[torch.FloatTensor | np.ndarray, torch.FloatTensor | np.ndarray]:
#     atoms_left = np.concatenate(list_left)
#     atoms_right = np.concatenate(list_right)
#     lengths_left = np.array([len(x) for x in list_left])
#     length_right = np.array([len(x) for x in list_right])

#     cl, cr, il, ir, sl, sr = fast_agg_indices(
#         atoms_left, lengths_left, atoms_right, length_right
#     )

#     # cast to longtensors
#     cl, cr, il, ir, sl, sr = [
#         torch.LongTensor(x).to(pos.device) for x in (cl, cr, il, ir, sl, sr)
#     ]

#     # # compute haussford by two aggregations
#     pos_left = pos[cl]
#     pos_right = pos[cr]
#     dists = torch.norm(pos_left - pos_right, dim=1)
#     hausdorff_left = scatter_min(dists, il)
#     hausdorff_left = scatter_max(hausdorff_left, sl)
#     hausdorff_right = scatter_min(dists, ir)
#     hausdorff_right = scatter_max(hausdorff_right, sr)

#     return hausdorff_left, hausdorff_right


def get_adjacency_types(
    max_dim: int,
    connectivity: str,
    neighbor_types: list[str],
    visible_dims: list[int] | None,
) -> list[str]:
    """
    Generate a list of adjacency type strings based on the specified connectivity pattern.

    Parameters
    ----------
    max_dim : int
        The maximum dimension (inclusive) for which to generate adjacency types. Represents the
        highest rank of cells in the connectivity pattern.
    connectivity : str
        The connectivity pattern to use. Must be one of the options defined below:
        - "self_and_next" generates adjacencies where each rank is connected to itself and the next
        (higher) rank.
        - "self_and_higher" generates adjacencies where each rank is connected to itself and all
        higher ranks.
        - "self_and_previous" generates adjacencies where each rank is connected to itself and the
        previous (lower) rank.
        - "self_and_lower" generates adjacencies where each rank is connected to itself and all
        lower ranks.
        - "self_and_neighbors" generates adjacencies where each rank is connected to itself, the
        next (higher) rank and the previous (lower) rank.
        - "all_to_all" generates adjacencies where each rank is connected to every other rank,
        including itself.
        - "legacy" ignores the max_dim parameter and returns ['0_0', '0_1', '1_1', '1_2'].
    neighbor_types : list[str]
        The types of adjacency between cells of the same rank. Must be one of the following:
        +1: two cells of same rank i are neighbors if they are both neighbors of a cell of rank i+1
        -1: two cells of same rank i are neighbors if they are both neighbors of a cell of rank i-1
        max: two cells of same rank i are neighbors if they are both neighbors of a cell of max rank
        min: two cells of same rank i are neighbors if they are both neighbors of a cell of min rank
    visible_dims: list[int] | None
        A list of ranks to explicitly represent as nodes. If None, all ranks are represented.

    Returns
    -------
    list[str]
        A list of strings representing the adjacency types for the specified connectivity pattern.
        Each string is in the format "i_j" where "i" and "j" are ranks indicating an adjacency
        from rank "i" to rank "j".

    Raises
    ------
    ValueError
        If `connectivity` is not one of the known connectivity patterns.

    Examples
    --------
    >>> get_adjacency_types(2, "self_and_next", ["+1"])
    ['0_0_1', '0_1', '1_1_2', '1_2']

    >>> get_adjacency_types(2, "self_and_higher", ["-1"])
    ['0_1', '0_2', '1_1_0', '1_2', '2_2_1']

    >>> get_adjacency_types(2, "all_to_all", ["-1", "+1", "max", "min"])
    ['0_0_1', '0_0_2','0_1', '0_2', '1_0', '1_1_0', '1_1_2', '1_2', '2_0', '2_1', '2_2_1', '2_2_0']
    """
    adj_types = []
    if connectivity not in [
        "self",
        "self_and_next",
        "self_and_higher",
        "self_and_previous",
        "self_and_lower",
        "self_and_neighbors",
        "all_to_all",
        "legacy",
    ]:
        raise ValueError(f"{connectivity} is not a known connectivity pattern!")

    if connectivity == "self":
        for i in range(max_dim + 1):
            adj_types.append(f"{i}_{i}")

    elif connectivity == "self_and_next":
        for i in range(max_dim + 1):
            adj_types.append(f"{i}_{i}")
            if i < max_dim:
                adj_types.append(f"{i}_{i+1}")

    elif connectivity == "self_and_higher":
        for i in range(max_dim + 1):
            for j in range(i, max_dim + 1):
                adj_types.append(f"{i}_{j}")

    elif connectivity == "self_and_previous":
        for i in range(max_dim + 1):
            adj_types.append(f"{i}_{i}")
            if i > 0:
                adj_types.append(f"{i}_{i-1}")

    elif connectivity == "self_and_lower":
        for i in range(max_dim + 1):
            for j in range(0, i):
                adj_types.append(f"{i}_{j}")

    elif connectivity == "self_and_neighbors":
        for i in range(max_dim + 1):
            adj_types.append(f"{i}_{i}")
            if i > 0:
                adj_types.append(f"{i}_{i-1}")
            if i < max_dim:
                adj_types.append(f"{i}_{i+1}")

    elif connectivity == "all_to_all":
        for i in range(max_dim + 1):
            for j in range(max_dim + 1):
                adj_types.append(f"{i}_{j}")

    else:
        adj_types = ["0_0", "0_1", "1_1", "1_2"]

    # Add one adjacency type for each neighbor type
    new_adj_types = []
    for adj_type in adj_types:
        i, j = map(int, adj_type.split("_"))
        if i == j:
            for neighbor_type in neighbor_types:
                if neighbor_type == "+1":
                    if i < max_dim:
                        new_adj_types.append(f"{i}_{i}_{i+1}")
                elif neighbor_type == "-1":
                    if i > 0:
                        new_adj_types.append(f"{i}_{i}_{i-1}")
                elif neighbor_type == "max":
                    if i < max_dim:
                        new_adj_types.append(f"{i}_{i}_{max_dim}")
                elif neighbor_type == "min":
                    if i > 0:
                        new_adj_types.append(f"{i}_{i}_0")
        else:
            new_adj_types.append(adj_type)
    new_adj_types = list(set(new_adj_types))
    adj_types = new_adj_types

    # Filter adjacencies with invisible ranks
    if visible_dims is not None:
        adj_types = [
            adj_type
            for adj_type in adj_types
            if all(int(dim) in visible_dims for dim in adj_type.split("_")[:2])
        ]

    return adj_types


def merge_adjacencies(adjacencies: list[str]) -> list[str]:
    """
    Merge all adjacency types i_i_j into a single i_i.

    We merge adjacencies of the form i_i_j into a single adjacency i_i. This is useful when we want
    to represent all rank i neighbors of a cell of rank i as a single adjacency matrix.

    Parameters
    ----------
    adjacencies : list[str]
        A list of adjacency types.

    Returns
    -------
    list[str]
        A list of merged adjacency types.

    """
    return list(set(["_".join(adj_type.split("_")[:2]) for adj_type in adjacencies]))


# def get_model(args: Namespace) -> nn.Module:
#     """Return model based on name."""
#     if args.dataset == "qm9":
#         num_features_per_rank = {0: 35, 1: 28, 2: 30, 3: 20}
#         num_out = 1
#     else:
#         raise ValueError(f"Do not recognize dataset {args.dataset}.")

#     model = ETNN(
#         num_features_per_rank=num_features_per_rank,
#         num_hidden=args.num_hidden,
#         num_out=num_out,
#         num_layers=args.num_layers,
#         max_dim=args.dim,
#         adjacencies=args.processed_adjacencies,
#         initial_features=args.initial_features,
#         normalize_invariants=args.normalize_invariants,
#         visible_dims=args.visible_dims,
#     )
#     return model


def get_loaders(args: Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return dataloaders based on dataset."""
    if args.dataset == "qm9":
        from qm9.utils import generate_loaders_qm9

        train_loader, val_loader, test_loader = generate_loaders_qm9(args)
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized.")

    return train_loader, val_loader, test_loader


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


class MessageLayer(nn.Module):
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
            messages * edge_weights, index=index_rec, dim=0, dim_size=x_rec.size(0)
        )

        return messages_aggr


class UpdateLayer(nn.Module):
    def __init__(self, num_hidden, num_mes):
        super().__init__()
        self.update_mlp = nn.Sequential(
            nn.Linear((num_mes + 1) * num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
        )

    def forward(self, x, bound_mes, upadj_mes):
        state = x

        if torch.is_tensor(bound_mes):
            state = torch.cat((state, bound_mes), dim=1)

        if torch.is_tensor(upadj_mes):
            state = torch.cat((state, upadj_mes), dim=1)

        update = self.update_mlp(state)
        return update


def compute_invariants_3d(feat_ind, pos, adj, inv_ind, device):

    # cast to int
    feat_ind_int = {key: value.int() for key, value in feat_ind.items()}
    inv_ind_int = {key: value.int() for key, value in inv_ind.items()}

    # angles
    angle = {}

    vecs = pos[feat_ind_int["1"][:, 0]] - pos[feat_ind_int["1"][:, 1]]
    send_vec, rec_vec = vecs[adj["1_1"][0]], vecs[adj["1_1"][1]]
    send_norm, rec_norm = torch.linalg.norm(send_vec, ord=2, dim=1), torch.linalg.norm(
        rec_vec, ord=2, dim=1
    )

    dot = torch.sum(send_vec * rec_vec, dim=1)
    cos_angle = dot / (send_norm * rec_norm)
    eps = 1e-6
    angle["1_1"] = torch.arccos(cos_angle.clamp(-1 + eps, 1 - eps)).unsqueeze(1)

    p1, p2, a = (
        pos[inv_ind_int["1_2"][0]],
        pos[inv_ind_int["1_2"][1]],
        pos[inv_ind_int["1_2"][2]],
    )
    v1, v2, b = p1 - a, p2 - a, p1 - p2
    v1_n, v2_n, b_n = (
        torch.linalg.norm(v1, dim=1),
        torch.linalg.norm(v2, dim=1),
        torch.linalg.norm(b, dim=1),
    )
    v1_a = torch.arccos(
        (torch.sum(v1 * b, dim=1) / (v1_n * b_n)).clamp(-1 + eps, 1 - eps)
    )
    v2_a = torch.arccos(
        (torch.sum(v2 * b, dim=1) / (v2_n * b_n)).clamp(-1 + eps, 1 - eps)
    )
    b_a = torch.arccos(
        (torch.sum(v1 * v2, dim=1) / (v1_n * v2_n)).clamp(-1 + eps, 1 - eps)
    )

    angle["1_2"] = torch.moveaxis(torch.vstack((v1_a + v2_a, b_a)), 0, 1)

    # areas
    area = {}
    area["0"] = torch.zeros(len(feat_ind_int["0"])).unsqueeze(1)
    area["1"] = torch.norm(
        pos[feat_ind_int["1"][:, 0]] - pos[feat_ind_int["1"][:, 1]], dim=1
    ).unsqueeze(1)
    area["2"] = (
        torch.norm(
            torch.cross(
                pos[feat_ind_int["2"][:, 0]] - pos[feat_ind_int["2"][:, 1]],
                pos[feat_ind_int["2"][:, 0]] - pos[feat_ind_int["2"][:, 2]],
                dim=1,
            ),
            dim=1,
        )
        / 2
    ).unsqueeze(1)

    area = {k: v.to(device) for k, v in area.items()}

    inv = {
        "0_0": torch.linalg.norm(
            pos[adj["0_0"][0]] - pos[adj["0_0"][1]], dim=1
        ).unsqueeze(1),
        "0_1": torch.linalg.norm(
            pos[inv_ind_int["0_1"][0]] - pos[inv_ind_int["0_1"][1]], dim=1
        ).unsqueeze(1),
        "1_1": torch.stack(
            [
                torch.linalg.norm(
                    pos[inv_ind_int["1_1"][0]] - pos[inv_ind_int["1_1"][1]], dim=1
                ),
                torch.linalg.norm(
                    pos[inv_ind_int["1_1"][0]] - pos[inv_ind_int["1_1"][2]], dim=1
                ),
                torch.linalg.norm(
                    pos[inv_ind_int["1_1"][1]] - pos[inv_ind_int["1_1"][2]], dim=1
                ),
            ],
            dim=1,
        ),
        "1_2": torch.stack(
            [
                torch.linalg.norm(
                    pos[inv_ind_int["1_2"][0]] - pos[inv_ind_int["1_2"][2]], dim=1
                )
                + torch.linalg.norm(
                    pos[inv_ind_int["1_2"][1]] - pos[inv_ind_int["1_2"][2]], dim=1
                ),
                torch.linalg.norm(
                    pos[inv_ind_int["1_2"][1]] - pos[inv_ind_int["1_2"][2]], dim=1
                ),
            ],
            dim=1,
        ),
    }

    for k, v in inv.items():
        area_send, area_rec = area[k[0]], area[k[2]]
        send, rec = adj[k]
        area_send, area_rec = area_send[send], area_rec[rec]
        inv[k] = torch.cat((v, area_send, area_rec), dim=1)

    inv["1_1"] = torch.cat((inv["1_1"], angle["1_1"].to(device)), dim=1)
    inv["1_2"] = torch.cat((inv["1_2"], angle["1_2"].to(device)), dim=1)

    return inv


compute_invariants_3d.num_features_map = {
    "0_0": 3,
    "0_1": 3,
    "1_1": 6,
    "1_2": 6,
}


@torch.jit.script
def compute_invariants(
    feat_ind: dict[str, list[Tensor]],
    pos: torch.FloatTensor,
    adj: dict[str, torch.LongTensor],
    haussdorf: bool = True,
    max_cell_size: int = 100,
    # inv_ind: dict[str, torch.FloatTensor] = None,
    # device: torch.device = None,
) -> dict[str, Tensor]:
    """
    Compute geometric invariants between pairs of cells specified in `adj`.

    This function calculates the following geometric features:

    Distance between centroids
        Euclidean distance between the centroids of cell pairs. The centroids are computed only once
        per rank and memoized to improve efficiency. The distances serve as geometric invariants
        that characterize the spatial relationships between cells of different ranks.

    Maximum pairwise distance within-cell
        For both the sender and receiver cell, the pairwise distances between their nodes are
        computed and the maximum distance is stored. This feature is meant to very loosely
        approximate the size of each cell.

    Two Hausdorff distances
        We compute two Hausdorff distances, one from the sender's point of view and one from the
        receiver's. The Hausdorff distance between two sets of points is typically defined as

            H(A, B) = max{sup_{a in A} inf_{b in B} d(a, b), sup_{b in B} inf_{a in A} d(b, a)}

        where A and B are two sets of points, d(a, b) is the Euclidean distance between points a and
        b, sup denotes the supremum (least upper bound) of a set, and inf denotes the infimum
        (greatest lower bound) of a set. Instead of taking the maximum, we instead return both of
        the terms. This choice allows us to implicitly encode the subset relationship into these
        features: the first Hausdorff distance is 0 iff A is a subset of B and the second Hausdorff
        distance is 0 iff B is a subset of A.


    Parameters
    ----------
    feat_ind : dict
        A dictionary mapping cell ranks to tensors of shape (num_cells, max_cardinality) containing
        indices of nodes for each cell. It is used to identify the cells for which centroids should
        be computed.
    pos : torch.FloatTensor
        A 2D tensor of shape (num_nodes, num_dimensions) containing the positions of each node.
    adj : dict
        A dictionary where each key is a string in the format 'sender_rank_receiver_rank' indicating
        the ranks of cell pairs, and each value is a tensor of shape (2, num_cell_pairs) containing
        indices for sender and receiver cells.

    Returns
    -------
    dict
        A dictionary where each key corresponds to a key in `adj` and each value is a 2D tensor
        holding the computed geometric features for each cell pair.

    Notes
    -----
    The `inv_ind` and `device` parameters are included for compatibility with a previous function
    interface and might be used in future versions of this function as it evolves. The computation
    of cell centroids is memoized based on cell rank to avoid redundant calculations, enhancing
    performance especially for large datasets with many cells. The current implementation focuses on
    Euclidean distances but may be extended to include other types of geometric invariants.
    """
    new_features = {}
    mean_cell_positions = {}
    max_pairwise_distances = {}

    dev = pos.device
    # compute centroids distance
    for rank_pair, cell_pairs in adj.items():
        centroid_dists = torch.zeros(cell_pairs.shape[1], device=dev)
        diameter_send = torch.zeros(cell_pairs.shape[1], device=dev)
        diameter_rec = torch.zeros(cell_pairs.shape[1], device=dev)
        hausdorff_dists_send = torch.zeros(cell_pairs.shape[1], device=dev)
        hausdorff_dists_rec = torch.zeros(cell_pairs.shape[1], device=dev)

        send_rank, rec_rank = rank_pair.split("_")[:2]

        for j in range(cell_pairs.shape[1]):
            index_send = feat_ind[send_rank][cell_pairs[0, j]]
            index_rec = feat_ind[rec_rank][cell_pairs[1, j]]

            if len(index_send) == 1 and len(index_rec) == 1:
                # trivial case, only one point
                dist = torch.norm(pos[index_send] - pos[index_rec])
                centroid_dists[j] = dist
                if haussdorf:
                    hausdorff_dists_send[j] = dist
                    hausdorff_dists_rec[j] = dist
                continue

            if len(index_send) > max_cell_size:
                new_pts_ix = torch.randperm(len(index_send))[:max_cell_size]
                index_send = index_send[new_pts_ix]
            if len(index_rec) > max_cell_size:
                new_pts_ix = torch.randperm(len(index_rec))[:max_cell_size]
                index_rec = index_rec[new_pts_ix]
            pos_send = pos[index_send]
            pos_rec = pos[index_rec]
            # centroids
            centroid_send = pos_send.mean(dim=0)
            centroid_rec = pos_rec.mean(dim=0)
            centroid_dists[j] = torch.norm(centroid_send - centroid_rec)
            # diameters
            diameter_send[j] = torch.norm(
                pos_send[:, None] - pos_send[None], dim=-1
            ).amax()
            diameter_rec[j] = torch.norm(
                pos_rec[:, None] - pos_rec[None], dim=-1
            ).amax()
            # hausdorff
            if haussdorf:
                distmat_cross = torch.norm(pos_send[:, None] - pos_rec[None], dim=-1)
                hausdorff_dists_send[j] = distmat_cross.amin(dim=1).max()
                hausdorff_dists_rec[j] = distmat_cross.amin(dim=0).max()

        f = torch.stack(
            [centroid_dists, diameter_send, diameter_rec],
            dim=1,
        )
        if haussdorf:
            f = torch.cat(
                [f, hausdorff_dists_send[:, None], hausdorff_dists_rec[:, None]],
                dim=1,
            )
        new_features[rank_pair] = f

    return new_features

    for rank_pair, cell_pairs in adj.items():

        # Compute mean cell positions memoized
        send_rank, rec_rank = rank_pair.split("_")[:2]

        for rank in [send_rank, rec_rank]:
            max_dim = max([len(x) for x in feat_ind[rank]])
            if max_dim == 1:
                if rank not in mean_cell_positions:
                    feats = torch.cat(feat_ind[rank])
                    mean_cell_positions[rank] = pos[feats]
                if rank not in max_pairwise_distances:
                    max_pairwise_distances[rank] = torch.zeros(
                        len(feat_ind[rank]), device=pos.device
                    )
            else:
                if rank not in mean_cell_positions:
                    mean_cell_positions[rank] = torch.stack(
                        [pos[f].mean(dim=0) for f in feat_ind[rank]]
                    )
                if rank not in max_pairwise_distances:
                    max_pairwise_distances[rank] = torch.stack(
                        [
                            torch.norm(pos[f][:, None] - pos[f], dim=2).max()
                            for f in feat_ind[rank]
                        ]
                    )

        # Compute mean distances
        indexed_sender_centroids = mean_cell_positions[send_rank][cell_pairs[0]]
        indexed_receiver_centroids = mean_cell_positions[rec_rank][cell_pairs[1]]
        differences = indexed_sender_centroids - indexed_receiver_centroids
        centroid_dists = torch.sqrt((differences**2).sum(dim=1))

        # Compute diameter
        max_dist_sender = max_pairwise_distances[send_rank][cell_pairs[0]]
        max_dist_receiver = max_pairwise_distances[rec_rank][cell_pairs[1]]

        # # Compute haussdorf distances
        max_dim_sender = max(len(x) for x in feat_ind[send_rank])
        max_dim_receiver = max(len(x) for x in feat_ind[rec_rank])

        if haussdorf:
            # easy case/graph
            if max_dim_sender == 1 and max_dim_receiver == 1:
                feats_sender = torch.cat(
                    [feat_ind[send_rank][c] for c in cell_pairs[0]]
                )
                feats_receiver = torch.cat(
                    [feat_ind[rec_rank][c] for c in cell_pairs[1]]
                )
                pos_sender = pos[feats_sender]
                pos_receiver = pos[feats_receiver]
                # cells have equal size, just get the positions and compute the distance
                # hausdorff is the trivially equal to the distance
                dists = torch.norm(pos_sender - pos_receiver, dim=1)
                hausdorff_dists_sender = dists
                hausdorff_dists_receiver = dists
            # general case
            else:
                hausdorff_dists_sender = torch.zeros_like(centroid_dists)
                hausdorff_dists_receiver = torch.zeros_like(centroid_dists)
                for j in range(cell_pairs.shape[1]):
                    index_left = feat_ind[send_rank][cell_pairs[0, j]]
                    index_right = feat_ind[rec_rank][cell_pairs[1, j]]
                    #
                    if len(index_left) > max_cell_size:
                        new_pts_ix = torch.randperm(len(index_left))[:max_cell_size]
                        index_left = index_left[new_pts_ix]
                    if len(index_right) > max_cell_size:
                        new_pts_ix = torch.randperm(len(index_right))[:max_cell_size]
                        index_right = index_right[new_pts_ix]
                    #
                    pos_sender = pos[index_left]
                    pos_receiver = pos[index_right]
                    distmat_cross = torch.norm(
                        pos_sender[:, None] - pos_receiver, dim=2
                    )
                    hausdorff_dists_sender[j] = distmat_cross.min(dim=1)[0].max()
                    hausdorff_dists_receiver[j] = distmat_cross.min(dim=0)[0].max()

            # Combine all features
            new_features[rank_pair] = torch.stack(
                [
                    centroid_dists,
                    max_dist_sender,
                    max_dist_receiver,
                    hausdorff_dists_sender,
                    hausdorff_dists_receiver,
                ],
                dim=1,
            )
        else:
            # Combine all features
            new_features[rank_pair] = torch.stack(
                [centroid_dists, max_dist_sender, max_dist_receiver], dim=1
            )

    return new_features


# compute_invariants.num_features_map = defaultdict(lambda: 5)


def compute_invariants2(
    feat_ind: dict[str, torch.LongTensor],
    pos: torch.FloatTensor,
    adj: dict[str, torch.LongTensor],
    agg_indices: dict[str, tuple[np.ndarray, ...]],
) -> dict[str, torch.FloatTensor]:
    new_features = {}
    mean_cell_positions = {}
    max_pairwise_distances = {}

    for rank_pair, cell_pairs in adj.items():

        # Compute mean cell positions memoized
        send_rank, rec_rank = rank_pair.split("_")[:2]

        for rank in [send_rank, rec_rank]:
            max_dim = max([len(x) for x in feat_ind[rank]])
            if max_dim == 1:
                if rank not in mean_cell_positions:
                    feats = torch.cat(feat_ind[rank])
                    mean_cell_positions[rank] = pos[feats]
                if rank not in max_pairwise_distances:
                    max_pairwise_distances[rank] = torch.zeros(
                        len(feat_ind[rank]), device=pos.device
                    )
            else:
                if rank not in mean_cell_positions:
                    mean_cell_positions[rank] = torch.stack(
                        [pos[f].mean(dim=0) for f in feat_ind[rank]]
                    )
                if rank not in max_pairwise_distances:
                    max_pairwise_distances[rank] = torch.stack(
                        [
                            torch.norm(pos[f][:, None] - pos[f], dim=2).max()
                            for f in feat_ind[rank]
                        ]
                    )

        # Compute mean distances
        indexed_sender_centroids = mean_cell_positions[send_rank][cell_pairs[0]]
        indexed_receiver_centroids = mean_cell_positions[rec_rank][cell_pairs[1]]
        differences = indexed_sender_centroids - indexed_receiver_centroids
        centroid_dists = torch.sqrt((differences**2).sum(dim=1))

        # Compute diameter
        max_dist_sender = max_pairwise_distances[send_rank][cell_pairs[0]]
        max_dist_receiver = max_pairwise_distances[rec_rank][cell_pairs[1]]

        # # Compute haussdorf distances
        max_dim_sender = max(len(x) for x in feat_ind[send_rank])
        max_dim_receiver = max(len(x) for x in feat_ind[rec_rank])

        # this part is a bit tedious since we want to solve independently the easy cases
        # for better performance

        if max_dim_sender == 1 and max_dim_receiver == 1:
            feats_sender = torch.cat([feat_ind[send_rank][c] for c in cell_pairs[0]])
            feats_receiver = torch.cat([feat_ind[rec_rank][c] for c in cell_pairs[1]])
            pos_sender = pos[feats_sender]
            pos_receiver = pos[feats_receiver]
            dists = torch.norm(pos_sender - pos_receiver, dim=1)
            hausdorff_dists_sender = dists
            hausdorff_dists_receiver = dists
        else:
            cl, cr, il, ir, sl, sr = agg_indices[rank_pair]
            pos_sender = pos[cl]
            pos_receiver = pos[cr]
            dists = torch.norm(pos_sender - pos_receiver, dim=1)
            hausdorff_dists_sender = scatter_max(scatter_min(dists, il), sl)
            hausdorff_dists_receiver = scatter_max(scatter_min(dists, ir), sr)

        # Combine all features
        new_features[rank_pair] = torch.stack(
            [
                centroid_dists,
                max_dist_sender,
                max_dist_receiver,
                hausdorff_dists_sender,
                hausdorff_dists_receiver,
            ],
            dim=1,
        )

    return new_features


def compute_max_pairwise_distances(
    cells: torch.FloatTensor, pos: torch.FloatTensor
) -> torch.FloatTensor:
    """
    Compute the maximum pairwise distance between nodes within each cell.

    Parameters
    ----------
    cells : torch.FloatTensor
        A 2D tensor of shape (n, k) containing indices of nodes for n cells. Indices may include
        NaNs to denote missing values.
    pos : torch.FloatTensor
        A 2D tensor of shape (m, 3) representing the 3D positions of m nodes.

    Returns
    -------
    torch.FloatTensor
        A tensor of shape (n, 1) containing the maximum pairwise distance within each of the n
        cells.

    Notes
    -----
    The function handles cells with varying numbers of nodes by using NaN values in the `cells`
    tensor to indicate missing nodes. It computes pairwise distances only for valid (non-NaN) nodes
    within each cell and ignores distances involving missing nodes.
    """

    dist_matrix = compute_intercell_distances(cells, cells, pos)
    dist_matrix = dist_matrix.nan_to_num(float("-inf"))
    max_distances = dist_matrix.max(dim=2)[0].max(dim=1)[0].unsqueeze(1)

    return max_distances


def compute_hausdorff_distances(
    sender_cells: torch.FloatTensor,
    receiver_cells: torch.FloatTensor,
    pos: torch.FloatTensor,
) -> torch.FloatTensor:
    """
    Compute the Hausdorff distances between two sets of cells.

    The Hausdorff distance is calculated based on the positions of sender and receiver cells. For
    two cells A and B where A is the sender and B is the receiver, the two Hausdorff distances
    computed by this function correspond to the maximum distance between any node in cell A and the
    node in cell B that is closest to it, and vice versa.

    Parameters
    ----------
    sender_cells : torch.FloatTensor
        A tensor representing the positions of the sender cells.
    receiver_cells : torch.FloatTensor
        A tensor representing the positions of the receiver cells.
    pos : torch.FloatTensor
        A tensor representing additional position information used in computing intercell distances.

    Returns
    -------
    torch.FloatTensor
        A tensor of shape (N, 2), where N is the number of sender cells. Each element contains the
        Hausdorff distance from sender to receiver cells in the first column, and receiver to sender
        cells in the second column.
    """
    dist_matrix = compute_intercell_distances(sender_cells, receiver_cells, pos)
    dist_matrix = dist_matrix.nan_to_num(float("inf"))

    sender_mins = dist_matrix.min(dim=2)[0]
    # Cast inf to -inf to correctly compute maxima
    sender_mins = torch.where(sender_mins == float("inf"), float("-inf"), sender_mins)
    sender_hausdorff = sender_mins.max(dim=1)[0]

    receiver_mins = dist_matrix.min(dim=1)[0]
    # Cast inf to -inf to correctly compute maxima
    receiver_mins = torch.where(
        receiver_mins == float("inf"), float("-inf"), receiver_mins
    )
    receiver_hausdorff = receiver_mins.max(dim=1)[0]

    hausdorff_distances = torch.stack([sender_hausdorff, receiver_hausdorff], dim=1)

    return hausdorff_distances


def compute_intercell_distances(
    sender_cells: torch.FloatTensor,
    receiver_cells: torch.FloatTensor,
    pos: torch.FloatTensor,
) -> torch.FloatTensor:
    """
    Compute the pairwise distances between nodes within each cell.

    Parameters
    ----------
    sender_cells : torch.FloatTensor
        A 2D tensor of shape (n, k_1) containing indices of nodes for n sender cells.
    receiver_cells : torch.FloatTensor
        A 2D tensor of shape (n, k_2) containing indices of nodes for n receiver cells.
    pos : torch.FloatTensor
        A 2D tensor of shape (m, 3) representing the 3D positions of m nodes.

    Returns
    -------
    torch.FloatTensor
        A 3D tensor of shape (n, k_1, k_2) containing the pairwise distances between nodes within
        each of the n cells.

    Notes
    -----
    This function handles cells with varying numbers of nodes by using NaN values in the
    `sender_cells` and `receiver_cells` tensors to indicate missing nodes. It computes distances for
    valid (non-NaN) nodes within each pair of sender and receiver cells. For combinations involving
    at least one NaN node, the computed distance is set to NaN.
    """
    # Cast nans to 0 to compute distances in a vectorized fashion
    sender_cells_filled = sender_cells.nan_to_num(0).to(torch.int64)
    receiver_cells_filled = receiver_cells.nan_to_num(0).to(torch.int64)
    sender_positions = pos[sender_cells_filled]
    receiver_positions = pos[receiver_cells_filled]
    dist_matrix = torch.norm(
        sender_positions.unsqueeze(2) - receiver_positions.unsqueeze(1), dim=3
    )

    # Set distances for invalid combinations to nan
    sender_mask = ~torch.isnan(sender_cells)
    receiver_mask = ~torch.isnan(receiver_cells)
    valid_combinations_mask = sender_mask.unsqueeze(2) & receiver_mask.unsqueeze(1)
    dist_matrix[~valid_combinations_mask] = torch.nan

    return dist_matrix


def compute_centroids(
    cells: torch.FloatTensor, features: torch.FloatTensor
) -> torch.FloatTensor:
    """
    Compute the centroids of cells based on their constituent nodes' features.

    This function calculates the mean feature vector (centroid) for each cell defined by `cells`,
    using the provided `features` for each node. It handles cells with varying numbers of nodes,
    including those with missing values (NaNs), by treating them as absent nodes.

    Parameters
    ----------
    cells : torch.FloatTensor
        A 2D tensor of shape (num_cells, max_cell_size) containing indices of nodes for each cell.
        Indices may include NaNs to indicate absent nodes in cells with fewer nodes than
        `max_cell_size`.
    features : torch.FloatTensor
        A 2D tensor of shape (num_nodes, num_features) containing feature vectors for each node.

    Returns
    -------
    torch.FloatTensor
        A 2D tensor of shape (num_cells, num_features) containing the computed centroids for each
        cell.

    Notes
    -----
    The function creates an intermediate tensor, `features_padded`, which is a copy of `features`
    with an additional row of zeros at the end. This increases the memory footprint, as it
    temporarily duplicates the `features` tensor. The purpose of this padding is to safely handle -1
    indices resulting from NaNs in `cells`, allowing vectorized operations without explicit NaN
    checks.
    """
    zeros_row = torch.zeros(1, features.shape[1], device=features.device)
    features_padded = torch.cat([features, zeros_row], dim=0)
    cells_int = cells.nan_to_num(-1).to(torch.int64)
    nodes_features = features_padded[cells_int]
    sum_features = torch.sum(nodes_features, dim=1)
    cell_cardinalities = torch.sum(cells_int != -1, dim=1, keepdim=True)
    centroids = sum_features / cell_cardinalities

    return centroids


# if __name__ == "__main__":
#     import timeit

#     list1 = [[0, 1], [2, 3, 4]]
#     list2 = [[2, 3, 4], [5]]
#     lengths_left = np.array([len(x) for x in list1])
#     lengths_right = np.array([len(x) for x in list2])
#     indices_left = np.concatenate(list1)
#     atoms_right = np.concatenate(list2)

#     # first run
#     fast_agg_indices(indices_left, lengths_left, atoms_right, lengths_right)

#     # second run
#     res = timeit.timeit(
#         lambda: fast_agg_indices(
#             indices_left, lengths_left, atoms_right, lengths_right
#         ),
#         number=100,
#     )
#     print(res)

#     # now test hausdorff, transform to torch
#     pos = torch.rand(6, 2)
#     with torch.no_grad():
#         hl, hr = haussdorff(list1, list2, pos)
#     print(hl, hr)
