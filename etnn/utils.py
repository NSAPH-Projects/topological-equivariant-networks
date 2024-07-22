from collections import defaultdict
from typing import Optional

import torch
from torch import Tensor


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


def scatter_mean(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
):
    src_shape = list(src.shape)
    src_shape[dim] = index.max().item() + 1 if dim_size is None else dim_size
    aux = src.new_zeros(src_shape)
    return aux.index_reduce(dim, index, src, reduce="mean", include_self=False)


def scatter_min(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
):
    src_shape = list(src.shape)
    src_shape[dim] = index.max().item() + 1 if dim_size is None else dim_size
    aux = src.new_zeros(src_shape)
    return aux.index_reduce(dim, index, src, reduce="amin", include_self=False)


def scatter_max(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
):
    src_shape = list(src.shape)
    src_shape[dim] = index.max().item() + 1 if dim_size is None else dim_size
    aux = src.new_zeros(src_shape)
    return aux.index_reduce(dim, index, src, reduce="amax", include_self=False)


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


def compute_invariants(
    feat_ind: dict[str, torch.FloatTensor],
    pos: torch.FloatTensor,
    adj: dict[str, torch.LongTensor],
    # inv_ind: dict[str, torch.FloatTensor],
    device: torch.device,
) -> dict[str, torch.FloatTensor]:
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
    device : Any
        Currently unused. Placeholder for potential device specification in future updates.

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
    for rank_pair, cell_pairs in adj.items():

        # Compute mean cell positions memoized
        sender_rank, receiver_rank = rank_pair.split("_")[:2]
        for rank in [sender_rank, receiver_rank]:
            if rank not in mean_cell_positions:
                mean_cell_positions[rank] = compute_centroids(feat_ind[rank], pos)
            if rank not in max_pairwise_distances:
                max_pairwise_distances[rank] = compute_max_pairwise_distances(
                    feat_ind[rank], pos
                )

        # Compute mean distances
        indexed_sender_centroids = mean_cell_positions[sender_rank][cell_pairs[0]]
        indexed_receiver_centroids = mean_cell_positions[receiver_rank][cell_pairs[1]]
        differences = indexed_sender_centroids - indexed_receiver_centroids
        distances = torch.sqrt((differences**2).sum(dim=1, keepdim=True))

        # Retrieve maximum pairwise distances
        max_dist_sender = max_pairwise_distances[sender_rank][cell_pairs[0]]
        max_dist_receiver = max_pairwise_distances[receiver_rank][cell_pairs[1]]

        # Compute Hausdorff distances
        sender_cells = feat_ind[sender_rank][cell_pairs[0]]
        receiver_cells = feat_ind[receiver_rank][cell_pairs[1]]
        hausdorff_distances = compute_hausdorff_distances(
            sender_cells, receiver_cells, pos
        )

        # Combine all features
        new_features[rank_pair] = torch.cat(
            [distances, max_dist_sender, max_dist_receiver, hausdorff_distances], dim=1
        )

    return new_features


compute_invariants.num_features_map = defaultdict(lambda: 5)


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
