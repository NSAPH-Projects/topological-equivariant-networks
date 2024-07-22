from copy import deepcopy
from dataclasses import dataclass
import numba
import numpy as np
import torch

from etnn import utils


def compute_invariants(
    pos: torch.FloatTensor,
    cell_ind: dict[str, torch.FloatTensor],
    adj: dict[str, torch.LongTensor],
    hausdorff: bool = True,
    # inv_ind: dict[str, torch.FloatTensor],
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
    pos : torch.FloatTensor
        A 2D tensor of shape (num_nodes, num_dimensions) containing the positions of each node.
    cell_ind : dict
        A dictionary mapping cell ranks to tensors of shape (num_cells, max_cardinality) containing
        indices of nodes for each cell. It is used to identify the cells for which centroids should
        be computed.
    adj : dict
        A dictionary where each key is a string in the format 'sender_rank_receiver_rank' indicating
        the ranks of cell pairs, and each value is a tensor of shape (2, num_cell_pairs) containing
        indices for sender and receiver cells.
    hausdorff : bool
        Whether to compute the Hausdorff distances between cells. Default is True

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
                mean_cell_positions[rank] = compute_centroids(cell_ind[rank], pos)
            if rank not in max_pairwise_distances:
                max_pairwise_distances[rank] = compute_max_pairwise_distances(
                    cell_ind[rank], pos
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
        sender_cells = cell_ind[sender_rank][cell_pairs[0]]
        receiver_cells = cell_ind[receiver_rank][cell_pairs[1]]

        new_feats = torch.cat([distances, max_dist_sender, max_dist_receiver], dim=1)

        if hausdorff:
            hausdorff_distances = compute_hausdorff_distances(
                sender_cells, receiver_cells, pos
            )
            new_feats = torch.cat([new_feats, hausdorff_distances], dim=1)

        # Combine all features
        new_features[rank_pair] = new_feats

    return new_features


# compute_invariants.num_features_map = defaultdict(lambda: 5)  # not needed, better to compute dynamic


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


@dataclass
class SparseInvariantComputationIndices:
    """This auxiliary class is used to compute the indices for the computation of geometric
    invariants. The agents permit to compute centroids, diameters, and Hausdorff distances between
    pairs of cells using only scatter add/min/max/mean operations in linear memory."""

    cell_ids: np.ndarray
    atoms_ids_send: np.ndarray
    atoms_ids_recv: np.ndarray
    haus_min_recv: np.ndarray
    haus_min_send: np.ndarray
    haus_max_recv: np.ndarray
    haus_max_send: np.ndarray


@numba.jit(nopython=True)
def _sparse_computation_indices(
    atoms_send: np.ndarray,
    slices_send: np.ndarray,
    atoms_recv: np.ndarray,
    slices_recv: np.ndarray,
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
    # inputs must be equal size
    splits_send = np.split(atoms_send, np.cumsum(slices_send)[:-1])
    splits_recv = np.split(atoms_recv, np.cumsum(slices_recv)[:-1])

    # must return a tuple of 6 arrays
    n = len(slices_send)  # must be the same as len(splits_recv)
    m = sum(slices_send * slices_recv)
    ids_send = np.empty(m, dtype=np.int64)
    ids_recv = np.empty(m, dtype=np.int64)
    minindex_send = np.empty(m, dtype=np.int64)
    minindex_recv = np.empty(m, dtype=np.int64)
    cell_ids = np.empty(m, dtype=np.int64)

    offset_index_send = 0
    offset_index_recv = 0
    step = 0
    for i in range(n):
        n_send = len(splits_send[i])
        n_recv = len(splits_recv[i])
        for j, sj in enumerate(splits_send[i]):
            for k, sk in enumerate(splits_recv[i]):
                ind = step + j * n_recv + k
                ids_send[ind] = sj
                ids_recv[ind] = sk
                minindex_send[ind] = offset_index_send + j
                minindex_recv[ind] = offset_index_recv + k
                cell_ids[ind] = i
        step += n_send * n_recv
        offset_index_send += n_send
        offset_index_recv += n_recv

    step = 0
    maxindex_send = np.empty(sum(slices_send), dtype=np.int64)
    for j, sj in enumerate(splits_send):
        maxindex_send[step : step + len(sj)] = j
        step += len(sj)

    step = 0
    maxindex_recv = np.empty(sum(slices_recv), dtype=np.int64)
    for j, sj in enumerate(splits_recv):
        maxindex_recv[step : step + len(sj)] = j
        step += len(sj)

    return (
        cell_ids,
        ids_send,
        ids_recv,
        minindex_send,
        minindex_recv,
        maxindex_send,
        maxindex_recv,
    )


def sparse_computation_indices_from_cc(cell_ind, adj, max_cell_size=100):
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
        slices = np.array([len(c) for c in cells])
        indices = _sparse_computation_indices(atoms, slices, atoms, slices)
        indices = SparseInvariantComputationIndices(*indices)
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
        slices_send = np.array([len(c) for c in cells_send])
        atoms_recv = np.concatenate(cells_recv)
        slices_recv = np.array([len(c) for c in cells_recv])

        # get the indices of the cells
        indices = _sparse_computation_indices(
            atoms_send, slices_send, atoms_recv, slices_recv
        )
        indices = SparseInvariantComputationIndices(*indices)
        agg_indices[adj_type] = indices

    return agg_indices, cell_ind


def compute_invariants_sparse(
    pos: torch.FloatTensor,
    cell_ind: dict[str, list[list[int]]],
    adj: dict[str, torch.LongTensor],
    rank_agg_indices: dict[str, SparseInvariantComputationIndices],
    hausdorff: bool = True,
    diff_high_order: bool = False,
) -> dict[str, torch.Tensor]:
    """This function computes the geometric invariants between pairs of cells specified in `adj`

    Parameters
    ----------
    pos : torch.FloatTensor
        A 2D tensor of shape (num_nodes, num_dimensions) containing the positions of each node.
    cell_ind : dict
        A dictionary mapping cell ranks to lists of lists of node indices for each cell.
    adj : dict
        A dictionary where each key is a string in the format 'sender_rank_receiver_rank' indicating
        the ranks of cell pairs, and each value is a tensor of shape (2, num_cell_pairs) containing
        indices for sender and receiver cells.
    rank_agg_indices : dict
        A dictionary where each key is a rank and each value is a SparseInvariantComputationIndices
        object that contains the indices for the computation of geometric invariants.
    hausdorff : bool
        Whether to compute the Hausdorff distances between cells. Default is True.
    diff_high_order : bool
        Whether to compute the distances between high-order nodes. Default is False.
    """
    # device
    dev = pos.device

    # placeholder
    inv_list: dict[str, list[torch.Tensor]] = {}

    # compute centroids and diameters
    centroids: dict[str, torch.Tensor] = {}
    diameters: dict[str, torch.Tensor] = {}
    for rank, cells in cell_ind.items():
        # compute centroids
        ids: list[int] = []
        for c in cells:
            ids.extend(c)

        sizes: list[int] = []
        index: list[int] = []
        for i, c in enumerate(cells):
            sizes.append(len(c))
            index.extend([i] * len(c))
        index = torch.tensor(index).to(dev)
        centroids[rank] = utils.scatter_mean(
            pos[ids], index, dim=0, dim_size=len(cells)
        )

        # compute diameters (max pairwise distance)
        agg = rank_agg_indices[rank]
        if diff_high_order:
            pos_send = pos[agg.atoms_ids_send]
            pos_recv = pos[agg.atoms_ids_recv]
        else:
            pos_send = pos.detach()[agg.atoms_ids_send]
            pos_recv = pos.detach()[agg.atoms_ids_recv]
        dist = torch.norm(pos_send - pos_recv, dim=-1)
        index = torch.tensor(agg.cell_ids).to(dev)
        diameters[rank] = utils.scatter_max(dist, index, dim=0, dim_size=len(cells))

    # compute distances
    for rank_pair, cell_pairs in adj.items():
        send_rank, recv_rank = rank_pair.split("_")

        # check if reverse message has been computed
        flipped_rank_pair = f"{recv_rank}_{send_rank}"
        if flipped_rank_pair in inv_list:
            inv_list[rank_pair] = inv_list[flipped_rank_pair]
            continue

        # centroid dist
        centroids_send = centroids[send_rank][cell_pairs[0]]
        centroids_recv = centroids[recv_rank][cell_pairs[1]]
        centroids_dist = torch.norm(centroids_send - centroids_recv, dim=1)

        # diameter
        diameter_send = diameters[send_rank][cell_pairs[0]]
        diameter_recv = diameters[recv_rank][cell_pairs[1]]

        inv_list[rank_pair] = [
            centroids_dist,
            diameter_send,
            diameter_recv,
        ]

        # hausdorff
        if hausdorff:
            # gather indices for positions
            agg = rank_agg_indices[rank_pair]
            if diff_high_order:
                pos_send = pos[agg.atoms_ids_send]
                pos_recv = pos[agg.atoms_ids_recv]
            else:
                pos_send = pos.detach()[agg.atoms_ids_send]
                pos_recv = pos.detach()[agg.atoms_ids_recv]
            dists = torch.norm(pos_send - pos_recv, dim=1)

            # move agg indices to device for scatter operations
            minix_send = torch.tensor(agg.haus_min_send).to(dev)
            minix_recv = torch.tensor(agg.haus_min_recv).to(dev)
            maxix_send = torch.tensor(agg.haus_max_send).to(dev)
            maxix_recv = torch.tensor(agg.haus_max_recv).to(dev)

            # compute hausdorff distances with scatter operations
            hausdorff_send = utils.scatter_max(
                utils.scatter_min(dists, minix_send), maxix_send
            )
            hausdorff_recv = utils.scatter_max(
                utils.scatter_min(dists, minix_recv), maxix_recv
            )

            inv_list[rank_pair].extend([hausdorff_send, hausdorff_recv])

    out: dict[str, torch.Tensor] = {
        k: torch.stack(v, dim=1) for k, v in inv_list.items()
    }

    return out
