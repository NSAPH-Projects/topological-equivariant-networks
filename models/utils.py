from collections import defaultdict

import torch
import torch.nn as nn
from torch_scatter import scatter_add


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
        messages_aggr = scatter_add(messages * edge_weights, index_rec, dim=0)

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

    p1, p2, a = pos[inv_ind_int["1_2"][0]], pos[inv_ind_int["1_2"][1]], pos[inv_ind_int["1_2"][2]]
    v1, v2, b = p1 - a, p2 - a, p1 - p2
    v1_n, v2_n, b_n = (
        torch.linalg.norm(v1, dim=1),
        torch.linalg.norm(v2, dim=1),
        torch.linalg.norm(b, dim=1),
    )
    v1_a = torch.arccos((torch.sum(v1 * b, dim=1) / (v1_n * b_n)).clamp(-1 + eps, 1 - eps))
    v2_a = torch.arccos((torch.sum(v2 * b, dim=1) / (v2_n * b_n)).clamp(-1 + eps, 1 - eps))
    b_a = torch.arccos((torch.sum(v1 * v2, dim=1) / (v1_n * v2_n)).clamp(-1 + eps, 1 - eps))

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
        "0_0": torch.linalg.norm(pos[adj["0_0"][0]] - pos[adj["0_0"][1]], dim=1).unsqueeze(1),
        "0_1": torch.linalg.norm(
            pos[inv_ind_int["0_1"][0]] - pos[inv_ind_int["0_1"][1]], dim=1
        ).unsqueeze(1),
        "1_1": torch.stack(
            [
                torch.linalg.norm(pos[inv_ind_int["1_1"][0]] - pos[inv_ind_int["1_1"][1]], dim=1),
                torch.linalg.norm(pos[inv_ind_int["1_1"][0]] - pos[inv_ind_int["1_1"][2]], dim=1),
                torch.linalg.norm(pos[inv_ind_int["1_1"][1]] - pos[inv_ind_int["1_1"][2]], dim=1),
            ],
            dim=1,
        ),
        "1_2": torch.stack(
            [
                torch.linalg.norm(pos[inv_ind_int["1_2"][0]] - pos[inv_ind_int["1_2"][2]], dim=1)
                + torch.linalg.norm(pos[inv_ind_int["1_2"][1]] - pos[inv_ind_int["1_2"][2]], dim=1),
                torch.linalg.norm(pos[inv_ind_int["1_2"][1]] - pos[inv_ind_int["1_2"][2]], dim=1),
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
    inv_ind: dict[str, torch.FloatTensor],
    device: torch.device,
):
    """
    Compute geometric invariants between pairs of cells specified in `adj`.

    This function calculates the Euclidean distance between the centroids of cell pairs. The
    centroids are computed only once per rank and memoized to improve efficiency. The distances
    serve as geometric invariants that characterize the spatial relationships between cells of
    different ranks.

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
    inv_ind : Any
        Currently unused. Placeholder for future use.
    device : Any
        Currently unused. Placeholder for potential device specification in future updates.

    Returns
    -------
    dict
        A dictionary where each key corresponds to a key in `adj` and each value is a tensor of
        Euclidean distances between the centroids of the sender-receiver cell pairs specified by the
        corresponding value in `adj`.

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
        sender_rank, receiver_rank = rank_pair.split("_")
        for rank in [sender_rank, receiver_rank]:
            if rank not in mean_cell_positions:
                mean_cell_positions[rank] = compute_centroids(feat_ind[rank], pos)
            if rank not in max_pairwise_distances:
                max_pairwise_distances[rank] = compute_max_pairwise_distances(feat_ind[rank], pos)
        # Compute mean distances
        indexed_sender_centroids = mean_cell_positions[sender_rank][cell_pairs[0]]
        indexed_receiver_centroids = mean_cell_positions[receiver_rank][cell_pairs[1]]
        differences = indexed_sender_centroids - indexed_receiver_centroids
        distances = torch.sqrt((differences**2).sum(dim=1, keepdim=True))
        max_dist_sender = max_pairwise_distances[sender_rank][cell_pairs[0]]
        max_dist_receiver = max_pairwise_distances[receiver_rank][cell_pairs[1]]
        new_features[rank_pair] = torch.cat([distances, max_dist_sender, max_dist_receiver], dim=1)

    return new_features


compute_invariants.num_features_map = defaultdict(lambda: 3)


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

    dist_matrix = compute_pairwise_distances(cells, pos)
    dist_matrix = dist_matrix.nan_to_num(float("-inf"))
    max_distances = dist_matrix.max(dim=2)[0].max(dim=1)[0].unsqueeze(1)

    return max_distances


def compute_pairwise_distances(
    cells: torch.FloatTensor, pos: torch.FloatTensor
) -> torch.FloatTensor:
    """
    Compute the pairwise distances between nodes within each cell.

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
        A 3D tensor of shape (n, k, k) containing the pairwise distances between nodes within each
        of the n cells.

    Notes
    -----
    The function handles cells with varying numbers of nodes by using NaN values in the `cells`
    tensor to indicate missing nodes. It computes pairwise distances for valid (non-NaN) nodes
    within each cell. For pairs where at least one node is NaN, the copmuted distance is also NaN.
    """
    # Cast nans to 0 to compute distances in a vectorized fashion
    cells_filled = cells.nan_to_num(0).to(torch.int64)
    positions = pos[cells_filled]
    dist_matrix = torch.norm(positions.unsqueeze(2) - positions.unsqueeze(1), dim=3)

    # Set distances for invalid combinations to nan
    mask = ~torch.isnan(cells)
    valid_combinations_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    dist_matrix[~valid_combinations_mask] = torch.nan

    return dist_matrix


def compute_centroids(cells: torch.FloatTensor, features: torch.FloatTensor) -> torch.FloatTensor:
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
