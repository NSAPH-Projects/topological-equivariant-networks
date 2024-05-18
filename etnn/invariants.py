from typing import Optional
import torch
from torch import Tensor


# @torch.jit.script
def _invariants_kernel(
    cells_send: list[Tensor],
    cells_rec: list[Tensor],
    pos: Tensor,
    haussdorf: bool = True,
    max_cell_size: Optional[int] = None,
) -> Tensor:
    out_dim = 5 if haussdorf else 3
    device = pos.device
    out = torch.zeros(len(cells_send), out_dim, device=device)

    # loop over all cells
    for i, (cell_send, cell_rec) in enumerate(zip(cells_send, cells_rec)):
        # trivial case, only one point
        if len(cell_send) == 1 and len(cell_rec) == 1:
            dist = torch.norm(pos[cell_send[0]] - pos[cell_send[0]])
            out[i, 0] = dist
            out[i, 1] = torch.tensor(0.0, device=device)
            out[i, 2] = torch.tensor(0.0, device=device)
            if haussdorf:
                out[i, 3] = torch.tensor(0.0, device=device)
                out[i, 4] = torch.tensor(0.0, device=device)
            continue

        # subsmple cell sizes if needed
        if max_cell_size is not None and len(cell_send) > max_cell_size:
            subsample = torch.randperm(len(cell_send))[:max_cell_size].to(device)
            cell_send = cell_send[subsample]
        if max_cell_size is not None and len(cell_rec) > max_cell_size:
            subsample = torch.randperm(len(cell_rec))[:max_cell_size].to(device)
            cell_rec = cell_rec[subsample]

        # obtain positions
        pos_send = pos[cell_send]
        pos_rec = pos[cell_rec]

        # compute centroid distance
        centroid_dist = torch.norm(pos_send.mean(0) - pos_rec.mean(0))
        out[i, 0] = centroid_dist

        # compute diameters (max distance per cell)
        send_dist = torch.norm(pos_send[:, None] - pos_send[None], dim=-1)
        rec_dist = torch.norm(pos_rec[:, None] - pos_rec[None], dim=-1)
        send_diameter = send_dist.max()
        rec_diameter = rec_dist.max()
        out[i, 1] = send_diameter
        out[i, 2] = rec_diameter

        # haussdorff distance
        if haussdorf:
            cross_dist = torch.norm(pos_send[:, None] - pos_rec[None], dim=-1)
            send_hausdorff = cross_dist.amin(dim=0).max()
            rec_hausdorff = cross_dist.amin(dim=1).max()
            out[i, 3] = send_hausdorff
            out[i, 4] = rec_hausdorff

    return out


def compute_invariants(
    cell_ind: dict[str, list[Tensor]],
    pos: Tensor,
    adj: dict[str, Tensor],
    haussdorf: bool = True,
    max_cell_size: Optional[int] = None,
) -> dict[str, Tensor]:
    # compute invariants for all adjacency types
    invariants = {}
    # compute centroids distance
    for rank_pair, cell_pairs in adj.items():
        send_rank, rec_rank = rank_pair.split("_")[:2]
        cell_send = [cell_ind[send_rank][c] for c in cell_pairs[0]]
        cell_rec = [cell_ind[rec_rank][c] for c in cell_pairs[1]]
        invariants[rank_pair] = _invariants_kernel(
            cell_send, cell_rec, pos, haussdorf, max_cell_size
        )

    return invariants


if __name__ == "__main__":
    import time

    # test invariants
    cell_0 = torch.tensor([[0, 1, 2], [3, 4, 5], [4, 3, 1]])
    cell_ind = {"0": cell_0}

    edges = torch.tensor([[0, 1, 2], [2, 0, 1]])
    adj = {"0_0": torch.cat([edges], dim=1)}
    pos = torch.randn(6, 3)

    for i in range(5):
        start = time.time()
        invariants = compute_invariants(
            cell_ind, adj, pos, max_cell_size=4, haussdorf=False
        )
        print(f"Attempt {i}: Time taken: {time.time() - start:.4f}s")

    adj = {"0_0": torch.cat([edges] * 10000, dim=1)}
    for i in range(5):
        start = time.time()
        invariants = compute_invariants(
            cell_ind, adj, pos, max_cell_size=4, haussdorf=False
        )
        print(f"Attempt {i}: Time taken: {time.time() - start:.4f}s")

    print(invariants)
