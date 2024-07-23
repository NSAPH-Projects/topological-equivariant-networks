from typing import Optional

import numpy as np
from numba import numba
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


def slices_to_pointer(slices: Tensor) -> Tensor:
    """This auxiliary function converts the the a slices object, which
    is a property of the torch_geometric.Data object, to a pointer index,
    which is a tensor of the same length as the number of objects where
    each element is the index of the pointer to which the entry belongs.
    For example, if the slices object is torch.tensor([0, 3, 5, 7]),
    then the output of this function is torch.tensor([0, 0, 0, 1, 1, 2, 2]).
    """
    n = slices.size(0) - 1
    return torch.arange(n, device=slices.device).repeat_interleave(
        (slices[1:] - slices[:-1]).long()
    )



class SparseAggIndices:

    @numba.jit(nopython=True)
    def __init__(
        self,
        atoms_send: np.ndarray,
        sizes_send: np.ndarray,
        atoms_recv: np.ndarray,
        sizes_recv: np.ndarray,
    ) -> None:
        # inputs must be equal size
        splits_send = np.split(atoms_send, np.cumsum(sizes_send)[:-1])
        splits_recv = np.split(atoms_recv, np.cumsum(sizes_recv)[:-1])

        # must return a tuple of 6 arrays
        n = len(sizes_send)  # must be the same as len(splits_recv)
        m = sum(sizes_send * sizes_recv)
        ids_send = np.empty(m, dtype=np.int64)
        ids_recv = np.empty(m, dtype=np.int64)
        minindex_send = np.empty(m, dtype=np.int64)
        minindex_recv = np.empty(m, dtype=np.int64)
        cell = np.empty(m, dtype=np.int64)

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
                    cell[ind] = i
            step += n_send * n_recv
            offset_index_send += n_send
            offset_index_recv += n_recv

        step = 0
        maxindex_send = np.empty(sum(sizes_send), dtype=np.int64)
        for j, sj in enumerate(splits_send):
            maxindex_send[step : step + len(sj)] = j
            step += len(sj)

        step = 0
        maxindex_recv = np.empty(sum(sizes_recv), dtype=np.int64)
        for j, sj in enumerate(splits_recv):
            maxindex_recv[step : step + len(sj)] = j
            step += len(sj)

        self.ids_send = list(ids_send)
        self.ids_recv = list(ids_recv)
        self.minindex_send = list(minindex_send)
        self.minindex_recv = list(minindex_recv)
        self.maxindex_send = list(maxindex_send)
        self.maxindex_recv = list(maxindex_recv)
        self.cell = list(cell)
