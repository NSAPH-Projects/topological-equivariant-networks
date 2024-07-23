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
