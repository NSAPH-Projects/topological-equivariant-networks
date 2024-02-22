import re

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader.dataloader import Collater


class CustomCollater(Collater):

    def __call__(self, batch: list[Data]) -> Batch:
        """
        Perform custom collation by pre-collating to pad tensors and using the superclass collation
        logic.

        Parameters
        ----------
        batch : list[Data]
            A list of Data objects to be collated.

        Returns
        -------
        Batch
            The collated list of Data objects, after pre-collation.
        """
        # Apply pre-collate logic
        batch = self.precollate(batch)

        # Use the original collation logic
        collated_batch = super().__call__(batch)

        return collated_batch

    def precollate(self, batch: list[Data]) -> list[Data]:
        """
        Pre-collate logic to pad tensors within each sample based on naming patterns.

        In particular, tensors that are named f'x_{i}' for some integer i are padded to have the
        same number of columns, and tensors that are named f'inv_{i}_{j}' for some integers i, j are
        padded to have the same number of rows. The variable number of columns/rows can arise if
        cells with the same rank may consist of a variable number of nodes. The padding value is -1
        which results in lookups to the last row of the feature matrix, which is overridden in
        postcollate() to be all zeros.

        Parameters
        ----------
        batch : list[Data]
            A list of Data objects to be pre-collated.

        Returns
        -------
        list[Data]
            The batch with tensors padded according to their naming patterns.
        """
        max_cols = {}
        max_rows = {}

        for data in batch:
            for attr_name in data.keys():
                if re.match(r"x_\d+", attr_name):
                    tensor = getattr(data, attr_name)
                    max_cols[attr_name] = max(max_cols.get(attr_name, 0), tensor.size(1))
                elif re.match(r"inv_\d+_\d+", attr_name):
                    tensor = getattr(data, attr_name)
                    max_rows[attr_name] = max(max_rows.get(attr_name, 0), tensor.size(0))

        for data in batch:
            for attr_name in data.keys():
                if attr_name in max_cols:
                    tensor = getattr(data, attr_name)
                    setattr(data, attr_name, pad_tensor(tensor, max_cols[attr_name], 1))
                elif attr_name in max_rows:
                    tensor = getattr(data, attr_name)
                    setattr(data, attr_name, pad_tensor(tensor, max_rows[attr_name], 0))

        return batch


class CombinatorialComplexData(Data):
    """
    Abstract combinatorial complex class that generalises the pytorch geometric graph (Data).
    Adjacency tensors are stacked in the same fashion as the standard edge_index.
    """

    def __inc__(self, key: str, value: any, *args, **kwargs) -> any:
        num_nodes = getattr(self, "x_0").size(0)
        if "adj" in key:
            i, j = key[4], key[6]
            return torch.tensor(
                [[getattr(self, f"x_{i}").size(0)], [getattr(self, f"x_{j}").size(0)]]
            )
        elif "inv" in key:
            return num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: any, *args, **kwargs) -> any:
        if "adj" in key or "inv" in key:
            return 1
        else:
            return 0


def pad_tensor(
    tensor: torch.Tensor, max_size: int, dim: int, pad_value: float = torch.nan
) -> torch.Tensor:
    """
    Pad a 2D tensor to the specified size along a given dimension with a pad value.

    Empty tensors are handled as a special case.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to be padded.
    max_size : int
        The maximum size to pad the tensor to.
    dim : int
        The dimension along which to pad.
    pad_value : float, optional
        The value to use for padding, by default torch.nan.

    Returns
    -------
    torch.Tensor
        The padded tensor.

    Raises
    ------
    ValueError
        If the input tensor is not 2D.
    """

    # Assert that the input tensor is 2D
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D.")

    # Check if the tensor is empty
    if torch.numel(tensor) == 0:
        # Return an empty tensor of shape (max_size, 0) or (0, max_size)
        # depending on the padding dimension
        if dim == 0:
            return torch.empty((max_size, 0), dtype=tensor.dtype)
        else:  # dim == 1
            return torch.empty((0, max_size), dtype=tensor.dtype)

    pad_sizes = [0] * (2 * len(tensor.size()))  # Pad size must be even, for each dim.
    pad_sizes[(1 - dim) * 2 + 1] = max_size - tensor.size(dim)  # Only pad at the end
    return torch.nn.functional.pad(tensor, pad=pad_sizes, value=pad_value)
