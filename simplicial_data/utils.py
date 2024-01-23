import numpy as np
import torch


def map_to_tensors(input_dict):
    output_dict = {}
    for key, value in input_dict.items():
        if value:
            tensor = torch.tensor(value, dtype=torch.float32)
        else:
            # For empty lists, create tensors with the specified size
            # The size is (0, key + 1) based on your example
            tensor = torch.empty((0, key + 1), dtype=torch.float32)
        output_dict[key] = tensor
    return output_dict


def sparse_to_dense(sparse_matrix):
    # Extract row and column indices of non-zero elements
    rows, cols = sparse_matrix.nonzero()

    # Convert to a 2D NumPy array
    dense_array = np.array([rows, cols])

    # Convert the NumPy array to a PyTorch tensor
    return torch.from_numpy(dense_array).type(torch.int64)
