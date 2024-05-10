# Type alias for a cell in a simplicial complex. Frozenset of node indices and a list of features.
Cell = tuple[frozenset[int], tuple[float]]
