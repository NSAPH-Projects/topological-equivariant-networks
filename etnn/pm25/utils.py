import torch
import json
from torch_geometric.data import InMemoryDataset
from etnn.combinatorial_complexes import (
    CombinatorialComplexData,
    CombinatorialComplexTransform,
    CombinatorialComplexCollater,
)


class SpatialCC(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["spatial_cc.json", "point_to_cell.csv"]

    @property
    def processed_file_names(self):
        return ["spatial_data.pt"]

    def process(self):
        # Read data into huge `Data` list.
        path = f"{self.raw_dir}/{self.raw_file_names[0]}"
        data = CombinatorialComplexData.from_json(path)
        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


def standardize_cc(data: CombinatorialComplexData) -> CombinatorialComplexData:
    pass


def add_virtual_node(data: CombinatorialComplexData) -> CombinatorialComplexData:
    pass


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from etnn.combinatorial_complexes import CombinatorialComplexCollater

    # quick test
    dataset = SpatialCC(root="data", force_reload=True)
    follow_batch = ["cell_0", "cell_1", "cell_2"]
    collate_fn = CombinatorialComplexCollater(dataset, follow_batch=follow_batch)
    loader = DataLoader(dataset, collate_fn=collate_fn)
    for batch in loader:
        pass
