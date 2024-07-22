import json

import pandas as pd
import requests
from torch_geometric.data import InMemoryDataset

from etnn.combinatorial_data import CombinatorialComplexData


class PM25CC(InMemoryDataset):
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
        return ["geospatialcc.json"]

    @property
    def processed_file_names(self):
        return [f"geospatialcc.pt"]

    def download(self):
        url = "https://raw.githubusercontent.com/NSAPH-Projects/topological-equivariant-networks/main/data/input/geospatialcc.json"
        path = f"{self.raw_dir}/{self.raw_file_names[0]}"
        response = requests.get(url)
        with open(path, "wb") as f:
            f.write(response.content)

    def process(self):
        # Read data into huge `Data` list.
        path = f"{self.raw_dir}/{self.raw_file_names[0]}"
        with open(path, "r") as f:
            ccdict = json.load(f)
        data = CombinatorialComplexData.from_ccdict(ccdict)
        data_list = [data]

        # Add road, tract indictor in the data
        node_cell_indicator = pd.DataFrame(ccdict["points_to_cells"])
        data.index_1 = node_cell_indicator.road
        data.index_2 = node_cell_indicator.tract

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     from etnn_spatial.combinatorial_complexes import CombinatorialComplexCollater
#     from torch_geometric.transforms import Compose

#     # quick test
#     dataset = GeospatialCC(
#         root="data",
#         transform=create_mask,
#         pre_transform=Compose([standardize_cc, squash_cc]),
#         force_reload=True,
#     )
#     follow_batch = ["cell_0", "cell_1", "cell_2"]
#     collate_fn = CombinatorialComplexCollater(dataset, follow_batch=follow_batch)
#     loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=1)
#     for batch in loader:
#         pass
