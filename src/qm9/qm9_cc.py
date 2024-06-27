import os
import os.path as osp
import sys
from typing import Callable, List, Optional

import torch
from torch import Tensor
from torch_geometric.data import Data, download_url, extract_zip
from torch_geometric.utils import one_hot, scatter
from tqdm import tqdm

from combinatorial_data.lifter import Lifter
from qm9.lifts.registry import lifter_registry
from combinatorial_data.combinatorial_data_utils import CombinatorialComplexTransform, InMemoryCCDataset

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor(
    [
        1.0,
        1.0,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        1.0,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        1.0,
        KCALMOL2EV,
        KCALMOL2EV,
        KCALMOL2EV,
        KCALMOL2EV,
        1.0,
        1.0,
        1.0,
    ]
)

atomrefs = {
    6: [0.0, 0.0, 0.0, 0.0, 0.0],
    7: [-13.61312172, -1029.86312267, -1485.30251237, -2042.61123593, -2713.48485589],
    8: [-13.5745904, -1029.82456413, -1485.26398105, -2042.5727046, -2713.44632457],
    9: [-13.54887564, -1029.79887659, -1485.2382935, -2042.54701705, -2713.42063702],
    10: [-13.90303183, -1030.25891228, -1485.71166277, -2043.01812778, -2713.88796536],
    11: [0.0, 0.0, 0.0, 0.0, 0.0],
}


def get_adjacency_types(
    max_dim: int, connectivity: str, neighbor_types: list[str], visible_dims: list[int] | None
) -> list[str]:
    """
    Generate a list of adjacency type strings based on the specified connectivity pattern.

    Parameters
    ----------
    max_dim : int
        The maximum dimension (inclusive) for which to generate adjacency types. Represents the
        highest rank of cells in the connectivity pattern.
    connectivity : str
        The connectivity pattern to use. Must be one of the options defined below:
        - "self_and_next" generates adjacencies where each rank is connected to itself and the next
        (higher) rank.
        - "self_and_higher" generates adjacencies where each rank is connected to itself and all
        higher ranks.
        - "self_and_previous" generates adjacencies where each rank is connected to itself and the
        previous (lower) rank.
        - "self_and_lower" generates adjacencies where each rank is connected to itself and all
        lower ranks.
        - "self_and_neighbors" generates adjacencies where each rank is connected to itself, the
        next (higher) rank and the previous (lower) rank.
        - "all_to_all" generates adjacencies where each rank is connected to every other rank,
        including itself.
        - "legacy" ignores the max_dim parameter and returns ['0_0', '0_1', '1_1', '1_2'].
    neighbor_types : list[str]
        The types of adjacency between cells of the same rank. Must be one of the following:
        +1: two cells of same rank i are neighbors if they are both neighbors of a cell of rank i+1
        -1: two cells of same rank i are neighbors if they are both neighbors of a cell of rank i-1
        max: two cells of same rank i are neighbors if they are both neighbors of a cell of max rank
        min: two cells of same rank i are neighbors if they are both neighbors of a cell of min rank
    visible_dims: list[int] | None
        A list of ranks to explicitly represent as nodes. If None, all ranks are represented.

    Returns
    -------
    list[str]
        A list of strings representing the adjacency types for the specified connectivity pattern.
        Each string is in the format "i_j" where "i" and "j" are ranks indicating an adjacency
        from rank "i" to rank "j".

    Raises
    ------
    ValueError
        If `connectivity` is not one of the known connectivity patterns.

    Examples
    --------
    >>> get_adjacency_types(2, "self_and_next", ["+1"])
    ['0_0_1', '0_1', '1_1_2', '1_2']

    >>> get_adjacency_types(2, "self_and_higher", ["-1"])
    ['0_1', '0_2', '1_1_0', '1_2', '2_2_1']

    >>> get_adjacency_types(2, "all_to_all", ["-1", "+1", "max", "min"])
    ['0_0_1', '0_0_2','0_1', '0_2', '1_0', '1_1_0', '1_1_2', '1_2', '2_0', '2_1', '2_2_1', '2_2_0']
    """
    adj_types = []
    if connectivity not in [
        "self",
        "self_and_next",
        "self_and_higher",
        "self_and_previous",
        "self_and_lower",
        "self_and_neighbors",
        "all_to_all",
        "legacy",
    ]:
        raise ValueError(f"{connectivity} is not a known connectivity pattern!")

    if connectivity == "self":
        for i in range(max_dim + 1):
            adj_types.append(f"{i}_{i}")

    elif connectivity == "self_and_next":
        for i in range(max_dim + 1):
            adj_types.append(f"{i}_{i}")
            if i < max_dim:
                adj_types.append(f"{i}_{i+1}")

    elif connectivity == "self_and_higher":
        for i in range(max_dim + 1):
            for j in range(i, max_dim + 1):
                adj_types.append(f"{i}_{j}")

    elif connectivity == "self_and_previous":
        for i in range(max_dim + 1):
            adj_types.append(f"{i}_{i}")
            if i > 0:
                adj_types.append(f"{i}_{i-1}")

    elif connectivity == "self_and_lower":
        for i in range(max_dim + 1):
            for j in range(0, i):
                adj_types.append(f"{i}_{j}")

    elif connectivity == "self_and_neighbors":
        for i in range(max_dim + 1):
            adj_types.append(f"{i}_{i}")
            if i > 0:
                adj_types.append(f"{i}_{i-1}")
            if i < max_dim:
                adj_types.append(f"{i}_{i+1}")

    elif connectivity == "all_to_all":
        for i in range(max_dim + 1):
            for j in range(max_dim + 1):
                adj_types.append(f"{i}_{j}")

    else:
        adj_types = ["0_0", "0_1", "1_1", "1_2"]

    # Add one adjacency type for each neighbor type
    new_adj_types = []
    for adj_type in adj_types:
        i, j = map(int, adj_type.split("_"))
        if i == j:
            for neighbor_type in neighbor_types:
                if neighbor_type == "+1":
                    if i < max_dim:
                        new_adj_types.append(f"{i}_{i}_{i+1}")
                elif neighbor_type == "-1":
                    if i > 0:
                        new_adj_types.append(f"{i}_{i}_{i-1}")
                elif neighbor_type == "max":
                    if i < max_dim:
                        new_adj_types.append(f"{i}_{i}_{max_dim}")
                elif neighbor_type == "min":
                    if i > 0:
                        new_adj_types.append(f"{i}_{i}_0")
        else:
            new_adj_types.append(adj_type)
    new_adj_types = list(set(new_adj_types))
    adj_types = new_adj_types

    # Filter adjacencies with invisible ranks
    if visible_dims is not None:
        adj_types = [
            adj_type
            for adj_type in adj_types
            if all(int(dim) in visible_dims for dim in adj_type.split("_")[:2])
        ]

    return adj_types

def merge_adjacencies(adjacencies: list[str]) -> list[str]:
    """
    Merge all adjacency types i_i_j into a single i_i.

    We merge adjacencies of the form i_i_j into a single adjacency i_i. This is useful when we want
    to represent all rank i neighbors of a cell of rank i as a single adjacency matrix.

    Parameters
    ----------
    adjacencies : list[str]
        A list of adjacency types.

    Returns
    -------
    list[str]
        A list of merged adjacency types.

    """
    return list(set(["_".join(adj_type.split("_")[:2]) for adj_type in adjacencies]))

class QM9_CC(InMemoryCCDataset):
    r"""
    Lift QM9 to a CombinatorialComplexData.

    Parameters
    ----------
    lifter_names : list[str]
        The names of the lifters to apply.
    neighbor_types : list[str]
        The types of neighbors to consider. Defines adjacency between cells of the same rank.
    connectivity : str
        The connectivity pattern between ranks.
    visible_dims : list[int]
        Specifies which ranks to explicitly represent as nodes.
    initial_features : list[str]
        The initial features to use.
    dim : int
        The ASC dimension.
    dis : bool
        Radius for Rips complex
    merge_neighbors : bool
        Whether to merge neighbors.
    
    The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 19 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    .. note::

        We also provide a pre-processed version of the dataset in case
        :class:`rdkit` is not installed. The pre-processed version matches with
        the manually processed version as outlined in :meth:`process`.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #tasks
        * - 130,831
          - ~18.0
          - ~37.3
          - 11
          - 19
    """  # noqa: E501

    raw_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/" "molnet_publish/qm9.zip"
    raw_url2 = "https://ndownloader.figshare.com/files/3195404"
    processed_url = "https://data.pyg.org/datasets/qm9_v3.zip"

    def __init__(
        self,
        root: str,
        lifter_names, neighbor_types, connectivity, visible_dims, initial_features, dim, dis, merge_neighbors,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        # Initialize subclass-specific attributes

        #dim : int
        #neighbor_types : list[str]
        #connectivity : str
        #visible_dims : list[int]
        adjacencies = get_adjacency_types(
            dim,
            connectivity,
            neighbor_types,
            visible_dims,
        )
        # If merge_neighbors is True, the adjacency types we feed to the model will be the merged ones
        if merge_neighbors:
            processed_adjacencies = merge_adjacencies(adjacencies)
        else:
            processed_adjacencies = adjacencies

        initial_features = sorted(initial_features)
        #lifter_names : list[str]
        #initial_features : str
        #dim : int
        #dis : bool
        self.lifter = Lifter(lifter_names, initial_features, dim, dis, lifter_registry)
        self.adjacencies = adjacencies
        self.processed_adjacencies = processed_adjacencies
        self.dim = dim
        self.merge_neighbors = merge_neighbors

        super().__init__(
            root, 
            transform, pre_transform, pre_filter, force_reload=force_reload
        )
        self.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def atomref(self, target: int) -> Optional[Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa

            return ["gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt"]
        except ImportError:
            return ["qm9_v3.pt"] # not generated

    @property
    def processed_file_names(self) -> str:
        return "data_v3.pt" # not generated

    def download(self) -> None:
        try:
            import rdkit  # noqa

            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(
                osp.join(self.raw_dir, "3195404"), osp.join(self.raw_dir, "uncharacterized.txt")
            )
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self) -> None:
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType

            RDLogger.DisableLog("rdApp.*")

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(
                (
                    "Using a pre-processed version of the dataset. Please "
                    "install 'rdkit' to alternatively process the raw data."
                ),
                file=sys.stderr,
            )

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.save(data_list, self.processed_paths[0])
            return

        types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        with open(self.raw_paths[1], "r") as f:
            target = [
                [float(x) for x in line.split(",")[1:20]] for line in f.read().split("\n")[1:-1]
            ]
            y = torch.tensor(target, dtype=torch.float)
            y = torch.cat([y[:, 3:], y[:, :3]], dim=-1)
            y = y * conversion.view(1, -1)

        with open(self.raw_paths[2], "r") as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

        # Create the transform lifter, dim, adjacencies, processed_adjacencies, merge_neighbors
        #dim : int
        #merge_neighbors : bool
        lift = CombinatorialComplexTransform(
            lifter=self.lifter,
            dim=self.dim,
            adjacencies=self.adjacencies,
            processed_adjacencies=self.processed_adjacencies,
            merge_neighbors=self.merge_neighbors,
        )

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue
            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            rows, cols, edge_types = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [start, end]
                cols += [end, start]
                edge_types += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce="sum").tolist()

            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = (
                torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs], dtype=torch.float)
                .t()
                .contiguous()
            )
            x = torch.cat([x1, x2], dim=-1)

            name = mol.GetProp("_Name")
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

            data = Data(
                x=x,
                z=z,
                pos=pos,
                edge_index=edge_index,
                smiles=smiles,
                edge_attr=edge_attr,
                y=y[i].unsqueeze(0),
                name=name,
                idx=i,
                mol=mol,
            )

            data = lift(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
