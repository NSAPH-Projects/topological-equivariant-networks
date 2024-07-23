import os
import os.path as osp
import sys
from typing import Callable, List, Optional

import torch
from torch import Tensor
from torch_geometric.data import Data, download_url, extract_zip, InMemoryDataset
from torch_geometric.utils import one_hot, scatter
from tqdm import tqdm
import yaml

from etnn.lifter import Lifter, get_adjacency_types
from etnn.qm9.lifts.registry import LIFTER_REGISTRY
from etnn.lifter import CombinatorialComplexTransform

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


class QM9CC(InMemoryDataset):
    r"""
    Lift QM9 to a CombinatorialComplexData.

    Parameters
    ----------
    lifters : list[str]
        The names of the lifters to apply.
    # neighbor_types : list[str]
    #     The types of neighbors to consider. Defines adjacency between cells of the same rank.
    # connectivity : str
    #     The connectivity pattern between ranks.
    # merge_neighbors : bool
    #     Whether to merge neighbors.
    **lifter_kwargs: Additional keyword arguments for the lifter.

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
        **lifter_kwargs: Additional keyword arguments for the lifter.

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

    raw_url = (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
        "molnet_publish/qm9.zip"
    )
    raw_url2 = "https://ndownloader.figshare.com/files/3195404"
    processed_url = "https://data.pyg.org/datasets/qm9_v3.zip"

    targets = [
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "U0",
        "U",
        "H",
        "G",
        "Cv",
        "U0_atom",
        "U_atom",
        "H_atom",
        "G_atom",
        "A",
        "B",
        "C",
    ]

    def __init__(
        self,
        root: str,
        lifters: list[str],
        neighbor_types: list[str],
        # dim,
        connectivity: str,
        # merge_neighbors: str,
        supercell: Optional[bool] = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        **lifter_kwargs,
    ) -> None:
        # Store subclass-specific attributes
        self.lifters = lifters
        self.neighbor_types = neighbor_types
        self.connectivity = connectivity
        # self.merge_neighbors = merge_neighbors
        self.supercell = supercell
        self.dim = len(lifters) - 1

        # Update dimension and lifters if supercell
        if supercell:
            self.dim += 1
            self.lifters.append("supercell:" + str(self.dim))

        # Get lifter and adjacencies
        self.adjacencies = get_adjacency_types(
            self.dim,
            connectivity,
            neighbor_types,
            # visible_dims,
        )

        self.lifter = Lifter(self.lifters, LIFTER_REGISTRY, self.dim, **lifter_kwargs)
        # self.adjacencies = adjacencies
        # self.processed_adjacencies = processed_adjacencies
        # self.merge_neighbors = merge_neighbors

        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
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
            return ["qm9_v3.pt"]  # not generated

    @property
    def processed_file_names(self) -> str:
        return ["data.pt"]  # not generated

    def download(self) -> None:
        try:
            import rdkit  # noqa

            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(
                osp.join(self.raw_dir, "3195404"),
                osp.join(self.raw_dir, "uncharacterized.txt"),
            )
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

        # save config for ease to interpret cached config
        cfg_path = f"{self.raw_dir}/config.yaml"
        with open(cfg_path, "w") as f:
            D = {
                "lifters": self.lifters,
                "neighbor_types": self.neighbor_types,
                "connectivity": self.connectivity,
            }
            yaml.dump(D, f)

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
                [float(x) for x in line.split(",")[1:20]]
                for line in f.read().split("\n")[1:-1]
            ]
            y = torch.tensor(target, dtype=torch.float)
            y = torch.cat([y[:, 3:], y[:, :3]], dim=-1)
            y = y * conversion.view(1, -1)

        with open(self.raw_paths[2], "r") as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

        # Create the transform lifter, dim, adjacencies, processed_adjacencies, merge_neighbors
        lift = CombinatorialComplexTransform(
            lifter=self.lifter,
            # dim=self.dim,
            adjacencies=self.adjacencies,
            # processed_adjacencies=self.processed_adjacencies,
            # merge_neighbors=self.merge_neighbors,
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
                torch.tensor(
                    [atomic_number, aromatic, sp, sp2, sp3, num_hs], dtype=torch.float
                )
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
