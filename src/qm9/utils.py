import hashlib
import json
import os
import pickle
import random
from argparse import Namespace

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from typing import List, Dict, Tuple, Optional


from combinatorial_data.combinatorial_data_utils import (
    CombinatorialComplexData,
    CombinatorialComplexTransform,
    CustomCollater,
)
from combinatorial_data.lifter import Lifter
from qm9.lifts.registry import lifter_registry


def calc_mean_mad(loader: DataLoader) -> tuple[Tensor, Tensor]:
    """Return mean and mean average deviation of target in loader."""
    values = [graph.y for graph in loader.dataset]
    mean = sum(values) / len(values)
    mad = sum([abs(v - mean) for v in values]) / len(values)
    return mean, mad


def prepare_data(graph: Data, index: int, target_name: str) -> Data:
    """
    Preprocesses the input graph data.

    Two main modifications are made:
    1. The target value is extracted and stored in the 'y' attribute. Since QM9 targets are
    graph-level, we throw away the vector of 'y' values and only keep the target value of the
    first node in the graph, at the given index.
    2. If the target name is 'zpve', the target value is multiplied by 1e3. This is consistent with
    EGNN.
    3. The feature vector of each node  is computed as a concatenation of the one-hot encoding of
    the atomic number, the atomic number scaled by 1/9, and the atomic number scaled by 1/9 squared.

    Parameters
    ----------
    graph : Data
        The input graph data. It should be an instance of the torch_geometric.data.Data class.
    index : int
        The index of the target value to extract. It should be a non-negative integer.
    target_name: str
        The name of the target.

    Returns
    -------
    Data
        The preprocessed graph data. It is an instance of the Data class with modified features.
    """
    graph.y = graph.y[0, index]
    one_hot = graph.x[:, :5]  # only get one_hot for cormorant
    if target_name == "zpve":
        graph.y *= 1e3
    Z_max = 9
    Z = graph.x[:, 5]
    Z_tilde = (Z / Z_max).unsqueeze(1).repeat(1, 5)

    graph.x = torch.cat((one_hot, Z_tilde * one_hot, Z_tilde * Z_tilde * one_hot), dim=1)

    return graph


def lift_qm9_to_cc(args: Namespace) -> list[dict]:
    """
    Lift QM9 dataset to CombinatorialComplexData format.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    Returns
    -------
    list[dict]
        List of Combinatorial Complex representations of QM9 molecules.

    Notes
    -----
    The QM9 dataset is loaded and each sample is transformed into a dictionary representation of
    the CombinatorialComplexData class. We transform to dictionary format to allow for storage as
    JSON files.
    """
    qm9 = QM9("./datasets/QM9")
    # Create the transform
    lifter = Lifter(args, lifter_registry)
    transform = CombinatorialComplexTransform(
        lifter=lifter,
        dim=args.dim,
        adjacencies=args.adjacencies,
        processed_adjacencies=args.processed_adjacencies,
        merge_neighbors=args.merge_neighbors,
    )
    qm9_cc = []
    for graph in tqdm(qm9, desc="Lifting QM9 samples"):
        qm9_cc.append(transform.graph_to_ccdict(graph))
    return qm9_cc


def save_lifted_qm9(storage_path: str, samples: list[dict]) -> None:
    """
    Save the lifted QM9 samples to individual JSON files.

    Parameters
    ----------
    storage_path : str
        The path to the directory where the JSON files will be saved.
    samples : list[dict]
        The list of lifted QM9 samples.

    Returns
    -------
    None
    """

    if os.path.exists(storage_path):
        raise FileExistsError(f"Path '{storage_path}' already exists.")
    os.makedirs(storage_path, exist_ok=True)

    for idx, sample in tqdm(enumerate(samples), desc="Saving lifted QM9 samples"):
        file_name = f"{idx}.json"
        file_path = f"{storage_path}/{file_name}"
        with open(file_path, "w") as f:
            json.dump(sample, f)


def generate_loaders_qm9(args: Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:

    # Load the QM9 dataset just to get the number of samples
    data_root = "./datasets/QM9"
    num_samples = len(QM9(root=data_root))

    # Compute split indices
    with open("misc/egnn_splits.pkl", "rb") as f:
        egnn_splits = pickle.load(f)

    if args.splits == "egnn":
        split_indices = egnn_splits
        for split in egnn_splits.keys():
            random.shuffle(egnn_splits[split])
    elif args.splits == "random":
        indices = list(range(num_samples))
        random.shuffle(indices)
        train_end_idx = len(egnn_splits["train"])
        val_end_idx = train_end_idx + len(egnn_splits["valid"])
        test_end_idx = val_end_idx + len(egnn_splits["test"])
        split_indices = {
            "train": indices[:train_end_idx],
            "valid": indices[train_end_idx:val_end_idx],
            "test": indices[val_end_idx:test_end_idx],
        }
    else:
        raise ValueError(f"Unknown split type: {args.splits}")

    # Subsample if requested
    for split in split_indices.keys():
        n_split = len(split_indices[split])
        if args.num_samples is not None:
            n_split = min(args.num_samples, n_split)
            split_indices[split] = split_indices[split][:n_split]

    # Compute the target index
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
    target_map = {target: target for target in targets}
    for key in ["U0", "U", "H", "G"]:
        target_map[key] = f"{key}_atom"
    assert target_map["U0"] == "U0_atom"
    index = targets.index(target_map[args.target_name])

    # Create DataLoader kwargs
    follow_batch = [f"cell_{i}" for i in range(args.dim + 1)] + ["x"]
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": True,
    }

    # Compute the data path
    relevant_args = [
        "lifters",
        "neighbor_types",
        "connectivity",
        "visible_dims",
        "merge_neighbors",
        "dim",
        "dis",
    ]
    data_path = "./datasets/QM9_" + generate_dataset_dir_name(args, relevant_args)

    # Check if data path already exists
    if not os.path.exists(data_path):
        qm9_cc = lift_qm9_to_cc(args)
        save_lifted_qm9(data_path, qm9_cc)

    # Process data splits
    loaders = {}
    data_files = sorted(os.listdir(data_path))
    for split in ["train", "valid", "test"]:

        # Filter out the relevant data files
        split_files = [data_files[i] for i in split_indices[split]]

        # Load the ccdicts from the data files
        split_ccdicts = []
        for file in tqdm(split_files, desc="Reading lifted QM9 samples"):
            with open(f"{data_path}/{file}", "r") as f:
                split_ccdicts.append(json.load(f))

        # Convert the dictionaries to CombinatorialComplexData objects
        split_dataset = []
        for ccdict in tqdm(
            split_ccdicts, desc="Converting ccdicts to CombinatorialComplexData objects"
        ):
            ccdata = CombinatorialComplexData().from_json(ccdict)
            split_dataset.append(ccdata)

        # Preprocess data
        processed_split_dataset = []
        for cc in tqdm(split_dataset, desc="Preparing data"):
            preprocessed_graph = prepare_data(cc, index, args.target_name)
            processed_split_dataset.append(preprocessed_graph)

        # Create DataLoader
        loaders[split] = torch.utils.data.DataLoader(
            processed_split_dataset,
            collate_fn=CustomCollater(processed_split_dataset, follow_batch=follow_batch),
            **dataloader_kwargs,
        )

    return tuple(loaders.values())


def generate_dataset_dir_name(args, relevant_args) -> str:
    """
    Generate a directory name based on a subset of script arguments.

    Parameters:
    args (dict): A dictionary of all script arguments.
    relevant_args (list): A list of argument names that are relevant to dataset generation.

    Returns:
    str: A hash-based directory name representing the relevant arguments.
    """
    # Convert Namespace to a dictionary
    args_dict = vars(args)

    # Filter the arguments, keeping only the relevant ones
    filtered_args = {key: args_dict[key] for key in relevant_args if key in args_dict}

    # Convert relevant arguments to a JSON string for consistent ordering
    args_str = json.dumps(filtered_args, sort_keys=True)

    # Create a hash of the relevant arguments string
    hash_obj = hashlib.sha256(args_str.encode())
    hash_hex = hash_obj.hexdigest()

    # Optional: truncate the hash for a shorter name
    short_hash = hash_hex[:16]  # First 16 characters

    return short_hash


def extract_bond_features(mol: Chem.Mol) -> List[Dict[str, any]]:
    """
    Extract bond features from a molecule.
    
    Parameters:
        mol (Chem.Mol): RDKit molecule object.

    Returns:
        List[Dict[str, any]]: A list of dictionaries containing bond features such as bond type,
                              whether the bond is conjugated, if it's in a ring, and its stereochemistry.
    """
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    bond_features = []
    for bond in mol.GetBonds():
        features = {
            'bond_type': str(bond.GetBondType()),
            'is_conjugated': bond.GetIsConjugated(),
            'is_in_ring': bond.IsInRing(),
            'stereo': str(bond.GetStereo())
        }
        bond_features.append(features)
    
    return bond_features

def extract_fg_features(mol: Chem.Mol) -> Dict[str, int]:
    """
    Extract counts of various functional groups from a molecule.

    Parameters:
        mol (Chem.Mol): RDKit molecule object.

    Returns:
        Dict[str, int]: A dictionary with keys as functional group names and values as their counts in the molecule.
    """
    patterns = {
        'hydroxyl': '[OH]',
        'carbonyl': 'C=O',
        'amine': 'N',
        'carboxyl': 'C(=O)O',
        'nitro': '[N+](=O)[O-]',
        'sulfonamide': 'S(=O)(=O)N',
    }

    results = {}
    for group_name, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern)
        results[group_name] = len(matches)
    
    return results

def extract_ring_features(mol: Chem.Mol) -> Dict[str, any]:
    """
    Extract features related to rings in the molecule.

    Parameters:
        mol (Chem.Mol): RDKit molecule object.

    Returns:
        Dict[str, any]: A dictionary containing the number of rings, information about fused systems,
                        and details about each ring such as size, aromaticity, presence of heteroatoms, and saturation.
    """
    ring_info = mol.GetRingInfo()
    features = {
        'number_of_rings': ring_info.NumRings(),
        'fused_systems': rdMolDescriptors.CalcNumRings(mol),
        'rings': []
    }

    for ring in ring_info.AtomRings():
        ring_atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
        ring_size = len(ring)
        is_aromatic = all(atom.GetIsAromatic() for atom in ring_atoms)
        has_heteroatom = any(atom.GetSymbol() not in ('C', 'H') for atom in ring_atoms)
        is_saturated = all(atom.GetHybridization() == Chem.HybridizationType.SP3 for atom in ring_atoms if not atom.GetIsAromatic())

        ring_details = {
            'size': ring_size,
            'is_aromatic': is_aromatic,
            'has_heteroatom': has_heteroatom,
            'is_saturated': is_saturated
        }
        features['rings'].append(ring_details)

    return features


def molecule_from_data(data: any) -> Optional[Chem.Mol]:
    """
    Create an RDKit molecule object from data containing a SMILES string.

    Parameters:
        data (any): Data object containing a SMILES string.

    Returns:
        Optional[Chem.Mol]: An RDKit molecule object or None if conversion fails.
    """
    if hasattr(data, 'smiles') and data.smiles:
        mol = Chem.MolFromSmiles(data.smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            return mol
    return None

def featurize_dataset(dataset: List[any]) -> Tuple[List[any], List[any], List[any]]:
    """
    Process a dataset to extract bond features, functional groups, and ring features for each molecule.

    Parameters:
        dataset (List[any]): A list of data objects containing SMILES strings.

    Returns:
        Tuple[List[any], List[any], List[any]]: Three lists containing bond features, functional group features,
                                                and ring features for all processed molecules.
    """
    all_functional_groups, all_ring_features, all_bond_features = [], [], []
    cnt = 0
    for data in tqdm(dataset):
        mol = molecule_from_data(data)
        if mol is None:
            cnt += 1
            print(f"{cnt}: Invalid or missing SMILES string for data item: {data}")
            continue

        bond_features = extract_bond_features(mol)
        all_bond_features.append(bond_features)
        
        functional_groups = extract_fg_features(mol)
        all_functional_groups.append(functional_groups)
        
        ring_features = extract_ring_features(mol)
        all_ring_features.append(ring_features)

    # Save the results
    with open('qm9_bond_features.pkl', 'wb') as f:
        pickle.dump(all_bond_features, f)
    with open('qm9_ring_features.pkl', 'wb') as f:
        pickle.dump(all_ring_features, f)
    with open('qm9_fg_features.pkl', 'wb') as f:
        pickle.dump(all_functional_groups, f)  # Corrected to save functional groups

    return all_bond_features, all_functional_groups, all_ring_features
