# E(n) Equivariant Topological Neural Network

Code for *Claudio Battiloro, Ege Karaismailoğlu, Mauricio Tec, George Dasoulas, Michelle Audirac, Francesca Dominici* "E(n) Equivariant Topological Neural Networks"
https://arxiv.org/abs/2405.15429

<img src="etnn.png" width="800">

**Abstract** Graph neural networks excel at modeling pairwise interactions, but they cannot flexibly accommodate higher-order interactions and features. Topological deep learning (TDL) has emerged recently as a promising tool for addressing this issue. TDL enables the principled modeling of arbitrary multi-way, hierarchical higher-order interactions by operating on combinatorial topological spaces, such as simplicial or cell complexes, instead of graphs. However, little is known about how to leverage geometric features such as positions and velocities for TDL. This paper introduces E(n)-Equivariant Topological Neural Networks (ETNNs), which are E(n)-equivariant message-passing networks operating on combinatorial complexes, formal objects unifying graphs, hypergraphs, simplicial, path, and cell complexes. ETNNs incorporate geometric node features while respecting rotation and translation equivariance. Moreover, ETNNs are natively ready for settings with heterogeneous interactions. We provide a theoretical analysis to show the improved expressiveness of ETNNs over architectures for geometric graphs. We also show how several E(n) equivariant variants of TDL models can be directly derived from our framework. The broad applicability of ETNNs is demonstrated through two tasks of vastly different nature: i) molecular property prediction on the QM9 benchmark and ii) land-use regression for hyper-local estimation of air pollution with multi-resolution irregular geospatial data. The experiment results indicate that ETNNs are an effective tool for learning from diverse types of richly structured data, highlighting the benefits of principled geometric inductive bias.

## Setup instructions
```
git clone <repo>
cd topological-equivariant-networks
conda env create -f environment.yaml
```

If you are running the code from a Mac use `conda env create -f environment-macos.yaml`

This will create and activate a new conda environment called "etnn" that contains all the packages necessary to run the code in this repository.

## Real-World Combinatorial Complexes
To showcase the modeling power of CCs and their versatility in conjunction with ETNNs, we introduced two Real-World Combinatorial Complexes useful to model molecules and irregular multi-resolution geospatial data:

**Molecular CC** In a Molecular CC, the set of nodes is the set of atoms, while the set of cells and the rank function are such that: atoms are rank 0 cells, bonds and functional groups made of two atoms are rank 1 cells, and rings and functional groups made of more than two atoms are rank 2 cells. A Molecular CC can jointly retain all the properties of graphs, cell complexes, and hypergraphs, thus overcoming their mutual limitations. 

**Spatial CC** In a Spatial CC, the set of cells is defined by hierarchical spatial objects. Two major examples are geometric descriptions (e.g. points, polylines, and polygons) or political partitions (e.g. zipcodes, counties, states) of a region. In the latter case, the listed objects can be modeled as cells of a CC and their rank correspond to their intrinsic dimensionality. Therefore, rank 0 cells are points, e.g. where air quality monitoring units are located, rank 1 cells are polylines, e.g. traffic roads, and rank 2 cells are polygons, e.g. census tracts.

<img src="benchmarks.png" width="800">

The scripts [create_qm9.py](create_qm9.py) and [create_geospatial.py](create_geospatial.py) use the `InMemoryDataset` pytorch-geometric objects to create and store the `Dataset` base class that contain these CCs.
