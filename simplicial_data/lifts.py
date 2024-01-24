from itertools import combinations

import gudhi


def clique_lift(graph, dim, dis) -> list[list[int]]:
    raise NotImplementedError


def rips_lift(graph, dim, dis, fc_nodes: bool = True) -> list[list[int]]:
    # create simplicial complex
    x_0, pos = graph.x, graph.pos
    points = [pos[i].tolist() for i in range(pos.shape[0])]
    rips_complex = gudhi.RipsComplex(points=points, max_edge_length=dis)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=dim)

    if fc_nodes:
        nodes = [i for i in range(x_0.shape[0])]
        for edge in combinations(nodes, 2):
            simplex_tree.insert(edge)

    # convert simplicial complex to list of lists
    simplexes = []
    for simplex, _ in simplex_tree.get_simplices():
        simplexes.append(simplex)

    return simplexes
