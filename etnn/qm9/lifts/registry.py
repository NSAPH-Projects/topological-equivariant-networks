from .atom import atom_lift, node_lift
from .bond import bond_lift, edge_lift
from .clique import clique_lift
from .functional_group import functional_group_lift
from .molecule import supercell_lift
from .ring import cycle_lift, ring_lift
from .rips_vietoris_complex import rips_lift

LIFTER_REGISTRY = {
    "atom": atom_lift,
    "bond": bond_lift,
    "clique": clique_lift,
    "cycle": cycle_lift,
    "edge": edge_lift,
    "functional_group": functional_group_lift,
    "node": node_lift,
    "ring": ring_lift,
    "rips": rips_lift,
    "supercell": supercell_lift,
}
