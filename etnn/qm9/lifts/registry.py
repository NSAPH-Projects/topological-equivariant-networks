from .atom import node_lift
from .bond import edge_lift
from .clique import clique_lift
from .functional_group import functional_group_lift
from .molecule import supercell_lift
from .ring import ring_lift
from .rips_vietoris_complex import rips_lift

lifter_registry = {
    "atom": node_lift,  # atom is an alias for node
    "bond": edge_lift,  # bond is an alias for edge
    "clique": clique_lift,
    "edge": edge_lift,
    "functional_group": functional_group_lift,
    "node": node_lift,
    "ring": ring_lift,
    "rips": rips_lift,
    "supercell": supercell_lift,
}
