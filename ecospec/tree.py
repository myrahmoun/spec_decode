"""The draft tree: the data structure every other component produces or consumes.

Drafter builds it -> predictor annotates it with expert sets -> selection picks
nodes from it -> verifier checks the picked nodes.
"""
from dataclasses import dataclass, field

# One (layer, expert) pair, e.g. (3, 17) = expert 17 in MoE layer 3.
LayerExpert = tuple[int, int]


@dataclass
class Node:
    id: int
    parent: int | None        # None for the root
    token: int                # token id in the TARGET's vocabulary
    path_prob: float          # cumulative draft probability from root to this node
    experts: set[LayerExpert] = field(default_factory=set)  # filled in by the predictor


@dataclass
class DraftTree:
    nodes: dict[int, Node]

    @property
    def root(self) -> Node:
        """The root is the bonus token from the previous round; it is always verified."""
        roots = [n for n in self.nodes.values() if n.parent is None]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root, found {len(roots)}")
        return roots[0]

    def path(self, node_id: int) -> list[Node]:
        """Nodes from the root down to node_id, inclusive."""
        path = []
        current: int | None = node_id
        while current is not None:
            node = self.nodes[current]
            path.append(node)
            current = node.parent
            # A well-formed tree can't have a path longer than its node count.
            if len(path) > len(self.nodes):
                raise ValueError(f"cycle detected on the path to node {node_id}")
        path.reverse()
        return path

    def path_experts(self, node_id: int) -> set[LayerExpert]:
        """Union of predicted experts along the path (paper Eq. 7)."""
        experts: set[LayerExpert] = set()
        for node in self.path(node_id):
            experts |= node.experts
        return experts