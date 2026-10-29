"""Cost-aware draft selection (paper Section 4.2, Algorithm 1).

Runs once per decoding round, between drafting and verification.
Pure Python: no model, no GPU.
"""
from collections.abc import Callable

from ecospec.tree import DraftTree, LayerExpert, Node

EPS = 1e-6  # paper does not state its epsilon; recorded as our decision


def _greedy_prefix_closed(
    tree: DraftTree,
    gamma: int,
    score: Callable[[Node], float],
    on_pick: Callable[[Node], None] = lambda node: None,
) -> list[int]:
    """Shared greedy loop: root first, then repeatedly the best eligible node.

    A node is eligible once its parent has been picked. Ties on score are broken
    by higher path_prob, then by lower node id, so results are deterministic.
    """
    if gamma < 1:
        raise ValueError(f"gamma must be >= 1 (the root is always verified), got {gamma}")

    root = tree.root
    selected = [root.id]
    on_pick(root)
    picked = {root.id}

    while len(selected) < min(gamma, len(tree.nodes)):
        candidates = [
            n for n in tree.nodes.values() if n.id not in picked and n.parent in picked
        ]
        best = max(candidates, key=lambda n: (score(n), n.path_prob, -n.id))
        selected.append(best.id)
        picked.add(best.id)
        on_pick(best)
    return selected


def select(
    tree: DraftTree,
    gamma: int,
    initial_buffer: set[LayerExpert] | None = None,
) -> list[int]:
    """Pick `gamma` node ids to verify, in the order they were picked.

    Score(node) = path_prob / (|path_experts - buffer| + EPS)
    After each pick, the node's path experts are added to the buffer.

    Our decision (not in the paper): a node is eligible only if its parent is
    already selected, so the result is always a valid prefix-closed subtree.
    """
    buffer = set(initial_buffer) if initial_buffer else set()

    def score(node: Node) -> float:
        return node.path_prob / (len(tree.path_experts(node.id) - buffer) + EPS)

    def on_pick(node: Node) -> None:
        buffer.update(tree.path_experts(node.id))

    return _greedy_prefix_closed(tree, gamma, score, on_pick)


def select_global_cost(tree: DraftTree, gamma: int) -> list[int]:
    """Ablation (paper Eq. 10, Table 4): Score = path_prob / (|path_experts| + EPS).

    Same as select() but with no buffer, so experts already covered by picked
    nodes are not discounted.
    """
    return _greedy_prefix_closed(
        tree, gamma, lambda node: node.path_prob / (len(tree.path_experts(node.id)) + EPS)
    )


def select_by_confidence(tree: DraftTree, gamma: int) -> list[int]:
    """Baseline: standard EAGLE selection, top-gamma by path_prob (prefix-closed)."""
    return _greedy_prefix_closed(tree, gamma, lambda node: node.path_prob)
