"""Tests for DraftTree and the two selection strategies."""
import pytest

from ecospec.selection import select, select_by_confidence, select_global_cost
from ecospec.tree import DraftTree, Node


def E(i: int) -> tuple[int, int]:
    """Expert i in MoE layer 0 (the paper's Figure 3 uses a single layer)."""
    return (0, i)


def make_tree(*nodes: Node) -> DraftTree:
    return DraftTree(nodes={n.id: n for n in nodes})


@pytest.fixture
def figure3_tree() -> DraftTree:
    """Paper Figure 3: root t0 with children t1 and t2."""
    return make_tree(
        Node(id=0, parent=None, token=100, path_prob=1.0, experts={E(1), E(2), E(3), E(4)}),
        Node(id=1, parent=0, token=101, path_prob=0.53, experts={E(4), E(6), E(7), E(8)}),
        Node(id=2, parent=0, token=102, path_prob=0.4, experts={E(3), E(4), E(5), E(6)}),
    )


def assert_prefix_closed(tree: DraftTree, selected: list[int]) -> None:
    """Every node's parent must be picked before the node itself."""
    for i, node_id in enumerate(selected):
        parent = tree.nodes[node_id].parent
        if parent is not None:
            assert parent in selected[:i], f"node {node_id} picked before its parent {parent}"


# --- DraftTree ---------------------------------------------------------------

def test_root_path_and_path_experts():
    tree = make_tree(
        Node(id=0, parent=None, token=1, path_prob=1.0, experts={E(1)}),
        Node(id=1, parent=0, token=2, path_prob=0.5, experts={E(2)}),
        Node(id=2, parent=1, token=3, path_prob=0.25, experts={E(1), E(3)}),
    )
    assert tree.root.id == 0
    assert [n.id for n in tree.path(2)] == [0, 1, 2]
    assert [n.id for n in tree.path(0)] == [0]
    assert tree.path_experts(2) == {E(1), E(2), E(3)}


def test_root_requires_exactly_one():
    no_root = make_tree(Node(id=0, parent=1, token=1, path_prob=1.0),
                        Node(id=1, parent=0, token=2, path_prob=1.0))
    with pytest.raises(ValueError):
        no_root.root
    two_roots = make_tree(Node(id=0, parent=None, token=1, path_prob=1.0),
                          Node(id=1, parent=None, token=2, path_prob=1.0))
    with pytest.raises(ValueError):
        two_roots.root


def test_path_detects_cycle():
    tree = make_tree(Node(id=0, parent=1, token=1, path_prob=1.0),
                     Node(id=1, parent=0, token=2, path_prob=1.0))
    with pytest.raises(ValueError):
        tree.path(0)


# --- EcoSpec selection --------------------------------------------------------

def test_figure3_example(figure3_tree):
    # After t0 the buffer is {E1..E4}: t1 adds 3 new experts (0.53/3 ≈ 0.177),
    # t2 adds 2 (0.4/2 = 0.2), so t2 wins despite its lower probability.
    assert select(figure3_tree, gamma=2) == [0, 2]


def test_figure3_continues_after_buffer_update(figure3_tree):
    # After t2 the buffer also holds E5, E6; t1 now adds only E7, E8.
    assert select(figure3_tree, gamma=3) == [0, 2, 1]


def test_gamma_one_is_just_the_root(figure3_tree):
    assert select(figure3_tree, gamma=1) == [0]


def test_gamma_larger_than_tree_returns_all_nodes(figure3_tree):
    selected = select(figure3_tree, gamma=10)
    assert sorted(selected) == [0, 1, 2]
    assert_prefix_closed(figure3_tree, selected)


def test_gamma_must_be_positive(figure3_tree):
    with pytest.raises(ValueError):
        select(figure3_tree, gamma=0)


def test_ties_broken_deterministically_by_lower_id():
    # Children 1 and 2 have identical probability and identical new-expert counts.
    tree = make_tree(
        Node(id=0, parent=None, token=1, path_prob=1.0, experts={E(1)}),
        Node(id=2, parent=0, token=3, path_prob=0.4, experts={E(3)}),
        Node(id=1, parent=0, token=2, path_prob=0.4, experts={E(2)}),
    )
    assert select(tree, gamma=2) == [0, 1]
    assert select_by_confidence(tree, gamma=2) == [0, 1]


def test_score_tie_broken_by_higher_path_prob():
    # Both children add zero new experts (score = p / EPS); EPS makes the higher
    # probability win, but we also want this to hold if scores were exactly equal.
    tree = make_tree(
        Node(id=0, parent=None, token=1, path_prob=1.0, experts={E(1), E(2)}),
        Node(id=1, parent=0, token=2, path_prob=0.3, experts={E(1)}),
        Node(id=2, parent=0, token=3, path_prob=0.6, experts={E(2)}),
    )
    assert select(tree, gamma=2) == [0, 2]


def test_prefix_closure_blocks_cheap_grandchild():
    # Grandchild 3 adds no experts beyond its parent, but its parent (1) is
    # expensive. It must not be picked before node 1.
    tree = make_tree(
        Node(id=0, parent=None, token=1, path_prob=1.0, experts={E(1)}),
        Node(id=1, parent=0, token=2, path_prob=0.5, experts={E(2), E(3), E(4)}),
        Node(id=2, parent=0, token=3, path_prob=0.3, experts={E(5)}),
        Node(id=3, parent=1, token=4, path_prob=0.45, experts={E(2)}),
    )
    # Scores after root: node1 = 0.5/3 ≈ 0.167, node2 = 0.3/1 = 0.3.
    assert select(tree, gamma=2) == [0, 2]
    assert select(tree, gamma=3) == [0, 2, 1]
    assert select(tree, gamma=4) == [0, 2, 1, 3]


def test_initial_buffer_counts_as_already_loaded(figure3_tree):
    buffer = {E(6), E(7), E(8)}
    # With E6-E8 already loaded, t1 adds nothing new and wins.
    assert select(figure3_tree, gamma=2, initial_buffer=buffer) == [0, 1]
    assert buffer == {E(6), E(7), E(8)}, "caller's buffer must not be mutated"


def test_selection_is_prefix_closed_on_deeper_tree():
    # Depth-3, top-2 tree like the paper's config, with varied experts.
    nodes = [Node(id=0, parent=None, token=0, path_prob=1.0, experts={E(0)})]
    next_id = 1
    frontier = [0]
    for depth in range(3):
        new_frontier = []
        for parent in frontier:
            for k in range(2):
                p = nodes[parent].path_prob * (0.7 if k == 0 else 0.2)
                experts = {E((next_id * 7 + j) % 16) for j in range(2)}
                nodes.append(Node(id=next_id, parent=parent, token=next_id,
                                  path_prob=p, experts=experts))
                new_frontier.append(next_id)
                next_id += 1
        frontier = new_frontier
    tree = make_tree(*nodes)
    for gamma in range(1, len(nodes) + 2):
        for selected in (select(tree, gamma), select_by_confidence(tree, gamma)):
            assert len(selected) == min(gamma, len(nodes))
            assert len(set(selected)) == len(selected)
            assert_prefix_closed(tree, selected)


# --- Confidence baseline -----------------------------------------------------

def test_confidence_baseline_ignores_experts(figure3_tree):
    assert select_by_confidence(figure3_tree, gamma=2) == [0, 1]
    assert select_by_confidence(figure3_tree, gamma=3) == [0, 1, 2]


def test_confidence_baseline_is_prefix_closed_on_ties():
    # Child and grandchild share the same path_prob (grandchild had draft prob 1.0).
    # The grandchild must still come after its parent.
    tree = make_tree(
        Node(id=0, parent=None, token=1, path_prob=1.0),
        Node(id=3, parent=0, token=2, path_prob=0.5),
        Node(id=1, parent=3, token=3, path_prob=0.5),
        Node(id=2, parent=0, token=4, path_prob=0.2),
    )
    assert select_by_confidence(tree, gamma=3) == [0, 3, 1]


# --- Global-Cost ablation ----------------------------------------------------

def test_global_cost_ignores_buffer():
    # Node b reuses exactly the experts node a brings in. EcoSpec sees b as free
    # after a is picked; Global Cost still charges b for its full path footprint.
    tree = make_tree(
        Node(id=0, parent=None, token=1, path_prob=1.0, experts={E(1)}),
        Node(id=1, parent=0, token=2, path_prob=0.5, experts={E(2), E(3)}),          # a
        Node(id=2, parent=0, token=3, path_prob=0.3, experts={E(2), E(3)}),          # b
        Node(id=3, parent=0, token=4, path_prob=0.42, experts={E(4), E(5), E(6)}),   # c
    )
    # Global: a = 0.5/3, b = 0.3/3 = 0.1, c = 0.42/4 = 0.105 -> a, then c.
    assert select_global_cost(tree, gamma=3) == [0, 1, 3]
    # EcoSpec: after a, b adds nothing new -> a, then b.
    assert select(tree, gamma=3) == [0, 1, 2]


def test_global_cost_figure3(figure3_tree):
    # t1 = 0.53/7 ≈ 0.076, t2 = 0.4/6 ≈ 0.067 (paper Fig. 3, step 1 values).
    assert select_global_cost(figure3_tree, gamma=2) == [0, 1]
