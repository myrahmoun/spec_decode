"""EAGLE-3 drafter: proposes a tree of candidate tokens.

Paper config: 3 draft steps, keep top-2 tokens per step.
Note: the RedHat checkpoint is in vLLM `speculators` format; loading it into
our own HF-Transformers loop may need a small weight-name conversion.
"""
from ecospec.tree import DraftTree


def load_drafter(model_id: str, revision: str, target):
    raise NotImplementedError


def draft_tree(drafter, target_hidden_states, last_token: int,
               depth: int = 3, top_k: int = 2) -> DraftTree:
    """Expand the tree `depth` steps, keeping `top_k` children per node.
    path_prob = product of draft probabilities along the path."""
    raise NotImplementedError