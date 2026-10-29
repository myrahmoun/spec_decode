"""The target MoE model (gpt-oss-20b): loading and tree verification."""
from ecospec.tree import DraftTree


def load_target(model_id: str, revision: str):
    """Load model + tokenizer from the pinned revision in models.lock."""
    raise NotImplementedError


def verify(model, context_ids, tree: DraftTree, selected: list[int]):
    """Run the target once over the selected nodes with a tree attention mask
    (each node attends to the context and its own ancestors only).

    Returns: accepted token ids (longest matching path + one bonus token),
    and the target hidden states EAGLE-3 needs for the next round.
    """
    raise NotImplementedError