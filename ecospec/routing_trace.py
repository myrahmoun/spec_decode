"""Record which experts the target's routers actually pick.

Used twice: as training data for the predictor, and as ground truth for
counting active experts (E) per verification pass.

Why we hook the MoE block, not the router: with MXFP4 weights, transformers
replaces each MoE block with one that computes `F.linear(h, router.weight,
router.bias)` directly and never calls `router.forward`, so a hook on the router
would silently record nothing on the real model. Instead we capture the MoE
block's input and recompute the same linear + top-k from the router's weights.
"""
import numpy as np
import torch
import torch.nn.functional as F


class RouterTraceLogger:
    """Context manager that hooks every router module in the target.

    with RouterTraceLogger(model) as log:
        model(input_ids)
    log.experts  # int16 array [num_tokens, num_layers, top_k]

    Tokens from successive forward calls are concatenated in call order; within
    a call, tokens are flattened batch-major. Experts within a row are ordered
    by descending router logit. `num_layers` counts MoE layers only.
    """

    def __init__(self, model):
        self.routers = _find_routers_with_parents(model)
        if not self.routers:
            raise ValueError("no router modules found in model")
        self._handles: list = []
        # _records[layer] is a list of [tokens_in_call, top_k] tensors, one per forward call.
        self._records: list[list[torch.Tensor]] = [[] for _ in self.routers]

    def __enter__(self):
        for slot, (_, router, moe_block) in enumerate(self.routers):
            hook = self._make_hook(slot, router)
            self._handles.append(moe_block.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False

    def _make_hook(self, slot: int, router):
        def hook(module, args):
            hidden_states = args[0]
            with torch.no_grad():
                h = hidden_states.reshape(-1, hidden_states.shape[-1])
                logits = F.linear(h, router.weight, router.bias)
                indices = torch.topk(logits, router.top_k, dim=-1).indices
            # Stay on device: a .cpu() here would force a GPU sync per layer per step.
            self._records[slot].append(indices.to(torch.int16))
        return hook

    @property
    def experts(self) -> np.ndarray:
        counts = {len(r) for r in self._records}
        if counts != {len(self._records[0])}:
            raise RuntimeError(f"layers saw different numbers of forward calls: {counts}")
        top_k = self.routers[0][1].top_k
        if not self._records[0]:
            return np.zeros((0, len(self.routers), top_k), dtype=np.int16)
        per_layer = [torch.cat(calls, dim=0).cpu() for calls in self._records]
        return torch.stack(per_layer, dim=1).numpy()


def find_router_modules(model) -> list[tuple[int, object]]:
    """(layer_index, module) for each router; check_env.py prints their names.

    `layer_index` is the router's position among MoE layers, in model order.
    """
    return [(i, router) for i, router, _ in _find_routers_with_parents(model)]


def _is_router(module) -> bool:
    """Duck-typed: a router has top_k, num_experts and a [num_experts, hidden] weight.

    Avoids depending on class or attribute names, which differ between the
    plain and MXFP4 code paths and between model families.
    """
    weight = getattr(module, "weight", None)
    num_experts = getattr(module, "num_experts", None)
    return (
        isinstance(getattr(module, "top_k", None), int)
        and isinstance(num_experts, int)
        and isinstance(weight, torch.Tensor)
        and weight.dim() == 2
        and weight.shape[0] == num_experts
    )


def _find_routers_with_parents(model) -> list[tuple[int, object, object]]:
    """(layer_index, router, moe_block) where moe_block is the router's parent module."""
    modules = dict(model.named_modules())
    found = []
    for name, module in modules.items():
        if _is_router(module):
            parent_name = name.rpartition(".")[0]
            found.append((len(found), module, modules[parent_name]))
    return found
