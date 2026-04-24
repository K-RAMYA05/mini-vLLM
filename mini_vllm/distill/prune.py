"""Layer pruning for Llama-family draft models.

Approach
--------
We keep a subset of the original decoder layers and discard the rest.
Embeddings, final norm, and lm_head stay in place (they dominate the
remaining param count and are expensive to retrain). Hidden size,
num_heads, num_kv_heads, head_dim are ALL unchanged — only depth is cut.

Why this beats width pruning for a draft:
  - Width pruning (shrinking hidden_size or num_heads) breaks the attention
    pattern the target learned. Token-level behavior diverges sharply.
  - Depth pruning preserves each layer's learned patterns; the draft just
    has fewer of them. Distillation then bridges the output gap.
  - For draft models specifically, we only care about output distribution
    agreement. Keeping the target's feature extractors (early layers) +
    its output projection (late layers) and pruning the middle is the
    standard trick (see Minitron, Sheared-LLaMA).

For an 8B Llama target, keeping 8 out of 32 layers gives a draft with the
same tokenizer, hidden size, head layout, embeddings, and lm_head, but much
lower decoder FLOPs. The absolute parameter count is still large because
embeddings are retained; speculative decoding mostly cares about
draft-forward-time vs. target-forward-time, not parameter count alone.

We expose this honest math in get_pruned_config().
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def _choose_layers_to_keep(total_layers: int, num_keep: int) -> List[int]:
    """Pick which layer indices to keep.

    Strategy: evenly-spaced sampling that always includes the first and
    last layers. This preserves the input-processing and output-shaping
    layers (which we know the target relies on for its final distribution)
    and uniformly samples the middle.

    For 16 -> 6, this picks [0, 3, 6, 9, 12, 15].
    """
    if num_keep >= total_layers:
        return list(range(total_layers))
    if num_keep < 2:
        raise ValueError("num_keep must be >= 2 (need at least first and last layer)")

    # Evenly spaced with endpoints at 0 and total_layers-1.
    step = (total_layers - 1) / (num_keep - 1)
    keep = [int(round(i * step)) for i in range(num_keep)]
    # De-duplicate while preserving order (rounding can collide for small configs).
    seen = set()
    out = []
    for idx in keep:
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def estimate_param_count(
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tied_embeddings: bool = True,
) -> int:
    """Closed-form param count for a Llama-style model. For sanity-printing."""
    embed = vocab_size * hidden_size
    lm_head = 0 if tied_embeddings else vocab_size * hidden_size

    # Per layer:
    q_proj = hidden_size * (num_heads * head_dim)
    k_proj = hidden_size * (num_kv_heads * head_dim)
    v_proj = hidden_size * (num_kv_heads * head_dim)
    o_proj = (num_heads * head_dim) * hidden_size
    attn = q_proj + k_proj + v_proj + o_proj

    # Llama MLP has gate, up, down (SwiGLU).
    gate = hidden_size * intermediate_size
    up = hidden_size * intermediate_size
    down = intermediate_size * hidden_size
    mlp = gate + up + down

    # Norms (~hidden_size each, 2 per layer) — negligible but we'll count them.
    norms = 2 * hidden_size

    per_layer = attn + mlp + norms
    return embed + lm_head + num_layers * per_layer + hidden_size  # + final norm


@torch.no_grad()
def prune_llama_to_n_layers(
    teacher_model,
    num_keep: int,
    keep_indices: Optional[List[int]] = None,
) -> Tuple[nn.Module, List[int]]:
    """Return a new Llama model with only `num_keep` of the teacher's decoder layers.

    The returned model is a proper transformers LlamaForCausalLM instance,
    so training it works with HF's Trainer, PEFT, standard DataLoaders, etc.

    We deep-copy the kept layers' weights (including q/k/v/o/mlp). Embeddings,
    final norm, and lm_head are also copied. Everything else — the `config`
    — is cloned and modified to have num_hidden_layers = num_keep.

    After pruning, the model is a valid autoregressive LM whose outputs will
    be WRONG until it's distilled. That's expected; see train_distill.py.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    import copy

    teacher_config = teacher_model.config
    total_layers = teacher_config.num_hidden_layers

    if keep_indices is None:
        keep_indices = _choose_layers_to_keep(total_layers, num_keep)
    assert len(keep_indices) == num_keep

    # Build new config with reduced depth.
    new_config = copy.deepcopy(teacher_config)
    new_config.num_hidden_layers = num_keep

    # Instantiate a fresh model with the new config (random init), then
    # overwrite the layers we want to copy.
    student_model = AutoModelForCausalLM.from_config(new_config)
    student_model = student_model.to(dtype=teacher_model.dtype, device=teacher_model.device)

    # Copy embeddings, final norm, lm_head.
    student_model.model.embed_tokens.load_state_dict(
        teacher_model.model.embed_tokens.state_dict()
    )
    student_model.model.norm.load_state_dict(
        teacher_model.model.norm.state_dict()
    )
    # lm_head is tied in Llama-3.2 so copying embeddings handles it, but
    # load_state_dict for lm_head is cheap and harmless if untied.
    student_model.lm_head.load_state_dict(teacher_model.lm_head.state_dict())

    # Copy the kept decoder layers.
    for student_idx, teacher_idx in enumerate(keep_indices):
        student_model.model.layers[student_idx].load_state_dict(
            teacher_model.model.layers[teacher_idx].state_dict()
        )

    return student_model, keep_indices
