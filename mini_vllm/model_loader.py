"""Load a HuggingFace Llama model and replace its attention with PagedAttention.

We keep HF's weights, tokenizer, embeddings, MLP, norms, and the overall
forward structure. We surgically replace the attention module in every
decoder layer with our PagedAttention, which shares the Q/K/V/O projections.

Also rewrites LlamaDecoderLayer.forward to pass kv_cache and attn_metadata
through to the attention module. This is a monkeypatch — we don't subclass,
to keep the diff against HF small and readable.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from mini_vllm.attention import PagedAttention


def load_model(
    model_name_or_path: str,
    dtype: torch.dtype,
    device: torch.device | str,
    use_triton: bool,
    prefill_backend: str = "auto",
    trust_remote_code: bool = False,
) -> Tuple[nn.Module, "AutoTokenizer", dict]:
    """Returns (model, tokenizer, model_info_dict)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    model.to(device)

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim = cfg.hidden_size // num_heads

    # Swap attention module in each decoder layer.
    for layer_idx, layer in enumerate(model.model.layers):
        orig = layer.self_attn
        paged = PagedAttention(
            hidden_size=cfg.hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope=model.model.rotary_emb if hasattr(model.model, "rotary_emb") else orig.rotary_emb,
            q_proj=orig.q_proj,
            k_proj=orig.k_proj,
            v_proj=orig.v_proj,
            o_proj=orig.o_proj,
            layer_idx=layer_idx,
            use_triton=use_triton,
            prefill_backend=prefill_backend,
        )
        layer.self_attn = paged

    info = dict(
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=cfg.hidden_size,
        vocab_size=cfg.vocab_size,
        eos_token_id=cfg.eos_token_id if isinstance(cfg.eos_token_id, int) else None,
    )
    return model, tokenizer, info
