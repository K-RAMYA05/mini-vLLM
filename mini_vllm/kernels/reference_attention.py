"""Reference implementation of paged attention.

Pure PyTorch. Slow but obviously correct. Used for:
  - validating the Triton kernel in tests,
  - running on machines without Triton (CPU, old GPUs),
  - as a sanity baseline in benchmarks.

Semantics: decode-style attention. Each sequence has exactly one query token;
keys/values for that sequence are scattered across its block table.
"""
from __future__ import annotations

import math

import torch


def reference_paged_attention(
    query: torch.Tensor,              # [num_seqs, num_heads, head_dim]
    key_cache: torch.Tensor,          # [num_blocks, num_kv_heads, block_size, head_dim]
    value_cache: torch.Tensor,        # [num_blocks, num_kv_heads, block_size, head_dim]
    block_tables: torch.Tensor,       # [num_seqs, max_num_blocks], int32
    context_lens: torch.Tensor,       # [num_seqs], int32 — valid K/V length per seq
    scale: float,
    key_scales: torch.Tensor | None = None,    # [num_blocks, num_kv_heads, block_size]
    value_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Returns output of shape [num_seqs, num_heads, head_dim].

    Handles GQA: if num_heads > num_kv_heads, Q heads are grouped so each
    group of (num_heads // num_kv_heads) shares one KV head.

    If key_cache/value_cache are int8, key_scales/value_scales must be
    provided. Dequantization happens per-token (per-(block, head, slot)).
    """
    num_seqs, num_heads, head_dim = query.shape
    num_blocks, num_kv_heads, block_size, _ = key_cache.shape
    assert num_heads % num_kv_heads == 0
    group = num_heads // num_kv_heads
    is_int8 = key_cache.dtype == torch.int8
    if is_int8 and (key_scales is None or value_scales is None):
        raise ValueError("int8 reference_paged_attention requires scales")
    out_dtype = query.dtype if not is_int8 else (key_scales.dtype)
    device = query.device

    out = torch.empty_like(query) if not is_int8 else torch.empty_like(query, dtype=out_dtype)

    for s in range(num_seqs):
        ctx_len = int(context_lens[s].item())
        if ctx_len == 0:
            out[s] = 0
            continue

        # Gather this sequence's K/V into [ctx_len, num_kv_heads, head_dim].
        n_blocks = (ctx_len + block_size - 1) // block_size
        block_ids = block_tables[s, :n_blocks].tolist()
        k_chunks, v_chunks = [], []
        for bi, block_id in enumerate(block_ids):
            take = block_size if bi < n_blocks - 1 else (ctx_len - bi * block_size)
            if is_int8:
                k_int = key_cache[block_id, :, :take, :].to(torch.float32)
                v_int = value_cache[block_id, :, :take, :].to(torch.float32)
                k_s = key_scales[block_id, :, :take].to(torch.float32).unsqueeze(-1)
                v_s = value_scales[block_id, :, :take].to(torch.float32).unsqueeze(-1)
                k_chunks.append((k_int * k_s).to(out_dtype))
                v_chunks.append((v_int * v_s).to(out_dtype))
            else:
                k_chunks.append(key_cache[block_id, :, :take, :])
                v_chunks.append(value_cache[block_id, :, :take, :])
        k = torch.cat(k_chunks, dim=1)  # [num_kv_heads, ctx_len, head_dim]
        v = torch.cat(v_chunks, dim=1)

        # Expand KV heads for GQA.
        if group > 1:
            k = k.repeat_interleave(group, dim=0)   # [num_heads, ctx_len, head_dim]
            v = v.repeat_interleave(group, dim=0)

        q = query[s]                                 # [num_heads, head_dim]
        # attn = softmax(q @ k^T * scale) @ v
        scores = torch.einsum("hd,hkd->hk", q, k).to(torch.float32) * scale
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        out[s] = torch.einsum("hk,hkd->hd", probs, v)

    return out
