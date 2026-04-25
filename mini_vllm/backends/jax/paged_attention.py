"""Pallas paged-attention decode kernel — SKELETON.

This file is a placeholder for the JAX/Pallas port of the Triton kernel in
mini_vllm/kernels/paged_attention.py. See ./DESIGN.md for the full plan.

Implementation order (track progress here):
  [ ] Phase 1: Pallas kernel running on JAX-CUDA, parity vs Triton.
  [ ] Phase 2: Same kernel running on JAX-TPU (v3-8 via TRC).
  [ ] Phase 3: Wire into a minimal greedy-decode loop in ./inference.py.

Until Phase 1 lands, calls to paged_attention_jax raise RuntimeError with a
clear message; this is intentional — silent fallback would mask whether the
JAX backend is actually being exercised.
"""
from __future__ import annotations


def paged_attention_jax(query, key_cache, value_cache,
                        block_tables, context_lens, scale):
    """Pallas paged-attention decode kernel.

    Mirrors the Triton kernel signature in mini_vllm/kernels/paged_attention.py
    but operates on JAX arrays. See DESIGN.md for the algorithmic plan.

    Args:
        query:        [num_seqs, num_heads, head_dim] bf16
        key_cache:    [num_blocks, num_kv_heads, block_size, head_dim] bf16
        value_cache:  same shape as key_cache
        block_tables: [num_seqs, max_num_blocks_per_seq] int32
        context_lens: [num_seqs] int32
        scale:        float — 1/sqrt(head_dim)

    Returns:
        [num_seqs, num_heads, head_dim] bf16
    """
    raise RuntimeError(
        "JAX/Pallas paged-attention backend is not yet implemented. "
        "See mini_vllm/backends/jax/DESIGN.md for the implementation plan. "
        "Use the Triton path (mini_vllm.kernels.paged_attention) on CUDA."
    )
