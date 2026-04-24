"""Triton paged attention kernel (decode phase).

Design
------
Grid: (num_seqs, num_heads).
Each program handles ONE query token for ONE head. It walks its sequence's
block table, loading one KV block at a time, accumulating attention with an
online softmax (Milakov & Gimelshein 2018). No materialized score matrix.

Why this is faster than calling SDPA in a loop over sequences:

1. Block-sparse gather: the kernel reads K/V directly from the paged cache
   via an indirection through `block_tables`. A dense SDPA path would need
   us to first gather every sequence's K/V into a contiguous [B, T, H, D]
   tensor — that gather is pure memory movement with no compute payoff.

2. One kernel launch for the whole batch, instead of one per sequence. At
   decode, arithmetic intensity is low (~1 FLOP per loaded byte) so launch
   overhead and global-memory bandwidth dominate. Fusing the batch saves
   launches AND keeps Q resident in registers while streaming KV.

3. Fused softmax + PV matmul. The "online softmax" trick means we never
   write the [H, T] score tensor to HBM — we only hold running max/sum
   per head, in registers.

GQA handling: we compute the KV head as `kv_head = head // group`. Each Q
head within a group loads the same KV — this is the standard GQA pattern
and it's fine because the repeat happens via pointer arithmetic, not via
an actual broadcast in memory.

Masking: the last block of a sequence is typically partially filled. We
apply a length mask against `context_len` to force the junk slots' scores
to -inf before softmax.

Limitations (by design, for research-grade):
  - decode only (query_len == 1 per sequence). Prefill uses a separate path.
  - head_dim must be a power of two ≤ 128 (64 and 128 are what Llama uses).
  - one query per sequence; speculative-decode verify uses a batched
    wrapper that tiles the γ+1 query tokens as γ+1 separate "sequences"
    sharing a block table.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _paged_attention_kernel(
        out_ptr,              # [num_seqs, num_heads, head_dim]
        q_ptr,                # [num_seqs, num_heads, head_dim]
        k_cache_ptr,          # [num_blocks, num_kv_heads, block_size, head_dim]
        v_cache_ptr,          # [num_blocks, num_kv_heads, block_size, head_dim]
        block_tables_ptr,     # [num_seqs, max_num_blocks_per_seq] int32
        context_lens_ptr,     # [num_seqs] int32
        scale,
        # strides
        q_stride_s, q_stride_h,
        k_stride_b, k_stride_h, k_stride_t,
        v_stride_b, v_stride_h, v_stride_t,
        o_stride_s, o_stride_h,
        bt_stride_s,
        # shapes (compile-time where possible)
        num_kv_heads: tl.constexpr,
        num_queries_per_kv: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        MAX_NUM_BLOCKS_PER_SEQ: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        kv_head_idx = head_idx // num_queries_per_kv

        ctx_len = tl.load(context_lens_ptr + seq_idx)
        if ctx_len == 0:
            # Write zeros and bail.
            off_out = seq_idx * o_stride_s + head_idx * o_stride_h + tl.arange(0, HEAD_DIM)
            tl.store(out_ptr + off_out, tl.zeros([HEAD_DIM], dtype=tl.float32).to(out_ptr.dtype.element_ty))
            return

        # Load Q for this (seq, head): [HEAD_DIM]
        d_range = tl.arange(0, HEAD_DIM)
        q_off = seq_idx * q_stride_s + head_idx * q_stride_h + d_range
        q = tl.load(q_ptr + q_off).to(tl.float32)

        # Online softmax accumulators.
        m_i = -float("inf")                                      # running max
        l_i = 0.0                                                # running denom
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)             # running output

        t_range = tl.arange(0, BLOCK_SIZE)                       # within-block token offs

        num_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        for blk in range(0, num_blocks):
            # Physical block id for this logical block.
            physical_block = tl.load(block_tables_ptr + seq_idx * bt_stride_s + blk)

            # Valid token count in this block (last block may be partial).
            block_start_pos = blk * BLOCK_SIZE
            valid = tl.minimum(BLOCK_SIZE, ctx_len - block_start_pos)
            mask_t = t_range < valid                             # [BLOCK_SIZE]

            # ---- load K block: [BLOCK_SIZE, HEAD_DIM] ----
            k_base = (
                physical_block * k_stride_b
                + kv_head_idx * k_stride_h
                + t_range[:, None] * k_stride_t
                + d_range[None, :]
            )
            k = tl.load(k_base, mask=mask_t[:, None], other=0.0).to(tl.float32)

            # ---- scores = q @ k^T * scale : [BLOCK_SIZE] ----
            s = tl.sum(q[None, :] * k, axis=1) * scale
            s = tl.where(mask_t, s, -float("inf"))

            # ---- online softmax update ----
            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            alpha = tl.exp(m_i - m_new)                          # rescale old acc
            p = tl.exp(s - m_new)                                # [BLOCK_SIZE], masked lanes ->0
            p = tl.where(mask_t, p, 0.0)

            # ---- load V block and accumulate ----
            v_base = (
                physical_block * v_stride_b
                + kv_head_idx * v_stride_h
                + t_range[:, None] * v_stride_t
                + d_range[None, :]
            )
            v = tl.load(v_base, mask=mask_t[:, None], other=0.0).to(tl.float32)

            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

        # Finalize.
        out = acc / l_i
        off_out = seq_idx * o_stride_s + head_idx * o_stride_h + d_range
        tl.store(out_ptr + off_out, out.to(out_ptr.dtype.element_ty))


def paged_attention(
    query: torch.Tensor,         # [num_seqs, num_heads, head_dim]
    key_cache: torch.Tensor,     # [num_blocks, num_kv_heads, block_size, head_dim]
    value_cache: torch.Tensor,   # [num_blocks, num_kv_heads, block_size, head_dim]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks] int32
    context_lens: torch.Tensor,  # [num_seqs] int32
    scale: float,
) -> torch.Tensor:
    """Triton paged attention. Falls back to reference if Triton isn't present."""
    if not _HAS_TRITON or not query.is_cuda:
        from mini_vllm.kernels.reference_attention import reference_paged_attention
        return reference_paged_attention(
            query, key_cache, value_cache, block_tables, context_lens, scale
        )

    num_seqs, num_heads, head_dim = query.shape
    num_blocks, num_kv_heads, block_size, _ = key_cache.shape
    max_num_blocks = block_tables.shape[1]
    assert num_heads % num_kv_heads == 0
    assert head_dim in (64, 128), f"head_dim {head_dim} not supported"

    out = torch.empty_like(query)

    grid = (num_seqs, num_heads)
    _paged_attention_kernel[grid](
        out, query, key_cache, value_cache, block_tables, context_lens,
        scale,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        out.stride(0), out.stride(1),
        block_tables.stride(0),
        num_kv_heads=num_kv_heads,
        num_queries_per_kv=num_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        MAX_NUM_BLOCKS_PER_SEQ=max_num_blocks,
    )
    return out
