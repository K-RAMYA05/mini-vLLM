"""Packed prefill attention kernel.

This is the owned prefill path for packed equal-shape groups:
  - query: [batch, heads, query_len, head_dim]
  - key/value: [batch, heads, kv_len, head_dim]

It handles both fresh prefill (prefix_len=0, kv_len=query_len) and grouped
prefix-cached prefill (prefix_len>0, kv_len=prefix_len+query_len). The kernel
uses an online softmax update so it never materializes the score matrix.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _packed_prefill_kernel(
        out_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        scale,
        q_stride_b, q_stride_h, q_stride_t,
        k_stride_b, k_stride_h, k_stride_t,
        v_stride_b, v_stride_h, v_stride_t,
        o_stride_b, o_stride_h, o_stride_t,
        batch_size,
        num_heads,
        QUERY_LEN: tl.constexpr,
        KV_LEN: tl.constexpr,
        PREFIX_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SLIDING_WINDOW: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch_idx = pid_bh // num_heads
        head_idx = pid_bh % num_heads
        if batch_idx >= batch_size:
            return

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_m = offs_m < QUERY_LEN

        q_ptrs = (
            q_ptr
            + batch_idx * q_stride_b
            + head_idx * q_stride_h
            + offs_m[:, None] * q_stride_t
            + offs_d[None, :]
        )
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        for start_n in range(0, KV_LEN, BLOCK_N):
            cur_n = start_n + offs_n
            mask_n = cur_n < KV_LEN
            k_ptrs = (
                k_ptr
                + batch_idx * k_stride_b
                + head_idx * k_stride_h
                + cur_n[:, None] * k_stride_t
                + offs_d[None, :]
            )
            v_ptrs = (
                v_ptr
                + batch_idx * v_stride_b
                + head_idx * v_stride_h
                + cur_n[:, None] * v_stride_t
                + offs_d[None, :]
            )
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

            scores = tl.dot(q, tl.trans(k)) * scale
            visible_hi = PREFIX_LEN + offs_m
            causal = cur_n[None, :] <= visible_hi[:, None]
            if SLIDING_WINDOW > 0:
                visible_lo = tl.maximum(visible_hi - (SLIDING_WINDOW - 1), 0)
                causal = causal & (cur_n[None, :] >= visible_lo[:, None])
            score_mask = mask_m[:, None] & mask_n[None, :] & causal
            scores = tl.where(score_mask, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            probs = tl.exp(scores - m_new[:, None])
            probs = tl.where(score_mask, probs, 0.0)

            acc = acc * alpha[:, None] + tl.dot(probs.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(probs, axis=1)
            m_i = m_new

        out = acc / l_i[:, None]
        out_ptrs = (
            out_ptr
            + batch_idx * o_stride_b
            + head_idx * o_stride_h
            + offs_m[:, None] * o_stride_t
            + offs_d[None, :]
        )
        tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=mask_m[:, None])


def packed_prefill_attention(
    query: torch.Tensor,   # [batch, heads, query_len, head_dim]
    key: torch.Tensor,     # [batch, heads, kv_len, head_dim]
    value: torch.Tensor,   # [batch, heads, kv_len, head_dim]
    scale: float,
    prefix_len: int = 0,
    sliding_window: int = 0,
) -> torch.Tensor:
    query_len = query.shape[2]
    kv_len = key.shape[2]
    if query_len < 1:
        return torch.empty_like(query)

    if (
        not _HAS_TRITON
        or not query.is_cuda
        or query.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or query.shape[-1] not in (64, 128)
    ):
        return _fallback_prefill_attention(
            query,
            key,
            value,
            scale=scale,
            prefix_len=prefix_len,
            sliding_window=sliding_window,
        )

    batch_size, num_heads, _, head_dim = query.shape
    out = torch.empty_like(query)
    grid = (triton.cdiv(query_len, 32), batch_size * num_heads)
    _packed_prefill_kernel[grid](
        out,
        query,
        key,
        value,
        scale,
        query.stride(0), query.stride(1), query.stride(2),
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        batch_size=batch_size,
        num_heads=num_heads,
        QUERY_LEN=query_len,
        KV_LEN=kv_len,
        PREFIX_LEN=prefix_len,
        HEAD_DIM=head_dim,
        BLOCK_M=32,
        BLOCK_N=32,
        SLIDING_WINDOW=int(sliding_window),
    )
    return out


def _fallback_prefill_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    prefix_len: int,
    sliding_window: int,
) -> torch.Tensor:
    if prefix_len <= 0 and sliding_window <= 0:
        return F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scale)

    query_len = query.shape[2]
    kv_len = key.shape[2]
    pos_q = torch.arange(query_len, device=query.device).unsqueeze(-1)
    pos_k = torch.arange(kv_len, device=query.device).unsqueeze(0)
    visible_hi = prefix_len + pos_q
    mask = pos_k <= visible_hi
    if sliding_window > 0:
        visible_lo = torch.clamp(visible_hi - (sliding_window - 1), min=0)
        mask &= pos_k >= visible_lo
    bias = torch.zeros((query_len, kv_len), dtype=query.dtype, device=query.device)
    bias.masked_fill_(~mask, float("-inf"))
    return F.scaled_dot_product_attention(query, key, value, attn_mask=bias, is_causal=False, scale=scale)


__all__ = ["packed_prefill_attention"]
