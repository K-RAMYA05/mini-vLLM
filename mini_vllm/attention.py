"""Attention module that reads/writes the paged KV cache.

We deliberately do NOT reuse HF's LlamaAttention. Instead we replace it
entirely at load time (see model_loader.py). The replacement:

  - computes Q, K, V using the original projections,
  - writes K, V into the paged cache at the right positions,
  - for prefill: runs a standard causal SDPA over the freshly-written tokens
    (prefill is compute-bound with long sequences, so we defer to the
    optimized CUDA/FlashAttention implementation of F.scaled_dot_product_attention),
  - for decode: calls our Triton paged-attention kernel,
  - applies the output projection.

Prefill and decode share no KV movement code — both write to the same cache
through `KVCache.write_prefill`. The difference is which attention kernel
consumes the cache afterward.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_vllm.kernels import paged_attention


class PagedAttention(nn.Module):
    """Drop-in replacement for LlamaAttention that uses the paged cache.

    Takes over from HF's attention at load time. Holds references to the
    original Q/K/V/O projections so weight loading is untouched.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope,                       # HF-style rotary embedding module
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        o_proj: nn.Linear,
        layer_idx: int,
        use_triton: bool = True,
        prefill_backend: str = "auto",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.rope = rope
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.layer_idx = layer_idx
        self.use_triton = use_triton
        self.prefill_backend = prefill_backend

    def forward(
        self,
        hidden_states: torch.Tensor,       # [total_tokens, hidden]
        position_ids: Optional[torch.Tensor] = None,
        kv_cache=None,                      # KVCache
        attn_metadata: Optional["AttentionMetadata"] = None,
        **kwargs,
    ) -> torch.Tensor:
        if kv_cache is None or attn_metadata is None:
            return self._forward_dense(
                hidden_states,
                position_ids,
                position_embeddings=kwargs.get("position_embeddings"),
            ), None

        total_tokens, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(total_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(total_tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(total_tokens, self.num_kv_heads, self.head_dim)

        # RoPE. HF's rope returns cos/sin broadcast-ready when given position_ids.
        cos, sin = self.rope(v, position_ids.unsqueeze(0))
        q, k = _apply_rope(q, k, cos.squeeze(0), sin.squeeze(0))

        # Split into prefill and decode chunks. The runner has arranged the
        # batch so that prefill tokens come first, then decode tokens.
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_seqs = attn_metadata.num_decode_seqs

        outputs = []

        # ---- PREFILL path ----
        if num_prefill_tokens > 0:
            q_pre = q[:num_prefill_tokens]
            k_pre = k[:num_prefill_tokens]
            v_pre = v[:num_prefill_tokens]

            # Write new K/V into paged cache before running attention.
            # Normally start_pos=0 (admitting a fresh sequence). Speculative
            # decoding reuses this path to write KV at an offset into a
            # sequence that's already partially cached — it sets
            # attn_metadata._prefill_write_offset for that case.
            write_offset = getattr(attn_metadata, "_prefill_write_offset", 0)
            for seq_info in attn_metadata.prefill_seq_infos:
                s, e = seq_info.token_range
                kv_cache.write_prefill(
                    self.layer_idx,
                    seq_info.block_table,
                    start_pos=seq_info.start_pos + write_offset,
                    keys=k_pre[s:e],
                    values=v_pre[s:e],
                )

            # Run causal SDPA per-sequence. Prefill batches are typically
            # few sequences × many tokens, so the per-seq Python overhead
            # is negligible. (Padded batch SDPA would also work here but
            # wastes flops on padding.)
            out_prefill = torch.empty_like(q_pre)
            group = self.num_heads // self.num_kv_heads
            for seq_info in attn_metadata.prefill_seq_infos:
                s, e = seq_info.token_range
                qs = q_pre[s:e].transpose(0, 1)  # [H, T, D]
                ks = k_pre[s:e].transpose(0, 1)  # [H_kv, T, D]
                vs = v_pre[s:e].transpose(0, 1)
                if group > 1:
                    ks = ks.repeat_interleave(group, dim=0)
                    vs = vs.repeat_interleave(group, dim=0)
                prefix_len = seq_info.start_pos + write_offset
                if prefix_len > 0:
                    prefix_k, prefix_v = kv_cache.read_tokens(
                        self.layer_idx, seq_info.block_table, prefix_len
                    )
                    prefix_k = prefix_k.transpose(0, 1)
                    prefix_v = prefix_v.transpose(0, 1)
                    if group > 1:
                        prefix_k = prefix_k.repeat_interleave(group, dim=0)
                        prefix_v = prefix_v.repeat_interleave(group, dim=0)
                    attn_out = _attention_prefill_with_prefix(
                        qs.unsqueeze(0),
                        torch.cat([prefix_k, ks], dim=1).unsqueeze(0),
                        torch.cat([prefix_v, vs], dim=1).unsqueeze(0),
                        prefix_len=prefix_len,
                        scale=self.scale,
                    )
                else:
                    attn_out = _attention_prefill(
                        qs.unsqueeze(0), ks.unsqueeze(0), vs.unsqueeze(0),
                        self.scale, self.prefill_backend,
                    )
                out_prefill[s:e] = attn_out.squeeze(0).transpose(0, 1)
            outputs.append(out_prefill)

        # ---- DECODE path ----
        if num_decode_seqs > 0:
            q_dec = q[num_prefill_tokens:]          # [num_decode_seqs, H, D]
            k_dec = k[num_prefill_tokens:]          # [num_decode_seqs, H_kv, D]
            v_dec = v[num_prefill_tokens:]

            # Append each decode token to its sequence's current position.
            for i, seq_info in enumerate(attn_metadata.decode_seq_infos):
                kv_cache.write_prefill(
                    self.layer_idx,
                    seq_info.block_table,
                    start_pos=seq_info.context_len - 1,  # just-added token pos
                    keys=k_dec[i:i + 1],
                    values=v_dec[i:i + 1],
                )

            # Now read back via paged attention.
            key_layer, value_layer = kv_cache.get_kv_tensors(self.layer_idx)
            if self.use_triton:
                attn_fn = paged_attention
            else:
                from mini_vllm.kernels import reference_paged_attention as attn_fn

            out_decode = attn_fn(
                q_dec, key_layer, value_layer,
                attn_metadata.decode_block_tables,
                attn_metadata.decode_context_lens,
                self.scale,
            )  # [num_decode_seqs, H, D]
            outputs.append(out_decode)

        merged = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
        merged = merged.reshape(total_tokens, self.num_heads * self.head_dim)
        return self.o_proj(merged)

    def _forward_dense(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        position_embeddings=None,
    ) -> torch.Tensor:
        """Dense causal attention fallback for HF forward/calibration calls."""
        was_flat = hidden_states.dim() == 2
        if was_flat:
            hidden_states = hidden_states.unsqueeze(0)
        batch, seq_len, _ = hidden_states.shape
        flat = hidden_states.reshape(batch * seq_len, -1)

        q = self.q_proj(flat).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(flat).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(flat).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        elif position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = self.rope(
                v.reshape(batch * seq_len, self.num_kv_heads, self.head_dim),
                position_ids,
            )
        q, k = _apply_rope(
            q.reshape(batch * seq_len, self.num_heads, self.head_dim),
            k.reshape(batch * seq_len, self.num_kv_heads, self.head_dim),
            cos.reshape(batch * seq_len, self.head_dim),
            sin.reshape(batch * seq_len, self.head_dim),
        )
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.transpose(1, 2)

        group = self.num_heads // self.num_kv_heads
        if group > 1:
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)

        out = _attention_prefill(q, k, v, self.scale, self.prefill_backend)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        return out.squeeze(0) if was_flat else out


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary embeddings.

    q, k: [T, H, D]. cos, sin: [T, D] (already position-selected by HF's rope).
    We broadcast over H.
    """
    cos = cos.unsqueeze(1)  # [T, 1, D]
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _sdpa_backend(backend: str):
    """Select PyTorch's SDPA backend when the installed torch exposes it."""
    if backend == "auto":
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:
        if not torch.cuda.is_available():
            return nullcontext()
        flags = {
            "flash": dict(enable_flash=True, enable_mem_efficient=False, enable_math=False),
            "mem_efficient": dict(enable_flash=False, enable_mem_efficient=True, enable_math=False),
            "math": dict(enable_flash=False, enable_mem_efficient=False, enable_math=True),
        }[backend]
        return torch.backends.cuda.sdp_kernel(**flags)

    selected = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }[backend]
    return sdpa_kernel(selected)


def _flash3_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run FlashAttention-3 on Hopper GPUs via the external hopper package."""
    if not q.is_cuda:
        raise RuntimeError("prefill_attention_backend='flash3' requires CUDA tensors")
    if torch.cuda.get_device_capability(q.device)[0] < 9:
        raise RuntimeError(
            "prefill_attention_backend='flash3' requires Hopper-class GPUs (SM90+, e.g. H100/H800)"
        )
    try:
        import flash_attn_interface
    except ImportError as exc:
        raise RuntimeError(
            "prefill_attention_backend='flash3' requires the FlashAttention-3 hopper package "
            "(install from flash-attention/hopper so `import flash_attn_interface` works)"
        ) from exc

    out = flash_attn_interface.flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        softmax_scale=scale,
        causal=True,
    )
    return out.transpose(1, 2)


def _flash2_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run FlashAttention-2 via the `flash_attn` package (Ampere, A100)."""
    if not q.is_cuda:
        raise RuntimeError("flash-attn backend requires CUDA tensors")
    try:
        from flash_attn import flash_attn_func
    except ImportError as exc:
        raise RuntimeError(
            "flash-attn backend requires the `flash-attn` package "
            "(install with: pip install flash-attn --no-build-isolation)"
        ) from exc
    out = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        causal=True,
        softmax_scale=scale,
    )
    return out.transpose(1, 2)


def _should_prefer_flash3(q: torch.Tensor) -> bool:
    """True when device is Hopper (SM90+). Does not check package install;
    _flash3_attention will raise with a clear message if it's missing."""
    if not q.is_cuda:
        return False
    return torch.cuda.get_device_capability(q.device)[0] >= 9


def _should_prefer_flash2(q: torch.Tensor) -> bool:
    """True when device is Ampere (SM80/SM86) — below Hopper."""
    if not q.is_cuda:
        return False
    major = torch.cuda.get_device_capability(q.device)[0]
    return 8 <= major < 9


def _attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    backend: str,
) -> torch.Tensor:
    """Run causal prefill attention with flash-attn (v2/v3) or PyTorch SDPA.

    q/k/v use PyTorch SDPA layout [B, H, T, D]. The flash-attn package uses
    [B, T, H, D], so this wrapper keeps the call sites backend-agnostic.

    Strict device routing (no silent fallback):
      - 'flash3'    : FA3, Hopper only.
      - 'flash_attn': FA2 via `flash_attn` package.
      - 'flash'     : FA3 on Hopper, FA2 on Ampere; raises on other CUDA devices.
      - 'auto'      : same as 'flash' on CUDA; falls back to SDPA on CPU.
      - other       : PyTorch SDPA with the requested kernel.
    """
    if backend == "flash3":
        return _flash3_attention(q, k, v, scale)
    if backend == "flash_attn":
        return _flash2_attention(q, k, v, scale)

    if backend in ("flash", "auto"):
        if _should_prefer_flash3(q):
            return _flash3_attention(q, k, v, scale)
        if _should_prefer_flash2(q):
            return _flash2_attention(q, k, v, scale)
        if backend == "flash":
            raise RuntimeError(
                "prefill_attention_backend='flash' requires an SM80+ CUDA GPU "
                "(A100 for FA2, H100 for FA3). Use 'math' or 'mem_efficient' "
                "for CPU / pre-Ampere devices."
            )

    with _sdpa_backend(backend):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


def _attention_prefill_with_prefix(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    prefix_len: int,
    scale: float,
) -> torch.Tensor:
    """Run prefill attention when the sequence already has cached prefix KV."""
    _, _, q_len, _ = q.shape
    total_k = k.shape[2]
    mask = torch.full((q_len, total_k), float("-inf"), dtype=q.dtype, device=q.device)
    for row in range(q_len):
        mask[row, : prefix_len + row + 1] = 0
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False, scale=scale)
