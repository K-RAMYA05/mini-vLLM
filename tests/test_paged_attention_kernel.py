"""Correctness tests for the paged attention kernel.

Compares Triton kernel output against the reference PyTorch implementation.
Only runs if CUDA + Triton are available; otherwise skips.
"""
import pytest
import torch

from mini_vllm.block_manager import BlockAllocator, BlockTable, build_block_tables_tensor
from mini_vllm.kernels import paged_attention, reference_paged_attention


CUDA_AVAILABLE = torch.cuda.is_available()
try:
    import triton  # noqa
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

skip_if_no_cuda_triton = pytest.mark.skipif(
    not (CUDA_AVAILABLE and TRITON_AVAILABLE),
    reason="CUDA + Triton required",
)


def _build_random_paged_cache(
    num_seqs, num_kv_heads, head_dim, block_size, seq_lens, dtype, device,
):
    """Build a paged KV cache populated with random data and corresponding
    block tables / context lens. Returns everything needed for a kernel call.
    """
    allocator = BlockAllocator(num_blocks=1024)
    block_tables = []
    for L in seq_lens:
        bt = BlockTable()
        num_blocks_needed = (L + block_size - 1) // block_size
        for _ in range(num_blocks_needed):
            bt.append(allocator.allocate())
        block_tables.append(bt)

    k_cache = torch.randn(
        (1024, num_kv_heads, block_size, head_dim), dtype=dtype, device=device
    )
    v_cache = torch.randn_like(k_cache)

    max_blocks = max(len(bt) for bt in block_tables)
    bt_tensor = build_block_tables_tensor(block_tables, max_blocks, device)
    ctx = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    return k_cache, v_cache, bt_tensor, ctx


@skip_if_no_cuda_triton
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("num_kv_heads,num_heads", [(8, 32), (4, 32), (2, 2)])
def test_triton_matches_reference(head_dim, block_size, num_kv_heads, num_heads):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    num_seqs = 4
    # Heterogeneous seq lengths — important to exercise partial last-block masking.
    seq_lens = [5, 16, 33, 47]

    k_cache, v_cache, bt, ctx = _build_random_paged_cache(
        num_seqs, num_kv_heads, head_dim, block_size, seq_lens, dtype, device,
    )
    q = torch.randn(num_seqs, num_heads, head_dim, dtype=dtype, device=device)
    scale = head_dim ** -0.5

    out_ref = reference_paged_attention(q, k_cache, v_cache, bt, ctx, scale)
    out_triton = paged_attention(q, k_cache, v_cache, bt, ctx, scale)

    # Small tolerance because fp16 accum vs fp32 accum paths differ.
    torch.testing.assert_close(out_triton, out_ref, rtol=1e-2, atol=1e-2)


@skip_if_no_cuda_triton
def test_triton_empty_context():
    """ctx=0 should return zeros, not NaN."""
    device = "cuda"
    dtype = torch.float16
    k_cache, v_cache, bt, _ = _build_random_paged_cache(
        1, 2, 64, 16, [16], dtype, device,
    )
    ctx = torch.tensor([0], dtype=torch.int32, device=device)
    q = torch.randn(1, 2, 64, dtype=dtype, device=device)
    out = paged_attention(q, k_cache, v_cache, bt, ctx, 64 ** -0.5)
    assert torch.all(out == 0)
