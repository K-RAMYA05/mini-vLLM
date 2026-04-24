"""Microbenchmark: Triton paged attention vs torch.nn.functional.SDPA.

Target claim: ~1.4x over F.scaled_dot_product_attention on A100.

To make the comparison fair, both kernels do the same workload — a
decode-step attention over a gathered KV cache. For SDPA we first gather
every seq's K/V into a contiguous [B, H, T, D] tensor (this matches what
a non-paged engine would need to do). The gather cost is part of SDPA's
path because it's unavoidable without paging.

Usage:
    python benchmarks/bench_kernel.py --head-dim 128
"""
import argparse
import time

import torch
import torch.nn.functional as F

from mini_vllm.block_manager import BlockAllocator, BlockTable, build_block_tables_tensor
from mini_vllm.kernels import paged_attention


def _gather_kv_for_sdpa(k_cache, v_cache, block_tables, context_lens, block_size):
    """Build dense [B, H, Tmax, D] from paged layout. Includes the gather cost."""
    B = block_tables.shape[0]
    _, H, _, D = k_cache.shape
    Tmax = int(context_lens.max().item())
    k_dense = torch.zeros(B, H, Tmax, D, dtype=k_cache.dtype, device=k_cache.device)
    v_dense = torch.zeros_like(k_dense)
    for b in range(B):
        L = int(context_lens[b].item())
        nblocks = (L + block_size - 1) // block_size
        for bi in range(nblocks):
            block_id = int(block_tables[b, bi].item())
            take = min(block_size, L - bi * block_size)
            k_dense[b, :, bi * block_size : bi * block_size + take, :] = \
                k_cache[block_id, :, :take, :]
            v_dense[b, :, bi * block_size : bi * block_size + take, :] = \
                v_cache[block_id, :, :take, :]
    return k_dense, v_dense


def bench(args):
    assert torch.cuda.is_available(), "Need CUDA for this benchmark"
    device = "cuda"
    dtype = torch.float16
    torch.manual_seed(0)

    B = args.batch_size
    H = args.num_heads
    H_kv = args.num_kv_heads
    D = args.head_dim
    T = args.seq_len
    block_size = args.block_size
    scale = D ** -0.5

    # Build a paged cache sized to hold B sequences of length T.
    num_blocks_needed = B * ((T + block_size - 1) // block_size) + 8
    allocator = BlockAllocator(num_blocks_needed)
    block_tables = []
    for _ in range(B):
        bt = BlockTable()
        for _ in range((T + block_size - 1) // block_size):
            bt.append(allocator.allocate())
        block_tables.append(bt)

    k_cache = torch.randn(num_blocks_needed, H_kv, block_size, D, dtype=dtype, device=device)
    v_cache = torch.randn_like(k_cache)
    bt_t = build_block_tables_tensor(
        block_tables, max(len(b) for b in block_tables), device
    )
    ctx = torch.full((B,), T, dtype=torch.int32, device=device)
    q = torch.randn(B, H, D, dtype=dtype, device=device)

    # ---- Triton ----
    for _ in range(5):
        _ = paged_attention(q, k_cache, v_cache, bt_t, ctx, scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        _ = paged_attention(q, k_cache, v_cache, bt_t, ctx, scale)
    torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - t0) * 1000 / args.iters

    # ---- SDPA (gathers included) ----
    group = H // H_kv
    # Pre-gather once for warmup.
    for _ in range(3):
        kd, vd = _gather_kv_for_sdpa(k_cache, v_cache, bt_t, ctx, block_size)
        if group > 1:
            kd = kd.repeat_interleave(group, dim=1)
            vd = vd.repeat_interleave(group, dim=1)
        _ = F.scaled_dot_product_attention(
            q.unsqueeze(2), kd, vd, is_causal=False, scale=scale
        ).squeeze(2)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        kd, vd = _gather_kv_for_sdpa(k_cache, v_cache, bt_t, ctx, block_size)
        if group > 1:
            kd = kd.repeat_interleave(group, dim=1)
            vd = vd.repeat_interleave(group, dim=1)
        _ = F.scaled_dot_product_attention(
            q.unsqueeze(2), kd, vd, is_causal=False, scale=scale
        ).squeeze(2)
    torch.cuda.synchronize()
    sdpa_ms = (time.perf_counter() - t0) * 1000 / args.iters

    print(f"Config: B={B} H={H} H_kv={H_kv} D={D} T={T} block={block_size}")
    print(f"  Triton paged:  {triton_ms:.3f} ms/iter")
    print(f"  SDPA+gather:   {sdpa_ms:.3f} ms/iter")
    print(f"  Speedup:       {sdpa_ms / triton_ms:.2f}x")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()
    bench(args)


if __name__ == "__main__":
    main()
