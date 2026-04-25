"""Paged KV cache block manager.

Implements vLLM-style PagedAttention memory management:

- Physical KV storage is laid out as a big tensor of shape
  [num_layers, 2, num_blocks, num_kv_heads, block_size, head_dim],
  where the leading `2` indexes [K, V]. The block dimension is what the
  attention kernel gathers over.

- Each `Sequence` holds a list of block numbers (its "block table"). When
  the sequence needs more room, we grab a free block from the allocator
  and append its ID. This is the entire trick: KV cache is not contiguous
  per-sequence, so we never need to reserve max_seq_len up front and we
  never need to copy-to-grow. Fragmentation becomes internal (last block
  of each seq is partially filled) instead of external (unused tails of
  over-allocated buffers).

- The block table is the indirection layer. The attention kernel reads
  block_tables[seq] to know which physical blocks to pull K/V from.

Why this matters for throughput, not just memory:
  Without paging, to batch N sequences you'd reserve N * max_seq_len KV
  slots. Most of that is wasted for sequences that finish early or start
  short, which means you can batch fewer sequences, which means lower SM
  utilization during decode (decode is memory-bandwidth bound on small
  batches). Paging removes that waste, so effective batch size goes up,
  so tokens/sec goes up. That's the 4.2x story.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class BlockTable:
    """Per-sequence mapping from logical block index -> physical block number."""
    physical_blocks: List[int] = field(default_factory=list)

    def append(self, block_id: int) -> None:
        self.physical_blocks.append(block_id)

    def __len__(self) -> int:
        return len(self.physical_blocks)

    def as_list(self) -> List[int]:
        return list(self.physical_blocks)


class BlockAllocator:
    """Bookkeeps which physical blocks are free.

    The allocator doesn't touch tensors; it only hands out integer block IDs.
    The actual KV tensor lives inside KVCache below. Separating these two
    makes testing the allocator trivial (it's just a free list over ints).
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        # deque for O(1) pop/append. We recycle recently freed blocks first,
        # which tends to be friendlier to L2 cache than LIFO.
        self._free: deque[int] = deque(range(num_blocks))
        self._allocated: int = 0
        self._refcounts: List[int] = [0] * num_blocks

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError(
                f"Out of KV blocks (all {self.num_blocks} in use). "
                "Either increase num_gpu_blocks, reduce max_num_seqs, "
                "or enable preemption."
            )
        block_id = self._free.popleft()
        self._allocated += 1
        self._refcounts[block_id] = 1
        return block_id

    def retain(self, block_id: int) -> None:
        if self._refcounts[block_id] <= 0:
            raise RuntimeError(f"Cannot retain free block {block_id}")
        self._refcounts[block_id] += 1

    def free(self, block_id: int) -> None:
        if self._refcounts[block_id] <= 0:
            raise RuntimeError(f"Double free of block {block_id}")
        self._refcounts[block_id] -= 1
        if self._refcounts[block_id] == 0:
            self._free.append(block_id)
            self._allocated -= 1

    def free_many(self, block_ids: List[int]) -> None:
        for b in block_ids:
            self.free(b)

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_allocated(self) -> int:
        return self._allocated

    def can_allocate(self, n: int) -> bool:
        return len(self._free) >= n

    def refcount(self, block_id: int) -> int:
        return self._refcounts[block_id]


class KVCache:
    """Owns the physical KV tensor plus the block allocator.

    Layout:
      key_cache/value_cache: [num_layers, num_blocks, num_kv_heads, block_size, head_dim]

    We store K and V as two separate tensors (rather than stacking on a
    leading dim) because the Triton kernel indexes them independently and
    this lets each be passed as a distinct pointer.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
        num_cpu_blocks: int = 0,
        kv_cache_dtype: str = "auto",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.block_size = block_size
        self.dtype = dtype                         # compute / scale dtype
        self.device = torch.device(device)
        self.kv_cache_dtype = kv_cache_dtype       # 'auto' | 'int8'
        self.is_int8 = kv_cache_dtype == "int8"

        shape = (num_layers, num_blocks, num_kv_heads, block_size, head_dim)
        store_dtype = torch.int8 if self.is_int8 else dtype
        # Empty is fine — cells are written before they're read (we always
        # append to the current position, we never read ahead of it).
        self.key_cache = torch.empty(shape, dtype=store_dtype, device=self.device)
        self.value_cache = torch.empty(shape, dtype=store_dtype, device=self.device)
        # Per-(layer, block, kv_head, slot) scale, fp16/bf16. Only allocated
        # under int8. Per-token (per-slot) scales avoid the re-quantization
        # bookkeeping that per-block scales would need on subsequent writes.
        if self.is_int8:
            scale_shape = (num_layers, num_blocks, num_kv_heads, block_size)
            self.key_scales = torch.empty(scale_shape, dtype=dtype, device=self.device)
            self.value_scales = torch.empty(scale_shape, dtype=dtype, device=self.device)
        else:
            self.key_scales = None
            self.value_scales = None

        self.allocator = BlockAllocator(num_blocks)
        self.cpu_allocator = BlockAllocator(num_cpu_blocks) if num_cpu_blocks > 0 else None
        self.cpu_key_cache = None
        self.cpu_value_cache = None
        if num_cpu_blocks > 0:
            cpu_shape = (num_layers, num_cpu_blocks, num_kv_heads, block_size, head_dim)
            self.cpu_key_cache = torch.empty(cpu_shape, dtype=dtype, device="cpu", pin_memory=torch.cuda.is_available())
            self.cpu_value_cache = torch.empty(cpu_shape, dtype=dtype, device="cpu", pin_memory=torch.cuda.is_available())

    # ----- size helpers -----

    def logical_to_physical(self, block_table: BlockTable, logical_pos: int) -> tuple[int, int]:
        """Translate a token position within a sequence to (block_id, offset)."""
        block_idx = logical_pos // self.block_size
        offset = logical_pos % self.block_size
        return block_table.physical_blocks[block_idx], offset

    def num_blocks_needed(self, seq_len: int) -> int:
        return (seq_len + self.block_size - 1) // self.block_size

    # ----- write path -----

    def write_prefill(
        self,
        layer_idx: int,
        block_table: BlockTable,
        start_pos: int,
        keys: torch.Tensor,    # [num_new_tokens, num_kv_heads, head_dim]
        values: torch.Tensor,
    ) -> None:
        """Scatter a contiguous run of new tokens into the paged cache.

        Used during prefill (many tokens at once) and also fine for decode
        (one token). Under kv_cache_dtype='int8' we quantize per-(token, head)
        with a symmetric scale = absmax / 127.
        """
        num_new = keys.shape[0]
        if self.is_int8:
            # Vectorized: per-(token, head) absmax across head_dim.
            k_absmax = keys.abs().amax(dim=-1).clamp_min(1e-8)   # [N, H]
            v_absmax = values.abs().amax(dim=-1).clamp_min(1e-8)
            k_scales = (k_absmax / 127.0).to(self.dtype)         # [N, H]
            v_scales = (v_absmax / 127.0).to(self.dtype)
            qk = (
                (keys / k_scales.to(keys.dtype).unsqueeze(-1))
                .round().clamp(-128, 127).to(torch.int8)
            )                                                    # [N, H, D]
            qv = (
                (values / v_scales.to(values.dtype).unsqueeze(-1))
                .round().clamp(-128, 127).to(torch.int8)
            )
            for i in range(num_new):
                pos = start_pos + i
                block_id, offset = self.logical_to_physical(block_table, pos)
                self.key_cache[layer_idx, block_id, :, offset, :] = qk[i]
                self.value_cache[layer_idx, block_id, :, offset, :] = qv[i]
                self.key_scales[layer_idx, block_id, :, offset] = k_scales[i]
                self.value_scales[layer_idx, block_id, :, offset] = v_scales[i]
            return
        for i in range(num_new):
            pos = start_pos + i
            block_id, offset = self.logical_to_physical(block_table, pos)
            self.key_cache[layer_idx, block_id, :, offset, :] = keys[i]
            self.value_cache[layer_idx, block_id, :, offset, :] = values[i]

    def get_kv_tensors(self, layer_idx: int):
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_kv_scales(self, layer_idx: int):
        """Return per-layer (key_scales, value_scales) or (None, None) under fp16."""
        if not self.is_int8:
            return None, None
        return self.key_scales[layer_idx], self.value_scales[layer_idx]

    def retain_blocks(self, block_ids: List[int]) -> None:
        for block_id in block_ids:
            self.allocator.retain(block_id)

    def read_tokens(
        self,
        layer_idx: int,
        block_table: BlockTable,
        end_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather logical token positions [0, end_pos) into dense K/V tensors.

        Under kv_cache_dtype='int8' the gather dequantizes back to self.dtype
        on the fly, so callers always receive fp16/bf16 tensors.
        """
        keys = torch.empty(
            (end_pos, self.num_kv_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        values = torch.empty_like(keys)
        if self.is_int8:
            for pos in range(end_pos):
                block_id, offset = self.logical_to_physical(block_table, pos)
                k_int = self.key_cache[layer_idx, block_id, :, offset, :].to(torch.float32)
                v_int = self.value_cache[layer_idx, block_id, :, offset, :].to(torch.float32)
                k_s = self.key_scales[layer_idx, block_id, :, offset].to(torch.float32)
                v_s = self.value_scales[layer_idx, block_id, :, offset].to(torch.float32)
                keys[pos] = (k_int * k_s.unsqueeze(-1)).to(self.dtype)
                values[pos] = (v_int * v_s.unsqueeze(-1)).to(self.dtype)
            return keys, values
        for pos in range(end_pos):
            block_id, offset = self.logical_to_physical(block_table, pos)
            keys[pos] = self.key_cache[layer_idx, block_id, :, offset, :]
            values[pos] = self.value_cache[layer_idx, block_id, :, offset, :]
        return keys, values

    # ----- CPU swap path -----

    @property
    def swap_enabled(self) -> bool:
        return self.cpu_allocator is not None

    def can_swap_out(self, block_table: BlockTable) -> bool:
        return self.swap_enabled and self.cpu_allocator.can_allocate(len(block_table))

    def can_swap_in(self, block_table: BlockTable) -> bool:
        return self.swap_enabled and self.allocator.can_allocate(len(block_table))

    def swap_out(self, block_table: BlockTable) -> None:
        """Move a sequence's KV blocks from GPU cache to CPU cache in-place."""
        if not self.swap_enabled:
            raise RuntimeError("CPU KV swap is disabled")
        num_blocks = len(block_table)
        if not self.cpu_allocator.can_allocate(num_blocks):
            raise RuntimeError("Out of CPU KV swap blocks")

        cpu_blocks = [self.cpu_allocator.allocate() for _ in range(num_blocks)]
        gpu_blocks = block_table.as_list()
        for gpu_block, cpu_block in zip(gpu_blocks, cpu_blocks):
            self.cpu_key_cache[:, cpu_block].copy_(self.key_cache[:, gpu_block].to("cpu"), non_blocking=True)
            self.cpu_value_cache[:, cpu_block].copy_(self.value_cache[:, gpu_block].to("cpu"), non_blocking=True)
        self.allocator.free_many(gpu_blocks)
        block_table.physical_blocks[:] = cpu_blocks

    def swap_in(self, block_table: BlockTable) -> None:
        """Move a sequence's KV blocks from CPU cache back to GPU cache in-place."""
        if not self.swap_enabled:
            raise RuntimeError("CPU KV swap is disabled")
        num_blocks = len(block_table)
        if not self.allocator.can_allocate(num_blocks):
            raise RuntimeError("Out of GPU KV blocks")

        gpu_blocks = [self.allocator.allocate() for _ in range(num_blocks)]
        cpu_blocks = block_table.as_list()
        for cpu_block, gpu_block in zip(cpu_blocks, gpu_blocks):
            self.key_cache[:, gpu_block].copy_(self.cpu_key_cache[:, cpu_block].to(self.device), non_blocking=True)
            self.value_cache[:, gpu_block].copy_(self.cpu_value_cache[:, cpu_block].to(self.device), non_blocking=True)
        self.cpu_allocator.free_many(cpu_blocks)
        block_table.physical_blocks[:] = gpu_blocks


def build_block_tables_tensor(
    block_tables: List[BlockTable],
    max_num_blocks: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Pack a list of variable-length block tables into a dense tensor.

    The Triton kernel expects a [num_seqs, max_num_blocks] int32 tensor.
    Unused slots are filled with 0 — they won't be read because the kernel
    also takes a context_lens tensor that bounds the valid range per seq.
    """
    out = torch.zeros((len(block_tables), max_num_blocks), dtype=torch.int32, device=device)
    for i, bt in enumerate(block_tables):
        n = len(bt)
        if n > 0:
            out[i, :n] = torch.tensor(bt.physical_blocks, dtype=torch.int32, device=device)
    return out
