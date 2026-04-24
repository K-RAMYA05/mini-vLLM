"""Tests for the block allocator and KV cache.

These are all CPU-runnable so you can sanity-check without a GPU.
"""
import pytest
import torch

from mini_vllm.block_manager import BlockAllocator, BlockTable, KVCache, build_block_tables_tensor


def test_allocator_allocates_and_frees():
    a = BlockAllocator(num_blocks=4)
    assert a.num_free == 4
    b0 = a.allocate()
    b1 = a.allocate()
    assert a.num_free == 2
    assert a.num_allocated == 2
    a.free(b0)
    assert a.num_free == 3
    a.free(b1)
    assert a.num_free == 4


def test_allocator_raises_when_exhausted():
    a = BlockAllocator(num_blocks=2)
    a.allocate()
    a.allocate()
    with pytest.raises(RuntimeError):
        a.allocate()


def test_allocator_recycles():
    """Freed blocks come back to the free pool in FIFO order."""
    a = BlockAllocator(num_blocks=3)
    xs = [a.allocate() for _ in range(3)]
    a.free_many(xs)
    ys = [a.allocate() for _ in range(3)]
    # FIFO freeing -> same order back out
    assert ys == xs


def test_logical_to_physical():
    cache = KVCache(
        num_layers=1, num_kv_heads=1, head_dim=8,
        num_blocks=8, block_size=4, dtype=torch.float32, device="cpu",
    )
    bt = BlockTable()
    bt.append(3)
    bt.append(5)
    # position 0 -> block 3 offset 0
    assert cache.logical_to_physical(bt, 0) == (3, 0)
    # position 3 -> block 3 offset 3
    assert cache.logical_to_physical(bt, 3) == (3, 3)
    # position 4 -> block 5 offset 0 (rolled over)
    assert cache.logical_to_physical(bt, 4) == (5, 0)
    # position 7 -> block 5 offset 3
    assert cache.logical_to_physical(bt, 7) == (5, 3)


def test_num_blocks_needed():
    cache = KVCache(
        num_layers=1, num_kv_heads=1, head_dim=8,
        num_blocks=8, block_size=16, dtype=torch.float32, device="cpu",
    )
    assert cache.num_blocks_needed(0) == 0
    assert cache.num_blocks_needed(1) == 1
    assert cache.num_blocks_needed(16) == 1
    assert cache.num_blocks_needed(17) == 2
    assert cache.num_blocks_needed(32) == 2
    assert cache.num_blocks_needed(33) == 3


def test_write_prefill_and_read_back():
    """Round-trip: write some K/V tokens, read them back via logical->physical."""
    cache = KVCache(
        num_layers=2, num_kv_heads=2, head_dim=4,
        num_blocks=8, block_size=4, dtype=torch.float32, device="cpu",
    )
    bt = BlockTable()
    bt.append(1)
    bt.append(2)

    keys = torch.arange(8 * 2 * 4, dtype=torch.float32).view(8, 2, 4)
    values = keys * -1.0
    cache.write_prefill(layer_idx=0, block_table=bt, start_pos=0, keys=keys, values=values)

    # Read position 5 back — should equal keys[5].
    block_id, offset = cache.logical_to_physical(bt, 5)
    got_k = cache.key_cache[0, block_id, :, offset, :]
    assert torch.equal(got_k, keys[5])

    got_v = cache.value_cache[0, block_id, :, offset, :]
    assert torch.equal(got_v, values[5])


def test_build_block_tables_tensor_pads_correctly():
    bts = [BlockTable(), BlockTable(), BlockTable()]
    bts[0].append(2); bts[0].append(3)
    bts[1].append(7)
    bts[2].append(1); bts[2].append(4); bts[2].append(5)
    t = build_block_tables_tensor(bts, max_num_blocks=4, device="cpu")
    assert t.shape == (3, 4)
    assert t[0, 0] == 2 and t[0, 1] == 3 and t[0, 2] == 0 and t[0, 3] == 0
    assert t[1, 0] == 7 and t[1, 1] == 0
    assert t[2, 0] == 1 and t[2, 1] == 4 and t[2, 2] == 5 and t[2, 3] == 0


def test_kv_cache_swaps_blocks_to_cpu_and_back():
    cache = KVCache(
        num_layers=1, num_kv_heads=1, head_dim=2,
        num_blocks=2, num_cpu_blocks=2, block_size=2,
        dtype=torch.float32, device="cpu",
    )
    bt = BlockTable()
    bt.append(cache.allocator.allocate())
    keys = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    values = keys * -1
    cache.write_prefill(0, bt, 0, keys, values)

    cache.swap_out(bt)
    assert cache.allocator.num_free == 2
    assert cache.cpu_allocator.num_free == 1

    cache.swap_in(bt)
    assert cache.allocator.num_free == 1
    assert cache.cpu_allocator.num_free == 2
    block_id, offset = cache.logical_to_physical(bt, 1)
    torch.testing.assert_close(cache.key_cache[0, block_id, :, offset, :], keys[1])
