"""Tests for the Tier-1 novelty additions:
   - LFU vs LRU prefix-cache eviction
   - int8 KV cache write+read round-trip
   - adaptive-γ selection in SpeculativeExecutor
   - Leviathan-formula helper used by the sweep analyzer
"""
from __future__ import annotations

import torch

from mini_vllm.block_manager import BlockTable, KVCache
from mini_vllm.config import EngineConfig
from mini_vllm.distill.analyze_sweep import leviathan_tokens_per_step
from mini_vllm.prefix_cache import PrefixCache


# ------------------------------------------------------------------
# LFU vs LRU
# ------------------------------------------------------------------

def _make_cache(num_blocks: int = 8) -> KVCache:
    return KVCache(
        num_layers=1, num_kv_heads=1, head_dim=4,
        num_blocks=num_blocks, block_size=4,
        dtype=torch.float32, device="cpu",
    )


def test_lfu_keeps_hot_prefix_under_churn():
    """Workload: one popular system prompt + many one-shot prompts.
    The hot prefix is hit several times before the cold flood, so its hit
    count beats every cold entry's. LFU must keep it; LRU would evict it."""
    kvc = _make_cache(num_blocks=64)
    cache = PrefixCache(kvc, block_size=4, max_entries=6, eviction="lfu")

    hot = list(range(20))                # reusable_blocks = 4
    hot_blocks = [kvc.allocator.allocate() for _ in range(4)]
    cache.register(hot, hot_blocks)
    for _ in range(5):
        cache.lookup(hot)                # boost hit counts on the hot blocks

    for i in range(10):
        cold = list(range(1000 + i * 10, 1000 + i * 10 + 7))   # 1-block prefix
        cold_blocks = [kvc.allocator.allocate()]
        cache.register(cold, cold_blocks)

    matched, n = cache.lookup(hot)
    assert n == 16, f"LFU should retain all 4 hot blocks (got {n}/16)"


def test_lru_evicts_hot_prefix_under_churn():
    """Mirror: same workload, LRU should drop the hot prefix entirely."""
    kvc = _make_cache(num_blocks=64)
    cache = PrefixCache(kvc, block_size=4, max_entries=6, eviction="lru")

    hot = list(range(20))
    hot_blocks = [kvc.allocator.allocate() for _ in range(4)]
    cache.register(hot, hot_blocks)
    for _ in range(5):
        cache.lookup(hot)

    for i in range(10):
        cold = list(range(1000 + i * 10, 1000 + i * 10 + 7))
        cold_blocks = [kvc.allocator.allocate()]
        cache.register(cold, cold_blocks)

    matched, n = cache.lookup(hot)
    assert n == 0, "LRU should have evicted the hot prefix's first block"


def test_engine_config_rejects_bad_eviction_policy():
    import pytest
    with pytest.raises(ValueError, match="prefix_cache_eviction"):
        EngineConfig(prefix_cache_eviction="random")


# ------------------------------------------------------------------
# int8 KV cache round-trip
# ------------------------------------------------------------------

def test_int8_kv_cache_round_trip_within_quant_error():
    kvc = KVCache(
        num_layers=1, num_kv_heads=2, head_dim=8,
        num_blocks=4, block_size=4,
        dtype=torch.float32, device="cpu",
        kv_cache_dtype="int8",
    )
    bt = BlockTable()
    bt.append(kvc.allocator.allocate())
    bt.append(kvc.allocator.allocate())

    torch.manual_seed(0)
    keys = torch.randn(8, 2, 8)            # 8 tokens fits in 2 blocks of 4
    values = torch.randn(8, 2, 8)

    kvc.write_prefill(0, bt, start_pos=0, keys=keys, values=values)
    rk, rv = kvc.read_tokens(0, bt, end_pos=8)

    assert rk.dtype == torch.float32
    assert rv.dtype == torch.float32
    # Symmetric int8 quantization with scale = absmax/127 has theoretical
    # max relative error ~1/127 ≈ 0.008. Allow some slack on top of that.
    k_err = (rk - keys).abs() / keys.abs().clamp_min(1e-4)
    v_err = (rv - values).abs() / values.abs().clamp_min(1e-4)
    # Looser bound because per-row absmax means small-magnitude lanes have
    # higher relative error after rounding. Absolute error scaled by absmax
    # is the right metric.
    k_abs_err = (rk - keys).abs() / keys.abs().amax(dim=-1, keepdim=True)
    v_abs_err = (rv - values).abs() / values.abs().amax(dim=-1, keepdim=True)
    assert k_abs_err.max().item() < 0.02, f"K abs-err too high: {k_abs_err.max().item()}"
    assert v_abs_err.max().item() < 0.02, f"V abs-err too high: {v_abs_err.max().item()}"


def test_kv_cache_dtype_validation():
    import pytest
    with pytest.raises(ValueError, match="kv_cache_dtype"):
        EngineConfig(kv_cache_dtype="fp4")


def test_int8_kv_cache_storage_is_actually_int8():
    kvc = KVCache(
        num_layers=1, num_kv_heads=1, head_dim=4,
        num_blocks=2, block_size=4,
        dtype=torch.float32, device="cpu",
        kv_cache_dtype="int8",
    )
    assert kvc.key_cache.dtype == torch.int8
    assert kvc.value_cache.dtype == torch.int8
    assert kvc.key_scales is not None
    assert kvc.key_scales.shape == (1, 2, 1, 4)


# ------------------------------------------------------------------
# Adaptive γ
# ------------------------------------------------------------------

def test_adaptive_gamma_picks_warmstart_for_unseen_seq():
    """Without touching real models, exercise just _select_gamma logic by
    constructing a SpeculativeExecutor-like stub."""

    class Stub:
        adaptive_gamma = True
        gamma = 4
        gamma_min = 1
        gamma_max = 8
        gamma_alpha = 0.3
        _accept_ewma: dict = {}

        # bind the real method
        from mini_vllm.speculative.spec_decode import SpeculativeExecutor
        _select_gamma = SpeculativeExecutor._select_gamma

    class FakeSeq:
        seq_id = 1

    stub = Stub()
    stub._accept_ewma = {}
    g = stub._select_gamma(FakeSeq())
    assert g == 4   # warm start to self.gamma


def test_adaptive_gamma_high_acceptance_picks_max():
    from mini_vllm.speculative.spec_decode import SpeculativeExecutor

    class Stub:
        adaptive_gamma = True
        gamma = 4
        gamma_min = 1
        gamma_max = 8
        gamma_alpha = 0.3

    class FakeSeq:
        seq_id = 7

    stub = Stub()
    stub._accept_ewma = {7: 1.0}   # perfect acceptance
    g = SpeculativeExecutor._select_gamma(stub, FakeSeq())
    assert g == 8


def test_adaptive_gamma_low_acceptance_picks_min():
    from mini_vllm.speculative.spec_decode import SpeculativeExecutor

    class Stub:
        adaptive_gamma = True
        gamma = 4
        gamma_min = 1
        gamma_max = 8
        gamma_alpha = 0.3

    class FakeSeq:
        seq_id = 7

    stub = Stub()
    stub._accept_ewma = {7: 0.0}   # nothing ever accepted
    g = SpeculativeExecutor._select_gamma(stub, FakeSeq())
    assert g == 1


def test_engine_config_validates_adaptive_bounds():
    import pytest
    with pytest.raises(ValueError, match="spec_gamma_min"):
        EngineConfig(
            use_speculative=True, draft_model_name_or_path="x",
            spec_adaptive_gamma=True, spec_gamma_min=5, spec_gamma_max=2,
        )


# ------------------------------------------------------------------
# Leviathan formula
# ------------------------------------------------------------------

def test_leviathan_alpha_zero_gives_one_token_per_step():
    assert leviathan_tokens_per_step(0.0, gamma=4) == 1.0


def test_leviathan_alpha_one_gives_gamma_plus_one():
    assert leviathan_tokens_per_step(1.0, gamma=4) == 5.0


def test_leviathan_monotonic_in_alpha():
    prev = -1.0
    for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
        cur = leviathan_tokens_per_step(a, gamma=4)
        assert cur > prev
        prev = cur
