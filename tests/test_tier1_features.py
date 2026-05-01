"""Tests for the Tier-1 novelty additions:
   - LFU vs LRU prefix-cache eviction
   - int8 KV cache write+read round-trip
   - lookahead decode advances multiple substeps
"""
from __future__ import annotations

import torch

from mini_vllm.block_manager import BlockTable, KVCache
from mini_vllm.config import EngineConfig
from mini_vllm.lookahead import LookaheadExecutor
from mini_vllm.model_runner import ModelRunner
from mini_vllm.prefix_cache import PrefixCache
from mini_vllm.quant.awq import _quantize_linear_awq
from mini_vllm.quant.gptq import GPTQLinear
from mini_vllm.sampling import SamplingParams
from mini_vllm.sequence import Sequence


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


def test_engine_config_accepts_fp8_kv_and_quant_method():
    cfg = EngineConfig(kv_cache_dtype="fp8", quant_method="fp8")
    assert cfg.kv_cache_dtype == "fp8"
    assert cfg.quant_method == "fp8"


def test_engine_config_validates_cuda_graph_batch_sizes():
    import pytest
    with pytest.raises(ValueError, match="cuda_graph_batch_sizes"):
        EngineConfig(enable_cuda_graphs=True, cuda_graph_batch_sizes=())


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
# Lookahead decoding
# ------------------------------------------------------------------

def test_engine_config_validates_lookahead_bounds():
    import pytest
    with pytest.raises(ValueError, match="lookahead_num_slots"):
        EngineConfig(
            enable_lookahead_decoding=True,
            lookahead_num_slots=1,
        )


def test_lookahead_executor_advances_multiple_decode_substeps():
    class StubScheduler:
        def _append_slot_if_needed(self, seq):
            return True

    class StubRunner:
        def execute_decode_plan(self, seqs, num_steps, eos_token_id):
            assert num_steps == 3
            return [[11, 12, 13] for _ in seqs]

    class StubKVCache:
        class Alloc:
            @staticmethod
            def can_allocate(n):
                return True

            @staticmethod
            def allocate():
                return 99

        allocator = Alloc()

        @staticmethod
        def num_blocks_needed(seq_len):
            return 1

    class StubEngine:
        config = EngineConfig(enable_lookahead_decoding=True, lookahead_num_slots=3, device="cpu")
        scheduler = StubScheduler()
        runner = StubRunner()
        kv_cache = StubKVCache()
        info = {"eos_token_id": None}

        def _observe_generated_token(self, seq):
            pass

    seq = Sequence(
        prompt="x",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=3, temperature=0.0),
    )

    LookaheadExecutor(StubEngine()).step([seq])
    assert seq.output_token_ids == [11, 12, 13]
    assert seq.finish_reason == "length"


# ------------------------------------------------------------------
# Chunked prefill + sliding window
# ------------------------------------------------------------------

def test_chunked_prefill_splits_long_prompt_and_updates_cache_count():
    class StubRunner:
        config = EngineConfig(
            enable_chunked_prefill=True,
            max_prefill_chunk_tokens=3,
            sliding_window=64,
            device="cpu",
        )
        device = torch.device("cpu")
        _execute_chunked_prefill = ModelRunner._execute_chunked_prefill
        _run_prefill_chunk = ModelRunner._run_prefill_chunk
        _sample = ModelRunner._sample

        def __init__(self):
            self.calls = []
            self.generator = torch.Generator(device=self.device).manual_seed(0)

        def _forward(self, input_ids, position_ids, attn_metadata):
            self.calls.append({
                "input_ids": input_ids.tolist(),
                "position_ids": position_ids.tolist(),
                "sliding_window": attn_metadata.sliding_window,
            })
            vocab = 32
            logits = torch.full((input_ids.shape[0], vocab), -1e9, device=input_ids.device)
            logits[:, input_ids.shape[0]] = 0.0
            return logits

    seq = Sequence(
        prompt="x",
        prompt_token_ids=list(range(7)),
        sampling_params=SamplingParams(max_tokens=8, temperature=0.0),
    )
    seq.block_table.append(0)
    seq.block_table.append(1)

    runner = StubRunner()
    sampled = runner._execute_chunked_prefill([seq])

    assert [call["input_ids"] for call in runner.calls] == [[0, 1, 2], [3, 4, 5], [6]]
    assert [call["position_ids"] for call in runner.calls] == [[0, 1, 2], [3, 4, 5], [6]]
    assert all(call["sliding_window"] == 64 for call in runner.calls)
    assert seq.num_cached_tokens == 7
    assert sampled == [1]


def test_run_prefill_chunk_uses_graph_helper_when_available():
    class StubRunner:
        config = EngineConfig(enable_cuda_graphs=True, device="cpu")
        device = torch.device("cpu")
        _run_prefill_chunk = ModelRunner._run_prefill_chunk

        def _execute_prefill_graph_single(self, seq, token_ids):
            return torch.ones((len(token_ids), 8), dtype=torch.float32)

        def _forward(self, input_ids, position_ids, attn_metadata):
            raise AssertionError("eager prefill path should not run when graph helper succeeds")

    seq = Sequence(
        prompt="x",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
    )
    seq.block_table.append(0)

    runner = StubRunner()
    runner._graph_enabled = True
    out = runner._run_prefill_chunk(seq, [1, 2, 3], start_pos=0)

    assert out.shape == (3, 8)


# ------------------------------------------------------------------
# AWQ
# ------------------------------------------------------------------

def test_awq_quantizes_linear_to_gptqlinear_runtime_module():
    lin = torch.nn.Linear(8, 4, bias=True)
    act = torch.linspace(0.5, 1.5, steps=8)
    q = _quantize_linear_awq(lin, activation_mean_abs=act, bits=4, group_size=4)

    assert isinstance(q, GPTQLinear)
    assert q.bits == 4
    x = torch.randn(2, 8)
    out = q(x)
    assert out.shape == (2, 4)
