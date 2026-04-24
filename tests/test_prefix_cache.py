import torch

from mini_vllm.block_manager import KVCache
from mini_vllm.config import EngineConfig
from mini_vllm.prefix_cache import PrefixCache
from mini_vllm.sampling import SamplingParams
from mini_vllm.scheduler import Scheduler
from mini_vllm.sequence import Sequence, SequenceStatus


def _make_scheduler(**kwargs) -> Scheduler:
    cfg = EngineConfig(
        block_size=4,
        num_gpu_blocks=8,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        max_model_len=64,
        enable_prefix_cache=True,
        prefix_cache_max_entries=32,
        **kwargs,
    )
    cache = KVCache(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        num_blocks=cfg.num_gpu_blocks,
        block_size=cfg.block_size,
        dtype=torch.float32,
        device="cpu",
    )
    return Scheduler(cfg, cache)


def test_prefix_cache_reuses_full_blocks_for_later_prompt():
    sched = _make_scheduler()
    seq1 = Sequence("a", list(range(10)), SamplingParams(max_tokens=1))
    sched.add_seq(seq1)
    out = sched.schedule()
    seq1 = out.prefill_seqs[0]
    sched.register_prefill_cache([seq1])
    seq1.status = SequenceStatus.FINISHED_STOPPED
    sched.finalize_finished()

    seq2 = Sequence("b", list(range(10)), SamplingParams(max_tokens=1))
    sched.add_seq(seq2)
    out = sched.schedule()
    seq2 = out.prefill_seqs[0]
    assert seq2.num_cached_tokens == 8
    assert seq2.prefix_cache_blocks == 2


def test_prefix_cache_lookup_matches_registered_blocks():
    cache = KVCache(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        dtype=torch.float32,
        device="cpu",
    )
    prefix_cache = PrefixCache(cache, block_size=4, max_entries=32)
    block_ids = [cache.allocator.allocate() for _ in range(3)]
    prefix_cache.register(list(range(10)), block_ids)
    matched_blocks, matched_tokens = prefix_cache.lookup(list(range(10)))
    assert matched_blocks == block_ids[:2]
    assert matched_tokens == 8
