"""Scheduler tests. CPU-only — no model, just the queue logic."""
import pytest
import torch

from mini_vllm.block_manager import KVCache
from mini_vllm.config import EngineConfig
from mini_vllm.sampling import SamplingParams
from mini_vllm.scheduler import Scheduler
from mini_vllm.sequence import Sequence, SequenceStatus


def _make_scheduler(**kwargs) -> Scheduler:
    cfg = EngineConfig(
        block_size=16,
        num_gpu_blocks=8,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        max_model_len=256,
        **kwargs,
    )
    cache = KVCache(
        num_layers=1, num_kv_heads=1, head_dim=8,
        num_blocks=cfg.num_gpu_blocks, block_size=cfg.block_size,
        dtype=torch.float32, device="cpu", num_cpu_blocks=cfg.num_cpu_blocks,
    )
    return Scheduler(cfg, cache)


def _make_seq(prompt_len: int, max_tokens: int = 16) -> Sequence:
    return Sequence(
        prompt="<prompt>",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def test_schedule_admits_waiting_seqs_as_prefill():
    sched = _make_scheduler()
    sched.add_seq(_make_seq(10))
    sched.add_seq(_make_seq(20))
    out = sched.schedule()
    assert len(out.prefill_seqs) == 2
    assert len(out.decode_seqs) == 0
    # Block tables should be populated.
    for s in out.prefill_seqs:
        assert len(s.block_table) > 0
        assert s.status == SequenceStatus.RUNNING


def test_token_budget_caps_prefill_admission():
    sched = _make_scheduler()
    # Three prompts of 60 tokens each; budget is 128. Only two should fit.
    for _ in range(3):
        sched.add_seq(_make_seq(60))
    out = sched.schedule()
    assert len(out.prefill_seqs) == 2


def test_seq_budget_caps_admission():
    sched = _make_scheduler()
    for _ in range(10):
        sched.add_seq(_make_seq(5))
    out = sched.schedule()
    # max_num_seqs=4
    assert len(out.prefill_seqs) == 4


def test_decode_grows_block_table_when_full():
    """Scheduler allocates a new block when the next token won't fit.

    The scheduler reserves KV space for seq.seq_len + 1 (the token about to
    be generated this step), so the block table grows one step BEFORE the
    current block is actually full.
    """
    sched = _make_scheduler()
    sched.add_seq(_make_seq(prompt_len=10))    # well inside the first 16-block
    out = sched.schedule()
    seq = out.prefill_seqs[0]
    assert len(seq.block_table) == 1

    # Decode step: seq_len=10, needs room for token 11 -> still fits in 1 block.
    seq.output_token_ids.append(42)            # seq_len now 11
    out = sched.schedule()
    assert len(out.decode_seqs) == 1
    assert len(seq.block_table) == 1

    # Grow the sequence until the next token would overflow block 0.
    # After adding tokens so seq_len == 16, the scheduler must allocate block 1
    # to hold the token at position 16.
    for i in range(5):
        seq.output_token_ids.append(100 + i)   # seq_len goes 12, 13, 14, 15, 16
        out = sched.schedule()
    assert len(seq.block_table) == 2


def test_finished_seqs_return_blocks():
    sched = _make_scheduler()
    sched.add_seq(_make_seq(prompt_len=20))
    out = sched.schedule()
    seq = out.prefill_seqs[0]
    free_before = sched.kv_cache.allocator.num_free
    seq.status = SequenceStatus.FINISHED_STOPPED
    sched.finalize_finished()
    free_after = sched.kv_cache.allocator.num_free
    # All blocks freed.
    assert free_after > free_before


def test_scheduler_swaps_instead_of_aborting_on_decode_oom():
    sched = _make_scheduler(num_gpu_blocks=1, num_cpu_blocks=2, block_size=4)
    sched.add_seq(_make_seq(prompt_len=4))
    out = sched.schedule()
    seq = out.prefill_seqs[0]
    seq.output_token_ids.append(1)

    out = sched.schedule()

    assert out.decode_seqs == []
    assert seq.status == SequenceStatus.SWAPPED
    assert list(sched.swapped) == [seq]
    assert sched.kv_cache.allocator.num_free == 1
