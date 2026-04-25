"""Continuous-batching scheduler.

Every step, the scheduler produces a batch of work for the model to run.
A "step" is one forward pass. The batch can mix two kinds of sequences:

  - PREFILL: sequences being admitted for the first time. They bring along
    their full prompt (many tokens) and need KV blocks allocated to hold it.
  - DECODE:  sequences already running. They contribute exactly one token
    (the last sampled one) and need at most one new block (when the current
    last block fills up).

Continuous batching == we don't wait for the whole batch to finish before
admitting new requests. The moment a sequence finishes, its slot is free
and we fill it with a waiting sequence on the next step. In practice this
keeps the GPU saturated in a way that static batching never does.

The scheduler is intentionally simple: FIFO over the waiting queue, with
a token budget to cap how much prefill we admit per step. When CPU swap is
configured, decode sequences that cannot grow their GPU KV block table are
preempted to CPU and resumed once GPU blocks free up.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

from mini_vllm.block_manager import KVCache
from mini_vllm.config import EngineConfig
from mini_vllm.prefix_cache import PrefixCache
from mini_vllm.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutputs:
    """What the scheduler hands to the model runner each step."""
    prefill_seqs: List[Sequence]
    decode_seqs: List[Sequence]

    @property
    def is_empty(self) -> bool:
        return not self.prefill_seqs and not self.decode_seqs

    @property
    def num_seqs(self) -> int:
        return len(self.prefill_seqs) + len(self.decode_seqs)


class Scheduler:
    def __init__(self, config: EngineConfig, kv_cache: KVCache, metrics=None):
        self.config = config
        self.kv_cache = kv_cache
        self.metrics = metrics

        self.waiting: deque[Sequence] = deque()
        self.swapped: deque[Sequence] = deque()
        self.running: List[Sequence] = []
        self.prefix_cache = (
            PrefixCache(
                kv_cache,
                config.block_size,
                config.prefix_cache_max_entries,
                eviction=config.prefix_cache_eviction,
            )
            if config.enable_prefix_cache
            else None
        )

    # ---------- admission ----------

    def add_seq(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def abort_seq(self, seq_id: int) -> None:
        for q in (self.waiting, self.swapped, self.running):
            for s in list(q):
                if s.seq_id == seq_id:
                    was_running = s in self.running
                    was_swapped = s in self.swapped
                    s.status = SequenceStatus.FINISHED_ABORTED
                    if was_running:
                        self._free(s)
                        self.running.remove(s)
                    elif was_swapped:
                        if self.kv_cache.cpu_allocator is not None and len(s.block_table) > 0:
                            self.kv_cache.cpu_allocator.free_many(s.block_table.as_list())
                            s.block_table.physical_blocks.clear()
                        self.swapped.remove(s)
                    else:
                        self.waiting.remove(s)
                    return

    # ---------- scheduling ----------

    def schedule(self) -> SchedulerOutputs:
        """Decide what runs this step."""
        prefill: List[Sequence] = []
        decode: List[Sequence] = []

        # 1) Grow resident decode sequences by one token's worth of KV room.
        #    With CPU swap enabled, a sequence that cannot grow is preempted
        #    instead of aborted.
        still_running: List[Sequence] = []
        for seq in self.running:
            if self._append_slot_if_needed(seq):
                decode.append(seq)
                still_running.append(seq)
            elif self._swap_out(seq):
                self.swapped.append(seq)
            else:
                # Out of memory for this seq; abort it rather than hang.
                seq.status = SequenceStatus.FINISHED_ABORTED
                seq.finish_reason = "oom"
                self._free(seq)
        self.running = still_running

        # 2) Resume swapped sequences when enough GPU blocks are available.
        token_budget = self.config.max_num_batched_tokens - len(decode)
        seq_budget = self.config.max_num_seqs - len(decode)
        while self.swapped and token_budget > 0 and seq_budget > 0:
            seq = self.swapped[0]
            blocks_needed = len(seq.block_table)
            needs_new_block = self.kv_cache.num_blocks_needed(seq.seq_len + 1) > blocks_needed
            if not self.kv_cache.allocator.can_allocate(blocks_needed + int(needs_new_block)):
                break
            self.swapped.popleft()
            self.kv_cache.swap_in(seq.block_table)
            if self.metrics is not None:
                self.metrics.swap_in_count += 1
            seq.status = SequenceStatus.RUNNING
            if self._append_slot_if_needed(seq):
                decode.append(seq)
                self.running.append(seq)
                token_budget -= 1
                seq_budget -= 1
            elif self._swap_out(seq):
                self.swapped.append(seq)
            else:
                seq.status = SequenceStatus.FINISHED_ABORTED
                seq.finish_reason = "oom"
                self._free(seq)

        # 3) Admit new (prefill) sequences under our token + seq budgets.
        #    Prefill tokens count against max_num_batched_tokens so that a
        #    single giant prompt can't blow the forward pass.
        while self.waiting and seq_budget > 0:
            seq = self.waiting[0]
            prompt_len = seq.num_prompt_tokens
            if prompt_len > self.config.max_model_len:
                # Too long even in isolation — drop it.
                self.waiting.popleft()
                seq.status = SequenceStatus.FINISHED_ABORTED
                seq.finish_reason = "too_long"
                continue
            if prompt_len > token_budget:
                break
            cached_blocks = []
            cached_tokens = 0
            if self.prefix_cache is not None:
                cached_blocks, cached_tokens = self.prefix_cache.lookup(seq.prompt_token_ids)
                if self.metrics is not None:
                    self.metrics.observe_prefix_cache(cached_tokens)
            uncached_tokens = prompt_len - cached_tokens
            if uncached_tokens > token_budget:
                break
            blocks_needed = self.kv_cache.num_blocks_needed(prompt_len)
            blocks_to_allocate = blocks_needed - len(cached_blocks)
            if not self.kv_cache.allocator.can_allocate(blocks_to_allocate):
                break

            # Commit.
            self.waiting.popleft()
            if cached_blocks:
                self.kv_cache.retain_blocks(cached_blocks)
                seq.block_table.physical_blocks.extend(cached_blocks)
                seq.num_cached_tokens = cached_tokens
                seq.prefix_cache_blocks = len(cached_blocks)
            for _ in range(blocks_to_allocate):
                seq.block_table.append(self.kv_cache.allocator.allocate())
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            prefill.append(seq)
            token_budget -= uncached_tokens
            seq_budget -= 1

        return SchedulerOutputs(prefill_seqs=prefill, decode_seqs=decode)

    # ---------- post-step hooks ----------

    def finalize_finished(self) -> List[Sequence]:
        """Called after a step; free blocks for sequences that stopped."""
        finished: List[Sequence] = []
        still: List[Sequence] = []
        for seq in self.running:
            if seq.status.is_finished:
                self._register_prefix_cache(seq)
                self._free(seq)
                finished.append(seq)
            else:
                still.append(seq)
        self.running = still
        return finished

    def register_prefill_cache(self, seqs: List[Sequence]) -> None:
        for seq in seqs:
            self._register_prefix_cache(seq)

    # ---------- internals ----------

    def _append_slot_if_needed(self, seq: Sequence) -> bool:
        """Ensure seq has a KV slot for its next token. Returns False on OOM."""
        needed = self.kv_cache.num_blocks_needed(seq.seq_len + 1)
        while len(seq.block_table) < needed:
            if not self.kv_cache.allocator.can_allocate(1):
                return False
            seq.block_table.append(self.kv_cache.allocator.allocate())
        return True

    def _swap_out(self, seq: Sequence) -> bool:
        if not self.kv_cache.can_swap_out(seq.block_table):
            return False
        self.kv_cache.swap_out(seq.block_table)
        if self.metrics is not None:
            self.metrics.swap_out_count += 1
        seq.status = SequenceStatus.SWAPPED
        return True

    def _free(self, seq: Sequence) -> None:
        if len(seq.block_table) > 0:
            if seq.status == SequenceStatus.SWAPPED and self.kv_cache.cpu_allocator is not None:
                self.kv_cache.cpu_allocator.free_many(seq.block_table.as_list())
            else:
                self.kv_cache.allocator.free_many(seq.block_table.as_list())
            seq.block_table.physical_blocks.clear()
            seq.prefix_cache_blocks = 0

    def _register_prefix_cache(self, seq: Sequence) -> None:
        if self.prefix_cache is None or len(seq.block_table) == 0:
            return
        self.prefix_cache.register(seq.prompt_token_ids, seq.block_table.as_list())

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.swapped or self.running)
