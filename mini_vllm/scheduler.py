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

The scheduler is decode-biased and admission-aware:
  - resident decode work is protected first;
  - new prefill admission scans a small waiting window to avoid
    head-of-line blocking on one giant prompt;
  - chunked-prefill runs get a dynamic chunk target based on the live
    decode pressure and current token budget;
  - when lookahead decoding is enabled, the scheduler briefly prefers
    decode-only steps so the resident batch can advance without churn.
CPU swap still acts as the last-resort preemption path when decode growth
would otherwise run out of GPU KV blocks.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
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
    adapter_name: str | None = None
    prefill_chunk_size: int = 0
    prefill_token_costs: List[int] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefill_seqs and not self.decode_seqs

    @property
    def num_seqs(self) -> int:
        return len(self.prefill_seqs) + len(self.decode_seqs)

    @property
    def total_prefill_tokens(self) -> int:
        if self.prefill_token_costs:
            return sum(self.prefill_token_costs)
        return sum(seq.num_prompt_tokens - seq.num_cached_tokens for seq in self.prefill_seqs)


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
        self._decode_only_streak = 0
        self._class_weights = {
            "latency": 2.0,
            "interactive": 1.0,
            "default": 0.5,
            "batch": 0.0,
            "background": -0.5,
        }
        self._last_adapter_served: str | None = None

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
        prefill_token_costs: List[int] = []
        adapter_name = self._choose_adapter_cohort()

        # 1) Grow resident decode sequences by one token's worth of KV room.
        #    With CPU swap enabled, a sequence that cannot grow is preempted
        #    instead of aborted.
        still_running: List[Sequence] = []
        for seq in self.running:
            if seq.lora_adapter_name != adapter_name:
                still_running.append(seq)
                continue
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
            if seq.lora_adapter_name != adapter_name:
                break
            blocks_needed = len(seq.block_table)
            needs_new_block = self.kv_cache.num_blocks_needed(seq.seq_len + 1) > blocks_needed
            if not self.kv_cache.allocator.can_allocate(blocks_needed + int(needs_new_block)):
                break
            self.swapped.popleft()
            self.kv_cache.swap_in(seq.block_table)
            if self.metrics is not None:
                self.metrics.swap_in_count += 1
            seq.status = SequenceStatus.RUNNING
            seq.admitted_time_s = time.perf_counter()
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
        prefill_chunk_size = self._choose_prefill_chunk_size(
            token_budget=token_budget,
            decode_count=len(decode),
            prefill_count=0,
        )

        if not self._should_hold_prefill_for_lookahead(
            decode_count=len(decode),
            token_budget=token_budget,
            seq_budget=seq_budget,
        ):
            while self.waiting and seq_budget > 0 and token_budget > 0:
                admission = self._select_waiting_candidate(
                    token_budget=token_budget,
                    chunk_size=prefill_chunk_size,
                    adapter_name=adapter_name,
                )
                if admission is None:
                    break
                seq, cached_blocks, cached_tokens, admission_tokens = admission
                prompt_len = seq.num_prompt_tokens
                blocks_needed = self.kv_cache.num_blocks_needed(prompt_len)
                blocks_to_allocate = blocks_needed - len(cached_blocks)

                self.waiting.remove(seq)
                if cached_blocks:
                    self.kv_cache.retain_blocks(cached_blocks)
                    seq.block_table.physical_blocks.extend(cached_blocks)
                    seq.num_cached_tokens = cached_tokens
                    seq.prefix_cache_blocks = len(cached_blocks)
                    if self.metrics is not None:
                        self.metrics.observe_prefix_cache(cached_tokens)
                for _ in range(blocks_to_allocate):
                    seq.block_table.append(self.kv_cache.allocator.allocate())
                seq.status = SequenceStatus.RUNNING
                seq.admitted_time_s = time.perf_counter()
                if self.metrics is not None:
                    self.metrics.observe_queue_wait(seq.admitted_time_s - seq.created_time_s)
                self.running.append(seq)
                prefill.append(seq)
                prefill_token_costs.append(admission_tokens)
                token_budget -= admission_tokens
                seq_budget -= 1
                prefill_chunk_size = self._choose_prefill_chunk_size(
                    token_budget=token_budget,
                    decode_count=len(decode),
                    prefill_count=len(prefill),
                )

        if decode and not prefill and self.waiting and self.config.enable_lookahead_decoding:
            self._decode_only_streak += 1
        else:
            self._decode_only_streak = 0
        self._last_adapter_served = adapter_name

        return SchedulerOutputs(
            prefill_seqs=prefill,
            decode_seqs=decode,
            adapter_name=adapter_name,
            prefill_chunk_size=prefill_chunk_size,
            prefill_token_costs=prefill_token_costs,
        )

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

    def _choose_prefill_chunk_size(
        self,
        token_budget: int,
        decode_count: int,
        prefill_count: int,
    ) -> int:
        if not self.config.enable_chunked_prefill:
            return self.config.max_prefill_chunk_tokens
        if token_budget <= 0:
            return self.config.block_size
        kv_stats = self.kv_cache.memory_stats()
        free_kv_tokens = kv_stats["gpu_kv_tokens_free"]
        total_kv_tokens = max(kv_stats["gpu_kv_tokens_total"], 1)
        kv_pressure = 1.0 - (free_kv_tokens / total_kv_tokens)
        participants = max(1, decode_count + prefill_count + 1)
        budget_share = max(self.config.block_size, token_budget // participants)
        kv_share = max(self.config.block_size, free_kv_tokens // participants)
        chunk = min(self.config.max_prefill_chunk_tokens, budget_share, kv_share)
        if kv_pressure >= 0.90:
            chunk = min(chunk, self.config.block_size)
        elif kv_pressure >= 0.75:
            chunk = min(chunk, max(self.config.block_size, self.config.max_prefill_chunk_tokens // 4))
        elif kv_pressure >= 0.50:
            chunk = min(chunk, max(self.config.block_size, self.config.max_prefill_chunk_tokens // 2))
        return max(self.config.block_size, chunk)

    def _should_hold_prefill_for_lookahead(
        self,
        decode_count: int,
        token_budget: int,
        seq_budget: int,
    ) -> bool:
        if not self.config.enable_lookahead_decoding:
            return False
        if decode_count <= 0 or not self.waiting:
            return False
        if token_budget <= 0 or seq_budget <= 0:
            return False
        oldest_wait_s = max(time.perf_counter() - seq.created_time_s for seq in self.waiting)
        if oldest_wait_s >= self.config.max_waiting_age_before_decode_priority_s:
            return False
        return self._decode_only_streak + 1 < self.config.lookahead_num_slots

    def _select_waiting_candidate(
        self,
        token_budget: int,
        chunk_size: int,
        adapter_name: str | None,
    ) -> Tuple[Sequence, List[int], int, int] | None:
        best = None
        best_score = None
        skipped_too_long: List[Sequence] = []
        now = time.perf_counter()
        scan = list(self.waiting)[: self.config.admission_window_size]
        for idx, seq in enumerate(scan):
            if seq.lora_adapter_name != adapter_name:
                continue
            prompt_len = seq.num_prompt_tokens
            if prompt_len > self.config.max_model_len:
                skipped_too_long.append(seq)
                continue

            cached_blocks: List[int] = []
            cached_tokens = 0
            if self.prefix_cache is not None:
                cached_blocks, cached_tokens = self.prefix_cache.lookup(seq.prompt_token_ids)
            uncached_tokens = prompt_len - cached_tokens
            admission_tokens = uncached_tokens
            if self.config.enable_chunked_prefill:
                admission_tokens = min(uncached_tokens, max(1, chunk_size))
            if admission_tokens > token_budget:
                continue

            blocks_needed = self.kv_cache.num_blocks_needed(prompt_len)
            blocks_to_allocate = blocks_needed - len(cached_blocks)
            if not self.kv_cache.allocator.can_allocate(blocks_to_allocate):
                continue

            age_s = max(now - seq.created_time_s, 0.0)
            class_bias = self._class_weights.get(seq.request_class, 0.0)
            score = (
                seq.priority,
                class_bias,
                cached_tokens,
                age_s * self.config.scheduler_age_bias,
                -admission_tokens,
                -prompt_len,
                -idx,
            )
            if best_score is None or score > best_score:
                best = (seq, cached_blocks, cached_tokens, admission_tokens)
                best_score = score
        for seq in skipped_too_long:
            if seq in self.waiting:
                self.waiting.remove(seq)
                seq.status = SequenceStatus.FINISHED_ABORTED
                seq.finish_reason = "too_long"
        return best

    def _choose_adapter_cohort(self) -> str | None:
        scores: dict[str | None, float] = {}
        now = time.perf_counter()
        for seq in self.running:
            score = 100.0 + float(seq.priority) * 10.0
            score += self._class_weights.get(seq.request_class, 0.0) * 10.0
            score += max(now - (seq.admitted_time_s or seq.created_time_s), 0.0)
            scores[seq.lora_adapter_name] = scores.get(seq.lora_adapter_name, 0.0) + score
        for seq in list(self.waiting)[: self.config.admission_window_size]:
            score = float(seq.priority) * 10.0
            score += self._class_weights.get(seq.request_class, 0.0) * 10.0
            score += max(now - seq.created_time_s, 0.0) * self.config.scheduler_age_bias
            scores[seq.lora_adapter_name] = scores.get(seq.lora_adapter_name, 0.0) + score
        if not scores:
            return None
        return max(
            scores.items(),
            key=lambda item: (
                item[1],
                item[0] == self._last_adapter_served,
                item[0] is None,
            ),
        )[0]

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.swapped or self.running)

    def estimate_admission_capacity(self) -> dict[str, int]:
        free_blocks = self.kv_cache.allocator.num_free
        free_tokens = free_blocks * self.config.block_size
        conservative_tokens = min(free_tokens, self.config.max_num_batched_tokens)
        avg_waiting_prompt = (
            max(1, sum(seq.num_prompt_tokens for seq in self.waiting) // len(self.waiting))
            if self.waiting
            else max(1, self.config.block_size)
        )
        return {
            "admission_capacity_tokens": conservative_tokens,
            "admission_capacity_seqs": max(
                0,
                min(
                    self.config.max_num_seqs - len(self.running),
                    conservative_tokens // avg_waiting_prompt,
                ),
            ),
        }
