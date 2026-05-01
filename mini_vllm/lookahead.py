"""Lookahead decoding executor.

This path keeps the current resident decode batch on GPU and advances it for
several decode substeps before returning control to the scheduler. It is a
single-model multi-step decode path: no draft model, just fewer round-trips
through Python and queue management.
"""
from __future__ import annotations

from typing import List

from mini_vllm.sequence import Sequence, SequenceStatus


class LookaheadExecutor:
    def __init__(self, engine: "LLMEngine"):
        self.engine = engine
        self.num_slots = engine.config.lookahead_num_slots

    def step(self, seqs: List[Sequence]) -> List[Sequence]:
        """Advance decode sequences for up to `num_slots` substeps."""
        active = [seq for seq in seqs if not seq.status.is_finished]
        if not active:
            return seqs

        plan_steps = self._reserve_plan_capacity(active)
        if plan_steps < 1:
            return seqs

        try:
            generated = self.engine.runner.execute_decode_plan(
                active,
                num_steps=plan_steps,
                eos_token_id=self.engine.info["eos_token_id"],
            )
        except RuntimeError:
            return self._legacy_step(active, plan_steps)
        for seq, toks in zip(active, generated):
            for tok in toks:
                seq.append_output_token(tok)
                self.engine._observe_generated_token(seq)
                if seq.check_stop(self.engine.info["eos_token_id"]):
                    break
        return seqs

    def _reserve_plan_capacity(self, seqs: List[Sequence]) -> int:
        for steps in range(self.num_slots, 0, -1):
            extra_needed = 0
            for seq in seqs:
                target_blocks = self.engine.kv_cache.num_blocks_needed(seq.seq_len + steps)
                extra_needed += max(target_blocks - len(seq.block_table), 0)
            if self.engine.kv_cache.allocator.can_allocate(extra_needed):
                for seq in seqs:
                    target_blocks = self.engine.kv_cache.num_blocks_needed(seq.seq_len + steps)
                    while len(seq.block_table) < target_blocks:
                        seq.block_table.append(self.engine.kv_cache.allocator.allocate())
                return steps
        for seq in list(seqs):
            if not self.engine.scheduler._append_slot_if_needed(seq):
                seq.status = SequenceStatus.FINISHED_ABORTED
                seq.finish_reason = "oom"
        return 1 if any(not seq.status.is_finished for seq in seqs) else 0

    def _legacy_step(self, active: List[Sequence], plan_steps: int) -> List[Sequence]:
        for _ in range(plan_steps):
            step_active = [seq for seq in active if not seq.status.is_finished]
            if not step_active:
                break
            for seq in list(step_active):
                if not self.engine.scheduler._append_slot_if_needed(seq):
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    seq.finish_reason = "oom"
                    step_active.remove(seq)
            if not step_active:
                break
            sampled = self.engine.runner.execute_decode_eager(step_active)
            for seq, tok in zip(step_active, sampled):
                seq.append_output_token(tok)
                self.engine._observe_generated_token(seq)
                seq.check_stop(self.engine.info["eos_token_id"])
        return active
