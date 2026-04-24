"""A Sequence is one generation request flowing through the engine.

Lifecycle:
  WAITING    -> sitting in the scheduler queue, no blocks allocated yet
  RUNNING    -> in the current batch, has a block table
  FINISHED   -> hit EOS, max tokens, or a stop token; blocks freed
"""
from __future__ import annotations

import enum
import itertools
import time
from dataclasses import dataclass, field
from typing import List, Optional

from mini_vllm.block_manager import BlockTable
from mini_vllm.sampling import SamplingParams


class SequenceStatus(enum.Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    SWAPPED = "SWAPPED"
    FINISHED_STOPPED = "FINISHED_STOPPED"
    FINISHED_LENGTH = "FINISHED_LENGTH"
    FINISHED_ABORTED = "FINISHED_ABORTED"

    @property
    def is_finished(self) -> bool:
        return self.name.startswith("FINISHED_")


_seq_id_counter = itertools.count()


@dataclass
class Sequence:
    prompt: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    seq_id: int = field(default_factory=lambda: next(_seq_id_counter))
    output_token_ids: List[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    block_table: BlockTable = field(default_factory=BlockTable)
    created_time_s: float = field(default_factory=time.perf_counter)
    first_token_time_s: Optional[float] = None
    last_token_time_s: Optional[float] = None

    # Number of tokens already committed into the KV cache. During prefill
    # this jumps from 0 to len(prompt); during decode it increments by 1
    # (or by the number of accepted draft tokens under speculative decoding).
    num_cached_tokens: int = 0
    prefix_cache_blocks: int = 0

    # Filled in when the sequence finishes.
    finish_reason: Optional[str] = None

    @property
    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def seq_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    def append_output_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def check_stop(self, eos_token_id: Optional[int]) -> bool:
        """Decide whether this sequence should finish on this step."""
        sp = self.sampling_params
        if self.num_output_tokens >= sp.max_tokens:
            self.status = SequenceStatus.FINISHED_LENGTH
            self.finish_reason = "length"
            return True
        if self.output_token_ids:
            last = self.output_token_ids[-1]
            if last in sp.stop_token_ids:
                self.status = SequenceStatus.FINISHED_STOPPED
                self.finish_reason = "stop"
                return True
            if not sp.ignore_eos and eos_token_id is not None and last == eos_token_id:
                self.status = SequenceStatus.FINISHED_STOPPED
                self.finish_reason = "eos"
                return True
        return False


@dataclass
class RequestOutput:
    """Returned to the caller when a sequence completes."""
    seq_id: int
    prompt: str
    prompt_token_ids: List[int]
    output_token_ids: List[int]
    output_text: str
    finish_reason: Optional[str]
