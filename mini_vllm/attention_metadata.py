"""Metadata passed from the runner to every attention layer each step.

Separated from attention.py so we can build it once per step and reuse
across all 32 layers without recomputing block tables, context lengths, etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch

from mini_vllm.block_manager import BlockTable


@dataclass
class PrefillSeqInfo:
    block_table: BlockTable
    token_range: Tuple[int, int]      # [start, end) in the packed hidden_states
    start_pos: int = 0


@dataclass
class DecodeSeqInfo:
    block_table: BlockTable
    context_len: int                   # length AFTER this step's token is appended


@dataclass
class AttentionMetadata:
    num_prefill_tokens: int
    num_decode_seqs: int
    prefill_seq_infos: List[PrefillSeqInfo]
    decode_seq_infos: List[DecodeSeqInfo]
    # Pre-built tensors consumed directly by the Triton kernel.
    decode_block_tables: torch.Tensor  # [num_decode_seqs, max_blocks] int32
    decode_context_lens: torch.Tensor  # [num_decode_seqs] int32
    prefill_block_tables: Optional[torch.Tensor] = None  # [num_prefill_seqs, max_blocks] int32
    # When set (>0), attention attends only to the last `sliding_window`
    # tokens. Plumbed through to both the prefill SDPA mask and the paged
    # decode kernel.
    sliding_window: int = 0
