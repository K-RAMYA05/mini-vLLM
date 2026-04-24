"""ModelRunner: orchestrates one forward pass per scheduler step.

Given a SchedulerOutputs (prefill_seqs + decode_seqs), the runner:
  1. Packs all query tokens into a single flat hidden_states tensor.
  2. Builds position_ids and AttentionMetadata.
  3. Runs the model's decoder stack (HF forward, with our attention layers
     intercepting the KV reads/writes).
  4. Takes hidden states at the LAST token of each sequence (the "sampling
     tokens") and projects to logits.
  5. Samples one token per sequence.

Packing layout (important):
  [ prefill_seq_0 tokens | prefill_seq_1 tokens | ... | decode_seq_0 | decode_seq_1 | ... ]
  So prefill tokens live in [0, num_prefill_tokens), decode tokens live in
  [num_prefill_tokens, total). The attention module splits on this boundary.

Sampling:
  Temperature / top_p / top_k / greedy. Done on-GPU batched. For sequences
  with different sampling params, we loop — cheap since the sampler tensor
  is only [B, V].
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from mini_vllm.attention_metadata import AttentionMetadata, DecodeSeqInfo, PrefillSeqInfo
from mini_vllm.block_manager import KVCache, build_block_tables_tensor
from mini_vllm.config import EngineConfig
from mini_vllm.scheduler import SchedulerOutputs
from mini_vllm.sequence import Sequence


class ModelRunner:
    def __init__(
        self,
        config: EngineConfig,
        model: nn.Module,
        kv_cache: KVCache,
        model_info: dict,
    ):
        self.config = config
        self.model = model
        self.kv_cache = kv_cache
        self.info = model_info
        self.device = torch.device(config.device)

        # Generator for stochastic sampling. Per-seq generators would be
        # nicer but a single global one keeps things simple for research use.
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)

    @torch.inference_mode()
    def execute(self, outputs: SchedulerOutputs) -> List[int]:
        """Run one step. Returns one sampled token_id per sequence, in the
        order [prefill_seqs..., decode_seqs...].
        """
        prefill_seqs = outputs.prefill_seqs
        decode_seqs = outputs.decode_seqs

        input_ids_list: List[int] = []
        position_ids_list: List[int] = []

        # Index into the packed token array of the LAST token of each seq.
        # That's the position we need logits from to sample.
        sampling_token_indices: List[int] = []

        prefill_infos: List[PrefillSeqInfo] = []
        cursor = 0
        for seq in prefill_seqs:
            toks = seq.prompt_token_ids[seq.num_cached_tokens:]
            n = len(toks)
            input_ids_list.extend(toks)
            position_ids_list.extend(range(seq.num_cached_tokens, seq.num_cached_tokens + n))
            prefill_infos.append(
                PrefillSeqInfo(
                    block_table=seq.block_table,
                    token_range=(cursor, cursor + n),
                    start_pos=seq.num_cached_tokens,
                )
            )
            cursor += n
            sampling_token_indices.append(cursor - 1)
            seq.num_cached_tokens += n
        num_prefill_tokens = cursor

        decode_infos: List[DecodeSeqInfo] = []
        for seq in decode_seqs:
            last_tok = seq.output_token_ids[-1] if seq.output_token_ids else seq.prompt_token_ids[-1]
            input_ids_list.append(last_tok)
            position_ids_list.append(seq.seq_len - 1)
            # context_len = tokens in cache AFTER this step writes the new one.
            ctx_len_after = seq.seq_len
            decode_infos.append(DecodeSeqInfo(block_table=seq.block_table, context_len=ctx_len_after))
            sampling_token_indices.append(cursor)
            cursor += 1

        if not input_ids_list:
            return []

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
        position_ids = torch.tensor(position_ids_list, dtype=torch.long, device=self.device)

        # Pre-build decode block tables + context_lens tensors for the kernel.
        if decode_seqs:
            max_blocks = max(len(s.block_table) for s in decode_seqs)
            decode_block_tables = build_block_tables_tensor(
                [s.block_table for s in decode_seqs], max_blocks, self.device
            )
            decode_context_lens = torch.tensor(
                [info.context_len for info in decode_infos], dtype=torch.int32, device=self.device
            )
        else:
            decode_block_tables = torch.empty((0, 0), dtype=torch.int32, device=self.device)
            decode_context_lens = torch.empty((0,), dtype=torch.int32, device=self.device)

        attn_metadata = AttentionMetadata(
            num_prefill_tokens=num_prefill_tokens,
            num_decode_seqs=len(decode_seqs),
            prefill_seq_infos=prefill_infos,
            decode_seq_infos=decode_infos,
            decode_block_tables=decode_block_tables,
            decode_context_lens=decode_context_lens,
        )

        # ---- Forward pass ----
        logits = self._forward(input_ids, position_ids, attn_metadata)
        # logits: [total_tokens, vocab]
        sampling_logits = logits[sampling_token_indices]

        # ---- Sample ----
        all_seqs = prefill_seqs + decode_seqs
        sampled = self._sample(sampling_logits, all_seqs)
        return sampled

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Run the HF decoder stack with our paged attention.

        Because we monkey-patched self_attn, but HF's LlamaDecoderLayer
        doesn't natively know how to pass kv_cache / attn_metadata to it,
        we call the layers manually here rather than going through the
        top-level HF forward. This gives us total control over the loop.
        """
        model = self.model
        hidden_states = model.model.embed_tokens(input_ids)

        # HF exposes the rotary embedding on the inner model; each layer's
        # PagedAttention has a reference to it already.
        for layer in model.model.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_states = layer.self_attn(
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache=self.kv_cache,
                attn_metadata=attn_metadata,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = model.model.norm(hidden_states)
        logits = model.lm_head(hidden_states)
        return logits

    def _sample(self, logits: torch.Tensor, seqs: List[Sequence]) -> List[int]:
        """Per-sequence sampling. Logits: [B, V]. Returns list of token ids."""
        out: List[int] = []
        for i, seq in enumerate(seqs):
            sp = seq.sampling_params
            row = logits[i]
            if sp.greedy:
                out.append(int(row.argmax().item()))
                continue
            # Temperature.
            row = row / sp.temperature
            # Top-k.
            if sp.top_k > 0:
                topk_vals, topk_idx = torch.topk(row, sp.top_k)
                mask = torch.full_like(row, float("-inf"))
                mask.scatter_(0, topk_idx, topk_vals)
                row = mask
            # Top-p.
            if sp.top_p < 1.0:
                sorted_vals, sorted_idx = torch.sort(row, descending=True)
                probs = torch.softmax(sorted_vals, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                cutoff = cum > sp.top_p
                # Keep first element that crosses threshold.
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_vals = sorted_vals.masked_fill(cutoff, float("-inf"))
                row = torch.full_like(row, float("-inf"))
                row.scatter_(0, sorted_idx, sorted_vals)
            probs = torch.softmax(row, dim=-1)
            tok = torch.multinomial(probs, 1, generator=self.generator).item()
            out.append(int(tok))
        return out
