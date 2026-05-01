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

from dataclasses import dataclass
import time
from typing import List, Tuple

import torch
import torch.nn as nn

from mini_vllm.attention_metadata import AttentionMetadata, DecodeSeqInfo, PrefillSeqInfo
from mini_vllm.block_manager import BlockTable, KVCache, build_block_tables_tensor
from mini_vllm.config import EngineConfig
from mini_vllm.scheduler import SchedulerOutputs
from mini_vllm.sequence import Sequence


@dataclass
class _DecodeGraphCapture:
    batch_size: int
    graph: "torch.cuda.CUDAGraph"
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    logits: torch.Tensor
    scratch_blocks: List[int]


@dataclass
class _PrefillGraphCapture:
    chunk_len: int
    graph: "torch.cuda.CUDAGraph"
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    prefill_block_tables: torch.Tensor
    logits: torch.Tensor


class ModelRunner:
    def __init__(
        self,
        config: EngineConfig,
        model: nn.Module,
        kv_cache: KVCache,
        model_info: dict,
        metrics=None,
    ):
        self.config = config
        self.model = model
        self.kv_cache = kv_cache
        self.info = model_info
        self.metrics = metrics
        self.device = torch.device(config.device)

        # Generator for stochastic sampling. Per-seq generators would be
        # nicer but a single global one keeps things simple for research use.
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
        self._decode_graphs: dict[int, _DecodeGraphCapture] = {}
        self._prefill_graphs: dict[tuple[int, int], _PrefillGraphCapture] = {}
        self._graph_batch_sizes = tuple(sorted(set(int(x) for x in config.cuda_graph_batch_sizes)))
        self._graph_max_blocks = self.kv_cache.num_blocks_needed(config.max_model_len)
        self._graph_enabled = (
            config.enable_cuda_graphs
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )

    @torch.inference_mode()
    def execute(self, outputs: SchedulerOutputs) -> List[int]:
        """Run one step. Returns one sampled token_id per sequence, in the
        order [prefill_seqs..., decode_seqs...].
        """
        prefill_seqs = outputs.prefill_seqs
        decode_seqs = outputs.decode_seqs

        if self._graph_enabled and decode_seqs and not prefill_seqs:
            return self._execute_decode_graph(decode_seqs)

        if prefill_seqs and (
            self.config.enable_chunked_prefill
            or any(seq.num_cached_tokens >= seq.num_prompt_tokens for seq in prefill_seqs)
        ):
            prefill_sampled = self._execute_chunked_prefill(
                prefill_seqs,
                max_chunk=outputs.prefill_chunk_size or self.config.max_prefill_chunk_tokens,
            )
            decode_sampled = self._execute_decode_only(decode_seqs)
            return prefill_sampled + decode_sampled

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
            sliding_window=self.config.sliding_window or 0,
        )

        # ---- Forward pass ----
        forward_started = time.perf_counter()
        logits = self._forward(input_ids, position_ids, attn_metadata)
        if self.metrics is not None:
            self.metrics.observe_stage_times(prefill_forward_s=time.perf_counter() - forward_started)
        # logits: [total_tokens, vocab]
        sampling_logits = logits[sampling_token_indices]

        # ---- Sample ----
        all_seqs = prefill_seqs + decode_seqs
        sample_started = time.perf_counter()
        sampled = self._sample(sampling_logits, all_seqs)
        if self.metrics is not None:
            self.metrics.observe_stage_times(sampling_s=time.perf_counter() - sample_started)
        return sampled

    @torch.inference_mode()
    def _execute_chunked_prefill(
        self,
        prefill_seqs: List[Sequence],
        max_chunk: int | None = None,
    ) -> List[int]:
        sampled: List[int] = []
        max_chunk = max_chunk or self.config.max_prefill_chunk_tokens

        for seq in prefill_seqs:
            start_pos = seq.num_cached_tokens
            remaining = seq.prompt_token_ids[start_pos:]
            replay_last_prompt = False
            if not remaining:
                replay_last_prompt = True
                if not seq.prompt_token_ids:
                    raise RuntimeError("cannot prefill an empty prompt")
                remaining = [seq.prompt_token_ids[-1]]
                start_pos = seq.num_prompt_tokens - 1

            last_logits = None
            cursor = 0
            while cursor < len(remaining):
                chunk = remaining[cursor: cursor + max_chunk]
                chunk_start = start_pos + cursor
                logits = self._run_prefill_chunk(seq, chunk, start_pos=chunk_start)
                last_logits = logits[-1]
                cursor += len(chunk)

            if not replay_last_prompt:
                seq.num_cached_tokens = seq.num_prompt_tokens
            sampled.extend(self._sample(last_logits.unsqueeze(0), [seq]))

        return sampled

    @torch.inference_mode()
    def _execute_decode_only(self, decode_seqs: List[Sequence]) -> List[int]:
        if not decode_seqs:
            return []
        outputs = SchedulerOutputs(prefill_seqs=[], decode_seqs=decode_seqs)
        return self.execute(outputs)

    def _run_prefill_chunk(
        self,
        seq: Sequence,
        token_ids: List[int],
        start_pos: int,
    ) -> torch.Tensor:
        if self._graph_enabled and start_pos == 0:
            graph_logits = self._execute_prefill_graph_single(seq, token_ids)
            if graph_logits is not None:
                if self.metrics is not None:
                    self.metrics.observe_graph(prefill_hit=True)
                return graph_logits
            if self.metrics is not None:
                self.metrics.observe_graph(prefill_hit=False)
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        position_ids = torch.arange(
            start_pos,
            start_pos + len(token_ids),
            dtype=torch.long,
            device=self.device,
        )
        attn_metadata = AttentionMetadata(
            num_prefill_tokens=len(token_ids),
            num_decode_seqs=0,
            prefill_seq_infos=[
                PrefillSeqInfo(
                    block_table=seq.block_table,
                    token_range=(0, len(token_ids)),
                    start_pos=start_pos,
                )
            ],
            decode_seq_infos=[],
            decode_block_tables=torch.empty((0, 0), dtype=torch.int32, device=self.device),
            decode_context_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            sliding_window=self.config.sliding_window or 0,
        )
        forward_started = time.perf_counter()
        logits = self._forward(input_ids, position_ids, attn_metadata)
        if self.metrics is not None:
            self.metrics.observe_stage_times(prefill_forward_s=time.perf_counter() - forward_started)
        return logits

    @torch.inference_mode()
    def _execute_prefill_graph_single(
        self,
        seq: Sequence,
        token_ids: List[int],
    ) -> torch.Tensor | None:
        chunk_len = len(token_ids)
        if chunk_len < 1 or len(seq.block_table) > self._graph_max_blocks:
            return None
        capture = self._get_or_create_prefill_graph(chunk_len, len(seq.block_table))
        if capture is None:
            return None
        capture.input_ids.copy_(torch.tensor(token_ids, dtype=torch.long, device=self.device))
        capture.prefill_block_tables.zero_()
        blocks = seq.block_table.as_list()
        if blocks:
            capture.prefill_block_tables[0, : len(blocks)] = torch.tensor(
                blocks, dtype=torch.int32, device=self.device
            )
        capture.graph.replay()
        return capture.logits[:chunk_len].clone()

    @torch.inference_mode()
    def _execute_decode_graph(self, decode_seqs: List[Sequence]) -> List[int]:
        actual_bs = len(decode_seqs)
        capture_bs = self._select_graph_batch_size(actual_bs)
        if capture_bs is None:
            if self.metrics is not None:
                self.metrics.observe_graph(decode_hit=False)
            return self._execute_decode_eager(decode_seqs)
        if any(len(seq.block_table) > self._graph_max_blocks for seq in decode_seqs):
            if self.metrics is not None:
                self.metrics.observe_graph(decode_hit=False)
            return self._execute_decode_eager(decode_seqs)

        capture = self._get_or_create_decode_graph(capture_bs)
        if capture is None:
            if self.metrics is not None:
                self.metrics.observe_graph(decode_hit=False)
            return self._execute_decode_eager(decode_seqs)
        if self.metrics is not None:
            self.metrics.observe_graph(decode_hit=True)

        capture.input_ids.zero_()
        capture.position_ids.zero_()
        capture.block_tables.zero_()
        capture.context_lens.fill_(1)
        for row, scratch_block in enumerate(capture.scratch_blocks):
            capture.block_tables[row, 0] = scratch_block

        for row, seq in enumerate(decode_seqs):
            last_tok = seq.output_token_ids[-1] if seq.output_token_ids else seq.prompt_token_ids[-1]
            capture.input_ids[row] = last_tok
            capture.position_ids[row] = seq.seq_len - 1
            capture.context_lens[row] = seq.seq_len
            blocks = seq.block_table.as_list()
            if blocks:
                capture.block_tables[row, : len(blocks)] = torch.tensor(
                    blocks, dtype=torch.int32, device=self.device
                )

        forward_started = time.perf_counter()
        capture.graph.replay()
        sampling_logits = capture.logits[:actual_bs].clone()
        sample_started = time.perf_counter()
        sampled = self._sample(sampling_logits, decode_seqs)
        if self.metrics is not None:
            self.metrics.observe_stage_times(
                decode_forward_s=time.perf_counter() - forward_started,
                sampling_s=time.perf_counter() - sample_started,
            )
        return sampled

    @torch.inference_mode()
    def execute_decode_plan(
        self,
        decode_seqs: List[Sequence],
        num_steps: int,
        eos_token_id: int | None,
    ) -> List[List[int]]:
        if not decode_seqs or num_steps < 1:
            return [[] for _ in decode_seqs]

        actual_bs = len(decode_seqs)
        capture_bs = self._select_graph_batch_size(actual_bs) if self._graph_enabled else None
        use_graph = capture_bs is not None and not any(
            len(seq.block_table) > self._graph_max_blocks for seq in decode_seqs
        )

        if use_graph:
            capture = self._get_or_create_decode_graph(capture_bs)
            if capture is None:
                use_graph = False
        else:
            capture = None

        max_blocks = max(len(seq.block_table) for seq in decode_seqs)
        batch_slots = capture_bs if use_graph else actual_bs
        input_ids = capture.input_ids if use_graph else torch.zeros(
            (batch_slots,), dtype=torch.long, device=self.device
        )
        position_ids = capture.position_ids if use_graph else torch.zeros(
            (batch_slots,), dtype=torch.long, device=self.device
        )
        block_tables = capture.block_tables if use_graph else torch.zeros(
            (batch_slots, max_blocks), dtype=torch.int32, device=self.device
        )
        context_lens = capture.context_lens if use_graph else torch.ones(
            (batch_slots,), dtype=torch.int32, device=self.device
        )
        scratch_blocks = capture.scratch_blocks if use_graph else self._allocate_scratch_blocks(batch_slots)

        generated: List[List[int]] = [[] for _ in decode_seqs]
        last_tokens = [
            seq.output_token_ids[-1] if seq.output_token_ids else seq.prompt_token_ids[-1]
            for seq in decode_seqs
        ]
        seq_lens = [seq.seq_len for seq in decode_seqs]
        active = [True] * actual_bs

        try:
            input_ids.zero_()
            position_ids.zero_()
            block_tables.zero_()
            context_lens.fill_(1)
            for row, scratch_block in enumerate(scratch_blocks):
                block_tables[row, 0] = scratch_block
            for row, seq in enumerate(decode_seqs):
                blocks = seq.block_table.as_list()
                block_tables[row, : len(blocks)] = torch.tensor(
                    blocks, dtype=torch.int32, device=self.device
                )

            for _ in range(num_steps):
                if not any(active):
                    break

                for row in range(actual_bs):
                    if not active[row]:
                        input_ids[row] = 0
                        position_ids[row] = 0
                        context_lens[row] = 1
                        block_tables[row].zero_()
                        block_tables[row, 0] = scratch_blocks[row]
                        continue
                    input_ids[row] = last_tokens[row]
                    position_ids[row] = seq_lens[row] - 1
                    context_lens[row] = seq_lens[row]

                if use_graph:
                    step_started = time.perf_counter()
                    capture.graph.replay()
                    logits = capture.logits[:actual_bs]
                    if self.metrics is not None:
                        self.metrics.observe_graph(decode_hit=True)
                else:
                    step_started = time.perf_counter()
                    logits = self._decode_forward_static(
                        input_ids[:actual_bs],
                        position_ids[:actual_bs],
                        block_tables[:actual_bs, :max_blocks],
                        context_lens[:actual_bs],
                    )
                    if self.metrics is not None:
                        self.metrics.observe_graph(decode_hit=False)

                active_rows = [i for i, flag in enumerate(active) if flag]
                sample_started = time.perf_counter()
                sampled = self._sample(
                    logits[active_rows],
                    [decode_seqs[i] for i in active_rows],
                )
                if self.metrics is not None:
                    self.metrics.observe_stage_times(
                        decode_forward_s=time.perf_counter() - step_started,
                        sampling_s=time.perf_counter() - sample_started,
                    )

                for row, tok in zip(active_rows, sampled):
                    generated[row].append(tok)
                    last_tokens[row] = tok
                    seq_lens[row] += 1
                    if self._would_stop(
                        decode_seqs[row],
                        generated_so_far=generated[row],
                        token_id=tok,
                        eos_token_id=eos_token_id,
                    ):
                        active[row] = False
        finally:
            if not use_graph:
                self.kv_cache.allocator.free_many(scratch_blocks)

        return generated

    def _execute_decode_eager(self, decode_seqs: List[Sequence]) -> List[int]:
        outputs = SchedulerOutputs(prefill_seqs=[], decode_seqs=decode_seqs)
        input_ids_list: List[int] = []
        position_ids_list: List[int] = []
        sampling_token_indices: List[int] = []
        decode_infos: List[DecodeSeqInfo] = []

        cursor = 0
        for seq in decode_seqs:
            last_tok = seq.output_token_ids[-1] if seq.output_token_ids else seq.prompt_token_ids[-1]
            input_ids_list.append(last_tok)
            position_ids_list.append(seq.seq_len - 1)
            decode_infos.append(DecodeSeqInfo(block_table=seq.block_table, context_len=seq.seq_len))
            sampling_token_indices.append(cursor)
            cursor += 1

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
        position_ids = torch.tensor(position_ids_list, dtype=torch.long, device=self.device)
        max_blocks = max(len(s.block_table) for s in decode_seqs)
        decode_block_tables = build_block_tables_tensor(
            [s.block_table for s in decode_seqs], max_blocks, self.device
        )
        decode_context_lens = torch.tensor(
            [info.context_len for info in decode_infos], dtype=torch.int32, device=self.device
        )
        attn_metadata = AttentionMetadata(
            num_prefill_tokens=0,
            num_decode_seqs=len(decode_seqs),
            prefill_seq_infos=[],
            decode_seq_infos=decode_infos,
            decode_block_tables=decode_block_tables,
            decode_context_lens=decode_context_lens,
            sliding_window=self.config.sliding_window or 0,
        )
        forward_started = time.perf_counter()
        logits = self._forward(input_ids, position_ids, attn_metadata)
        sample_started = time.perf_counter()
        sampled = self._sample(logits[sampling_token_indices], decode_seqs)
        if self.metrics is not None:
            self.metrics.observe_stage_times(
                decode_forward_s=time.perf_counter() - forward_started,
                sampling_s=time.perf_counter() - sample_started,
            )
        return sampled

    def _decode_forward_static(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        decode_block_tables: torch.Tensor,
        decode_context_lens: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = AttentionMetadata(
            num_prefill_tokens=0,
            num_decode_seqs=input_ids.shape[0],
            prefill_seq_infos=[],
            decode_seq_infos=[],
            decode_block_tables=decode_block_tables,
            decode_context_lens=decode_context_lens,
            sliding_window=self.config.sliding_window or 0,
        )
        return self._forward(input_ids, position_ids, attn_metadata)

    def _select_graph_batch_size(self, actual_bs: int) -> int | None:
        for bs in self._graph_batch_sizes:
            if actual_bs <= bs:
                return bs
        return None

    def _get_or_create_decode_graph(self, batch_size: int) -> _DecodeGraphCapture | None:
        if batch_size in self._decode_graphs:
            return self._decode_graphs[batch_size]
        if self._graph_max_blocks < 1:
            return None
        if not self.kv_cache.allocator.can_allocate(batch_size):
            return None

        scratch_blocks = [self.kv_cache.allocator.allocate() for _ in range(batch_size)]
        input_ids = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        position_ids = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        block_tables = torch.zeros(
            (batch_size, self._graph_max_blocks), dtype=torch.int32, device=self.device
        )
        context_lens = torch.ones((batch_size,), dtype=torch.int32, device=self.device)
        for row, scratch_block in enumerate(scratch_blocks):
            block_tables[row, 0] = scratch_block

        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self._decode_forward_static(input_ids, position_ids, block_tables, context_lens)
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=warmup_stream):
            logits = self._decode_forward_static(input_ids, position_ids, block_tables, context_lens)
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)

        capture = _DecodeGraphCapture(
            batch_size=batch_size,
            graph=graph,
            input_ids=input_ids,
            position_ids=position_ids,
            block_tables=block_tables,
            context_lens=context_lens,
            logits=logits,
            scratch_blocks=scratch_blocks,
        )
        self._decode_graphs[batch_size] = capture
        return capture

    def _get_or_create_prefill_graph(
        self,
        chunk_len: int,
        max_blocks: int,
    ) -> _PrefillGraphCapture | None:
        key = (chunk_len, max_blocks)
        if key in self._prefill_graphs:
            return self._prefill_graphs[key]
        if max_blocks < 1 or max_blocks > self._graph_max_blocks:
            return None

        input_ids = torch.zeros((chunk_len,), dtype=torch.long, device=self.device)
        position_ids = torch.arange(chunk_len, dtype=torch.long, device=self.device)
        prefill_block_tables = torch.zeros((1, max_blocks), dtype=torch.int32, device=self.device)
        attn_metadata = AttentionMetadata(
            num_prefill_tokens=chunk_len,
            num_decode_seqs=0,
            prefill_seq_infos=[
                PrefillSeqInfo(
                    block_table=BlockTable(),
                    token_range=(0, chunk_len),
                    start_pos=0,
                )
            ],
            decode_seq_infos=[],
            decode_block_tables=torch.empty((0, 0), dtype=torch.int32, device=self.device),
            decode_context_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            prefill_block_tables=prefill_block_tables,
            sliding_window=self.config.sliding_window or 0,
        )

        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self._forward(input_ids, position_ids, attn_metadata)
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=warmup_stream):
            logits = self._forward(input_ids, position_ids, attn_metadata)
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)

        capture = _PrefillGraphCapture(
            chunk_len=chunk_len,
            graph=graph,
            input_ids=input_ids,
            position_ids=position_ids,
            prefill_block_tables=prefill_block_tables,
            logits=logits,
        )
        self._prefill_graphs[key] = capture
        return capture

    def _allocate_scratch_blocks(self, num_blocks: int) -> List[int]:
        if not self.kv_cache.allocator.can_allocate(num_blocks):
            raise RuntimeError("insufficient KV blocks for lookahead scratch space")
        return [self.kv_cache.allocator.allocate() for _ in range(num_blocks)]

    def _would_stop(
        self,
        seq: Sequence,
        generated_so_far: List[int],
        token_id: int,
        eos_token_id: int | None,
    ) -> bool:
        total_output = seq.num_output_tokens + len(generated_so_far)
        sp = seq.sampling_params
        if total_output >= sp.max_tokens:
            return True
        if token_id in sp.stop_token_ids:
            return True
        if not sp.ignore_eos and eos_token_id is not None and token_id == eos_token_id:
            return True
        return False

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
