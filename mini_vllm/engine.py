"""Top-level inference engine.

Usage:
    engine = LLMEngine(EngineConfig(model_name_or_path=...))
    engine.add_request("Once upon a time", SamplingParams(max_tokens=64))
    outputs = engine.run_until_done()     # blocks, returns List[RequestOutput]

Or streaming-style:
    while engine.has_pending_work():
        step_outputs = engine.step()       # returns completed seqs this step

The engine owns: config, model, KV cache, scheduler, runner. It is single-
threaded and synchronous — there's no async queue, no separate worker
process. All concurrency is via batching.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

import torch

from mini_vllm.block_manager import KVCache
from mini_vllm.config import EngineConfig
from mini_vllm.model_loader import load_model
from mini_vllm.model_runner import ModelRunner
from mini_vllm.metrics import EngineMetrics
from mini_vllm.sampling import SamplingParams
from mini_vllm.scheduler import Scheduler
from mini_vllm.sequence import RequestOutput, Sequence


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class LLMEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        if config.tensor_parallel_size != 1 or config.pipeline_parallel_size != 1:
            raise NotImplementedError(
                "tensor_parallel_size and pipeline_parallel_size are config placeholders; "
                "this engine currently supports one full model replica per process. "
                "Use generate_data_parallel for request-level multi-GPU sharding."
            )
        logging.basicConfig(level=config.log_level)
        self.logger = logging.getLogger("mini_vllm")

        dtype = _DTYPE_MAP[config.dtype]

        self.logger.info(f"Loading model {config.model_name_or_path} ...")
        self.model, self.tokenizer, self.info = load_model(
            config.model_name_or_path,
            dtype=dtype,
            device=config.device,
            use_triton=config.use_triton_attention,
            prefill_backend=config.prefill_attention_backend,
            trust_remote_code=config.trust_remote_code,
        )

        # Optional weight-only GPTQ quantization, applied post-load so the HF loader
        # doesn't have to understand our format.
        if config.use_quantization:
            from mini_vllm.quant import apply_gptq_quantization
            self.logger.info(f"Quantizing weights to {config.quant_bits}-bit GPTQ ...")
            apply_gptq_quantization(
                self.model,
                bits=config.quant_bits,
                group_size=config.quant_group_size,
                tokenizer=self.tokenizer,
            )

        self.kv_cache = KVCache(
            num_layers=self.info["num_layers"],
            num_kv_heads=self.info["num_kv_heads"],
            head_dim=self.info["head_dim"],
            num_blocks=config.num_gpu_blocks,
            block_size=config.block_size,
            dtype=dtype,
            device=config.device,
            num_cpu_blocks=config.num_cpu_blocks,
        )

        self.metrics = EngineMetrics()
        self.scheduler = Scheduler(config, self.kv_cache, metrics=self.metrics)
        self.runner = ModelRunner(config, self.model, self.kv_cache, self.info)

        # Speculative decoding (optional).
        self.spec_executor = None
        if config.use_speculative:
            from mini_vllm.speculative.spec_decode import SpeculativeExecutor
            self.spec_executor = SpeculativeExecutor(config, self)

        self.finished_outputs: List[RequestOutput] = []

    # -------------------------- public API --------------------------

    def add_request(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> int:
        """Queue a generation request. Returns the seq_id."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        token_ids = self.tokenizer.encode(prompt)
        seq = Sequence(
            prompt=prompt,
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
        )
        self.scheduler.add_seq(seq)
        self.metrics.request_started()
        return seq.seq_id

    def has_pending_work(self) -> bool:
        return self.scheduler.has_work

    def step(self) -> List[RequestOutput]:
        """Run one scheduler step; return any sequences that finished."""
        outputs = self.scheduler.schedule()
        if outputs.is_empty:
            return []
        self.metrics.observe_step(
            prefill_tokens=sum(s.num_prompt_tokens for s in outputs.prefill_seqs),
            decode_tokens=len(outputs.decode_seqs),
        )

        if self.spec_executor is not None and outputs.decode_seqs and not outputs.prefill_seqs:
            # Speculative decode path (decode-only steps). Prefill goes
            # through the normal runner.
            sampled_per_seq = self.spec_executor.step(outputs.decode_seqs)
            # sampled_per_seq: List[List[int]] — accepted tokens per seq
            all_seqs = outputs.prefill_seqs + outputs.decode_seqs
            # prefill is empty here, so index aligns with decode_seqs.
            for seq, tokens in zip(outputs.decode_seqs, sampled_per_seq):
                for tok in tokens:
                    seq.append_output_token(tok)
                    self._observe_generated_token(seq)
                    if seq.check_stop(self.info["eos_token_id"]):
                        break
        else:
            sampled = self.runner.execute(outputs)
            all_seqs = outputs.prefill_seqs + outputs.decode_seqs
            for seq, tok in zip(all_seqs, sampled):
                seq.append_output_token(tok)
                self._observe_generated_token(seq)
                seq.check_stop(self.info["eos_token_id"])

        self.scheduler.register_prefill_cache(outputs.prefill_seqs)

        # Collect any sequences that finished this step.
        finished = self.scheduler.finalize_finished()
        results = [self._build_output(s) for s in finished]
        for s in finished:
            self.metrics.request_finished(aborted=s.finish_reason == "oom")
        self.finished_outputs.extend(results)
        return results

    def run_until_done(self) -> List[RequestOutput]:
        """Run repeated steps until all queued requests complete."""
        while self.has_pending_work():
            self.step()
        out = self.finished_outputs
        self.finished_outputs = []
        # Sort by seq_id so output order matches insertion order (mostly).
        out.sort(key=lambda r: r.seq_id)
        return out

    # -------------------------- internals --------------------------

    def _build_output(self, seq: Sequence) -> RequestOutput:
        text = self.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True)
        return RequestOutput(
            seq_id=seq.seq_id,
            prompt=seq.prompt,
            prompt_token_ids=seq.prompt_token_ids,
            output_token_ids=seq.output_token_ids,
            output_text=text,
            finish_reason=seq.finish_reason,
        )

    def _observe_generated_token(self, seq: Sequence) -> None:
        now = time.perf_counter()
        ttft = None
        itl = None
        if seq.first_token_time_s is None:
            seq.first_token_time_s = now
            ttft = now - seq.created_time_s
        elif seq.last_token_time_s is not None:
            itl = now - seq.last_token_time_s
        seq.last_token_time_s = now
        self.metrics.observe_token(ttft_s=ttft, itl_s=itl)

    def get_metrics(self) -> dict:
        return self.metrics.snapshot(kv_cache=self.kv_cache, scheduler=self.scheduler)

    def get_prometheus_metrics(self) -> str:
        return self.metrics.prometheus(kv_cache=self.kv_cache, scheduler=self.scheduler)
