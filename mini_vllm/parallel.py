"""Multi-process data-parallel inference helpers.

This module keeps the core engine single-GPU and runs one independent
LLMEngine replica per worker. It is the right kind of parallelism for this
codebase today: requests are naturally independent, paged KV cache ownership
stays local to each process, and no tensor-parallel collectives are needed in
the hot path.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import replace
from typing import Iterable, List, Optional, Sequence as TypingSequence, Tuple

import torch

from mini_vllm.config import EngineConfig
from mini_vllm.engine import LLMEngine
from mini_vllm.sampling import SamplingParams
from mini_vllm.sequence import RequestOutput


def generate_data_parallel(
    prompts: TypingSequence[str],
    config: EngineConfig,
    sampling_params: Optional[SamplingParams] = None,
    num_workers: Optional[int] = None,
    devices: Optional[TypingSequence[str]] = None,
) -> List[RequestOutput]:
    """Generate for prompts using one engine process per worker.

    Args:
        prompts: Input prompts. Output order matches this input order.
        config: Base engine config. Each worker gets a copy with its local
            device filled in.
        sampling_params: Sampling parameters shared by all prompts.
        num_workers: Number of worker processes. Defaults to all CUDA devices
            when CUDA is available, otherwise one CPU worker.
        devices: Explicit device strings, e.g. ["cuda:0", "cuda:1"].
    """
    if sampling_params is None:
        sampling_params = SamplingParams()
    if not prompts:
        return []

    if devices is None:
        if num_workers is None:
            num_workers = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        devices = [f"cuda:{i}" for i in range(num_workers)] if torch.cuda.is_available() else ["cpu"]
    else:
        devices = list(devices)
        num_workers = len(devices)
        if num_workers == 0:
            raise ValueError("devices must not be empty")

    indexed_prompts = list(enumerate(prompts))
    shards = _round_robin(indexed_prompts, len(devices))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(devices)) as pool:
        jobs = [
            pool.apply_async(_worker_generate, (device, config, sampling_params, shard))
            for device, shard in zip(devices, shards)
            if shard
        ]
        flat = [item for job in jobs for item in job.get()]

    flat.sort(key=lambda item: item[0])
    return [out for _, out in flat]


def _round_robin(items: TypingSequence[Tuple[int, str]], n: int) -> List[List[Tuple[int, str]]]:
    shards: List[List[Tuple[int, str]]] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        shards[i % n].append(item)
    return shards


def _worker_generate(
    device: str,
    config: EngineConfig,
    sampling_params: SamplingParams,
    indexed_prompts: Iterable[Tuple[int, str]],
) -> List[Tuple[int, RequestOutput]]:
    worker_config = replace(config, device=device)
    engine = LLMEngine(worker_config)
    request_order: List[Tuple[int, int]] = []
    for original_idx, prompt in indexed_prompts:
        seq_id = engine.add_request(prompt, sampling_params)
        request_order.append((original_idx, seq_id))

    outputs = {out.seq_id: out for out in engine.run_until_done()}
    return [(original_idx, outputs[seq_id]) for original_idx, seq_id in request_order]
