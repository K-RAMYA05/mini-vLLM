"""HTTP benchmark harness for the OpenAI-compatible server.

Measures:
  - TTFT (time to first token)
  - ITL (inter-token latency)
  - request latency
  - completion-token throughput
  - mixed-load steady-state under concurrent arrivals
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from benchmarks.schema import (
    BenchmarkSummary,
    PercentileSummary,
    ThroughputSummary,
    TokenSummary,
    WorkloadSummary,
)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return values[lo]
    frac = rank - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


@dataclass
class BenchResult:
    ttft_s: float
    itls_s: list[float]
    latency_s: float
    completion_tokens: int
    prompt_tokens: int
    finish_reason: str | None


def _builtin_prompts(mix: str) -> list[str]:
    short = [
        "Explain paged attention in two sentences.",
        "List three uses of CUDA graphs.",
        "What is TTFT?",
    ]
    medium = [
        "Write a concise explanation of how chunked prefill interacts with paged KV cache.",
        "Compare decode throughput, TTFT, and ITL for continuous batching in an LLM server.",
        "Describe why H100 FP8 KV cache can help decode performance and what still bottlenecks it.",
    ]
    long = [
        "Summarize the architecture of a vLLM-style server including scheduler, paged attention, prefix cache, chunked prefill, and continuous batching. Explain the main performance bottlenecks and tradeoffs in each stage.",
        "Explain how an OpenAI-compatible serving layer should implement request queueing, backpressure, mixed-load benchmarking, and operational metrics for a single-GPU LLM engine.",
    ]
    pieces = []
    for part in mix.split(","):
        part = part.strip()
        if part == "short":
            pieces.extend(short)
        elif part == "medium":
            pieces.extend(medium)
        elif part == "long":
            pieces.extend(long)
    return pieces or short + medium + long


def _load_prompts(path: str | None, mix: str) -> list[str]:
    if path is None:
        return _builtin_prompts(mix)
    prompts = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                prompts.append(data.get("prompt") or data.get("text") or "")
            else:
                prompts.append(str(data))
        except json.JSONDecodeError:
            prompts.append(line)
    return [p for p in prompts if p]


def _apply_preset(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    preset_name = None
    for idx, arg in enumerate(argv):
        if arg == "--preset" and idx + 1 < len(argv):
            preset_name = argv[idx + 1]
            break
    defaults = {}
    if preset_name is not None:
        preset_path = Path("configs/benchmarks") / f"{preset_name}.json"
        defaults = json.loads(preset_path.read_text())
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    setattr(args, "_preset_defaults", defaults)
    return args


async def _run_one(
    client,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    priority: int,
    request_class: str,
    lora_adapter: str | None,
) -> BenchResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "priority": priority,
        "request_class": request_class,
        "lora_adapter": lora_adapter,
        "stream": True,
    }
    started = time.perf_counter()
    first_token_at = None
    last_chunk_at = None
    itls: list[float] = []
    completion_tokens = 0
    prompt_tokens = 0
    finish_reason = None

    async with client.stream("POST", url, json=payload, timeout=None) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            message = json.loads(data)
            choice = message["choices"][0]
            text = choice.get("text", "")
            finish_reason = choice.get("finish_reason")
            usage = message.get("usage")
            if usage is not None:
                prompt_tokens = int(usage["prompt_tokens"])
                completion_tokens = int(usage["completion_tokens"])
            if text:
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                elif last_chunk_at is not None:
                    itls.append(now - last_chunk_at)
                last_chunk_at = now
    ended = time.perf_counter()
    if first_token_at is None:
        first_token_at = ended
    return BenchResult(
        ttft_s=first_token_at - started,
        itls_s=itls,
        latency_s=ended - started,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        finish_reason=finish_reason,
    )


def _flatten_numeric(data: dict, prefix: str = "") -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten_numeric(value, prefix=full_key))
        elif isinstance(value, (int, float)):
            out[full_key] = value
    return out


def _diff_numeric(after: dict, before: dict) -> dict[str, float | int]:
    after_flat = _flatten_numeric(after)
    before_flat = _flatten_numeric(before)
    out = {}
    for key, value in after_flat.items():
        before_value = before_flat.get(key)
        if isinstance(before_value, (int, float)):
            out[key] = value - before_value
    return out


async def _get_metrics(client, base_url: str) -> dict | None:
    try:
        response = await client.get(base_url.rstrip("/") + "/metrics/structured")
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


async def _benchmark(args) -> dict:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("benchmark harness requires `pip install -e '.[bench]'` with httpx available") from exc

    prompts = _load_prompts(args.prompts, args.mix)
    rng = random.Random(args.seed)
    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[BenchResult] = []

    async with httpx.AsyncClient() as client:
        metrics_before = await _get_metrics(client, args.base_url)

        async def launch_request(prompt: str):
            async with semaphore:
                result = await _run_one(
                    client,
                    args.base_url.rstrip("/") + "/v1/completions",
                    args.model,
                    prompt,
                    args.max_tokens,
                    args.priority,
                    args.request_class,
                    args.lora_adapter,
                )
                results.append(result)

        tasks = []
        bench_started = time.perf_counter()
        for i in range(args.num_requests):
            prompt = prompts[i % len(prompts)]
            if args.arrival_rate > 0 and i > 0:
                await asyncio.sleep(rng.expovariate(args.arrival_rate))
            tasks.append(asyncio.create_task(launch_request(prompt)))
        await asyncio.gather(*tasks)
        bench_ended = time.perf_counter()
        metrics_after = await _get_metrics(client, args.base_url)

    ttfts = sorted(r.ttft_s for r in results)
    latencies = sorted(r.latency_s for r in results)
    itls = sorted(itl for r in results for itl in r.itls_s)
    completion_tokens = sum(r.completion_tokens for r in results)
    prompt_tokens = sum(r.prompt_tokens for r in results)
    wall = max(bench_ended - bench_started, 1e-9)
    summary = BenchmarkSummary(
        preset=args.preset,
        wall_time_s=wall,
        workload=WorkloadSummary(
            requests=len(results),
            concurrency=args.concurrency,
            arrival_rate_rps=args.arrival_rate,
            max_tokens=args.max_tokens,
            mix=args.mix,
            priority=args.priority,
            request_class=args.request_class,
            lora_adapter=args.lora_adapter,
        ),
        ttft=PercentileSummary(
            p50_s=_percentile(ttfts, 0.50),
            p95_s=_percentile(ttfts, 0.95),
            p99_s=_percentile(ttfts, 0.99),
        ),
        latency=PercentileSummary(
            p50_s=_percentile(latencies, 0.50),
            p95_s=_percentile(latencies, 0.95),
            p99_s=_percentile(latencies, 0.99),
        ),
        itl=PercentileSummary(
            p50_s=_percentile(itls, 0.50),
            p95_s=_percentile(itls, 0.95),
            p99_s=_percentile(itls, 0.99),
        ),
        throughput=ThroughputSummary(
            completion_tokens_per_s=completion_tokens / wall,
            total_tokens_per_s=(completion_tokens + prompt_tokens) / wall,
        ),
        tokens=TokenSummary(
            avg_completion_tokens=statistics.fmean([r.completion_tokens for r in results]) if results else 0.0,
            finish_reasons={reason: sum(1 for r in results if r.finish_reason == reason) for reason in {r.finish_reason for r in results}},
        ),
        server_metrics_before=metrics_before,
        server_metrics_after=metrics_after,
        server_metrics_delta=(
            _diff_numeric(metrics_after, metrics_before)
            if isinstance(metrics_before, dict) and isinstance(metrics_after, dict)
            else None
        ),
    )
    return summary.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--preset", default=None)
    parser.add_argument("--prompts", default=None, help="Optional JSONL or plain-text prompt file.")
    parser.add_argument("--mix", default="short,medium,long")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--arrival-rate", type=float, default=0.0, help="Poisson arrival rate (req/s). 0 = closed loop.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--request-class", default="default")
    parser.add_argument("--lora-adapter", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--baseline", default=None, help="Optional baseline JSON to compare against.")
    import sys
    args = _apply_preset(parser, sys.argv[1:])

    summary = asyncio.run(_benchmark(args))
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        summary["baseline_delta"] = _diff_numeric(summary, baseline)
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
