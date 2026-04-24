"""Runtime metrics for serving and benchmark runs."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EngineMetrics:
    requests_started: int = 0
    requests_finished: int = 0
    requests_aborted: int = 0
    output_tokens: int = 0
    scheduler_steps: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    swap_out_count: int = 0
    swap_in_count: int = 0
    prefix_cache_queries: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_hit_tokens: int = 0
    total_time_to_first_token_s: float = 0.0
    total_inter_token_latency_s: float = 0.0
    inter_token_latency_count: int = 0
    start_time_s: float = field(default_factory=time.perf_counter)

    def request_started(self) -> None:
        self.requests_started += 1

    def request_finished(self, aborted: bool = False) -> None:
        self.requests_finished += 1
        if aborted:
            self.requests_aborted += 1

    def observe_step(self, prefill_tokens: int, decode_tokens: int) -> None:
        self.scheduler_steps += 1
        self.prefill_tokens += prefill_tokens
        self.decode_tokens += decode_tokens

    def observe_token(self, ttft_s: float | None, itl_s: float | None) -> None:
        self.output_tokens += 1
        if ttft_s is not None:
            self.total_time_to_first_token_s += ttft_s
        if itl_s is not None:
            self.total_inter_token_latency_s += itl_s
            self.inter_token_latency_count += 1

    def observe_prefix_cache(self, hit_tokens: int) -> None:
        self.prefix_cache_queries += 1
        if hit_tokens > 0:
            self.prefix_cache_hits += 1
            self.prefix_cache_hit_tokens += hit_tokens

    def snapshot(self, kv_cache=None, scheduler=None) -> Dict[str, float | int]:
        elapsed = max(time.perf_counter() - self.start_time_s, 1e-9)
        out: Dict[str, float | int] = {
            "uptime_s": elapsed,
            "requests_started": self.requests_started,
            "requests_finished": self.requests_finished,
            "requests_aborted": self.requests_aborted,
            "output_tokens": self.output_tokens,
            "output_tokens_per_s": self.output_tokens / elapsed,
            "scheduler_steps": self.scheduler_steps,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "swap_out_count": self.swap_out_count,
            "swap_in_count": self.swap_in_count,
            "prefix_cache_queries": self.prefix_cache_queries,
            "prefix_cache_hits": self.prefix_cache_hits,
            "prefix_cache_hit_tokens": self.prefix_cache_hit_tokens,
            "avg_ttft_s": self.total_time_to_first_token_s / max(self.requests_started, 1),
            "avg_itl_s": self.total_inter_token_latency_s / max(self.inter_token_latency_count, 1),
        }
        if kv_cache is not None:
            out.update({
                "gpu_kv_blocks_total": kv_cache.allocator.num_blocks,
                "gpu_kv_blocks_free": kv_cache.allocator.num_free,
                "gpu_kv_blocks_used": kv_cache.allocator.num_allocated,
                "cpu_kv_blocks_total": kv_cache.cpu_allocator.num_blocks if kv_cache.cpu_allocator else 0,
                "cpu_kv_blocks_free": kv_cache.cpu_allocator.num_free if kv_cache.cpu_allocator else 0,
                "cpu_kv_blocks_used": kv_cache.cpu_allocator.num_allocated if kv_cache.cpu_allocator else 0,
            })
        if scheduler is not None:
            out.update({
                "waiting_requests": len(scheduler.waiting),
                "running_requests": len(scheduler.running),
                "swapped_requests": len(scheduler.swapped),
            })
            if getattr(scheduler, "prefix_cache", None) is not None:
                out["prefix_cache_entries"] = scheduler.prefix_cache.num_entries
        return out

    def prometheus(self, kv_cache=None, scheduler=None) -> str:
        snap = self.snapshot(kv_cache=kv_cache, scheduler=scheduler)
        lines = []
        for key, value in snap.items():
            lines.append(f"mini_vllm_{key} {float(value)}")
        return "\n".join(lines) + "\n"
