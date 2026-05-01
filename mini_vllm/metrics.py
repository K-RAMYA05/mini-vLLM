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
    total_queue_wait_s: float = 0.0
    queue_wait_count: int = 0
    scheduler_prefill_forward_s: float = 0.0
    scheduler_decode_forward_s: float = 0.0
    scheduler_sampling_s: float = 0.0
    prefill_graph_hits: int = 0
    prefill_graph_misses: int = 0
    decode_graph_hits: int = 0
    decode_graph_misses: int = 0
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

    def observe_queue_wait(self, queue_wait_s: float) -> None:
        self.total_queue_wait_s += queue_wait_s
        self.queue_wait_count += 1

    def observe_stage_times(
        self,
        prefill_forward_s: float = 0.0,
        decode_forward_s: float = 0.0,
        sampling_s: float = 0.0,
    ) -> None:
        self.scheduler_prefill_forward_s += prefill_forward_s
        self.scheduler_decode_forward_s += decode_forward_s
        self.scheduler_sampling_s += sampling_s

    def observe_graph(self, *, decode_hit: bool | None = None, prefill_hit: bool | None = None) -> None:
        if decode_hit is not None:
            if decode_hit:
                self.decode_graph_hits += 1
            else:
                self.decode_graph_misses += 1
        if prefill_hit is not None:
            if prefill_hit:
                self.prefill_graph_hits += 1
            else:
                self.prefill_graph_misses += 1

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
            "avg_queue_wait_s": self.total_queue_wait_s / max(self.queue_wait_count, 1),
            "prefill_forward_time_s": self.scheduler_prefill_forward_s,
            "decode_forward_time_s": self.scheduler_decode_forward_s,
            "sampling_time_s": self.scheduler_sampling_s,
            "prefill_graph_hits": self.prefill_graph_hits,
            "prefill_graph_misses": self.prefill_graph_misses,
            "decode_graph_hits": self.decode_graph_hits,
            "decode_graph_misses": self.decode_graph_misses,
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
            out.update(kv_cache.memory_stats())
        if scheduler is not None:
            out.update({
                "waiting_requests": len(scheduler.waiting),
                "running_requests": len(scheduler.running),
                "swapped_requests": len(scheduler.swapped),
            })
            if scheduler.running:
                now = time.perf_counter()
                active_ages = [
                    now - (seq.admitted_time_s or seq.created_time_s)
                    for seq in scheduler.running
                ]
                out["active_request_age_max_s"] = max(active_ages)
                out["active_request_age_avg_s"] = sum(active_ages) / len(active_ages)
            else:
                out["active_request_age_max_s"] = 0.0
                out["active_request_age_avg_s"] = 0.0
            if getattr(scheduler, "prefix_cache", None) is not None:
                out["prefix_cache_entries"] = scheduler.prefix_cache.num_entries
            out.update(scheduler.estimate_admission_capacity())
            out["active_adapter_cohort"] = scheduler._last_adapter_served or "__base__"
        return out

    def structured_snapshot(self, kv_cache=None, scheduler=None) -> Dict[str, object]:
        flat = self.snapshot(kv_cache=kv_cache, scheduler=scheduler)
        structured: Dict[str, object] = {
            "runtime": {
                "uptime_s": flat["uptime_s"],
                "output_tokens_per_s": flat["output_tokens_per_s"],
            },
            "requests": {
                "started": flat["requests_started"],
                "finished": flat["requests_finished"],
                "aborted": flat["requests_aborted"],
                "output_tokens": flat["output_tokens"],
            },
            "latency": {
                "avg_ttft_s": flat["avg_ttft_s"],
                "avg_itl_s": flat["avg_itl_s"],
                "avg_queue_wait_s": flat["avg_queue_wait_s"],
                "active_request_age_max_s": flat.get("active_request_age_max_s", 0.0),
                "active_request_age_avg_s": flat.get("active_request_age_avg_s", 0.0),
            },
            "scheduler": {
                "steps": flat["scheduler_steps"],
                "prefill_tokens": flat["prefill_tokens"],
                "decode_tokens": flat["decode_tokens"],
                "waiting_requests": flat.get("waiting_requests", 0),
                "running_requests": flat.get("running_requests", 0),
                "swapped_requests": flat.get("swapped_requests", 0),
                "admission_capacity_tokens": flat.get("admission_capacity_tokens", 0),
                "admission_capacity_seqs": flat.get("admission_capacity_seqs", 0),
                "active_adapter_cohort": flat.get("active_adapter_cohort", "__base__"),
            },
            "stages": {
                "prefill_forward_time_s": flat["prefill_forward_time_s"],
                "decode_forward_time_s": flat["decode_forward_time_s"],
                "sampling_time_s": flat["sampling_time_s"],
            },
            "graphs": {
                "prefill_hits": flat["prefill_graph_hits"],
                "prefill_misses": flat["prefill_graph_misses"],
                "decode_hits": flat["decode_graph_hits"],
                "decode_misses": flat["decode_graph_misses"],
            },
            "prefix_cache": {
                "queries": flat["prefix_cache_queries"],
                "hits": flat["prefix_cache_hits"],
                "hit_tokens": flat["prefix_cache_hit_tokens"],
                "entries": flat.get("prefix_cache_entries", 0),
            },
            "swap": {
                "swap_out_count": flat["swap_out_count"],
                "swap_in_count": flat["swap_in_count"],
            },
        }
        if kv_cache is not None:
            structured["kv_cache"] = {
                "gpu_blocks_total": flat["gpu_kv_blocks_total"],
                "gpu_blocks_free": flat["gpu_kv_blocks_free"],
                "gpu_blocks_used": flat["gpu_kv_blocks_used"],
                "cpu_blocks_total": flat["cpu_kv_blocks_total"],
                "cpu_blocks_free": flat["cpu_kv_blocks_free"],
                "cpu_blocks_used": flat["cpu_kv_blocks_used"],
                "gpu_kv_tokens_total": flat.get("gpu_kv_tokens_total", 0),
                "gpu_kv_tokens_used": flat.get("gpu_kv_tokens_used", 0),
                "gpu_kv_tokens_free": flat.get("gpu_kv_tokens_free", 0),
                "gpu_kv_bytes_total": flat.get("gpu_kv_bytes_total", 0),
                "gpu_kv_bytes_used": flat.get("gpu_kv_bytes_used", 0),
                "gpu_kv_bytes_free": flat.get("gpu_kv_bytes_free", 0),
            }
        return structured

    def prometheus(self, kv_cache=None, scheduler=None) -> str:
        snap = self.snapshot(kv_cache=kv_cache, scheduler=scheduler)
        lines = []
        for key, value in snap.items():
            if isinstance(value, (int, float)):
                lines.append(f"mini_vllm_{key} {float(value)}")
        return "\n".join(lines) + "\n"
