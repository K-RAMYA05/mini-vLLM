"""Structured benchmark output helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class PercentileSummary:
    p50_s: float
    p95_s: float
    p99_s: float


@dataclass
class ThroughputSummary:
    completion_tokens_per_s: float
    total_tokens_per_s: float


@dataclass
class WorkloadSummary:
    requests: int
    concurrency: int
    arrival_rate_rps: float
    max_tokens: int
    mix: str
    priority: int
    request_class: str
    lora_adapter: str | None


@dataclass
class TokenSummary:
    avg_completion_tokens: float
    finish_reasons: Dict[str | None, int]


@dataclass
class BenchmarkSummary:
    preset: str | None
    wall_time_s: float
    workload: WorkloadSummary
    ttft: PercentileSummary
    latency: PercentileSummary
    itl: PercentileSummary
    throughput: ThroughputSummary
    tokens: TokenSummary
    server_metrics_before: Dict[str, Any] | None = None
    server_metrics_after: Dict[str, Any] | None = None
    server_metrics_delta: Dict[str, float | int] | None = None
    baseline_delta: Dict[str, float | int] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
