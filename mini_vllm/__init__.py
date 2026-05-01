"""Mini-vLLM: Paged KV-cache inference engine for Llama-3.1-8B."""
from __future__ import annotations

from mini_vllm.config import EngineConfig
from mini_vllm.sampling import SamplingParams

__all__ = ["LLMEngine", "EngineConfig", "SamplingParams", "generate_data_parallel"]


def __getattr__(name: str):
    if name == "LLMEngine":
        from mini_vllm.engine import LLMEngine

        return LLMEngine
    if name == "generate_data_parallel":
        from mini_vllm.parallel import generate_data_parallel

        return generate_data_parallel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
