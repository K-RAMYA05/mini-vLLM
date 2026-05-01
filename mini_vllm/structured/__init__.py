"""Structured-output logits processors.

Constrained decoding for the API surface — `response_format=json_object` and
regex-shaped outputs. The processor is applied to each row of logits inside
the sampler before softmax/argmax.
"""
from mini_vllm.structured.json_logits import (
    JSONLogitsProcessor,
    RegexLogitsProcessor,
    build_processor,
)

__all__ = ["JSONLogitsProcessor", "RegexLogitsProcessor", "build_processor"]
