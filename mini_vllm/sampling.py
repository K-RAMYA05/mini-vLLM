"""Per-request sampling parameters.

Kept deliberately simple — temperature/top_p/top_k/greedy plus stop tokens.
Extend as needed; the sampler reads these fields directly.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 1.0        # 0.0 -> greedy
    top_p: float = 1.0
    top_k: int = -1                 # -1 -> disabled
    stop_token_ids: List[int] = field(default_factory=list)
    ignore_eos: bool = False
    seed: Optional[int] = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not (0 < self.top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k == 0:
            raise ValueError("top_k must be -1 (off) or >= 1")
