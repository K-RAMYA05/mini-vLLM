"""Configuration for the inference engine.

All knobs that affect behavior, memory, or throughput live here so that
experiments are reproducible from a single config object.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EngineConfig:
    # ---- Model ----
    model_name_or_path: str = "meta-llama/Llama-3.1-8B"
    dtype: str = "bfloat16"  # float16 | bfloat16 | float32
    device: str = "cuda"
    trust_remote_code: bool = False

    # ---- Paged KV cache ----
    # Block size is the number of tokens per KV block. 16 is what vLLM uses
    # by default on most GPUs — good balance between fragmentation and the
    # fixed per-block overhead in the attention kernel.
    block_size: int = 16
    # Total number of blocks to pre-allocate. Total tokens cacheable =
    # num_gpu_blocks * block_size. For an 8B Llama-class model with 32 layers,
    # 8 KV heads, 128 head_dim, bf16/fp16: per block is about 2 MiB.
    # 8192 blocks is about 16 GiB of KV cache.
    num_gpu_blocks: int = 8192
    # Swap space on CPU for preemption. 0 disables swapping.
    num_cpu_blocks: int = 0
    # 'auto' = match model dtype (fp16/bf16). 'int8' = symmetric per-token
    # quantization with one fp16 scale per (block, kv_head, slot). ~50%
    # memory saving at rest, enabling ~2x longer context. Quantization error
    # is bounded by max-abs of each row.
    kv_cache_dtype: str = "auto"  # auto | int8

    # ---- Scheduler / continuous batching ----
    max_num_seqs: int = 256                # max concurrent sequences
    max_num_batched_tokens: int = 8192     # token budget per forward pass
    max_model_len: int = 4096              # max context length per sequence
    enable_chunked_prefill: bool = False
    max_prefill_chunk_tokens: int = 2048
    enable_prefix_cache: bool = False
    prefix_cache_max_entries: int = 16384
    # LRU is fine for single-tenant. LFU is much better when one popular
    # system prompt is repeatedly hit while many one-shot prompts churn.
    prefix_cache_eviction: str = "lru"  # lru | lfu

    # ---- Multi-GPU placeholders ----
    # Real TP/PP requires sharded weight loading, distributed collectives, and
    # KV ownership changes. Values >1 fail fast in LLMEngine today.
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # ---- Speculative decoding ----
    use_speculative: bool = False
    draft_model_name_or_path: Optional[str] = None
    spec_num_draft_tokens: int = 4          # γ: fixed tokens-per-step
    # Adaptive γ: per-seq EWMA of recent acceptance fraction picks γ in
    # [spec_gamma_min, spec_gamma_max]. spec_num_draft_tokens is used as
    # the warm-start γ for newly-seen sequences.
    spec_adaptive_gamma: bool = False
    spec_gamma_min: int = 1
    spec_gamma_max: int = 8
    spec_gamma_ewma_alpha: float = 0.3      # weight of latest observation

    # ---- Quantization ----
    use_quantization: bool = False
    quant_method: str = "gptq"              # only gptq supported
    quant_bits: int = 8                     # 4 or 8
    quant_group_size: int = 128

    # ---- Kernels ----
    use_triton_attention: bool = True       # False -> fallback reference kernel
    prefill_attention_backend: str = "auto" # auto | flash | flash_attn | mem_efficient | math

    # ---- Misc ----
    seed: int = 0
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.use_speculative and self.draft_model_name_or_path is None:
            raise ValueError("draft_model_name_or_path required when use_speculative=True")
        if self.spec_adaptive_gamma:
            if not (1 <= self.spec_gamma_min <= self.spec_gamma_max):
                raise ValueError(
                    f"spec_gamma_min ({self.spec_gamma_min}) must be in "
                    f"[1, spec_gamma_max] ({self.spec_gamma_max})"
                )
            if not (0.0 < self.spec_gamma_ewma_alpha <= 1.0):
                raise ValueError("spec_gamma_ewma_alpha must be in (0, 1]")
        if self.block_size not in (4, 8, 16, 32):
            # The Triton kernel is written assuming power-of-two block sizes
            # that fit nicely in a warp. 4 is allowed for tests/small-batch
            # sanity runs; 8/16/32 are the tuned production sizes.
            raise ValueError(f"block_size must be one of 4, 8, 16, 32 (got {self.block_size})")
        if self.quant_bits not in (4, 8):
            raise ValueError(f"quant_bits must be 4 or 8 (got {self.quant_bits})")
        if self.quant_group_size <= 0:
            raise ValueError("quant_group_size must be positive")
        if self.prefill_attention_backend not in (
            "auto",
            "flash",
            "flash_attn",
            "mem_efficient",
            "math",
        ):
            raise ValueError(
                "prefill_attention_backend must be one of "
                "auto, flash, flash_attn, mem_efficient, math"
            )
        if self.tensor_parallel_size < 1 or self.pipeline_parallel_size < 1:
            raise ValueError("parallel sizes must be >= 1")
        if self.enable_chunked_prefill:
            raise NotImplementedError("chunked prefill is not implemented yet")
        if self.prefix_cache_max_entries < 0:
            raise ValueError("prefix_cache_max_entries must be >= 0")
        if self.prefix_cache_eviction not in ("lru", "lfu"):
            raise ValueError(
                f"prefix_cache_eviction must be 'lru' or 'lfu' (got {self.prefix_cache_eviction})"
            )
        if self.kv_cache_dtype not in ("auto", "int8"):
            raise ValueError(
                f"kv_cache_dtype must be 'auto' or 'int8' (got {self.kv_cache_dtype})"
            )
