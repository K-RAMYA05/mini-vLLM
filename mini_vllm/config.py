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
    # quantization with one fp16 scale per (block, kv_head, slot). 'fp8' is
    # Hopper-only E4M3 storage plus the same per-token scales.
    kv_cache_dtype: str = "auto"  # auto | int8 | fp8

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
    admission_window_size: int = 32
    scheduler_age_bias: float = 0.25
    max_waiting_age_before_decode_priority_s: float = 0.050

    # ---- Multi-GPU placeholders ----
    # Real TP/PP requires sharded weight loading, distributed collectives, and
    # KV ownership changes. Values >1 fail fast in LLMEngine today.
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # ---- Lookahead decoding ----
    # Decode-only steps can advance the resident batch for several substeps
    # before handing control back to the scheduler. This removes repeated
    # Python/scheduler overhead without introducing a draft model.
    enable_lookahead_decoding: bool = False
    lookahead_num_slots: int = 4

    # ---- Quantization ----
    use_quantization: bool = False
    quant_method: str = "gptq"              # gptq | awq | fp8
    quant_bits: int = 8                     # 4 or 8
    quant_group_size: int = 128
    lora_adapters: tuple[str, ...] = field(default_factory=tuple)

    # ---- Kernels ----
    use_triton_attention: bool = True       # False -> fallback reference kernel
    prefill_attention_backend: str = "auto" # auto | flash | flash2 | flash3 | flash_attn | mem_efficient | math

    # ---- Sliding window attention ----
    # When set, attention attends only to the last `sliding_window` tokens.
    # None disables it and the model uses full causal context. The flag is
    # read at runtime; weights and architecture are unchanged.
    sliding_window: Optional[int] = None

    # ---- CUDA graphs (decode capture) ----
    # When True, decode-only forward passes are captured into CUDA graphs at
    # a small set of padded batch sizes. Replays eliminate Python launch
    # overhead and typically lift decode tok/s by 1.3-2x on small models.
    # Decode-only forwards can be captured and replayed at a small set of
    # padded batch sizes. Prefill remains eager.
    enable_cuda_graphs: bool = False
    cuda_graph_batch_sizes: tuple = (1, 2, 4, 8, 16, 32)

    # ---- Misc ----
    seed: int = 0
    log_level: str = "INFO"

    def __post_init__(self) -> None:
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
            "flash2",
            "flash3",
            "flash_attn",
            "mem_efficient",
            "math",
        ):
            raise ValueError(
                "prefill_attention_backend must be one of "
                "auto, flash, flash2, flash3, flash_attn, mem_efficient, math"
            )
        if self.tensor_parallel_size < 1 or self.pipeline_parallel_size < 1:
            raise ValueError("parallel sizes must be >= 1")
        if self.enable_chunked_prefill and self.max_prefill_chunk_tokens < 1:
            raise ValueError("max_prefill_chunk_tokens must be >= 1 when chunked prefill is on")
        if self.enable_lookahead_decoding and self.lookahead_num_slots < 2:
            raise ValueError("lookahead_num_slots must be >= 2 when lookahead decoding is on")
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ValueError("sliding_window must be >= 1 when set")
        if self.quant_method not in ("gptq", "awq", "fp8"):
            raise ValueError(f"quant_method must be 'gptq', 'awq', or 'fp8' (got {self.quant_method})")
        if any("=" not in item for item in self.lora_adapters):
            raise ValueError("lora_adapters entries must use name=path format")
        if self.enable_cuda_graphs:
            if not self.cuda_graph_batch_sizes:
                raise ValueError("cuda_graph_batch_sizes must be non-empty when enable_cuda_graphs=True")
            if any(int(x) < 1 for x in self.cuda_graph_batch_sizes):
                raise ValueError("cuda_graph_batch_sizes entries must be >= 1")
        if self.prefix_cache_max_entries < 0:
            raise ValueError("prefix_cache_max_entries must be >= 0")
        if self.prefix_cache_eviction not in ("lru", "lfu"):
            raise ValueError(
                f"prefix_cache_eviction must be 'lru' or 'lfu' (got {self.prefix_cache_eviction})"
            )
        if self.admission_window_size < 1:
            raise ValueError("admission_window_size must be >= 1")
        if self.scheduler_age_bias < 0.0:
            raise ValueError("scheduler_age_bias must be >= 0")
        if self.max_waiting_age_before_decode_priority_s < 0.0:
            raise ValueError("max_waiting_age_before_decode_priority_s must be >= 0")
        if self.kv_cache_dtype not in ("auto", "int8", "fp8"):
            raise ValueError(
                f"kv_cache_dtype must be 'auto', 'int8', or 'fp8' (got {self.kv_cache_dtype})"
            )
