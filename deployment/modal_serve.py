"""Modal deployment entrypoint for the OpenAI-compatible server."""
from __future__ import annotations

import os

try:
    import modal
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("Modal deployment requires `pip install modal`") from exc

from mini_vllm import EngineConfig, LLMEngine
from mini_vllm.entrypoints.openai_server import create_app


APP_NAME = os.environ.get("MINI_VLLM_MODAL_APP", "mini-vllm-server")
MODEL_NAME = os.environ.get("MINI_VLLM_MODEL", "meta-llama/Llama-3.1-8B")
GPU_NAME = os.environ.get("MINI_VLLM_GPU", "H100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.1.0", "triton>=2.1.0", "transformers>=4.40.0", "safetensors", "fastapi", "uvicorn[standard]", "pydantic")
    .add_local_dir(".", remote_path="/root/mini_vllm")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu=GPU_NAME,
    scaledown_window=300,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.asgi_app()
def fastapi_app():
    os.chdir("/root/mini_vllm")
    engine = LLMEngine(
        EngineConfig(
            model_name_or_path=MODEL_NAME,
            dtype=os.environ.get("MINI_VLLM_DTYPE", "bfloat16"),
            device="cuda",
            num_gpu_blocks=int(os.environ.get("MINI_VLLM_NUM_GPU_BLOCKS", "4096")),
            max_num_seqs=int(os.environ.get("MINI_VLLM_MAX_NUM_SEQS", "128")),
            max_num_batched_tokens=int(os.environ.get("MINI_VLLM_MAX_BATCHED_TOKENS", "8192")),
            enable_chunked_prefill=os.environ.get("MINI_VLLM_ENABLE_CHUNKED_PREFILL", "1") != "0",
            max_prefill_chunk_tokens=int(os.environ.get("MINI_VLLM_MAX_PREFILL_CHUNK_TOKENS", "1024")),
            enable_lookahead_decoding=os.environ.get("MINI_VLLM_ENABLE_LOOKAHEAD", "1") != "0",
            lookahead_num_slots=int(os.environ.get("MINI_VLLM_LOOKAHEAD_SLOTS", "4")),
            enable_prefix_cache=os.environ.get("MINI_VLLM_ENABLE_PREFIX_CACHE", "1") != "0",
            admission_window_size=int(os.environ.get("MINI_VLLM_ADMISSION_WINDOW_SIZE", "32")),
            scheduler_age_bias=float(os.environ.get("MINI_VLLM_SCHEDULER_AGE_BIAS", "0.25")),
            max_waiting_age_before_decode_priority_s=float(
                os.environ.get("MINI_VLLM_MAX_WAITING_AGE_BEFORE_DECODE_PRIORITY_S", "0.05")
            ),
            use_quantization=os.environ.get("MINI_VLLM_USE_QUANTIZATION", "1") != "0",
            quant_method=os.environ.get("MINI_VLLM_QUANT_METHOD", "fp8"),
            lora_adapters=tuple(
                item for item in os.environ.get("MINI_VLLM_LORA_ADAPTERS", "").split(",") if item
            ),
            kv_cache_dtype=os.environ.get("MINI_VLLM_KV_CACHE_DTYPE", "fp8"),
            prefill_attention_backend=os.environ.get("MINI_VLLM_PREFILL_BACKEND", "flash3"),
            sliding_window=int(os.environ["MINI_VLLM_SLIDING_WINDOW"]) if "MINI_VLLM_SLIDING_WINDOW" in os.environ else None,
        )
    )
    return create_app(
        engine,
        max_queue_size=int(os.environ.get("MINI_VLLM_MAX_QUEUE_SIZE", "2048")),
        max_active_requests=int(os.environ.get("MINI_VLLM_MAX_ACTIVE_REQUESTS", "512")),
        request_timeout_s=float(os.environ["MINI_VLLM_REQUEST_TIMEOUT_S"]) if "MINI_VLLM_REQUEST_TIMEOUT_S" in os.environ else None,
        queue_wait_timeout_s=float(os.environ.get("MINI_VLLM_QUEUE_WAIT_TIMEOUT_S", "5.0")),
    )
