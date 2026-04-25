"""OpenAI-compatible completion server.

This is a synchronous single-engine server. It is useful for local testing,
benchmarks, and integration experiments; it serializes generation through a
lock because LLMEngine is not thread-safe.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from typing import Any, Dict, Iterable, List

from mini_vllm import EngineConfig, LLMEngine, SamplingParams


_ENGINE: LLMEngine | None = None
_LOCK = threading.Lock()


def create_app(engine: LLMEngine):
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("OpenAI server requires `pip install -e '.[serve]'`") from exc

    global _ENGINE
    _ENGINE = engine
    app = FastAPI(title="mini-vLLM OpenAI-compatible server")

    class CompletionRequest(BaseModel):
        model: str | None = None
        prompt: str | List[str]
        max_tokens: int = Field(default=128, ge=1)
        temperature: float = Field(default=0.0, ge=0.0)
        top_p: float = Field(default=1.0, gt=0.0, le=1.0)
        top_k: int = -1
        stream: bool = False

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str | None = None
        messages: List[ChatMessage]
        max_tokens: int = Field(default=128, ge=1)
        temperature: float = Field(default=0.0, ge=0.0)
        top_p: float = Field(default=1.0, gt=0.0, le=1.0)
        top_k: int = -1
        stream: bool = False

    @app.get("/health")
    def health():
        return {"status": "ok", "model": engine.config.model_name_or_path}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{
                "id": engine.config.model_name_or_path,
                "object": "model",
                "owned_by": "mini-vllm",
            }],
        }

    @app.get("/metrics")
    def metrics():
        return PlainTextResponse(engine.get_prometheus_metrics(), media_type="text/plain")

    @app.get("/metrics.json")
    def metrics_json():
        return engine.get_metrics()

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        prompts = [req.prompt] if isinstance(req.prompt, str) else req.prompt
        if req.stream:
            if len(prompts) != 1:
                return JSONResponse({"error": "streaming supports one prompt per request"}, status_code=400)
            return StreamingResponse(
                _stream_completion(engine, prompts[0], _sampling(req)),
                media_type="text/event-stream",
            )
        with _LOCK:
            choices = []
            for i, prompt in enumerate(prompts):
                out = _run_one(engine, prompt, _sampling(req))
                choices.append({
                    "index": i,
                    "text": out.output_text,
                    "finish_reason": out.finish_reason,
                })
        return _completion_response(engine, choices)

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        prompt = _messages_to_prompt([_model_to_dict(m) for m in req.messages])
        if req.stream:
            return StreamingResponse(
                _stream_chat(engine, prompt, _sampling(req)),
                media_type="text/event-stream",
            )
        with _LOCK:
            out = _run_one(engine, prompt, _sampling(req))
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": engine.config.model_name_or_path,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": out.output_text},
                "finish_reason": out.finish_reason,
            }],
        }

    return app


def _sampling(req) -> SamplingParams:
    return SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
    )


def _run_one(engine: LLMEngine, prompt: str, sp: SamplingParams):
    seq_id = engine.add_request(prompt, sp)
    while engine.has_pending_work():
        finished = engine.step()
        for out in finished:
            if out.seq_id == seq_id:
                return out
    raise RuntimeError("request finished without output")


def _stream_completion(engine: LLMEngine, prompt: str, sp: SamplingParams) -> Iterable[str]:
    yield from _stream(engine, prompt, sp, chat=False)


def _stream_chat(engine: LLMEngine, prompt: str, sp: SamplingParams) -> Iterable[str]:
    yield from _stream(engine, prompt, sp, chat=True)


def _stream(engine: LLMEngine, prompt: str, sp: SamplingParams, chat: bool) -> Iterable[str]:
    with _LOCK:
        seq_id = engine.add_request(prompt, sp)
        emitted = 0
        while engine.has_pending_work():
            finished = engine.step()
            seq = _find_seq(engine, seq_id)
            if seq is not None:
                new_ids = seq.output_token_ids[emitted:]
                if new_ids:
                    text = engine.tokenizer.decode(new_ids, skip_special_tokens=True)
                    emitted += len(new_ids)
                    yield _sse(_chunk(engine, text, chat=chat, finish_reason=None))
            for out in finished:
                if out.seq_id == seq_id:
                    tail_ids = out.output_token_ids[emitted:]
                    if tail_ids:
                        text = engine.tokenizer.decode(tail_ids, skip_special_tokens=True)
                        yield _sse(_chunk(engine, text, chat=chat, finish_reason=None))
                    yield _sse(_chunk(engine, "", chat=chat, finish_reason=out.finish_reason))
                    yield "data: [DONE]\n\n"
                    return


def _find_seq(engine: LLMEngine, seq_id: int):
    for group in (engine.scheduler.running, engine.scheduler.waiting, engine.scheduler.swapped):
        for seq in group:
            if seq.seq_id == seq_id:
                return seq
    return None


def _completion_response(engine: LLMEngine, choices: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": engine.config.model_name_or_path,
        "choices": choices,
    }


def _chunk(engine: LLMEngine, text: str, chat: bool, finish_reason: str | None) -> Dict[str, Any]:
    base = {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "created": int(time.time()),
        "model": engine.config.model_name_or_path,
    }
    if chat:
        base.update({
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"content": text} if text else {},
                "finish_reason": finish_reason,
            }],
        })
    else:
        base.update({
            "object": "text_completion.chunk",
            "choices": [{
                "index": 0,
                "text": text,
                "finish_reason": finish_reason,
            }],
        })
    return base


def _sse(payload: Dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, separators=(",", ":")) + "\n\n"


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    lines = []
    for msg in messages:
        lines.append(f"{msg['role']}: {msg['content']}")
    lines.append("assistant:")
    return "\n".join(lines)


def _model_to_dict(model) -> Dict[str, str]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def build_engine_from_args(args) -> LLMEngine:
    cfg = EngineConfig(
        model_name_or_path=args.model,
        dtype=args.dtype,
        device=args.device,
        block_size=args.block_size,
        num_gpu_blocks=args.num_gpu_blocks,
        num_cpu_blocks=args.num_cpu_blocks,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        enable_prefix_cache=args.enable_prefix_cache,
        prefill_attention_backend=args.prefill_backend,
        use_triton_attention=not args.no_triton,
        trust_remote_code=args.trust_remote_code,
    )
    return LLMEngine(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-gpu-blocks", type=int, default=8192)
    parser.add_argument("--num-cpu-blocks", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enable-prefix-cache", action="store_true")
    parser.add_argument(
        "--prefill-backend",
        default="flash",
        choices=["auto", "flash", "flash_attn", "mem_efficient", "math"],
    )
    parser.add_argument("--no-triton", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("OpenAI server requires `pip install -e '.[serve]'`") from exc

    app = create_app(build_engine_from_args(args))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
