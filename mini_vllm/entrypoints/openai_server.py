"""OpenAI-compatible completion server.

Requests are admitted through a bounded queue and executed by a single
background engine worker so engine state remains single-threaded while the
HTTP layer can provide async waiting, backpressure, cancellation, and expiry.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from mini_vllm import EngineConfig, LLMEngine, SamplingParams

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = object  # type: ignore[assignment]

    def Field(*args, **kwargs):  # type: ignore[no-redef]
        return None

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
except ImportError:
    FastAPI = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]
    StreamingResponse = None  # type: ignore[assignment]
    PlainTextResponse = None  # type: ignore[assignment]

_SERVICE = None


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | List[str]
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = -1
    priority: int = 0
    request_class: str = "default"
    lora_adapter: str | None = None
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
    priority: int = 0
    request_class: str = "default"
    lora_adapter: str | None = None
    stream: bool = False


@dataclass
class _ServeRequest:
    prompt: str
    sampling_params: SamplingParams
    priority: int = 0
    request_class: str = "default"
    lora_adapter: str | None = None
    created_at_s: float = field(default_factory=time.perf_counter)
    request_deadline_s: float | None = None
    queue_deadline_s: float | None = None
    seq_id: int | None = None
    emitted_tokens: int = 0
    output: Any | None = None
    error: Exception | None = None
    finish_reason: str | None = None
    done: bool = False
    cancelled: bool = False
    condition: threading.Condition = field(default_factory=threading.Condition)
    stream_queue: "queue.Queue[tuple[str, str | None, dict[str, int] | None] | None]" = field(
        default_factory=queue.Queue
    )


class AsyncEngineService:
    def __init__(
        self,
        engine: LLMEngine,
        max_queue_size: int = 1024,
        max_active_requests: int | None = None,
        request_timeout_s: float | None = None,
        queue_wait_timeout_s: float | None = None,
    ):
        self.engine = engine
        self.max_active_requests = max_active_requests or max(1, engine.config.max_num_seqs * 4)
        self.request_timeout_s = request_timeout_s
        self.queue_wait_timeout_s = queue_wait_timeout_s
        self._incoming: "queue.Queue[_ServeRequest]" = queue.Queue(maxsize=max_queue_size)
        self._cancelled: "queue.Queue[_ServeRequest]" = queue.Queue()
        self._pending: list[_ServeRequest] = []
        self._active: dict[int, _ServeRequest] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="mini-vllm-server")
        self._thread.start()

    def submit(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        priority: int = 0,
        request_class: str = "default",
        lora_adapter: str | None = None,
    ) -> _ServeRequest:
        now = time.perf_counter()
        req = _ServeRequest(
            prompt=prompt,
            sampling_params=sampling_params,
            request_deadline_s=(now + self.request_timeout_s) if self.request_timeout_s else None,
            queue_deadline_s=(now + self.queue_wait_timeout_s) if self.queue_wait_timeout_s else None,
        )
        req.priority = priority
        req.request_class = request_class
        req.lora_adapter = lora_adapter
        self._incoming.put_nowait(req)
        return req

    def wait(self, req: _ServeRequest):
        with req.condition:
            while not req.done and req.error is None:
                req.condition.wait(timeout=0.05)
        if req.error is not None:
            raise req.error
        return req.output

    def cancel(self, req: _ServeRequest, reason: str = "cancelled") -> None:
        if req.done or req.cancelled:
            return
        req.cancelled = True
        req.finish_reason = reason
        self._cancelled.put(req)

    def iter_stream(self, req: _ServeRequest) -> Iterable[tuple[str, str | None, dict[str, int] | None]]:
        while True:
            item = req.stream_queue.get()
            if item is None:
                return
            yield item

    @property
    def queue_depth(self) -> int:
        return self._incoming.qsize() + len(self._pending)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._drain_incoming()
            self._drain_cancelled()
            self._expire_requests()
            self._admit_pending()

            if self.engine.has_pending_work():
                try:
                    finished = self.engine.step()
                    self._expire_requests()
                    self._flush_partial_outputs()
                    for out in finished:
                        req = self._active.pop(out.seq_id, None)
                        if req is not None:
                            self._complete(req, out)
                except Exception as exc:  # pragma: no cover - defensive server path
                    self._fail_all(exc)
            elif not self._pending:
                try:
                    req = self._incoming.get(timeout=0.02)
                    self._pending.append(req)
                except queue.Empty:
                    continue
            else:
                time.sleep(0.001)

    def _drain_incoming(self) -> None:
        while True:
            try:
                self._pending.append(self._incoming.get_nowait())
            except queue.Empty:
                return

    def _drain_cancelled(self) -> None:
        while True:
            try:
                req = self._cancelled.get_nowait()
            except queue.Empty:
                return
            if req.seq_id is not None:
                self.engine.scheduler.abort_seq(req.seq_id)
                self._active.pop(req.seq_id, None)
            else:
                try:
                    self._pending.remove(req)
                except ValueError:
                    pass
            self._timeout(req, req.finish_reason or "cancelled")

    def _admit_pending(self) -> None:
        while self._pending and len(self._active) < self.max_active_requests:
            req = self._pending.pop(0)
            if self._request_expired(req, queue_only=True):
                self._timeout(req, "queue_timeout")
                continue
            req.seq_id = self.engine.add_request(
                req.prompt,
                req.sampling_params,
                priority=req.priority,
                request_class=req.request_class,
                lora_adapter_name=req.lora_adapter,
            )
            self._active[req.seq_id] = req

    def _flush_partial_outputs(self) -> None:
        for seq_id, req in list(self._active.items()):
            seq = _find_seq(self.engine, seq_id)
            if seq is None:
                continue
            new_ids = seq.output_token_ids[req.emitted_tokens:]
            if not new_ids:
                continue
            text = self.engine.tokenizer.decode(new_ids, skip_special_tokens=True)
            req.emitted_tokens += len(new_ids)
            req.stream_queue.put((text, None, None))

    def _complete(self, req: _ServeRequest, out) -> None:
        if req.cancelled:
            return
        tail_ids = out.output_token_ids[req.emitted_tokens:]
        if tail_ids:
            text = self.engine.tokenizer.decode(tail_ids, skip_special_tokens=True)
            req.stream_queue.put((text, None, None))
            req.emitted_tokens += len(tail_ids)
        usage = _usage_from_output(out)
        req.stream_queue.put(("", out.finish_reason, usage))
        req.stream_queue.put(None)
        with req.condition:
            req.output = out
            req.done = True
            req.condition.notify_all()

    def _expire_requests(self) -> None:
        for req in list(self._pending):
            if self._request_expired(req, queue_only=True):
                self._pending.remove(req)
                self._timeout(req, "queue_timeout")
        for seq_id, req in list(self._active.items()):
            if self._request_expired(req, queue_only=False):
                self.engine.scheduler.abort_seq(seq_id)
                self._active.pop(seq_id, None)
                self._timeout(req, "request_timeout")

    def _request_expired(self, req: _ServeRequest, queue_only: bool) -> bool:
        now = time.perf_counter()
        if queue_only:
            return req.queue_deadline_s is not None and now >= req.queue_deadline_s
        return req.request_deadline_s is not None and now >= req.request_deadline_s

    def _timeout(self, req: _ServeRequest, reason: str) -> None:
        req.cancelled = True
        req.finish_reason = reason
        with req.condition:
            req.error = TimeoutError(reason)
            req.done = True
            req.condition.notify_all()
        req.stream_queue.put(None)

    def _fail_all(self, exc: Exception) -> None:
        for req in self._pending:
            with req.condition:
                req.error = exc
                req.done = True
                req.condition.notify_all()
            req.stream_queue.put(None)
        self._pending.clear()
        for req in self._active.values():
            with req.condition:
                req.error = exc
                req.done = True
                req.condition.notify_all()
            req.stream_queue.put(None)
        self._active.clear()


def create_app(
    engine: LLMEngine,
    max_queue_size: int = 1024,
    max_active_requests: int | None = None,
    request_timeout_s: float | None = None,
    queue_wait_timeout_s: float | None = None,
):
    if FastAPI is None:
        raise RuntimeError("OpenAI server requires `pip install -e '.[serve]'`")

    global _SERVICE
    _SERVICE = AsyncEngineService(
        engine,
        max_queue_size=max_queue_size,
        max_active_requests=max_active_requests,
        request_timeout_s=request_timeout_s,
        queue_wait_timeout_s=queue_wait_timeout_s,
    )
    app = FastAPI(title="mini-vLLM OpenAI-compatible server")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": engine.config.model_name_or_path,
            "queue_depth": _SERVICE.queue_depth,
            "active_requests": len(_SERVICE._active),
        }

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{
                "id": engine.config.model_name_or_path,
                "object": "model",
                "owned_by": "mini-vllm",
            }],
            "adapters": sorted(engine.lora_manager.available_adapters) if engine.lora_manager else [],
        }

    @app.get("/metrics")
    def metrics():
        return PlainTextResponse(engine.get_prometheus_metrics(), media_type="text/plain")

    @app.get("/metrics.json")
    def metrics_json():
        return engine.get_metrics()

    @app.get("/metrics/structured")
    def metrics_structured():
        return engine.get_metrics_structured()

    @app.post("/v1/completions")
    async def completions(request: Request):
        req = CompletionRequest.model_validate(await request.json())
        prompts = [req.prompt] if isinstance(req.prompt, str) else req.prompt
        if req.stream:
            if len(prompts) != 1:
                return JSONResponse({"error": "streaming supports one prompt per request"}, status_code=400)
            try:
                handle = _SERVICE.submit(
                    prompts[0],
                    _sampling(req),
                    priority=req.priority,
                    request_class=req.request_class,
                    lora_adapter=req.lora_adapter,
                )
            except queue.Full:
                return JSONResponse({"error": "server queue full"}, status_code=429)
            return StreamingResponse(_stream_completion(engine, request, handle), media_type="text/event-stream")
        handles = []
        try:
            for prompt in prompts:
                handles.append(
                    _SERVICE.submit(
                        prompt,
                        _sampling(req),
                        priority=req.priority,
                        request_class=req.request_class,
                        lora_adapter=req.lora_adapter,
                    )
                )
        except queue.Full:
            return JSONResponse({"error": "server queue full"}, status_code=429)
        outputs = await asyncio.gather(*[asyncio.to_thread(_SERVICE.wait, handle) for handle in handles])
        choices = []
        for i, out in enumerate(outputs):
            choices.append({
                "index": i,
                "text": out.output_text,
                "finish_reason": out.finish_reason,
                "_usage": _usage_from_output(out),
            })
        return _completion_response(engine, choices)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        req = ChatCompletionRequest.model_validate(await request.json())
        prompt = _messages_to_prompt([_model_to_dict(m) for m in req.messages])
        if req.stream:
            try:
                handle = _SERVICE.submit(
                    prompt,
                    _sampling(req),
                    priority=req.priority,
                    request_class=req.request_class,
                    lora_adapter=req.lora_adapter,
                )
            except queue.Full:
                return JSONResponse({"error": "server queue full"}, status_code=429)
            return StreamingResponse(_stream_chat(engine, request, handle), media_type="text/event-stream")
        try:
            handle = _SERVICE.submit(
                prompt,
                _sampling(req),
                priority=req.priority,
                request_class=req.request_class,
                lora_adapter=req.lora_adapter,
            )
        except queue.Full:
            return JSONResponse({"error": "server queue full"}, status_code=429)
        out = await asyncio.to_thread(_SERVICE.wait, handle)
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
            "usage": _usage_from_output(out),
        }

    return app


def _sampling(req) -> SamplingParams:
    return SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
    )


async def _stream_completion(engine: LLMEngine, request: Request, req: _ServeRequest):
    async for chunk in _stream(engine, request, req, chat=False):
        yield chunk


async def _stream_chat(engine: LLMEngine, request: Request, req: _ServeRequest):
    async for chunk in _stream(engine, request, req, chat=True):
        yield chunk


async def _stream(engine: LLMEngine, request: Request, req: _ServeRequest, chat: bool):
    while True:
        if await request.is_disconnected():
            _SERVICE.cancel(req, reason="client_disconnect")
            return
        try:
            item = await asyncio.to_thread(req.stream_queue.get, True, 0.1)
        except queue.Empty:
            continue
        if item is None:
            return
        text, finish_reason, usage = item
        yield _sse(_chunk(engine, text, chat=chat, finish_reason=finish_reason, usage=usage))
        if finish_reason is not None:
            yield "data: [DONE]\n\n"
            return


def _find_seq(engine: LLMEngine, seq_id: int):
    for group in (engine.scheduler.running, engine.scheduler.waiting, engine.scheduler.swapped):
        for seq in group:
            if seq.seq_id == seq_id:
                return seq
    return None


def _completion_response(engine: LLMEngine, choices: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for choice in choices:
        usage = choice.pop("_usage", None)
        if usage is not None:
            total_prompt_tokens += usage["prompt_tokens"]
            total_completion_tokens += usage["completion_tokens"]
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": engine.config.model_name_or_path,
        "choices": choices,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
    }


def _chunk(
    engine: LLMEngine,
    text: str,
    chat: bool,
    finish_reason: str | None,
    usage: dict[str, int] | None = None,
) -> Dict[str, Any]:
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
    if usage is not None:
        base["usage"] = usage
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


def _usage_from_output(out) -> Dict[str, int]:
    prompt_tokens = len(out.prompt_token_ids)
    completion_tokens = len(out.output_token_ids)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def build_engine_from_args(args) -> LLMEngine:
    cfg = EngineConfig(
        model_name_or_path=args.model,
        dtype=args.dtype,
        device=args.device,
        block_size=args.block_size,
        num_gpu_blocks=args.num_gpu_blocks,
        num_cpu_blocks=args.num_cpu_blocks,
        kv_cache_dtype=args.kv_cache_dtype,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_prefill_chunk_tokens=args.max_prefill_chunk_tokens,
        enable_lookahead_decoding=args.enable_lookahead_decoding,
        lookahead_num_slots=args.lookahead_num_slots,
        enable_prefix_cache=args.enable_prefix_cache,
        admission_window_size=args.admission_window_size,
        scheduler_age_bias=args.scheduler_age_bias,
        max_waiting_age_before_decode_priority_s=args.max_waiting_age_before_decode_priority_s,
        use_quantization=args.use_quantization,
        quant_method=args.quant_method,
        lora_adapters=tuple(args.lora_adapter),
        sliding_window=args.sliding_window,
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
    parser.add_argument("--max-queue-size", type=int, default=1024)
    parser.add_argument("--max-active-requests", type=int, default=None)
    parser.add_argument("--request-timeout-s", type=float, default=None)
    parser.add_argument("--queue-wait-timeout-s", type=float, default=5.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-gpu-blocks", type=int, default=8192)
    parser.add_argument("--num-cpu-blocks", type=int, default=0)
    parser.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "int8", "fp8"])
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--max-prefill-chunk-tokens", type=int, default=2048)
    parser.add_argument("--enable-lookahead-decoding", action="store_true")
    parser.add_argument("--lookahead-num-slots", type=int, default=4)
    parser.add_argument("--enable-prefix-cache", action="store_true")
    parser.add_argument("--admission-window-size", type=int, default=32)
    parser.add_argument("--scheduler-age-bias", type=float, default=0.25)
    parser.add_argument("--max-waiting-age-before-decode-priority-s", type=float, default=0.05)
    parser.add_argument("--use-quantization", action="store_true")
    parser.add_argument("--quant-method", default="gptq", choices=["gptq", "awq", "fp8"])
    parser.add_argument(
        "--lora-adapter",
        action="append",
        default=[],
        help="Repeatable LoRA adapter spec in name=path format.",
    )
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument(
        "--prefill-backend",
        default="flash",
        choices=["auto", "flash", "flash2", "flash3", "flash_attn", "mem_efficient", "math"],
    )
    parser.add_argument("--no-triton", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("OpenAI server requires `pip install -e '.[serve]'`") from exc

    app = create_app(
        build_engine_from_args(args),
        max_queue_size=args.max_queue_size,
        max_active_requests=args.max_active_requests,
        request_timeout_s=args.request_timeout_s,
        queue_wait_timeout_s=args.queue_wait_timeout_s,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
