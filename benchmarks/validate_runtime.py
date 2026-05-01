"""Runtime validation checks against a running server."""
from __future__ import annotations

import argparse
import asyncio
import json


async def _validate(args) -> dict:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("runtime validation requires httpx") from exc

    async with httpx.AsyncClient(timeout=30.0) as client:
        health = (await client.get(args.base_url.rstrip("/") + "/health")).json()
        metrics_before = (await client.get(args.base_url.rstrip("/") + "/metrics.json")).json()
        payload = {
            "model": args.model,
            "prompt": "Explain CUDA graphs in one sentence.",
            "max_tokens": 16,
            "temperature": 0.0,
        }
        completion = (await client.post(args.base_url.rstrip("/") + "/v1/completions", json=payload)).json()
        metrics_after = (await client.get(args.base_url.rstrip("/") + "/metrics.json")).json()
    return {
        "health": health,
        "completion_finish_reason": completion["choices"][0]["finish_reason"],
        "requests_started_delta": metrics_after["requests_started"] - metrics_before["requests_started"],
        "requests_finished_delta": metrics_after["requests_finished"] - metrics_before["requests_finished"],
        "decode_graph_hits_after": metrics_after.get("decode_graph_hits", 0),
        "prefill_graph_hits_after": metrics_after.get("prefill_graph_hits", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_validate(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
