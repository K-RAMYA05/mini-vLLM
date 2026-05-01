"""Async server stress/timeout validation against a running deployment."""
from __future__ import annotations

import argparse
import asyncio
import json


async def _stress(args) -> dict:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("async server stress validation requires httpx") from exc

    async with httpx.AsyncClient(timeout=None) as client:
        async def fire(i: int):
            payload = {
                "model": args.model,
                "prompt": f"Request {i}: explain paged attention briefly.",
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "priority": 1 if i % 5 == 0 else 0,
                "request_class": "latency" if i % 5 == 0 else "default",
            }
            resp = await client.post(args.base_url.rstrip("/") + "/v1/completions", json=payload)
            return resp.status_code

        statuses = await asyncio.gather(*[fire(i) for i in range(args.requests)])
        metrics = (await client.get(args.base_url.rstrip("/") + "/metrics.json")).json()
    return {
        "requests": args.requests,
        "status_counts": {str(code): statuses.count(code) for code in sorted(set(statuses))},
        "avg_queue_wait_s": metrics.get("avg_queue_wait_s", 0.0),
        "active_request_age_max_s": metrics.get("active_request_age_max_s", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_stress(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
