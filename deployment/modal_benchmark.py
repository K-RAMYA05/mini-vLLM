"""Run the HTTP benchmark harness against a Modal deployment."""
from __future__ import annotations

import os

try:
    import modal
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("Modal benchmark entrypoint requires `pip install modal`") from exc

app = modal.App(os.environ.get("MINI_VLLM_MODAL_BENCH_APP", "mini-vllm-bench"))


@app.local_entrypoint()
def main(
    base_url: str,
    model: str = "meta-llama/Llama-3.1-8B",
    preset: str = "",
    num_requests: int = 64,
    concurrency: int = 8,
    arrival_rate: float = 0.0,
    max_tokens: int = 128,
    priority: int = 0,
    request_class: str = "default",
    lora_adapter: str = "",
    prompts: str = "",
    mix: str = "short,medium,long",
    baseline: str = "",
    output: str = "",
):
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "benchmarks.run_benchmark",
        "--base-url",
        base_url,
        "--model",
        model,
    ]
    if preset:
        cmd.extend(["--preset", preset])
    else:
        cmd.extend([
            "--num-requests", str(num_requests),
            "--concurrency", str(concurrency),
            "--arrival-rate", str(arrival_rate),
            "--max-tokens", str(max_tokens),
            "--mix", mix,
            "--priority", str(priority),
            "--request-class", request_class,
        ])
    if lora_adapter:
        cmd.extend(["--lora-adapter", lora_adapter])
    if prompts:
        cmd.extend(["--prompts", prompts])
    if baseline:
        cmd.extend(["--baseline", baseline])
    if output:
        cmd.extend(["--output", output])
    subprocess.run(cmd, check=True)
