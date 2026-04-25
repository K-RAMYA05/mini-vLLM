"""Production-comparison benchmark: mini_vllm vs vLLM vs TensorRT-LLM.

Runs the same workload (same prompts, same dtype, same sampling) through up
to three engines and reports tok/s, TTFT, and ITL side-by-side. Output is
both JSON (for downstream tooling) and a markdown table (for the writeup).

Standardized workload — keep this constant across runs:
  - Model:         Llama-3.1-8B (or any HF id passed via --model)
  - Dtype:         bfloat16
  - Prompt count:  100 (configurable)
  - Prompt length: ~256–512 input tokens (truncated/right-padded)
  - Output:        128 max new tokens
  - Sampling:      greedy (temperature=0, removes RNG variance)
  - Batches swept: 1, 8, 32

Greedy is non-negotiable — without it numbers across engines aren't comparable
because each engine seeds RNG differently.

Usage:
    python benchmarks/bench_vs_production.py \\
        --model meta-llama/Llama-3.1-8B \\
        --engines mini_vllm vllm trt_llm \\
        --batch-sizes 1 8 32 \\
        --num-prompts 100 \\
        --json-out results/vs_production.json \\
        --markdown-out results/vs_production.md

Engines requested but not installed are skipped with a note in the output —
this lets you run mini_vllm-only on a machine without vllm/trt_llm and add
them later.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------


def make_prompts(num_prompts: int, target_len_tokens: int = 384, seed: int = 0) -> List[str]:
    """Deterministic prompt set: a mix of WikiText-style prose and HumanEval
    code stubs, all truncated/padded to ~target_len_tokens for fairness.

    We don't tokenize here (engines may use different tokenizers); we generate
    text and trust each engine to truncate to its own ~target_len. The padding
    text is repeatable so every engine sees the same input.
    """
    import random
    rng = random.Random(seed)

    base_prose = [
        "The history of inference engines for large language models begins with the realization that ",
        "Quantization, in the context of deep learning, refers to the process of mapping high-precision ",
        "Speculative decoding accelerates autoregressive generation by ",
        "Paged attention addresses a memory-fragmentation problem that arises when ",
        "FlashAttention reduces the memory footprint of self-attention by ",
        "GPTQ is a one-shot post-training quantization method that ",
        "Continuous batching, also called rolling batching, differs from static batching in that ",
        "The Triton language allows researchers to write GPU kernels in Python by ",
        "An LLM serving system must balance latency and throughput because ",
        "Prefix caching reuses computation when multiple requests share a common prompt prefix; ",
    ]
    base_code = [
        "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\n",
        "import torch\n\nclass Attention(torch.nn.Module):\n    def __init__(self, dim, num_heads):\n        super().__init__()\n",
        "def quicksort(arr):\n    \"\"\"In-place quicksort.\n",
        "from typing import List\n\ndef binary_search(xs: List[int], target: int) -> int:\n",
        "import numpy as np\n\ndef softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:\n",
    ]

    prompts: List[str] = []
    for i in range(num_prompts):
        seed_text = (base_prose if i % 2 == 0 else base_code)[i % len(base_prose if i % 2 == 0 else base_code)]
        # Pad with deterministic filler to hit target length-ish
        filler_lines = []
        for _ in range(target_len_tokens // 8):
            filler_lines.append(rng.choice([
                "This is a continuation of the prior thought.",
                "We now consider the implications for system design.",
                "x = 1; y = 2; z = x + y",
                "for i in range(10):",
                "    accumulator += weights[i] * values[i]",
            ]))
        prompts.append(seed_text + " ".join(filler_lines))
    return prompts


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    engine: str
    batch_size: int
    num_prompts: int
    max_new_tokens: int
    decode_tokens_per_sec: float        # decode-only tokens/sec (post-prefill)
    end_to_end_tokens_per_sec: float    # includes prefill in the wall time
    ttft_ms_p50: float
    ttft_ms_p95: float
    itl_ms_p50: Optional[float]         # inter-token latency
    elapsed_s: float
    notes: str = ""


@dataclass
class BenchOutput:
    model: str
    dtype: str
    num_prompts: int
    max_new_tokens: int
    batch_sizes: List[int]
    results: List[RunResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine adapters — each returns a RunResult, or raises NotImplementedError
# if the engine isn't installed (we catch and skip).
# ---------------------------------------------------------------------------


def run_mini_vllm(prompts: List[str], batch_size: int, max_new_tokens: int,
                  model: str, dtype: str) -> RunResult:
    from mini_vllm import EngineConfig, LLMEngine, SamplingParams

    cfg = EngineConfig(
        model_name_or_path=model,
        dtype=dtype,
        num_gpu_blocks=4096,
        max_num_seqs=batch_size,
        max_num_batched_tokens=8192,
    )
    engine = LLMEngine(cfg)
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)

    # Submit `batch_size` prompts at a time and time the engine.
    t0 = time.perf_counter()
    submitted = 0
    while submitted < len(prompts):
        for p in prompts[submitted:submitted + batch_size]:
            engine.add_request(p, sp)
        submitted += batch_size
        outs = engine.run_until_done()  # drain this wave
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o.output_token_ids) for o in outs)
    metrics = engine.get_metrics()
    return RunResult(
        engine="mini_vllm",
        batch_size=batch_size,
        num_prompts=len(prompts),
        max_new_tokens=max_new_tokens,
        decode_tokens_per_sec=total_tokens / max(elapsed, 1e-9),
        end_to_end_tokens_per_sec=total_tokens / max(elapsed, 1e-9),
        ttft_ms_p50=metrics.get("ttft_p50_ms", 0.0),
        ttft_ms_p95=metrics.get("ttft_p95_ms", 0.0),
        itl_ms_p50=metrics.get("itl_p50_ms"),
        elapsed_s=elapsed,
    )


def run_vllm(prompts: List[str], batch_size: int, max_new_tokens: int,
             model: str, dtype: str) -> RunResult:
    try:
        from vllm import LLM, SamplingParams as VLLMSamplingParams
    except ImportError as e:
        raise NotImplementedError(f"vllm not installed: {e}")

    llm = LLM(model=model, dtype=dtype, max_model_len=2048)
    sp = VLLMSamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    t0 = time.perf_counter()
    # vLLM batches internally; pass the whole list and let it schedule.
    outs = llm.generate(prompts, sp)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    return RunResult(
        engine="vllm",
        batch_size=batch_size,
        num_prompts=len(prompts),
        max_new_tokens=max_new_tokens,
        decode_tokens_per_sec=total_tokens / max(elapsed, 1e-9),
        end_to_end_tokens_per_sec=total_tokens / max(elapsed, 1e-9),
        ttft_ms_p50=0.0,   # vLLM's offline API doesn't expose per-request TTFT cleanly
        ttft_ms_p95=0.0,
        itl_ms_p50=None,
        elapsed_s=elapsed,
        notes="vLLM offline LLM API; TTFT not reported (use AsyncLLMEngine for tail latency).",
    )


def run_trt_llm(prompts: List[str], batch_size: int, max_new_tokens: int,
                model: str, dtype: str) -> RunResult:
    """TensorRT-LLM via the LLM API (>=0.13).

    NOTE: TRT-LLM has a fragile dep tree and may need its own venv.
    The engine build is cached after the first run.
    """
    try:
        from tensorrt_llm import LLM as TRTLLM
        from tensorrt_llm import SamplingParams as TRTSamplingParams
    except ImportError as e:
        raise NotImplementedError(f"tensorrt_llm not installed: {e}")

    llm = TRTLLM(model=model)
    sp = TRTSamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    return RunResult(
        engine="trt_llm",
        batch_size=batch_size,
        num_prompts=len(prompts),
        max_new_tokens=max_new_tokens,
        decode_tokens_per_sec=total_tokens / max(elapsed, 1e-9),
        end_to_end_tokens_per_sec=total_tokens / max(elapsed, 1e-9),
        ttft_ms_p50=0.0,
        ttft_ms_p95=0.0,
        itl_ms_p50=None,
        elapsed_s=elapsed,
        notes="TRT-LLM LLM API; engine cached after first build (~10 min one-time cost).",
    )


ENGINE_ADAPTERS: Dict[str, Callable[..., RunResult]] = {
    "mini_vllm": run_mini_vllm,
    "vllm": run_vllm,
    "trt_llm": run_trt_llm,
}


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def render_markdown(out: BenchOutput) -> str:
    lines: List[str] = []
    lines.append(f"# Production comparison — {out.model}, {out.dtype}\n")
    lines.append(f"Workload: {out.num_prompts} prompts, max_new_tokens={out.max_new_tokens}, "
                 f"greedy sampling. Batch sizes: {out.batch_sizes}.\n")

    # Group by batch size, then by engine
    by_batch: Dict[int, Dict[str, RunResult]] = {}
    for r in out.results:
        by_batch.setdefault(r.batch_size, {})[r.engine] = r

    lines.append("## Throughput (tokens/sec, end-to-end including prefill)\n")
    engines = sorted({r.engine for r in out.results})
    header = "| batch | " + " | ".join(engines) + " |" + " | ".join(
        [f" mini/{e} " for e in engines if e != "mini_vllm"]
    ) + " |"
    lines.append(header)
    lines.append("|---" * (1 + len(engines) + max(0, len(engines) - 1)) + "|")
    for batch in sorted(by_batch.keys()):
        row = [str(batch)]
        for e in engines:
            r = by_batch[batch].get(e)
            row.append(f"{r.end_to_end_tokens_per_sec:.1f}" if r else "—")
        for e in engines:
            if e == "mini_vllm":
                continue
            mini = by_batch[batch].get("mini_vllm")
            other = by_batch[batch].get(e)
            if mini and other and other.end_to_end_tokens_per_sec > 0:
                ratio = mini.end_to_end_tokens_per_sec / other.end_to_end_tokens_per_sec
                row.append(f"{ratio*100:.0f}%")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Notes")
    seen_notes = set()
    for r in out.results:
        if r.notes and r.notes not in seen_notes:
            lines.append(f"- **{r.engine}**: {r.notes}")
            seen_notes.add(r.notes)

    lines.append("")
    lines.append("## Gap analysis (fill in after run)")
    lines.append("- At batch=1 we are at X% of vLLM. Gap dominated by [...].")
    lines.append("- At batch=32 we are at Y% of vLLM. Gap shrinks because [...].")
    lines.append("- Versus TRT-LLM at batch=32: Z%. Their advantage is [...].")
    lines.append("")
    lines.append("## What we'd do next to close the gap")
    lines.append("- Vectorize `write_prefill` (currently a Python scatter loop).")
    lines.append("- Batched top-k/top-p sampler (currently per-seq Python loop).")
    lines.append("- CUDA Graphs for decode (single-launch instead of N kernels per token).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--engines", nargs="+", default=["mini_vllm", "vllm", "trt_llm"],
                   choices=list(ENGINE_ADAPTERS.keys()))
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--target-prompt-tokens", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", default=None)
    p.add_argument("--markdown-out", default=None)
    args = p.parse_args()

    prompts = make_prompts(args.num_prompts, args.target_prompt_tokens, args.seed)
    print(f"Generated {len(prompts)} prompts (~{args.target_prompt_tokens} tokens each)")

    out = BenchOutput(
        model=args.model, dtype=args.dtype,
        num_prompts=args.num_prompts, max_new_tokens=args.max_new_tokens,
        batch_sizes=args.batch_sizes,
    )

    for batch in args.batch_sizes:
        for engine in args.engines:
            print(f"\n=== Running {engine} at batch={batch} ===")
            try:
                r = ENGINE_ADAPTERS[engine](
                    prompts, batch_size=batch,
                    max_new_tokens=args.max_new_tokens,
                    model=args.model, dtype=args.dtype,
                )
                print(f"  {engine}: {r.end_to_end_tokens_per_sec:.1f} tok/s "
                      f"(elapsed {r.elapsed_s:.1f}s)")
                out.results.append(r)
            except NotImplementedError as e:
                print(f"  SKIP {engine}: {e}")
            except Exception as e:
                print(f"  ERROR {engine}: {e!r}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "model": out.model,
                    "dtype": out.dtype,
                    "num_prompts": out.num_prompts,
                    "max_new_tokens": out.max_new_tokens,
                    "batch_sizes": out.batch_sizes,
                    "results": [asdict(r) for r in out.results],
                },
                f, indent=2,
            )
        print(f"\nWrote {args.json_out}")

    md = render_markdown(out)
    if args.markdown_out:
        os.makedirs(os.path.dirname(args.markdown_out) or ".", exist_ok=True)
        with open(args.markdown_out, "w") as f:
            f.write(md + "\n")
        print(f"Wrote {args.markdown_out}")
    else:
        print("\n" + md)


if __name__ == "__main__":
    main()
