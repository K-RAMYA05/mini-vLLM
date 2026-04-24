"""Speculative decoding benchmark.

Measures:
  - Token acceptance rate α (averaged across decode steps).
  - End-to-end latency vs baseline (no spec-decode) on the same prompts.

Target numbers from your project writeup:
  ~68% acceptance on code, ~2.3x latency reduction.

Usage:
    python benchmarks/bench_speculative.py \\
        --target meta-llama/Llama-3.1-8B \\
        --draft  outputs/llama-3.1-8b-draft-8layer \\  # replace with your draft
        --gamma 4
"""
import argparse
import time
from typing import List, Tuple

import torch

from mini_vllm import EngineConfig, LLMEngine, SamplingParams


DEFAULT_CODE_PROMPTS = [
    "def fibonacci(n):\n    ",
    "def is_prime(n):\n    ",
    "class BinaryTree:\n    def __init__(self, val):\n        ",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    ",
    "import torch\n\nclass MLP(torch.nn.Module):\n    def __init__(self, hidden):\n        ",
    "def merge_sort(arr):\n    ",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    ",
    "def dfs(graph, start, visited=None):\n    ",
]


def run_and_time(engine: LLMEngine, prompts: List[str], max_tokens: int) -> Tuple[float, int, float]:
    """Run engine on prompts, return (elapsed_seconds, total_output_tokens, avg_tokens_per_step)."""
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    for p in prompts:
        engine.add_request(p, sp)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Instrument tokens-per-step by stepping manually.
    total_tokens = 0
    total_steps = 0
    # We capture tokens generated per step by diffing output length before/after.
    # For non-spec engines, this is always 1 per active sequence per step.
    # For spec engines, it's 1..γ+1 per active sequence per step.
    while engine.has_pending_work():
        lengths_before = {s.seq_id: len(s.output_token_ids) for s in engine.scheduler.running}
        lengths_before.update({s.seq_id: len(s.output_token_ids) for s in engine.scheduler.waiting})
        engine.step()
        for s in engine.scheduler.running:
            total_tokens += len(s.output_token_ids) - lengths_before.get(s.seq_id, 0)
        for fin in engine.finished_outputs:
            # Already counted above? We track generation per step so finished
            # sequences' tokens are captured during the step that finished them.
            pass
        total_steps += 1

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # For completed sequences, recompute total output tokens authoritatively.
    total_output = sum(len(o.output_token_ids) for o in engine.finished_outputs)
    avg_tps_per_step = total_output / max(total_steps, 1)

    return elapsed, total_output, avg_tps_per_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--draft", required=True,
                   help="Path to draft model, e.g. outputs/llama-3.1-8b-draft-8layer/final")
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    # ---- Baseline (no spec) ----
    print("=== Baseline (no speculative decoding) ===")
    cfg_base = EngineConfig(
        model_name_or_path=args.target,
        dtype=args.dtype,
        max_num_seqs=8,
        max_model_len=2048,
        seed=0,
    )
    engine_base = LLMEngine(cfg_base)
    base_elapsed, base_tokens, base_tps_step = run_and_time(
        engine_base, DEFAULT_CODE_PROMPTS, args.max_tokens
    )
    print(f"  {base_tokens} tokens in {base_elapsed:.2f}s "
          f"({base_tokens/base_elapsed:.1f} tok/s)")
    print(f"  Avg tokens/step across generation: {base_tps_step:.2f}")

    del engine_base
    torch.cuda.empty_cache()

    # ---- Speculative ----
    print(f"\n=== Speculative decoding (γ={args.gamma}, draft={args.draft}) ===")
    cfg_spec = EngineConfig(
        model_name_or_path=args.target,
        dtype=args.dtype,
        use_speculative=True,
        draft_model_name_or_path=args.draft,
        spec_num_draft_tokens=args.gamma,
        max_num_seqs=8,
        max_model_len=2048,
        seed=0,
    )
    engine_spec = LLMEngine(cfg_spec)
    spec_elapsed, spec_tokens, spec_tps_step = run_and_time(
        engine_spec, DEFAULT_CODE_PROMPTS, args.max_tokens
    )
    print(f"  {spec_tokens} tokens in {spec_elapsed:.2f}s "
          f"({spec_tokens/spec_elapsed:.1f} tok/s)")
    print(f"  Avg tokens/step across generation: {spec_tps_step:.2f}")

    # Expected acceptance rate α from tokens-per-step:
    #   tokens/step = (1 - α^{γ+1}) / (1 - α)   (geometric sum of accept probs)
    # Solve numerically.
    from scipy.optimize import brentq
    target_tps = spec_tps_step

    def f(alpha):
        if alpha >= 1 - 1e-9:
            return (args.gamma + 1) - target_tps
        return (1 - alpha ** (args.gamma + 1)) / (1 - alpha) - target_tps

    try:
        alpha = brentq(f, 0.01, 0.999)
        print(f"\n  Implied acceptance rate α ≈ {alpha:.3f}")
    except Exception as e:
        print(f"  Couldn't solve for α: {e}")

    print(f"\nLatency speedup: {base_elapsed / spec_elapsed:.2f}x")


if __name__ == "__main__":
    main()
