"""HumanEval pass@1 quality evaluation across optimization configurations.

Demonstrates that mini_vllm's optimizations (GPTQ, int8 KV, lookahead
decoding) preserve generation quality. Every speed claim should come paired
with a quality claim — otherwise it's not a real comparison.

Procedure:
  1. Load HumanEval (164 Python problems).
  2. For each engine config, generate completions for all problems using
     greedy decoding.
  3. Run the official HumanEval evaluator to get pass@1.
  4. Compare against vLLM as the reference baseline.

Configs swept by default:
  - vllm-reference          (vLLM, baseline reference)
  - mini-vllm-baseline      (mini_vllm, no optimizations beyond paged KV)
  - mini-vllm-int8kv        (+ int8 KV cache)
  - mini-vllm-gptq8         (+ GPTQ 8-bit weights)
  - mini-vllm-gptq4         (+ GPTQ 4-bit weights)
  - mini-vllm-lookahead     (+ lookahead decoding)

Usage:
    python benchmarks/eval_humaneval.py \\
        --model meta-llama/Llama-3.1-8B \\
        --configs vllm-reference mini-vllm-baseline mini-vllm-int8kv mini-vllm-gptq8 \\
        --json-out results/quality_eval.json \\
        --markdown-out results/quality_eval.md

Note: requires the `human-eval` package (pip install human-eval) and CUDA.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ConfigResult:
    config_name: str
    pass_at_1: float
    num_passed: int
    num_total: int
    elapsed_s: float
    notes: str = ""


@dataclass
class EvalOutput:
    model: str
    configs: List[str]
    results: List[ConfigResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HumanEval glue
# ---------------------------------------------------------------------------


def load_humaneval():
    try:
        from human_eval.data import read_problems
    except ImportError as e:
        raise RuntimeError(
            "human-eval not installed. Run: pip install human-eval"
        ) from e
    return read_problems()


def evaluate_completions(problems: dict, completions: List[dict]) -> tuple[int, int]:
    """Run the official HumanEval correctness check. Returns (passed, total)."""
    from human_eval.execution import check_correctness

    passed = 0
    for c in completions:
        problem = problems[c["task_id"]]
        result = check_correctness(problem, c["completion"], timeout=10.0)
        if result["passed"]:
            passed += 1
    return passed, len(completions)


def truncate_completion(text: str) -> str:
    """HumanEval scoring expects only the function body; strip anything after.

    The model often continues past the function with `def` for the next
    function or a `class` block. Cut at the first such marker.
    """
    stop_patterns = [
        re.compile(r"^def\s", re.MULTILINE),
        re.compile(r"^class\s", re.MULTILINE),
        re.compile(r"^if __name__", re.MULTILINE),
        re.compile(r"^\s*```", re.MULTILINE),
    ]
    earliest = len(text)
    for pat in stop_patterns:
        m = pat.search(text)
        if m and m.start() > 0 and m.start() < earliest:
            earliest = m.start()
    return text[:earliest]


# ---------------------------------------------------------------------------
# Config adapters — each returns a list of {"task_id", "completion"}
# ---------------------------------------------------------------------------


def gen_with_vllm(problems: dict, model: str, max_new_tokens: int = 512) -> List[dict]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise NotImplementedError(f"vllm not installed: {e}")
    llm = LLM(model=model, dtype="bfloat16", max_model_len=2048)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    prompts = [p["prompt"] for p in problems.values()]
    outs = llm.generate(prompts, sp)
    return [
        {"task_id": tid, "completion": truncate_completion(o.outputs[0].text)}
        for tid, o in zip(problems.keys(), outs)
    ]


def _make_mini_vllm_engine(model: str, **engine_kwargs):
    from mini_vllm import EngineConfig, LLMEngine
    cfg = EngineConfig(
        model_name_or_path=model,
        dtype="bfloat16",
        num_gpu_blocks=4096,
        max_num_seqs=16,
        **engine_kwargs,
    )
    return LLMEngine(cfg)


def gen_with_mini_vllm(problems: dict, model: str,
                        engine_kwargs: dict, max_new_tokens: int = 512) -> List[dict]:
    from mini_vllm import SamplingParams
    engine = _make_mini_vllm_engine(model, **engine_kwargs)
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    for tid, prob in problems.items():
        engine.add_request(prob["prompt"], sp)
    outs = engine.run_until_done()
    # Output order matches insertion order (engine sorts by seq_id).
    return [
        {"task_id": tid, "completion": truncate_completion(o.output_text)}
        for tid, o in zip(problems.keys(), outs)
    ]


# Registry: config_name -> generator function
def make_generators(model: str) -> Dict[str, Callable[[dict], List[dict]]]:
    gens: Dict[str, Callable[[dict], List[dict]]] = {
        "vllm-reference": lambda probs: gen_with_vllm(probs, model),
        "mini-vllm-baseline": lambda probs: gen_with_mini_vllm(probs, model, {}),
        "mini-vllm-int8kv": lambda probs: gen_with_mini_vllm(
            probs, model, {"kv_cache_dtype": "int8"}
        ),
        "mini-vllm-gptq8": lambda probs: gen_with_mini_vllm(
            probs, model, {"use_quantization": True, "quant_bits": 8}
        ),
        "mini-vllm-gptq4": lambda probs: gen_with_mini_vllm(
            probs, model, {"use_quantization": True, "quant_bits": 4}
        ),
        "mini-vllm-lookahead": lambda probs: gen_with_mini_vllm(
            probs, model, {"enable_lookahead_decoding": True, "lookahead_num_slots": 4}
        ),
    }
    return gens


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def render_markdown(out: EvalOutput) -> str:
    lines: List[str] = []
    lines.append(f"# Quality preservation — HumanEval pass@1\n")
    lines.append(f"Model: `{out.model}`. Greedy decoding, max_new_tokens=512, "
                 "164 HumanEval problems.\n")
    baseline = next((r for r in out.results if r.config_name == "vllm-reference"), None)
    lines.append("| Config | pass@1 | passed/total | Δ vs vLLM ref |")
    lines.append("|---|---:|---:|---:|")
    for r in out.results:
        delta = ""
        if baseline and baseline is not r:
            d = r.pass_at_1 - baseline.pass_at_1
            delta = f"{d*100:+.1f} pp"
        lines.append(f"| `{r.config_name}` | {r.pass_at_1:.3f} | "
                     f"{r.num_passed}/{r.num_total} | {delta} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- Baseline mini_vllm (no optimizations) should match vLLM within ±1 problem. "
                 "If it doesn't, there's a correctness regression in the engine — fix before "
                 "trusting any speed numbers.")
    lines.append("- GPTQ 4-bit typically loses 1–3 problems vs. fp16 (consistent with "
                 "published 4-bit quality trade-off).")
    lines.append("- int8 KV cache should be essentially lossless at this scale.")
    lines.append("- Lookahead decoding with greedy sampling should preserve the baseline "
                 "token stream exactly; if it diverges, the multi-substep loop is wrong.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--configs", nargs="+", default=[
        "vllm-reference",
        "mini-vllm-baseline",
        "mini-vllm-int8kv",
        "mini-vllm-gptq8",
    ])
    p.add_argument("--num-problems", type=int, default=164,
                   help="Subset for fast iteration; default = full 164.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--json-out", default=None)
    p.add_argument("--markdown-out", default=None)
    args = p.parse_args()

    problems = load_humaneval()
    if args.num_problems < len(problems):
        keys = list(problems.keys())[:args.num_problems]
        problems = {k: problems[k] for k in keys}
    print(f"Loaded {len(problems)} HumanEval problems")

    generators = make_generators(args.model)
    out = EvalOutput(model=args.model, configs=args.configs)

    for cfg_name in args.configs:
        if cfg_name not in generators:
            print(f"\n=== SKIP {cfg_name}: not registered")
            continue
        print(f"\n=== Running {cfg_name} ===")
        t0 = time.perf_counter()
        try:
            completions = generators[cfg_name](problems)
            passed, total = evaluate_completions(problems, completions)
            elapsed = time.perf_counter() - t0
            r = ConfigResult(
                config_name=cfg_name,
                pass_at_1=passed / max(total, 1),
                num_passed=passed,
                num_total=total,
                elapsed_s=elapsed,
            )
            print(f"  {cfg_name}: pass@1 = {r.pass_at_1:.3f} ({passed}/{total}), "
                  f"elapsed {elapsed:.1f}s")
            out.results.append(r)
        except NotImplementedError as e:
            print(f"  SKIP {cfg_name}: {e}")
        except Exception as e:
            print(f"  ERROR {cfg_name}: {e!r}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "model": out.model,
                    "configs": out.configs,
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
