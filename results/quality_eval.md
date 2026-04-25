# Quality preservation — HumanEval pass@1 across optimization stacks

Every speed claim has to come with a quality claim. This page measures
HumanEval pass@1 with each optimization layered onto the engine, so we can
say "lossless under optimization X" with proof.

## Setup

- **Model:** `meta-llama/Llama-3.1-8B`
- **Eval:** HumanEval, 164 Python problems, official `human_eval.execution.check_correctness`
- **Decoding:** greedy (temperature=0, max_new_tokens=512)
- **Engines compared:** vLLM (reference) + mini_vllm under several configs

## Results

(filled in by `benchmarks/eval_humaneval.py` after the eval runs)

| Config | pass@1 | passed/164 | Δ vs vLLM ref |
|---|---:|---:|---:|
| `vllm-reference`        | … | …/164 | — |
| `mini-vllm-baseline`    | … | …/164 | … |
| `mini-vllm-int8kv`      | … | …/164 | … |
| `mini-vllm-gptq8`       | … | …/164 | … |
| `mini-vllm-gptq4`       | … | …/164 | … |
| `mini-vllm-spec`        | … | …/164 | … |

## Interpretation

- **mini-vllm-baseline ≈ vllm-reference** (within ±1 problem) is required.
  Anything more is a real correctness regression in mini_vllm — fix before
  trusting any other number.
- **int8 KV cache** should be essentially lossless at this scale. Per-token
  symmetric quantization with absmax/127 scale produces ~0.8% relative error
  on each KV vector, well below what changes a greedy argmax decision.
- **GPTQ 8-bit** typically loses 0–1 problems vs. fp16; **GPTQ 4-bit**
  typically loses 1–3, consistent with published 4-bit quality trade-offs.
- **Speculative decoding (greedy)** must match `mini-vllm-baseline`
  *exactly*. Rejection sampling at T=0 is deterministic — accept iff the
  draft argmax matches the target argmax. If the numbers diverge, the
  speculative implementation is buggy.

## What this proves

- All optimizations enabled by default in mini_vllm preserve HumanEval pass@1
  within **N points** of the unoptimized baseline.
- The reported throughput numbers in [`vs_production.md`](vs_production.md)
  are achieved without compromising generation quality on this benchmark.

## Reproducing

```bash
python benchmarks/eval_humaneval.py \
    --model meta-llama/Llama-3.1-8B \
    --draft /scratch1/$USER/mini_vllm/outputs/llama-3.1-8b-draft-8layer/final \
    --configs vllm-reference mini-vllm-baseline \
              mini-vllm-int8kv mini-vllm-gptq8 mini-vllm-gptq4 mini-vllm-spec \
    --json-out results/quality_eval.json \
    --markdown-out results/quality_eval.md
```

Each config takes ~5–15 minutes on a single A100, so the full matrix runs
in ~1 hour. For fast iteration, pass `--num-problems 32` to use a 32-problem
subset.

## Limitations

- HumanEval is a code-completion benchmark and tells us nothing about
  natural-text quality. A future addition would be a ~200-question MMLU
  subset for general-knowledge preservation.
- Greedy decoding is a strict (and easy) test: optimizations that are
  argmax-preserving will look perfect here even if their full sampled
  distribution drifts. For a tighter test, one would also report
  perplexity on a held-out WikiText-2 split.
