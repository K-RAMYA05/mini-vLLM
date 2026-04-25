# Production comparison — mini_vllm vs vLLM vs TensorRT-LLM

Honest head-to-head on the same A100 with identical workload. Numbers below
are placeholders until the benchmark runs; the methodology is fixed.

## Setup

- **Hardware:** 1× NVIDIA A100 80GB
- **Software:** torch 2.5.1+cu124, vllm 0.6.3, tensorrt_llm 0.13.0, mini_vllm @ `<git sha>`
- **Model:** `meta-llama/Llama-3.1-8B`
- **Dtype:** bfloat16
- **Workload:** 100 prompts (50 prose / 50 code stubs), ~384 input tokens each
- **Output:** 128 max new tokens, greedy decoding (temperature=0)
- **Batches swept:** 1, 8, 32

Greedy is non-negotiable for cross-engine comparison — without it, RNG
state differs between engines and tokens-per-second numbers aren't
comparable on identical work.

## Throughput (end-to-end tokens/sec)

| batch | mini_vllm | vLLM | TRT-LLM | mini/vLLM | mini/TRT-LLM |
|---:|---:|---:|---:|---:|---:|
| 1  | … | … | … | …% | …% |
| 8  | … | … | … | …% | …% |
| 32 | … | … | … | …% | …% |

## TTFT (ms, p50)

| batch | mini_vllm | vLLM | TRT-LLM |
|---:|---:|---:|---:|
| 1  | … | … | … |
| 8  | … | … | … |
| 32 | … | … | … |

## Gap analysis

(fill in after running)

- **At batch=1**, mini_vllm is at **__%** of vLLM and **__%** of TRT-LLM. The
  gap at batch=1 is dominated by per-token overhead — kernel launches, the
  Python sampling loop, and the lack of CUDA Graph capture. TRT-LLM's edge
  here is largely from CUDA Graphs replaying the decode forward as one launch.
- **At batch=32**, mini_vllm is at **__%** of vLLM and **__%** of TRT-LLM. The
  gap shrinks because fixed per-step overhead amortizes across more sequences,
  and our paged-attention Triton kernel is genuinely competitive for the
  decode hot path. Most remaining gap is in prefill (we use SDPA/FA2; vLLM
  uses fused QKV + FA2; TRT-LLM uses a custom prefill kernel).

## What we'd do next to close the gap

In rough order of expected impact:

1. **CUDA Graphs for decode.** Capture the decode forward per batch-size
   bucket, replay per token. Estimated +10–15% at batch=1.
2. **Vectorize `write_prefill` + `read_tokens`.** Currently a Python loop
   scattering per-token KV ([block_manager.py:187](../mini_vllm/block_manager.py)).
   Estimated +20% prefill at batch=32.
3. **Batched top-k/top-p sampler.** Currently a per-seq Python loop
   ([model_runner.py:182](../mini_vllm/model_runner.py)). Estimated +5–10%
   decode at batch ≥ 16.
4. **Fused QKV projection.** The HF Llama path computes Q, K, V as three
   separate matmuls. A single fused Q+K+V GEMM saves a kernel launch.

## Reproducing

```bash
python benchmarks/bench_vs_production.py \
    --model meta-llama/Llama-3.1-8B \
    --engines mini_vllm vllm trt_llm \
    --batch-sizes 1 8 32 \
    --num-prompts 100 \
    --json-out results/vs_production.json \
    --markdown-out results/vs_production.md
```

The script auto-skips any engine whose Python package isn't installed; you
can run mini_vllm-only first and add vLLM/TRT-LLM later.
