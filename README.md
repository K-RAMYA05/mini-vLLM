# mini_vllm

A from-scratch Python + Triton reimplementation of vLLM's LLM inference
stack, with an original speculative-decoding methodology study using a
**self-distilled Llama-3.1-8B draft model**. ~7,700 lines, single-A100
target, 51 unit tests.

```
                ┌──────────────────────────────────────────────────────┐
  request   →   │ Scheduler  →  ModelRunner  →  Attention  →  Sampler │  →  tokens
                │   (paged KV, continuous batching, prefix cache)      │
                └──────────────────────────────────────────────────────┘
                          ↑                       ↑
                          │                       │
                  speculative-decode        FA2 prefill (A100)
                  with self-distilled       Triton paged decode
                  draft + adaptive γ        int8 KV (kernel-fused)
```

## What's novel

The headline contribution is a **measured speculative-decoding study**:
distill Llama-3.1-8B layers down to {4, 6, 8, 10, 12} layers via KL+CE
distillation, run rejection sampling at γ=4 against the target, and compare
empirical tokens-per-step to the Leviathan et al. (2022) theoretical
prediction `(1 − α^(γ+1)) / (1 − α)`.

Headline result *(filled in once the CARC sweep job finishes)*:

| draft depth | α (code) | α (natural) | empirical tok/step | adj. speedup |
|---:|---:|---:|---:|---:|
| 4  | … | … | … | … |
| 6  | … | … | … | … |
| 8  | … | … | … | … |
| 10 | … | … | … | … |
| 12 | … | … | … | … |

Full methodology + residual analysis: [`results/methodology.md`](results/methodology.md).

## What's measured (vs. vLLM and TensorRT-LLM)

Same workload, same A100, same dtype:

| batch | mini_vllm | vLLM | TensorRT-LLM | mini/vLLM |
|---:|---:|---:|---:|---:|
| 1  | … tok/s | … tok/s | … tok/s | …% |
| 8  | … tok/s | … tok/s | … tok/s | …% |
| 32 | … tok/s | … tok/s | … tok/s | …% |

Full numbers + gap analysis: [`results/vs_production.md`](results/vs_production.md).

## Quality preservation under optimizations

HumanEval pass@1 across optimization stacks (proves the speed numbers come
without correctness regressions):

| Config | pass@1 | Δ vs vLLM ref |
|---|---:|---:|
| vLLM (reference) | … | — |
| mini_vllm baseline | … | … |
| + GPTQ 4-bit | … | … |
| + int8 KV cache | … | … |
| + speculative decoding | … | … |

Full matrix: [`results/quality_eval.md`](results/quality_eval.md).

---

## What's inside

- **Paged KV cache + block allocator** ([block_manager.py](mini_vllm/block_manager.py)) —
  fixed-size blocks + per-sequence indirection, the trick that makes batched
  inference memory-efficient.
- **Continuous-batching scheduler** ([scheduler.py](mini_vllm/scheduler.py)) — admits new
  requests every step; supports CPU swap for preemption.
- **Triton paged-attention decode kernel** ([kernels/paged_attention.py](mini_vllm/kernels/paged_attention.py)) —
  one launch per (sequence, head), online softmax, GQA-aware, **int8 KV
  dequantization fused in-register**.
- **FlashAttention-2 prefill** ([attention.py](mini_vllm/attention.py)) — strict device
  routing (raises on unsupported GPUs, no silent SDPA fallback).
- **GPTQ 4/8-bit quantization** ([quant/gptq.py](mini_vllm/quant/gptq.py)) — calibration
  + Cholesky-based Hessian inverse update; INT4 weights bit-packed two per
  byte with optional Triton matmul.
- **Prefix caching** ([prefix_cache.py](mini_vllm/prefix_cache.py)) — cumulative
  digest indexing; both **LRU** and **LFU** eviction policies.
- **Speculative decoding** ([speculative/spec_decode.py](mini_vllm/speculative/spec_decode.py)) —
  draft-verify with rejection sampling; **adaptive γ** picks per-sequence
  draft depth from a rolling EWMA of acceptance rate.
- **Distillation pipeline** ([distill/](mini_vllm/distill/)) — layer prune → KL+CE distill
  → eval acceptance → depth sweep + Leviathan analysis.
- **Int8 KV cache** ([block_manager.py](mini_vllm/block_manager.py)) — symmetric per-token
  quantization with per-(block, head, slot) fp16 scales; ~50% KV memory
  saving.
- **OpenAI-compatible HTTP server** ([entrypoints/openai_server.py](mini_vllm/entrypoints/openai_server.py)) —
  FastAPI + Prometheus metrics.

## Limits (deliberate cuts)

- **Single GPU.** No tensor or pipeline parallelism. Multi-GPU support is a
  config placeholder that fails fast.
- **A100 / FA2 only.** FA3 / Hopper paths were removed after the project
  scope narrowed. Strict device routing raises if the target backend is
  missing — no silent fallback.
- **Single-process synchronous serving.** The OpenAI server is good for
  integration testing, not multi-tenant production.
- **Speculative decoding verifies one sequence per step.** Batched verify
  would give an additional throughput win; the acceptance-rate study is
  independent of this implementation choice.

---

## Quick start

```bash
pip install -e ".[serve,bench,test]"
pip install flash-attn --no-build-isolation   # GPU node only

python -m pytest tests/ -q
```

```python
from mini_vllm import EngineConfig, LLMEngine, SamplingParams

engine = LLMEngine(EngineConfig(
    model_name_or_path="meta-llama/Llama-3.1-8B",
    dtype="bfloat16",
    num_gpu_blocks=4096,
    max_num_seqs=16,
    enable_prefix_cache=True,
    prefix_cache_eviction="lfu",
    kv_cache_dtype="int8",
))
engine.add_request("Explain paged attention.", SamplingParams(max_tokens=64))
for out in engine.run_until_done():
    print(out.output_text)
```

## Reproducing the headline experiments

### Self-distilled draft depth sweep (the novelty)

```bash
DATA_DIR=/scratch1/$USER/mini_vllm/distill_data \
OUT_ROOT=/scratch1/$USER/mini_vllm/sweep_depth \
DEPTHS="4 6 8 10 12" GAMMA=4 EPOCHS=3 \
bash scripts/sweep_draft_depth.sh
```

Produces `sweep_depth/report.md` with per-depth α, empirical tok/step,
Leviathan prediction, residual, and adjusted speedup.

### Production comparison (vLLM + TRT-LLM)

```bash
python benchmarks/bench_vs_production.py \
    --model meta-llama/Llama-3.1-8B \
    --engines mini_vllm vllm trt_llm \
    --batch-sizes 1 8 32 \
    --markdown-out results/vs_production.md
```

### HumanEval quality eval

```bash
python benchmarks/eval_humaneval.py \
    --model meta-llama/Llama-3.1-8B \
    --draft /scratch1/$USER/mini_vllm/outputs/llama-3.1-8b-draft-8layer/final \
    --markdown-out results/quality_eval.md
```

### CARC / SLURM end-to-end

```bash
HF_TOKEN=hf_xxx sbatch scripts/slurm_full_pipeline.sbatch       # env + distill + bench
HF_TOKEN=hf_xxx sbatch --dependency=afterok:<id> \
    scripts/slurm_sweep_depth.sbatch                             # depth sweep
```

## OpenAI-compatible server

```bash
python -m mini_vllm.entrypoints.openai_server \
    --model meta-llama/Llama-3.1-8B \
    --dtype bfloat16 \
    --num-gpu-blocks 4096 \
    --prefill-backend flash_attn \
    --enable-prefix-cache \
    --host 0.0.0.0 --port 8000

curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"meta-llama/Llama-3.1-8B","prompt":"Explain paged attention:","max_tokens":64}'

curl http://localhost:8000/metrics.json    # tok/s, TTFT, ITL, KV usage, swap counters
```

---

## Code layout (reading order)

For a tour of the codebase, follow this order:

1. [`config.py`](mini_vllm/config.py), [`sampling.py`](mini_vllm/sampling.py), [`sequence.py`](mini_vllm/sequence.py) — data types
2. [`block_manager.py`](mini_vllm/block_manager.py) — paging machinery + int8 KV
3. [`scheduler.py`](mini_vllm/scheduler.py) — batch formation
4. [`attention.py`](mini_vllm/attention.py) — paged attention module + FA2 routing
5. [`kernels/paged_attention.py`](mini_vllm/kernels/paged_attention.py) — Triton kernel + int8 fusion
6. [`model_runner.py`](mini_vllm/model_runner.py), [`engine.py`](mini_vllm/engine.py) — control flow
7. [`speculative/spec_decode.py`](mini_vllm/speculative/spec_decode.py) — adaptive γ + rejection sampling
8. [`distill/`](mini_vllm/distill/) — pruning, distillation, sweep + Leviathan analysis
9. [`prefix_cache.py`](mini_vllm/prefix_cache.py) — LRU / LFU eviction
10. [`quant/gptq.py`](mini_vllm/quant/gptq.py) — weight quantization

## Tests

```bash
pytest tests/                    # CPU-only (51 fast tests)
pytest tests/ -k "triton or cuda" # GPU tests (require CUDA)
```

The HF parity test ([`tests/test_parity_with_hf.py`](tests/test_parity_with_hf.py))
runs only on CUDA — it loads a small Llama and asserts greedy decoding through
mini_vllm matches `transformers.generate` token-for-token. Without this gate,
no speed claim is trustworthy.

---

## Roadmap

- [ ] Vectorize the per-token Python loops in `write_prefill` / `read_tokens`
- [ ] CUDA Graphs for the decode forward
- [ ] JAX / Pallas paged-attention port for TPU v3-8 (TRC application submitted)
- [ ] AWQ alongside GPTQ for activation-aware quantization
- [ ] Batched speculative-decode verify (currently one sequence at a time)

## Acknowledgements

This is a re-implementation; the design is owed to vLLM (Kwon et al. 2023),
PagedAttention, FlashAttention-2 (Dao 2023), and the speculative-decoding
work of Leviathan et al. (2022) and Chen et al. (2023).
