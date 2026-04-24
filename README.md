# mini-vLLM

Paged KV-cache inference engine for Llama-3.1-8B, with continuous batching,
a custom Triton paged-attention kernel, FlashAttention-backed prefill through
PyTorch SDPA or `flash-attn`, speculative decoding, data-parallel request sharding, and GPTQ
quantization. Built as a research-grade reference implementation — clean
enough to read end-to-end, complete enough to benchmark.

## What's inside

```
mini_vllm/
├── engine.py                 # Top-level LLMEngine
├── config.py                 # EngineConfig
├── sampling.py               # SamplingParams
├── sequence.py               # Sequence lifecycle
├── block_manager.py          # Paged KV cache, block allocator, block tables
├── scheduler.py              # Continuous batching
├── attention.py              # PagedAttention module (replaces HF LlamaAttention)
├── attention_metadata.py     # Per-step metadata shared across layers
├── model_loader.py           # HF load + monkey-patch attention layers
├── model_runner.py           # Forward pass + sampling
├── parallel.py               # Multi-process data-parallel generation
├── kernels/
│   ├── paged_attention.py    # Triton decode kernel (online softmax, block-sparse gather)
│   └── reference_attention.py# Pure-PyTorch reference for correctness tests
├── quant/
│   └── gptq.py               # GPTQ calibration + weight-only quantized linear layer
├── speculative/
│   └── spec_decode.py        # Draft-verify with rejection sampling
└── distill/
    ├── prune.py              # Layer-prune the target to create a small initial draft
    ├── distill_loss.py       # KL + CE loss on top-k teacher logits
    ├── generate_teacher_data.py   # Pre-compute teacher top-k across a corpus
    ├── dataset.py            # Sharded on-disk dataset loader
    ├── train_distill.py      # Training loop
    └── eval_acceptance.py    # Measure α (acceptance rate) on held-out data
```

## Design notes

### Paged KV cache

Each layer's KV is stored as one big tensor
`[num_blocks, num_kv_heads, block_size, head_dim]`. Each sequence has a
**block table** — a list of physical block IDs it owns. When a sequence
needs more room, the scheduler asks the `BlockAllocator` for a new block
and appends its ID to the table. This eliminates the need to reserve
`max_seq_len` slots up front.

Why it matters for throughput, not just memory: without paging, batching
N sequences forces you to reserve `N × max_seq_len` KV slots, most of
which is wasted for short sequences. Paging removes that waste, so your
effective batch size grows, so tokens/sec grow (decode is memory-bandwidth
bound — bigger batches amortize the HBM traffic).

### Triton paged-attention kernel

`mini_vllm/kernels/paged_attention.py` implements a decode-phase attention
kernel with:

- **Block-sparse gather**: reads K/V directly from the paged cache via a
  `block_tables` indirection, no dense gather pre-step.
- **One launch for the whole batch**: grid is `(num_seqs, num_heads)`, each
  program handles one (seq, head) pair.
- **Online softmax** (Milakov & Gimelshein): streams KV blocks, maintains
  running max/sum in registers, never materializes the score matrix.
- **Fused softmax + PV**: single kernel for both reductions.
- **GQA-aware**: `kv_head = head // group`, handled via pointer math.

The reference implementation in `reference_attention.py` is used to
validate correctness in `tests/test_paged_attention_kernel.py`.

### Continuous batching

The scheduler (`scheduler.py`) admits new sequences from a waiting queue
on every step, subject to token and sequence budgets. Prefill and decode
tokens share a single forward pass — the batch is packed as
`[prefill_tokens | decode_tokens]` and the attention module splits on that
boundary, running causal SDPA on the prefill half and the Triton paged
kernel on the decode half.

### Speculative decoding

Standard draft-verify loop with rejection sampling (Leviathan '22 / Chen '23).
Draft model has its own KV cache, shares the tokenizer with the target.
Each step per sequence: draft proposes γ tokens autoregressively; target
verifies with one γ+1 forward pass; rejection walk accepts a prefix and
optionally samples a bonus.

### FlashAttention-backed prefill

Prefill uses PyTorch `scaled_dot_product_attention`, which dispatches to
FlashAttention on supported CUDA builds. Set
`prefill_attention_backend="flash"` to require PyTorch's FlashAttention SDPA
backend, `prefill_attention_backend="flash_attn"` to call the external
`flash-attn` CUDA extension directly, or leave it at `"auto"` to let PyTorch
choose among flash, memory-efficient, and math backends.

### Data-parallel serving

`mini_vllm.generate_data_parallel` shards independent prompts across one
engine replica per worker. Each process owns its model and paged KV cache
on a local device, avoiding cross-device KV traffic in the decode hot path.

```python
from mini_vllm import EngineConfig, SamplingParams, generate_data_parallel

outs = generate_data_parallel(
    ["Explain paged attention.", "Write a CUDA haiku."],
    EngineConfig(model_name_or_path="meta-llama/Llama-3.1-8B"),
    SamplingParams(max_tokens=64),
    devices=["cuda:0", "cuda:1"],
)
```

### GPTQ

Per-channel, per-group (group_size=128) symmetric 4-bit or 8-bit with the Cholesky-
based Hessian inverse update. Calibration runs a handful of prompts to
collect activation Hessians; the quantization pass walks columns left-to-
right within each group applying the GPTQ error-compensation rule.

INT8 weight-only quantization dequantizes to fp16 before matmul. INT4 weights
are bit-packed two per byte and use an optional Triton matmul path on CUDA;
CPU and non-Triton environments fall back to portable unpack/dequant matmul.

### Draft model distillation (for speculative decoding)

`mini_vllm/distill/` produces a small draft model aligned to the target
via layer pruning + KL distillation:

1. **Prune** the target to keep N evenly-spaced layers (for example 8 out of
   32 for Llama-3.1-8B). The pruned model is initialized from the
   target's own weights — only depth is reduced, hidden size and head
   counts are unchanged. This preserves the target's attention patterns
   and lets the draft inherit the tokenizer by construction.

2. **Generate teacher data**: run the target over a balanced corpus
   (WikiText + CodeSearchNet) and save top-k logits per token position.
   Top-k=50 cuts storage ~2500x vs full-vocab logits with negligible
   accuracy impact.

3. **Distill** the pruned student with combined loss:
   α · KL(student || teacher_topk) + (1-α) · CE(student, next_token).
   Temperature-softened (T=2.0), Hinton-scaled. α=0.9 standard.

4. **Evaluate** α (acceptance rate) on held-out prompts, per domain.

See `scripts/run_distillation.sh` for the full pipeline driver and
`scripts/slurm_distill.sbatch` for a Discovery-ready SLURM script.

## Installation

```bash
pip install -r requirements.txt
# or
pip install -e .
```

Requires CUDA + Triton for the kernel path. On CPU or non-CUDA GPUs the
engine falls back to the reference attention implementation (slow, correct).

## Quick start

```python
from mini_vllm import EngineConfig, LLMEngine, SamplingParams

engine = LLMEngine(EngineConfig(
    model_name_or_path="meta-llama/Llama-3.1-8B",
    dtype="bfloat16",
    num_gpu_blocks=8192,
    max_num_seqs=16,
    prefill_attention_backend="flash",
))
engine.add_request("Tell me a joke about compilers.", SamplingParams(max_tokens=64))
for out in engine.run_until_done():
    print(out.output_text)
```

## OpenAI-Compatible Server

Install serving extras:

```bash
pip install -e ".[serve]"
```

Start a single-engine server:

```bash
python -m mini_vllm.entrypoints.openai_server \
    --model meta-llama/Llama-3.1-8B \
    --dtype bfloat16 \
    --num-gpu-blocks 8192 \
    --prefill-backend flash \
    --host 0.0.0.0 \
    --port 8000
```

Completion request:

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"meta-llama/Llama-3.1-8B","prompt":"Explain paged attention:","max_tokens":64,"stream":true}'
```

Metrics:

```bash
curl http://localhost:8000/metrics
curl http://localhost:8000/metrics.json
```

The server exposes TTFT, ITL, tokens/sec, scheduler steps, KV block usage,
and CPU swap counters.

## Reproducing the benchmark numbers

Numbers in the project writeup are A100-class GPU figures at specific
configurations. Whether your GPU hits them depends on hardware, dtype,
and tuning. Each benchmark prints its measured values — don't trust
anything you didn't re-run.

### Throughput vs HuggingFace `generate()`

```bash
python benchmarks/bench_throughput.py \
    --model meta-llama/Llama-3.1-8B \
    --dtype bfloat16 \
    --batch-size 32 --num-prompts 128 \
    --prompt-len 512 --max-tokens 128 \
    --block-size 16 --num-gpu-blocks 8192 \
    --prefill-backend flash
```

Expected: mini-vLLM wins because HF does static batching (pads every
sequence to max length; big prompts block small ones from starting).

### Kernel vs SDPA

```bash
python benchmarks/bench_kernel.py \
    --batch-size 32 --num-heads 32 --num-kv-heads 8 --head-dim 128 \
    --seq-len 512 --block-size 16
```

The SDPA baseline includes the gather cost from paged-cache to dense,
which is the cost you'd actually pay in a paged engine if you didn't
have a custom kernel.

### Speculative decoding

```bash
python benchmarks/bench_speculative.py \
    --target meta-llama/Llama-3.1-8B \
    --draft  outputs/llama-3.1-8b-draft-8layer \
    --gamma 4
```

Prints per-step tokens and implied acceptance rate α. Code prompts (the
default set) give α ≈ 0.6–0.7 with a reasonable draft; natural language
typically lands lower

### GPTQ perplexity

```bash
python benchmarks/bench_gptq.py --model meta-llama/Llama-3.1-8B --bits 8
```

Reports FP16 vs quantized perplexity on WikiText-2 and the VRAM reduction on
the quantized layers.

### Long Stress / GPU Matrix

Run the main A100/H100 validation matrix on an allocated GPU:

```bash
bash scripts/run_gpu_matrix.sh
```

Or run just the long stress test:

```bash
python benchmarks/stress_long_run.py \
    --model meta-llama/Llama-3.1-8B \
    --dtype bfloat16 \
    --waves 10 \
    --requests-per-wave 32 \
    --num-gpu-blocks 8192 \
    --prefill-backend flash
```

## Tests

```bash
pytest tests/              # CPU-only tests always run
pytest tests/ -v -k triton # kernel tests require CUDA + Triton
```

## Known limitations

These are deliberate cuts for a research-grade build:

- CPU KV swap/preemption is implemented for decode growth pressure. Waiting
  prompt admission still waits when GPU blocks are full instead of forcing a
  running sequence out.
- OpenAI-compatible serving is synchronous and single-engine. It is useful
  for integration testing, but it is not an async multi-worker production
  server.
- Chunked prefill and prefix caching are config placeholders that fail fast
  if enabled.
- Speculative decoding verifies one sequence at a time. Batched verify
  would give an additional throughput win; the acceptance-rate and
  latency-reduction claims are independent of this.
- GPTQ is weight-only. No activation quantization; int4 CUDA speedup depends
  on the optional Triton kernel path.
- Prefill uses PyTorch SDPA/FlashAttention or the external `flash-attn`
  package per sequence; there is no project-owned CUDA/CUTLASS prefill kernel.
- Parallelism is data-parallel request sharding. No tensor parallelism or
  pipeline parallelism.

## File organization for reading

Start here, in order:
1. `config.py`, `sampling.py`, `sequence.py` — data types
2. `block_manager.py` — the paging machinery
3. `scheduler.py` — how batches form
4. `attention_metadata.py` + `attention.py` — how the batch reaches the kernel
5. `kernels/paged_attention.py` — the kernel itself
6. `model_runner.py` + `engine.py` — control flow
7. `speculative/spec_decode.py`, `quant/gptq.py` — the optional features
