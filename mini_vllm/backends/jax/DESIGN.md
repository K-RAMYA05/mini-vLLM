# JAX / Pallas backend — design doc

This document is a port plan, not yet an implementation. The goal is a
second backend at `mini_vllm.backends.jax` that runs the paged-attention
decode kernel on JAX (CPU/GPU/TPU) using **Pallas**, JAX's kernel-author
language.

## Why this exists

- **Cross-platform credibility.** "Same algorithm runs on A100 (Triton)
  and TPU v3-8 (Pallas), validated against each other" is a stronger
  claim than "we ship on NVIDIA GPUs only."
- **Hardware portability is a research-engineer signal.** Google /
  Apple / Amazon (AWS Neuron is XLA-backed) all care about it.
- **TPU access is granted.** Google TRC application submitted; ~30 days
  of free TPU v3-8 starting on approval.

## Scope (v1)

What's in:

1. A Pallas paged-attention decode kernel, math-equivalent to the Triton
   one in [`mini_vllm/kernels/paged_attention.py`](../../kernels/paged_attention.py).
   Same online-softmax algorithm, same per-(seq, head) parallelism.
2. A numerical-equivalence test against the Triton kernel (parity within
   bf16 tolerance).
3. A minimal JAX/Flax inference loop using HuggingFace `FlaxLlamaForCausalLM`
   for the model and the Pallas kernel for decode-time attention.
4. A cross-platform benchmark page reporting the same algorithm on:
   - Triton on A100 (existing)
   - Pallas on JAX-CUDA on A100 (new)
   - Pallas on JAX-TPU on v3-8 (new)

What's out (deferred):

- Paged KV on TPU. TPUs reward larger contiguous tiles than GPUs do —
  the per-block gather pattern that wins on Triton may not win on Pallas/TPU.
  v1 uses a contiguous KV view; paging on TPU is future work.
- Continuous batching, scheduler integration, prefix cache. v1 runs one
  sequence at a time.
- Speculative decoding on JAX. The acceptance theory carries unchanged;
  the executor doesn't.
- INT8 KV on TPU. v1 is bf16 only.
- Multi-host SPMD. v3-8 is single-host. Multi-host is future work.

## Architecture

```
mini_vllm/backends/jax/
├── DESIGN.md               (this file)
├── __init__.py             (public exports)
├── paged_attention.py      Pallas decode kernel + Python wrapper
├── inference.py            Minimal JAX/Flax model + greedy decode loop
└── tests/
    └── test_parity_with_triton.py   (numerical-equivalence test)
```

## Pallas kernel — what changes from Triton

The math is identical: online softmax over per-block KV tiles, accumulating
running (max, sum, output_vec) in registers, finalizing at the end.

The mechanical translation:

| Triton | Pallas equivalent |
|---|---|
| `tl.program_id(0)` / `(1)` | `pl.program_id(0)` / `(1)` |
| `tl.load(ptr + offset, mask=…)` | `pl.load(ref, idx, mask=…)` |
| `tl.store(ptr + offset, val)` | `pl.store(ref, idx, val)` |
| `triton.jit` | `pl.pallas_call(kernel, out_shape, grid)` |
| Strides as kernel args | `pl.BlockSpec` declares tile shape + indexing |

Specific gotchas to plan for:

1. **No raw pointer arithmetic.** Pallas works on `Ref` objects with
   block-spec indexing rather than ptr+offset. The kernel body is
   structurally cleaner but requires re-thinking the access pattern.
2. **Block-table gather.** The Triton kernel gathers KV via the block
   table inside the kernel. On Pallas/CUDA this is straightforward; on
   Pallas/TPU we may need to lift the gather to the host (compute a
   contiguous-KV view per sequence outside the kernel) because TPU memory
   indirection has different cost characteristics.
3. **Fori-loop vs Python for-loop.** Triton lets you write a Python
   `for blk in range(num_blocks):` loop. Pallas requires `lax.fori_loop`
   or `pl.when` for variable-iteration loops on TPU.
4. **No `int8` in v1.** TPU v3 doesn't have native int8 matmul; we'll
   keep KV as bf16 for the JAX backend. INT8 KV is a feature only of the
   Triton path.

## Sequenced plan

### Phase 1 — local development on JAX-CUDA (week 1)

Develop on the existing A100 — JAX has a CUDA backend. No TPU access
needed yet. Goals:

- Port `_paged_attention_kernel` to Pallas. Start with the no-int8 path.
- Wire `paged_attention_jax(query, key_cache, value_cache, block_tables, context_lens, scale)` Python entry.
- Write `tests/test_parity_with_triton.py`: same Q/K/V/block_tables, run
  through both kernels, assert max abs diff < 1e-2 (bf16 tolerance).

Exit criterion: parity test green on A100.

### Phase 2 — TPU validation (week 2)

Once TRC access lands:

- `gcloud compute tpus tpu-vm create` a v3-8.
- Install JAX-TPU on the VM.
- Run the same Pallas kernel — JAX dispatches to TPU automatically.
- Investigate any TPU-specific failures (block-table gather, tile sizes).
- Restructure if needed (e.g. lift gather to host).

Exit criterion: kernel runs on TPU v3-8, produces correct output on a
small smoke test (Llama-3.2-1B, 1 sequence, 32 tokens).

### Phase 3 — minimal inference loop + benchmark (week 2 end)

- `inference.py`: load `FlaxLlamaForCausalLM`, prefill with `jnp.numpy`
  SDPA, decode with the Pallas kernel.
- Greedy-only generation, single sequence at a time.
- Benchmark on Llama-3.2-1B (faster iteration; 8B may fit on v3-8 but
  iteration becomes painful):
  - Triton on A100 (baseline)
  - Pallas on JAX-CUDA on A100
  - Pallas on JAX-TPU on v3-8
- Write `results/cross_platform.md` with the three-row table + a
  "what had to change for TPU" narrative.

Exit criterion: README has a "Cross-platform" section with real numbers.

## Parity test

Pseudocode for `tests/test_parity_with_triton.py`:

```python
@pytest.mark.skipif(not (jax_available and torch_cuda_available), reason="...")
def test_pallas_matches_triton():
    torch.manual_seed(0)
    Q = torch.randn(num_seqs, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    Kc = torch.randn(num_blocks, num_kv_heads, block_size, head_dim,
                     dtype=torch.bfloat16, device="cuda")
    Vc = torch.randn_like(Kc)
    bt = build_block_tables(...)
    cl = context_lens(...)

    triton_out = paged_attention(Q, Kc, Vc, bt, cl, scale)

    # Convert torch tensors to JAX (zero-copy via dlpack)
    Q_j = jnp.asarray(Q); Kc_j = jnp.asarray(Kc); ...
    pallas_out = paged_attention_jax(Q_j, Kc_j, Vc_j, bt_j, cl_j, scale)

    assert torch.allclose(
        triton_out.float(),
        torch.from_dlpack(pallas_out).cuda().float(),
        atol=1e-2, rtol=1e-2,
    )
```

Tolerance 1e-2 because both backends use fp32 accumulators but bf16 inputs;
`a@b^T*c` in bf16 has well-known ~1% error vs fp32-throughout.

## What lands at companies

- **Google:** direct hit. TPU + Pallas experience is the gap-closer for
  Vertex AI / Cloud TPU / DeepMind-applied roles.
- **Apple:** adjacent — JAX-influenced thinking transfers to MLX. "I can
  switch numerical-array stacks."
- **Amazon:** AWS Neuron uses XLA, same ecosystem. Partial credit toward
  Inferentia/Trainium.
- **NVIDIA / Meta:** breadth signal, not direct fit.

## References

- Pallas docs: https://docs.jax.dev/en/latest/pallas/index.html
- JAX TPU programming guide: https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
- TRC (TPU Research Cloud): https://sites.research.google/trc/
- Existing Triton kernel to mirror: [`mini_vllm/kernels/paged_attention.py`](../../kernels/paged_attention.py)
