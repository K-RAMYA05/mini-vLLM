"""Throughput benchmark: mini-vLLM vs HuggingFace generate().

Measures tokens per second at a fixed batch size. Target number from your
project writeup: ~4.2x over HuggingFace generate() at batch size 32.

Whether you hit that exact number depends on:
  - GPU model (A100 vs H100 vs consumer cards)
  - block_size and num_gpu_blocks tuning
  - prompt length distribution
  - output length
  - dtype

Run:
    python benchmarks/bench_throughput.py --batch-size 32 --num-prompts 128 \\
        --prompt-len 512 --max-tokens 128 --model meta-llama/Llama-3.1-8B
"""
import argparse
import time
from typing import List

import torch

from mini_vllm import EngineConfig, LLMEngine, SamplingParams


def build_prompts(num_prompts: int, prompt_len: int, tokenizer) -> List[str]:
    """Make a deterministic set of prompts of roughly `prompt_len` tokens each."""
    seed_text = (
        "The history of computing began with mechanical calculators in the 17th century. "
        "Charles Babbage designed the Difference Engine and the Analytical Engine. "
        "Ada Lovelace wrote the first algorithm intended for a machine. "
        "Alan Turing formalized the concept of computation and introduced the Turing test. "
        "Von Neumann proposed the stored-program architecture that underlies modern computers. "
    )
    base_ids = tokenizer.encode(seed_text)
    # Repeat and truncate to reach target length.
    out = []
    for i in range(num_prompts):
        ids = (base_ids * ((prompt_len // len(base_ids)) + 2))[:prompt_len]
        # Slight per-prompt perturbation so the model doesn't see identical inputs.
        ids[0] = (ids[0] + i) % tokenizer.vocab_size
        out.append(tokenizer.decode(ids))
    return out


def bench_mini_vllm(args, prompts):
    cfg = EngineConfig(
        model_name_or_path=args.model,
        dtype=args.dtype,
        block_size=args.block_size,
        num_gpu_blocks=args.num_gpu_blocks,
        num_cpu_blocks=args.num_cpu_blocks,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.batch_size * args.prompt_len,
        max_model_len=args.prompt_len + args.max_tokens + 32,
        use_triton_attention=not args.no_triton,
        prefill_attention_backend=args.prefill_backend,
        seed=0,
    )
    engine = LLMEngine(cfg)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Warmup.
    engine.add_request(prompts[0], sp)
    engine.run_until_done()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for p in prompts:
        engine.add_request(p, sp)
    outputs = engine.run_until_done()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_out_tokens = sum(len(o.output_token_ids) for o in outputs)
    elapsed = t1 - t0
    tps = total_out_tokens / elapsed
    print(f"[mini-vLLM] {total_out_tokens} output tokens in {elapsed:.2f}s -> {tps:.1f} tok/s")
    return tps


def bench_hf(args, prompts):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    _DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=_DTYPE_MAP[args.dtype]
    ).to("cuda")
    model.eval()

    # HF .generate() batches statically — all prompts padded to max length
    # and run in one call. This is the standard apples-to-apples comparison.
    # Warmup.
    batch = tok(prompts[:args.batch_size], return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        _ = model.generate(**batch, max_new_tokens=8, do_sample=False)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_out = 0
    # Process in batch-sized chunks (HF can't do continuous batching).
    with torch.inference_mode():
        for start in range(0, len(prompts), args.batch_size):
            chunk = prompts[start:start + args.batch_size]
            batch = tok(chunk, return_tensors="pt", padding=True).to("cuda")
            out = model.generate(
                **batch,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            total_out += (out.shape[1] - batch["input_ids"].shape[1]) * out.shape[0]
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    tps = total_out / elapsed
    print(f"[HF generate] {total_out} output tokens in {elapsed:.2f}s -> {tps:.1f} tok/s")
    return tps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-prompts", type=int, default=128)
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-gpu-blocks", type=int, default=8192)
    p.add_argument("--num-cpu-blocks", type=int, default=0)
    p.add_argument(
        "--prefill-backend",
        default="auto",
        choices=["auto", "flash", "flash_attn", "mem_efficient", "math"],
    )
    p.add_argument("--no-triton", action="store_true")
    p.add_argument("--skip-hf", action="store_true")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(args.num_prompts, args.prompt_len, tok)

    print(f"Running with batch_size={args.batch_size}, num_prompts={args.num_prompts}, "
          f"prompt_len={args.prompt_len}, max_tokens={args.max_tokens}")

    mini_tps = bench_mini_vllm(args, prompts)
    if not args.skip_hf:
        hf_tps = bench_hf(args, prompts)
        print(f"\nSpeedup: {mini_tps / hf_tps:.2f}x")


if __name__ == "__main__":
    main()
