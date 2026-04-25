"""Prefix-cache benchmark: repeated shared-prefix prompts with cache on/off."""
from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoTokenizer

from mini_vllm import EngineConfig, LLMEngine, SamplingParams


def build_shared_prefix_prompts(tokenizer, num_prompts: int, prefix_len: int, suffix_len: int) -> list[str]:
    prefix_seed = (
        "Paged KV cache keeps memory compact while continuous batching improves throughput. "
        "This benchmark measures prefix cache reuse across prompts with the same long prefix. "
    )
    suffix_seed = "Variant suffix for request number "
    prefix_ids = tokenizer.encode(prefix_seed)
    prefix_ids = (prefix_ids * ((prefix_len // max(len(prefix_ids), 1)) + 2))[:prefix_len]
    prompts = []
    for i in range(num_prompts):
        suffix_text = f"{suffix_seed}{i}. " * 8
        suffix_ids = tokenizer.encode(suffix_text)[:suffix_len]
        prompts.append(tokenizer.decode(prefix_ids + suffix_ids))
    return prompts


def run_case(args, enable_prefix_cache: bool) -> None:
    cfg = EngineConfig(
        model_name_or_path=args.model,
        dtype=args.dtype,
        block_size=args.block_size,
        num_gpu_blocks=args.num_gpu_blocks,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.batch_size * (args.prefix_len + args.suffix_len),
        max_model_len=args.prefix_len + args.suffix_len + args.max_tokens + 32,
        prefill_attention_backend=args.prefill_backend,
        use_triton_attention=not args.no_triton,
        enable_prefix_cache=enable_prefix_cache,
        seed=0,
    )
    engine = LLMEngine(cfg)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = build_shared_prefix_prompts(tokenizer, args.num_prompts, args.prefix_len, args.suffix_len)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    engine.add_request(prompts[0], sp)
    engine.run_until_done()
    before = engine.get_metrics()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for prompt in prompts[1:]:
        engine.add_request(prompt, sp)
    outputs = engine.run_until_done()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    after = engine.get_metrics()

    output_tokens = sum(len(out.output_token_ids) for out in outputs)
    delta_tokens = after["output_tokens"] - before["output_tokens"]
    delta_prefix_hits = after["prefix_cache_hits"] - before["prefix_cache_hits"]
    delta_prefix_hit_tokens = after["prefix_cache_hit_tokens"] - before["prefix_cache_hit_tokens"]
    delta_started = after["requests_started"] - before["requests_started"]
    print(f"=== prefix_cache={'on' if enable_prefix_cache else 'off'} ===")
    print(f"requests={delta_started} output_tokens={output_tokens} measured_output_tokens={delta_tokens}")
    print(f"elapsed={elapsed:.2f}s tok/s={output_tokens / elapsed:.1f}")
    print(f"prefix_cache_hits={delta_prefix_hits} prefix_cache_hit_tokens={delta_prefix_hit_tokens}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-prompts", type=int, default=33)
    p.add_argument("--prefix-len", type=int, default=256)
    p.add_argument("--suffix-len", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-gpu-blocks", type=int, default=8192)
    p.add_argument("--prefill-backend", default="flash", choices=["auto", "flash", "flash_attn", "mem_efficient", "math"])
    p.add_argument("--no-triton", action="store_true")
    args = p.parse_args()

    run_case(args, enable_prefix_cache=False)
    torch.cuda.empty_cache()
    run_case(args, enable_prefix_cache=True)


if __name__ == "__main__":
    main()
