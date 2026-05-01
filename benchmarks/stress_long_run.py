"""Long-running stress test for mini-vLLM.

This exercises scheduler stability, KV cache usage, optional CPU swap, and
basic latency metrics over repeated request waves.
"""
from __future__ import annotations

import argparse
import time

from mini_vllm import EngineConfig, LLMEngine, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--waves", type=int, default=10)
    parser.add_argument("--requests-per-wave", type=int, default=32)
    parser.add_argument("--prompt-len-words", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-gpu-blocks", type=int, default=8192)
    parser.add_argument("--num-cpu-blocks", type=int, default=0)
    parser.add_argument(
        "--prefill-backend",
        default="flash",
        choices=["auto", "flash", "flash2", "flash3", "flash_attn", "mem_efficient", "math"],
    )
    args = parser.parse_args()

    cfg = EngineConfig(
        model_name_or_path=args.model,
        dtype=args.dtype,
        num_gpu_blocks=args.num_gpu_blocks,
        num_cpu_blocks=args.num_cpu_blocks,
        prefill_attention_backend=args.prefill_backend,
        max_num_seqs=args.requests_per_wave,
        max_num_batched_tokens=args.requests_per_wave * args.prompt_len_words,
    )
    engine = LLMEngine(cfg)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    seed = " ".join(["attention"] * args.prompt_len_words)

    t0 = time.perf_counter()
    for wave in range(args.waves):
        for i in range(args.requests_per_wave):
            engine.add_request(f"{seed} wave={wave} request={i}", sp)
        outputs = engine.run_until_done()
        metrics = engine.get_metrics()
        print(
            f"wave={wave} completed={len(outputs)} "
            f"tok/s={metrics['output_tokens_per_s']:.1f} "
            f"avg_ttft={metrics['avg_ttft_s']:.4f}s "
            f"avg_itl={metrics['avg_itl_s']:.4f}s "
            f"gpu_blocks_used={metrics['gpu_kv_blocks_used']} "
            f"swap_out={metrics['swap_out_count']} swap_in={metrics['swap_in_count']}"
        )

    elapsed = time.perf_counter() - t0
    print(f"stress_done elapsed={elapsed:.2f}s metrics={engine.get_metrics()}")


if __name__ == "__main__":
    main()
