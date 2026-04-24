"""Minimal example: generate text with mini-vLLM.

    python examples/generate.py
"""
from mini_vllm import EngineConfig, LLMEngine, SamplingParams


def main():
    engine = LLMEngine(
        EngineConfig(
            model_name_or_path="meta-llama/Llama-3.1-8B",
            dtype="bfloat16",
            num_gpu_blocks=8192,
            max_num_seqs=16,
            prefill_attention_backend="flash",
        )
    )
    prompts = [
        "The theory of relativity states that",
        "def hello_world():",
        "Once upon a time in a distant galaxy,",
    ]
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=64)
    for p in prompts:
        engine.add_request(p, sp)
    outputs = engine.run_until_done()
    for o in outputs:
        print(f"\n--- seq {o.seq_id} ---")
        print(o.prompt + o.output_text)


if __name__ == "__main__":
    main()
