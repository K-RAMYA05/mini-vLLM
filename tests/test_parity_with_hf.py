"""Greedy-decoding parity vs Hugging Face `transformers.generate`.

Lossless-correctness gate: same model, same prompt, same seed, same greedy
decoding rule must produce the same first-N tokens through mini_vllm and
through transformers. If it doesn't, every speed claim is suspect.

This test only runs on CUDA. On CPU it skips (loading a real Llama model on
CPU is too slow for a unit test, and the engine's CUDA-specific paths can't
exercise on CPU anyway).

Use a small open model so the test runs in a few seconds — we're testing the
*pipeline*, not the model. Llama-3.2-1B is the smallest Llama-class model
that exercises the same architecture as the 8B target.

Override MODEL via env var if 1B isn't available locally:
    MINI_VLLM_PARITY_MODEL=Qwen/Qwen2.5-0.5B pytest tests/test_parity_with_hf.py
"""
from __future__ import annotations

import os

import pytest


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# Function-level skip so we can also skip when transformers isn't installed.
def _require_cuda_and_hf():
    import importlib.util
    if not _has_cuda():
        pytest.skip("CUDA not available")
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("transformers not installed")


PARITY_MODEL = os.environ.get("MINI_VLLM_PARITY_MODEL", "meta-llama/Llama-3.2-1B")
PARITY_PROMPT = "The capital of France is"
PARITY_NUM_TOKENS = 32


def _generate_with_hf(model_name: str, prompt: str, num_tokens: int) -> list[int]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    input_ids = tok.encode(prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            max_new_tokens=num_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.eos_token_id,
        )
    generated = out[0, input_ids.shape[1]:].tolist()
    return generated


def _generate_with_mini_vllm(model_name: str, prompt: str, num_tokens: int) -> list[int]:
    from mini_vllm import EngineConfig, LLMEngine, SamplingParams

    cfg = EngineConfig(
        model_name_or_path=model_name,
        dtype="bfloat16",
        num_gpu_blocks=2048,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
    )
    engine = LLMEngine(cfg)
    engine.add_request(prompt, SamplingParams(max_tokens=num_tokens, temperature=0.0))
    outs = engine.run_until_done()
    assert len(outs) == 1
    return outs[0].output_token_ids


def test_greedy_decoding_matches_hf_for_first_n_tokens():
    _require_cuda_and_hf()

    hf_tokens = _generate_with_hf(PARITY_MODEL, PARITY_PROMPT, PARITY_NUM_TOKENS)
    mini_tokens = _generate_with_mini_vllm(PARITY_MODEL, PARITY_PROMPT, PARITY_NUM_TOKENS)

    # Compare the longest common prefix; allow divergence after stop tokens.
    n = min(len(hf_tokens), len(mini_tokens))
    common = 0
    for i in range(n):
        if hf_tokens[i] != mini_tokens[i]:
            break
        common += 1

    # The bar: at least 16 tokens must match exactly. After that, sampling
    # noise from numerics (matmul order, fp accumulation) can diverge.
    # If we drop below 16, we have a real correctness bug.
    min_required = 16
    assert common >= min_required, (
        f"mini_vllm diverged from HF after {common} tokens "
        f"(required >= {min_required}). "
        f"HF: {hf_tokens[:n]}\nmini: {mini_tokens[:n]}"
    )


def test_greedy_decoding_matches_hf_full_sequence_when_short():
    """Stronger test on a short generation: full match expected."""
    _require_cuda_and_hf()

    hf_tokens = _generate_with_hf(PARITY_MODEL, PARITY_PROMPT, num_tokens=8)
    mini_tokens = _generate_with_mini_vllm(PARITY_MODEL, PARITY_PROMPT, num_tokens=8)
    assert hf_tokens == mini_tokens, (
        f"Full sequence mismatch:\n  HF:   {hf_tokens}\n  mini: {mini_tokens}"
    )
