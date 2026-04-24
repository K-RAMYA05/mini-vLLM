"""Evaluate perplexity on WikiText-2 before and after GPTQ quantization.

Target from writeup: <0.4 perplexity degradation, ~3.8x VRAM reduction.
(VRAM depends on how much of the model we quantize — embeddings and
lm_head are kept in fp16 here, so the ratio is ~3.8x on the model weights
portion, not on total GPU memory which includes KV cache.)

Usage:
    python benchmarks/bench_gptq.py --model meta-llama/Llama-3.1-8B --bits 8
"""
import argparse
import math

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def eval_perplexity(model, tokenizer, stride: int = 1024, max_length: int = 2048) -> float:
    """Sliding-window PPL on WikiText-2 test split."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(next(model.parameters()).device)

    nlls = []
    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + max_length, input_ids.size(1))
        trg_len = end - begin
        ids = input_ids[:, begin:end]
        tgt = ids.clone()
        out = model(ids, labels=tgt)
        nlls.append(out.loss.float() * trg_len)
        if end == input_ids.size(1):
            break
    return math.exp(torch.stack(nlls).sum().item() / input_ids.size(1))


def vram_mb(model: nn.Module) -> float:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total / (1024 ** 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--bits", type=int, default=8, choices=[4, 8])
    p.add_argument("--group-size", type=int, default=128)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)

    print("Loading FP16 model ...")
    model_fp = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to("cuda").eval()
    fp_vram = vram_mb(model_fp)
    print(f"  FP16 weights: {fp_vram:.0f} MiB")
    fp_ppl = eval_perplexity(model_fp, tok)
    print(f"  FP16 PPL on WikiText-2: {fp_ppl:.3f}")

    # Quantize in place.
    from mini_vllm.quant import apply_gptq_quantization
    print(f"\nApplying GPTQ INT{args.bits} quantization (this collects a Hessian; takes a minute) ...")
    apply_gptq_quantization(model_fp, bits=args.bits, group_size=args.group_size, tokenizer=tok)
    q_vram = vram_mb(model_fp)
    print(f"  Quantized model: {q_vram:.0f} MiB  ({fp_vram / q_vram:.2f}x smaller)")
    q_ppl = eval_perplexity(model_fp, tok)
    print(f"  INT{args.bits} PPL on WikiText-2: {q_ppl:.3f}")

    print(f"\nPPL degradation: +{q_ppl - fp_ppl:.3f}")


if __name__ == "__main__":
    main()
