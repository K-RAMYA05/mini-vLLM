"""Generate distillation training data from the teacher model.

For each sequence in the prompt corpus, runs the teacher model forward
and saves:
  - input_ids  : [T]      token ids
  - topk_values: [T, K]   teacher logits at top-K positions per token
  - topk_indices: [T, K]  token ids of the top-K at each position

These are saved as sharded .pt files to avoid one giant file. Each shard
holds ~1000 sequences.

Usage:
    python -m mini_vllm.distill.generate_teacher_data \\
        --teacher meta-llama/Llama-3.1-8B \\
        --corpus balanced \\
        --num-sequences 50000 \\
        --seq-len 512 \\
        --output-dir ./distill_data \\
        --topk 50
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterator, List

import torch
from torch.utils.data import Dataset


def _load_balanced_corpus(tokenizer, num_sequences: int, seq_len: int) -> List[torch.Tensor]:
    """50/50 natural-text (C4) + code (CodeSearchNet Python).

    Falls back to WikiText-103 for natural text if C4 isn't available and
    to the GitHub-Code dataset for code. Both are on the HF hub.
    """
    from datasets import load_dataset

    half = num_sequences // 2

    print(f"Loading {half} natural-text sequences from wikitext ...")
    try:
        nat = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    except Exception as e:
        print(f"  wikitext load failed ({e}); trying c4 ...")
        nat = load_dataset("allenai/c4", "en", split="train", streaming=True)

    print(f"Loading {half} code sequences from code_search_net (python) ...")
    try:
        code = load_dataset("code_search_net", "python", split="train", streaming=True,
                            trust_remote_code=True)
        code_field = "whole_func_string"
    except Exception as e:
        print(f"  code_search_net load failed ({e}); trying codeparrot/github-code-clean ...")
        code = load_dataset("codeparrot/github-code-clean", "python", split="train",
                            streaming=True, trust_remote_code=True)
        code_field = "code"

    def pack(stream, field, target_n):
        out = []
        buf: List[int] = []
        for ex in stream:
            text = ex[field] if field in ex else ex.get("text", "")
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            buf.extend(ids + [tokenizer.eos_token_id or 0])
            while len(buf) >= seq_len:
                out.append(torch.tensor(buf[:seq_len], dtype=torch.long))
                buf = buf[seq_len:]
                if len(out) >= target_n:
                    return out
        return out

    nat_seqs = pack(nat, "text", half)
    code_seqs = pack(code, code_field, half)
    all_seqs = nat_seqs + code_seqs
    # Shuffle so training batches mix domains.
    import random
    random.seed(0)
    random.shuffle(all_seqs)
    return all_seqs


@torch.inference_mode()
def generate_teacher_data(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)

    print(f"Loading teacher {args.teacher} ...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.float16
    ).to("cuda").eval()

    seqs = _load_balanced_corpus(tokenizer, args.num_sequences, args.seq_len)
    print(f"Got {len(seqs)} sequences of length {args.seq_len}")

    shard = {"input_ids": [], "topk_values": [], "topk_indices": []}
    shard_idx = 0
    saved = 0
    t0 = time.time()

    for i in range(0, len(seqs), args.batch_size):
        batch = seqs[i : i + args.batch_size]
        input_ids = torch.stack(batch).to("cuda")              # [B, T]
        logits = teacher(input_ids).logits                      # [B, T, V]
        topk = torch.topk(logits, k=args.topk, dim=-1)          # values, indices

        # Store in CPU fp16 to keep disk usage down.
        shard["input_ids"].append(input_ids.cpu())
        shard["topk_values"].append(topk.values.to(torch.float16).cpu())
        shard["topk_indices"].append(topk.indices.to(torch.int32).cpu())

        if sum(t.shape[0] for t in shard["input_ids"]) >= args.shard_size:
            _write_shard(args.output_dir, shard_idx, shard)
            shard_idx += 1
            saved += sum(t.shape[0] for t in shard["input_ids"])
            shard = {"input_ids": [], "topk_values": [], "topk_indices": []}
            elapsed = time.time() - t0
            rate = saved / elapsed if elapsed > 0 else 0
            print(f"  [shard {shard_idx}] saved {saved}/{len(seqs)} "
                  f"({rate:.1f} seq/s)")

    if shard["input_ids"]:
        _write_shard(args.output_dir, shard_idx, shard)
        shard_idx += 1

    print(f"Wrote {shard_idx} shards to {args.output_dir}")


def _write_shard(out_dir: str, idx: int, shard: dict) -> None:
    import torch
    merged = {k: torch.cat(v, dim=0) for k, v in shard.items()}
    path = Path(out_dir) / f"shard_{idx:04d}.pt"
    torch.save(merged, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--corpus", default="balanced", choices=["balanced"])
    p.add_argument("--num-sequences", type=int, default=50000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--output-dir", default="./distill_data")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=1000)
    args = p.parse_args()
    generate_teacher_data(args)


if __name__ == "__main__":
    main()
