"""Evaluate draft-model acceptance rate for the legacy draft/verify pipeline.

Measures α — the fraction of draft tokens accepted by the target during
the acceptance walk — on a held-out eval set. Reports:
  - Overall α (averaged across positions and sequences).
  - Per-domain α (natural text vs. code), if both domains are present.
  - Expected tokens-per-step = (1 - α^{γ+1}) / (1 - α), and implied latency
    reduction vs. baseline (ignoring draft forward cost) = tokens_per_step.

The "latency reduction" number is an upper bound: in practice the draft
forward pass also takes time, so real speedup is slightly lower. For an
8-layer draft vs. a 32-layer 8B target, the draft is much cheaper per
forward, so the amortized overhead is small but non-zero.

Usage:
    python -m mini_vllm.distill.eval_acceptance \\
        --target meta-llama/Llama-3.1-8B \\
        --draft ./outputs/llama-3.1-8b-draft-8layer/final \\
        --gamma 4 \\
        --num-sequences 200 --seq-len 256
"""
from __future__ import annotations

import argparse
import time
from typing import List

import torch


@torch.inference_mode()
def compute_acceptance_rate(target, draft, tokenizer, prompts: List[str],
                            gamma: int, max_new_tokens: int, device: str) -> dict:
    """Run the draft/target over each prompt, counting accepted tokens.

    We don't use the full SpeculativeExecutor here because this script's
    job is specifically to measure α cleanly. We implement the core
    propose-verify-accept loop directly.
    """
    target.eval()
    draft.eval()

    total_proposed = 0
    total_accepted = 0
    total_generated = 0   # includes bonus tokens on full-accept
    total_steps = 0
    t0 = time.time()

    for prompt_idx, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        L_prompt = input_ids.shape[1]
        generated = input_ids.clone()

        # Simple greedy spec decoding loop. For evaluation purposes greedy
        # is cleaner — α corresponds directly to the fraction of draft
        # argmaxes that match target argmaxes.
        while generated.shape[1] - L_prompt < max_new_tokens:
            # Draft proposes gamma tokens autoregressively.
            draft_tokens = []
            draft_ctx = generated.clone()
            for _ in range(gamma):
                draft_logits = draft(draft_ctx).logits[:, -1, :]
                next_tok = int(draft_logits.argmax(dim=-1).item())
                draft_tokens.append(next_tok)
                draft_ctx = torch.cat(
                    [draft_ctx, torch.tensor([[next_tok]], device=device)], dim=1
                )

            # Target verifies: one forward pass over generated + draft tokens.
            verify_ctx = torch.cat(
                [generated, torch.tensor([draft_tokens], device=device)], dim=1
            )
            target_logits = target(verify_ctx).logits
            # target_logits[:, L-1, :] predicts position L; we want positions
            # [L_cur - 1, ..., L_cur + gamma - 1] which predict [L_cur, ..., L_cur + gamma].
            # But we're comparing draft argmax at positions [L_cur .. L_cur+gamma-1]
            # to target argmax at those same positions.
            L_cur = generated.shape[1]
            target_argmax = target_logits[:, L_cur - 1 : L_cur - 1 + gamma + 1, :].argmax(dim=-1)
            # target_argmax[:, i] is the token target would choose at position L_cur + i.

            accepted_this_step = 0
            for i, t in enumerate(draft_tokens):
                if int(target_argmax[0, i].item()) == t:
                    accepted_this_step += 1
                else:
                    break

            total_proposed += gamma
            total_accepted += accepted_this_step

            # Append accepted prefix; on full-accept also append the bonus.
            if accepted_this_step == gamma:
                # Bonus token comes from target argmax at the final position.
                bonus = int(target_argmax[0, gamma].item())
                appended = draft_tokens + [bonus]
            else:
                # Rejected at position `accepted_this_step` — replace with target's choice.
                corrected = int(target_argmax[0, accepted_this_step].item())
                appended = draft_tokens[:accepted_this_step] + [corrected]

            total_generated += len(appended)
            total_steps += 1
            generated = torch.cat(
                [generated, torch.tensor([appended], device=device)], dim=1
            )

            # Stop if EOS appeared.
            eos = tokenizer.eos_token_id
            if eos is not None and eos in appended:
                break

        if (prompt_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {prompt_idx + 1}/{len(prompts)} prompts, "
                  f"running α = {total_accepted / max(total_proposed, 1):.3f} "
                  f"({elapsed:.1f}s elapsed)")

    alpha = total_accepted / max(total_proposed, 1)
    tokens_per_step = total_generated / max(total_steps, 1)
    return {
        "alpha": alpha,
        "tokens_per_step": tokens_per_step,
        "total_accepted": total_accepted,
        "total_proposed": total_proposed,
        "total_steps": total_steps,
        "total_generated": total_generated,
    }


def build_eval_prompts(num_sequences: int, tokenizer) -> dict:
    """Return a dict of {"natural": [...], "code": [...]} prompt lists."""
    from datasets import load_dataset

    half = num_sequences // 2

    # Natural-text eval: WikiText test split, clean paragraphs.
    nat_ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    nat_prompts = []
    for ex in nat_ds:
        t = ex["text"].strip()
        if len(t) > 100 and not t.startswith("="):   # skip section headers
            nat_prompts.append(t[:400])
        if len(nat_prompts) >= half:
            break

    # Code eval: HumanEval's prompt field is perfect — partial code for the model to complete.
    try:
        code_ds = load_dataset("openai_humaneval", split="test")
        code_prompts = [ex["prompt"] for ex in code_ds][:half]
    except Exception:
        code_ds = load_dataset("code_search_net", "python", split="test",
                               trust_remote_code=True)
        code_prompts = []
        for ex in code_ds:
            t = ex.get("whole_func_string", "")
            if t and len(t) > 50:
                code_prompts.append(t[:400])
            if len(code_prompts) >= half:
                break

    return {"natural": nat_prompts, "code": code_prompts}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--draft", required=True, help="Path to distilled draft checkpoint")
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--num-sequences", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--json-out", default=None,
                   help="If set, write per-domain results to this JSON file.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target)

    print(f"Loading target {args.target} ...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=dtype
    ).to(device)
    print(f"Loading draft {args.draft} ...")
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft, torch_dtype=dtype
    ).to(device)

    prompts_by_domain = build_eval_prompts(args.num_sequences, tokenizer)

    all_results: dict = {
        "draft": args.draft,
        "target": args.target,
        "gamma": args.gamma,
        "num_sequences": args.num_sequences,
        "domains": {},
    }

    for domain, prompts in prompts_by_domain.items():
        if not prompts:
            continue
        print(f"\n=== Domain: {domain} ({len(prompts)} prompts) ===")
        r = compute_acceptance_rate(
            target, draft, tokenizer, prompts,
            gamma=args.gamma, max_new_tokens=args.max_new_tokens, device=device,
        )
        print(f"  α (acceptance rate): {r['alpha']:.3f}")
        print(f"  tokens/step:         {r['tokens_per_step']:.3f}  (max = {args.gamma + 1})")
        print(f"  total steps:         {r['total_steps']}")
        print(f"  total generated:     {r['total_generated']}")
        # Max theoretical speedup = tokens/step. Adjusted speedup =
        # tokens/step / (1 + c) where c = draft_cost / target_cost.
        # For an 8-layer/32-layer pair, c is roughly the draft/target forward cost ratio.
        adjusted = r["tokens_per_step"] / 1.12
        print(f"  est. latency speedup: {adjusted:.2f}x (assuming draft_cost ≈ 12% target_cost)")
        all_results["domains"][domain] = r

    if args.json_out:
        import json
        with open(args.json_out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
