"""Train a draft model via layer-pruning + distillation.

Pipeline
--------
1. Load teacher (Llama-3.1-8B).
2. Layer-prune to num_keep layers (default 8) -> student.
3. Train student on pre-generated teacher data with KL + CE loss.

The teacher isn't used at training time — we've already run it to produce
top-k logits per token (see generate_teacher_data.py). This decouples the
expensive forward pass from the training loop and lets us do many
student epochs without re-running the teacher.

Usage:
    python -m mini_vllm.distill.train_distill \\
        --teacher meta-llama/Llama-3.1-8B \\
        --data-dir ./distill_data \\
        --num-keep-layers 8 \\
        --output-dir ./outputs/llama-3.1-8b-draft-8layer \\
        --epochs 3 \\
        --batch-size 8 \\
        --lr 3e-4
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mini_vllm.distill.dataset import DistillDataset, collate_distill
from mini_vllm.distill.distill_loss import DistillConfig, distillation_loss
from mini_vllm.distill.prune import estimate_param_count, prune_llama_to_n_layers


def build_student(teacher_path: str, num_keep: int, dtype: torch.dtype, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path, torch_dtype=dtype
    ).to(device).eval()

    print(f"Teacher: {teacher.config.num_hidden_layers} layers, "
          f"{sum(p.numel() for p in teacher.parameters()) / 1e6:.0f}M params")

    student, keep_idx = prune_llama_to_n_layers(teacher, num_keep=num_keep)
    print(f"Student: {student.config.num_hidden_layers} layers "
          f"(kept teacher layers {keep_idx}), "
          f"{sum(p.numel() for p in student.parameters()) / 1e6:.0f}M params")

    # Free the teacher — training only needs the student and pre-computed data.
    del teacher
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return student, tokenizer


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    student, tokenizer = build_student(
        args.teacher, args.num_keep_layers, dtype=dtype, device=device
    )
    # For training, we want fp32 master weights even with fp16 storage.
    # Use bfloat16 if your GPU supports it — avoids the mixed-precision loss scaler.
    if args.use_bf16 and device == "cuda":
        student = student.to(torch.bfloat16)
        dtype = torch.bfloat16

    ds = DistillDataset(args.data_dir)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_distill, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    print(f"Dataset: {len(ds)} sequences, {len(loader)} batches/epoch")

    distill_cfg = DistillConfig(
        temperature=args.temperature,
        kl_weight=args.kl_weight,
        ce_weight=1.0 - args.kl_weight,
        teacher_topk=ds[0]["topk_values"].shape[-1],
    )
    print(f"Distill config: T={distill_cfg.temperature}, "
          f"kl_weight={distill_cfg.kl_weight}, top-k={distill_cfg.teacher_topk}")

    # AdamW with cosine schedule and linear warmup. Standard for transformer finetuning.
    optim = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95)
    )
    total_steps = len(loader) * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda step: _cosine_with_warmup(step, warmup_steps, total_steps),
    )

    # GradScaler only needed for fp16. For bf16 we don't use it.
    use_scaler = (dtype == torch.float16 and device == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    student.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)                   # [B, T]
            topk_values = batch["topk_values"].to(device, dtype=torch.float32)
            topk_indices = batch["topk_indices"].to(device, dtype=torch.long)

            # Causal-LM targets: predict token t+1 from tokens <= t.
            # The teacher's top-k at position t are the distribution the
            # teacher assigned over what comes next AT t, i.e., over token t+1.
            # So we shift student logits too: student_logits[:, :-1] predicts
            # input_ids[:, 1:]; teacher top-k should be slice [:, :-1] as well.
            ground_truth = input_ids[:, 1:].contiguous()

            # Forward pass. Let HF handle autocast if we're in bf16/fp16 storage.
            if use_scaler:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    student_logits = student(input_ids).logits[:, :-1, :].contiguous()
                    loss, metrics = distillation_loss(
                        student_logits=student_logits,
                        teacher_topk_values=topk_values[:, :-1, :],
                        teacher_topk_indices=topk_indices[:, :-1, :],
                        ground_truth_ids=ground_truth,
                        cfg=distill_cfg,
                    )
                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                student_logits = student(input_ids).logits[:, :-1, :].contiguous()
                loss, metrics = distillation_loss(
                    student_logits=student_logits,
                    teacher_topk_values=topk_values[:, :-1, :],
                    teacher_topk_indices=topk_indices[:, :-1, :],
                    ground_truth_ids=ground_truth,
                    cfg=distill_cfg,
                )
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optim.step()
            scheduler.step()

            step += 1
            if step % args.log_every == 0:
                elapsed = time.time() - t0
                tps = step * args.batch_size / elapsed
                print(
                    f"[ep {epoch+1}/{args.epochs} step {step}/{total_steps}] "
                    f"loss={metrics['loss'].item():.4f} "
                    f"kl={metrics['kl_loss'].item():.4f} "
                    f"ce={metrics['ce_loss'].item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"({tps:.1f} seq/s)"
                )

            if step % args.save_every == 0:
                _save(student, tokenizer, args.output_dir, step)

    _save(student, tokenizer, args.output_dir, "final")
    print(f"Done. Checkpoint at {args.output_dir}/final")


def _cosine_with_warmup(step, warmup, total):
    import math
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _save(model, tokenizer, output_dir, tag):
    path = Path(output_dir) / str(tag)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"  saved -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--num-keep-layers", type=int, default=8)
    p.add_argument("--output-dir", default="./outputs/llama-3.1-8b-draft-8layer")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--kl-weight", type=float, default=0.9)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--use-bf16", action="store_true", help="Use bfloat16 instead of fp16+scaler")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=2000)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
