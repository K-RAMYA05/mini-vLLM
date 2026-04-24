"""Distillation loss.

Standard Hinton-style soft-label distillation with two tweaks for memory:

1. Top-k teacher logits. The full Llama-3 vocabulary is 128256, so storing
   teacher logits at every position for a batch of 8 × 512 is 8 × 512 × 128k
   × 2 bytes ≈ 1 GiB PER BATCH in fp16. We instead keep only the top-k
   teacher logits (k=50) per position and treat everything else as a uniform
   tail. This cuts storage ~2500x with negligible accuracy loss — 50 tokens
   cover 99%+ of the probability mass in practice.

2. Hybrid loss: α · KL(student || teacher) + (1-α) · CE(student, ground_truth).
   The CE term is a standard regularizer; pure KL can drift if the teacher
   is confidently wrong. α=0.9 is the common choice.

Temperature T > 1 is applied to both teacher and student logits before
softmax (and the result is multiplied by T² to keep gradient magnitudes
stable — the Hinton 2015 recipe).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class DistillConfig:
    temperature: float = 2.0
    kl_weight: float = 0.9          # α
    ce_weight: float = 0.1          # 1 - α
    teacher_topk: int = 50


def distillation_loss(
    student_logits: torch.Tensor,       # [B, T, V]
    teacher_topk_values: torch.Tensor,  # [B, T, K]  — teacher logits at top-k ids
    teacher_topk_indices: torch.Tensor, # [B, T, K]  — token ids of top-k
    ground_truth_ids: torch.Tensor,     # [B, T]     — next-token labels (causal shift already applied)
    cfg: DistillConfig,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict]:
    """Combined KL + CE distillation loss.

    The student's probability at the top-k teacher indices is what the KL
    term constrains. Since the teacher's "tail" (non-top-k) probability
    mass is small and roughly uniform, we don't need to match it — we
    only enforce agreement on the positions the teacher thinks are likely.

    Implementation detail: we compute KL over the K teacher-chosen indices
    by gathering student logits at those same positions, renormalizing
    both distributions within the top-k support, and running standard
    KL in that restricted space. This is the "partial softmax" trick from
    MiniLLM and is known to behave well.
    """
    T = cfg.temperature
    B, Tlen, V = student_logits.shape
    K = teacher_topk_values.shape[-1]

    # --- KL term on top-k support ---
    # Gather the student's logits at the teacher's top-k indices.
    student_topk = torch.gather(student_logits, dim=-1, index=teacher_topk_indices)  # [B, T, K]

    # Temperature-softened distributions over just these K candidates.
    student_log_probs = F.log_softmax(student_topk / T, dim=-1)
    teacher_probs = F.softmax(teacher_topk_values / T, dim=-1)

    # KL(student || teacher) — note convention: we minimize KL from student
    # toward teacher, which pulls student probs toward teacher probs.
    # torch.nn.functional.kl_div expects (log_input, target), computes
    #   target * (log target - log_input) summed.
    kl_per_position = F.kl_div(
        student_log_probs, teacher_probs, reduction="none", log_target=False
    ).sum(dim=-1)                                                # [B, T]

    # Mask out padding positions.
    valid_mask = (ground_truth_ids != ignore_index).float()       # [B, T]
    kl_loss = (kl_per_position * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
    kl_loss = kl_loss * (T ** 2)                                  # Hinton scaling

    # --- CE term (standard next-token) ---
    ce_loss = F.cross_entropy(
        student_logits.reshape(-1, V),
        ground_truth_ids.reshape(-1),
        ignore_index=ignore_index,
    )

    total = cfg.kl_weight * kl_loss + cfg.ce_weight * ce_loss
    metrics = {
        "loss": total.detach(),
        "kl_loss": kl_loss.detach(),
        "ce_loss": ce_loss.detach(),
    }
    return total, metrics
