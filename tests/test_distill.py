"""Tests for the distillation pipeline.

Pruning is covered by a small mock Llama config (no actual weight loading
from disk required). Loss is tested against hand-computed values.
"""
import pytest
import torch

from mini_vllm.distill.distill_loss import DistillConfig, distillation_loss
from mini_vllm.distill.prune import _choose_layers_to_keep, estimate_param_count


# ------------------- prune.py -------------------

def test_choose_layers_keeps_endpoints():
    keep = _choose_layers_to_keep(total_layers=16, num_keep=6)
    assert keep[0] == 0
    assert keep[-1] == 15
    assert len(keep) == 6


def test_choose_layers_evenly_spaced():
    keep = _choose_layers_to_keep(total_layers=16, num_keep=4)
    # Even spacing: 0, 5, 10, 15
    assert keep == [0, 5, 10, 15]


def test_choose_layers_no_duplicates():
    keep = _choose_layers_to_keep(total_layers=4, num_keep=3)
    assert len(keep) == len(set(keep))


def test_choose_layers_all_if_num_keep_exceeds():
    keep = _choose_layers_to_keep(total_layers=4, num_keep=10)
    assert keep == [0, 1, 2, 3]


def test_estimate_param_count_matches_llama_3_2_1b():
    """Sanity check against published Llama-3.2-1B param count."""
    n = estimate_param_count(
        vocab_size=128256, hidden_size=2048, intermediate_size=8192,
        num_layers=16, num_heads=32, num_kv_heads=8, head_dim=64,
        tied_embeddings=True,
    )
    # Published: ~1.24B. Our formula should be close.
    assert 1.15e9 < n < 1.30e9, f"Got {n/1e9:.2f}B, expected ~1.24B"


def test_estimate_param_count_6_layer_pruned():
    n = estimate_param_count(
        vocab_size=128256, hidden_size=2048, intermediate_size=8192,
        num_layers=6, num_heads=32, num_kv_heads=8, head_dim=64,
        tied_embeddings=True,
    )
    # Rough target: ~617M per the comments in prune.py.
    assert 0.55e9 < n < 0.70e9


# ------------------- distill_loss.py -------------------

def test_distillation_loss_zero_when_student_matches_teacher():
    """If student logits == teacher logits, KL should be ~0."""
    torch.manual_seed(0)
    B, T, V, K = 2, 4, 100, 10
    # Teacher: random logits, take top-K.
    teacher_logits = torch.randn(B, T, V)
    topk_values, topk_indices = torch.topk(teacher_logits, K, dim=-1)

    # Build student logits that agree with teacher on the top-K (equal values
    # at those indices; -inf everywhere else).
    student_logits = torch.full((B, T, V), -1e9)
    student_logits.scatter_(-1, topk_indices, topk_values)

    gt_ids = torch.randint(0, V, (B, T - 1))
    cfg = DistillConfig()
    total, m = distillation_loss(
        student_logits[:, :-1, :],
        topk_values[:, :-1, :],
        topk_indices[:, :-1, :],
        gt_ids,
        cfg,
    )
    # KL should be essentially zero (up to floating point noise).
    assert m["kl_loss"].item() < 1e-3, f"KL={m['kl_loss'].item()}"


def test_distillation_loss_positive_when_student_differs():
    torch.manual_seed(0)
    B, T, V, K = 2, 4, 100, 10
    student_logits = torch.randn(B, T - 1, V)
    teacher_logits = torch.randn(B, T, V)
    topk_values, topk_indices = torch.topk(teacher_logits, K, dim=-1)
    gt_ids = torch.randint(0, V, (B, T - 1))

    cfg = DistillConfig()
    total, m = distillation_loss(
        student_logits,
        topk_values[:, :-1, :],
        topk_indices[:, :-1, :],
        gt_ids,
        cfg,
    )
    assert m["kl_loss"].item() > 0
    assert m["ce_loss"].item() > 0
    assert m["loss"].item() == pytest.approx(
        cfg.kl_weight * m["kl_loss"].item() + cfg.ce_weight * m["ce_loss"].item(),
        rel=1e-4,
    )


def test_distillation_loss_respects_ignore_index():
    """Padding positions (label = -100) should not contribute to the loss."""
    torch.manual_seed(0)
    B, T, V, K = 1, 4, 50, 5
    student_logits = torch.randn(B, T, V)
    teacher_logits = torch.randn(B, T, V)
    topk_values, topk_indices = torch.topk(teacher_logits, K, dim=-1)

    gt_all_ignored = torch.full((B, T), -100)
    cfg = DistillConfig()
    _, m_ignored = distillation_loss(
        student_logits, topk_values, topk_indices, gt_all_ignored, cfg
    )
    # With everything ignored, KL should be 0 (division by max(1, 0) = 1
    # and the masked sum is 0).
    assert m_ignored["kl_loss"].item() == 0.0
