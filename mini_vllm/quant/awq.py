"""Simplified AWQ weight-only quantization.

This is a compact activation-aware path for this repo's research engine:
we calibrate on a small prompt set, collect per-input-channel activation
magnitudes, and use them to choose more conservative groupwise scales than
plain max-abs weight quantization.

The runtime module is shared with GPTQ (`GPTQLinear`) because the storage
format is the same: low-bit groupwise symmetric weights plus fp16/bf16
scales. The difference is only in how the quantization scales are chosen.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from mini_vllm.quant.gptq import DEFAULT_CALIB_PROMPTS, GPTQLinear, _iter_target_linears


class _ActivationHook:
    """Tracks average absolute activation magnitude per input channel."""

    def __init__(self, in_features: int, device):
        self.mean_abs = torch.zeros((in_features,), dtype=torch.float32, device=device)
        self.n = 0

    def __call__(self, module, inputs, output):
        x = inputs[0]
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        x = x.float().abs()
        cur = x.mean(dim=0)
        n_new = x.shape[0]
        self.mean_abs = self.mean_abs * (self.n / (self.n + n_new)) + cur * (n_new / (self.n + n_new))
        self.n += n_new


def _quantize_linear_awq(
    linear: nn.Linear,
    activation_mean_abs: torch.Tensor,
    bits: int,
    group_size: int,
) -> GPTQLinear:
    if bits not in (4, 8):
        raise ValueError(f"only 4-bit and 8-bit AWQ are implemented, got {bits}")

    W = linear.weight.data.clone().float()
    O, I = W.shape
    qmax = 2 ** (bits - 1) - 1
    num_groups = (I + group_size - 1) // group_size
    scales = torch.zeros((O, num_groups), device=W.device, dtype=torch.float32)
    qW = torch.zeros_like(W)

    act = activation_mean_abs.to(W.device).float().clamp_min(1e-6)
    act = act / act.mean().clamp_min(1e-6)

    for g in range(num_groups):
        g_start = g * group_size
        g_end = min(g_start + group_size, I)
        w_block = W[:, g_start:g_end]
        act_block = act[g_start:g_end]

        weighted_abs = w_block.abs() * act_block.unsqueeze(0)
        scale = weighted_abs.amax(dim=1).clamp_min(1e-8) / qmax
        scales[:, g] = scale

        q = torch.clamp(torch.round(w_block / scale.unsqueeze(1)), -qmax, qmax)
        qW[:, g_start:g_end] = q

    return GPTQLinear.from_linear(
        linear,
        qW.to(torch.int8),
        scales,
        group_size=group_size,
        bits=bits,
    )


@torch.inference_mode()
def apply_awq_quantization(
    model: nn.Module,
    bits: int = 8,
    group_size: int = 128,
    calibration_prompts: Optional[List[str]] = None,
    tokenizer=None,
) -> None:
    """In-place: replace eligible linear layers with AWQ-style low-bit layers."""
    if calibration_prompts is None:
        calibration_prompts = DEFAULT_CALIB_PROMPTS
    if tokenizer is None:
        raise RuntimeError(
            "apply_awq_quantization requires a tokenizer; "
            "call it from LLMEngine or pass tokenizer= explicitly."
        )

    targets = list(_iter_target_linears(model))
    hooks = {}
    handles = []
    for parent, attr, lin in targets:
        hk = _ActivationHook(lin.in_features, lin.weight.device)
        h = lin.register_forward_hook(hk)
        hooks[(id(parent), attr)] = hk
        handles.append(h)

    device = next(model.parameters()).device
    model.eval()
    try:
        for prompt in calibration_prompts:
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            _ = model(ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    for parent, attr, lin in targets:
        hk = hooks[(id(parent), attr)]
        q = _quantize_linear_awq(
            lin,
            activation_mean_abs=hk.mean_abs,
            bits=bits,
            group_size=group_size,
        )
        setattr(parent, attr, q)
        del lin
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


__all__ = ["apply_awq_quantization"]
