"""FP8 weight-only quantization for Hopper GPUs."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from mini_vllm.fp8 import get_fp8_dtype, require_hopper_fp8
from mini_vllm.quant.gptq import _iter_target_linears


class FP8Linear(nn.Module):
    """Groupwise-scaled FP8 weight-only linear.

    We store quantized weights in `float8_e4m3fn` and keep one scale per
    output-channel group. Forward dequantizes on the fly back to the layer's
    compute dtype.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int,
        bias: bool,
        dtype: torch.dtype,
        device,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = (in_features + group_size - 1) // group_size
        self.fp8_dtype = get_fp8_dtype()

        self.register_buffer(
            "qweight",
            torch.zeros((out_features, in_features), dtype=self.fp8_dtype, device=device),
        )
        self.register_buffer(
            "scales",
            torch.zeros((out_features, self.num_groups), dtype=dtype, device=device),
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x_2d = x.reshape(-1, self.in_features)
        expanded_scales = self.scales.repeat_interleave(self.group_size, dim=1)
        expanded_scales = expanded_scales[:, : self.in_features]
        w = self.qweight.to(self.scales.dtype) * expanded_scales
        out = x_2d @ w.t()
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape, self.out_features)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
    ) -> "FP8Linear":
        q = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            group_size=group_size,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        q.qweight.copy_(qweight.to(q.fp8_dtype))
        q.scales.copy_(scales.to(q.scales.dtype))
        if linear.bias is not None:
            q.bias.data.copy_(linear.bias.data)
        return q


def _quantize_linear_fp8(linear: nn.Linear, group_size: int) -> FP8Linear:
    fp8_dtype = get_fp8_dtype()
    max_fp8 = torch.finfo(fp8_dtype).max
    W = linear.weight.data.clone().float()
    O, I = W.shape
    num_groups = (I + group_size - 1) // group_size
    scales = torch.zeros((O, num_groups), device=W.device, dtype=torch.float32)
    qW = torch.zeros_like(W)

    for g in range(num_groups):
        g_start = g * group_size
        g_end = min(g_start + group_size, I)
        w_block = W[:, g_start:g_end]
        scale = w_block.abs().amax(dim=1).clamp_min(1e-8) / max_fp8
        scales[:, g] = scale
        qW[:, g_start:g_end] = (w_block / scale.unsqueeze(1)).clamp(-max_fp8, max_fp8)

    return FP8Linear.from_linear(linear, qW.to(fp8_dtype), scales, group_size)


@torch.inference_mode()
def apply_fp8_quantization(
    model: nn.Module,
    group_size: int = 128,
    calibration_prompts: List[str] | None = None,
    tokenizer=None,
) -> None:
    """In-place: replace eligible linears with FP8Linear.

    `calibration_prompts` and `tokenizer` are accepted for API symmetry with
    GPTQ/AWQ, but this path uses weight-only group scales.
    """
    del calibration_prompts, tokenizer
    device = next(model.parameters()).device
    require_hopper_fp8(device, what="FP8 weight quantization")

    targets = list(_iter_target_linears(model))
    for parent, attr, lin in targets:
        q = _quantize_linear_fp8(lin, group_size=group_size)
        setattr(parent, attr, q)
        del lin
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


__all__ = ["FP8Linear", "apply_fp8_quantization", "_quantize_linear_fp8"]
