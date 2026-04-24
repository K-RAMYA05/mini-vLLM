"""GPTQ weight-only quantization.

Research-grade implementation of Frantar et al., "GPTQ: Accurate Post-Training
Quantization for Generative Pre-trained Transformers" (2022).

This is the *simplified* GPTQ path:
  - per-output-channel, per-group (group_size=128) symmetric 4-bit or 8-bit quantization,
  - Cholesky-based Hessian inverse for the update rule,
  - activations collected from a handful of calibration prompts.

For INT8 weight-only quant the accuracy loss is small even with naive
round-to-nearest; GPTQ gets us the last bit of perplexity headroom. For
INT4 it matters much more.

Runtime path: GPTQLinear stores int8 weights directly, and stores int4
weights as two signed nibbles packed into each uint8 byte. Forward
uses a Triton int4 matmul on CUDA when Triton is available; otherwise it
falls back to portable unpack/dequant plus a standard fp16 matmul. INT8
currently uses the portable dequant path.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# --------------------------------------------------------------------------
# Quantized linear layer
# --------------------------------------------------------------------------

def _pack_int4(qweight: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values [-8, 7] into uint8 nibbles row-wise."""
    if qweight.dtype != torch.int8:
        qweight = qweight.to(torch.int8)
    out_features, in_features = qweight.shape
    if in_features % 2:
        pad = torch.zeros((out_features, 1), dtype=torch.int8, device=qweight.device)
        qweight = torch.cat([qweight, pad], dim=1)
    q = qweight.clamp(-8, 7).to(torch.int16)
    q = torch.where(q < 0, q + 16, q).to(torch.uint8)
    low = q[:, 0::2]
    high = q[:, 1::2] << 4
    return low | high


def _unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Unpack uint8 nibbles into signed int8 values with shape [O, I]."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    vals = torch.empty(
        (packed.shape[0], packed.shape[1] * 2),
        dtype=torch.int8,
        device=packed.device,
    )
    vals[:, 0::2] = torch.where(low >= 8, low.to(torch.int16) - 16, low.to(torch.int16)).to(torch.int8)
    vals[:, 1::2] = torch.where(high >= 8, high.to(torch.int16) - 16, high.to(torch.int16)).to(torch.int8)
    return vals[:, :in_features]


if _HAS_TRITON:

    @triton.jit
    def _int4_linear_kernel(
        x_ptr,
        qweight_ptr,
        scales_ptr,
        out_ptr,
        M: tl.constexpr,
        I: tl.constexpr,
        O: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        PACKED_I: tl.constexpr,
        x_stride_m: tl.constexpr,
        q_stride_o: tl.constexpr,
        s_stride_o: tl.constexpr,
        out_stride_m: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, I, BLOCK_K):
            k = k0 + offs_k
            x = tl.load(
                x_ptr + offs_m[:, None] * x_stride_m + k[None, :],
                mask=(offs_m[:, None] < M) & (k[None, :] < I),
                other=0.0,
            ).to(tl.float32)

            byte_idx = k // 2
            packed = tl.load(
                qweight_ptr + offs_n[:, None] * q_stride_o + byte_idx[None, :],
                mask=(offs_n[:, None] < O) & (k[None, :] < I),
                other=0,
            )
            nibble = tl.where((k[None, :] & 1) == 0, packed & 0x0F, (packed >> 4) & 0x0F)
            signed = tl.where(nibble >= 8, nibble.to(tl.int16) - 16, nibble.to(tl.int16))
            group_idx = k // GROUP_SIZE
            scale = tl.load(
                scales_ptr + offs_n[:, None] * s_stride_o + group_idx[None, :],
                mask=(offs_n[:, None] < O) & (k[None, :] < I),
                other=0.0,
            ).to(tl.float32)
            w = signed.to(tl.float32) * scale
            acc += tl.dot(x, tl.trans(w))

        tl.store(
            out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < O),
        )


def _triton_int4_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    in_features: int,
    out_features: int,
    group_size: int,
) -> torch.Tensor:
    M = x.shape[0]
    out = torch.empty((M, out_features), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, 16), triton.cdiv(out_features, 32))
    _int4_linear_kernel[grid](
        x,
        qweight,
        scales,
        out,
        M,
        in_features,
        out_features,
        group_size,
        qweight.shape[1],
        x.stride(0),
        qweight.stride(0),
        scales.stride(0),
        out.stride(0),
        BLOCK_M=16,
        BLOCK_N=32,
        BLOCK_K=64,
    )
    return out


class GPTQLinear(nn.Module):
    """Low-bit weight, fp16 activation linear layer (symmetric, groupwise).

    Storage:
      qweight: int8 [out_features, in_features] for int8, or uint8
               [out_features, ceil(in_features / 2)] for packed int4
      scales:  fp16, shape [out_features, num_groups]
      bias:    fp16, shape [out_features] or None

    Forward dequantizes per-group on the fly. Not the fastest possible
    path, but correct and memory-saving.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int,
        bits: int,
        bias: bool,
        dtype: torch.dtype,
        device,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.bits = bits
        self.num_groups = (in_features + group_size - 1) // group_size

        if bits == 4:
            qweight_shape = (out_features, (in_features + 1) // 2)
            qweight_dtype = torch.uint8
        elif bits == 8:
            qweight_shape = (out_features, in_features)
            qweight_dtype = torch.int8
        else:
            raise ValueError(f"bits must be 4 or 8, got {bits}")
        self.register_buffer("qweight", torch.zeros(qweight_shape, dtype=qweight_dtype, device=device))
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
        if self.bits == 4 and _HAS_TRITON and x_2d.is_cuda:
            out = _triton_int4_linear(
                x_2d,
                self.qweight,
                self.scales,
                self.in_features,
                self.out_features,
                self.group_size,
            )
            if self.bias is not None:
                out = out + self.bias
            return out.reshape(*orig_shape, self.out_features)

        # Dequantize groupwise. Use repeat_interleave instead of a padded view
        # so arbitrary in_features/group_size combinations work.
        qweight = _unpack_int4(self.qweight, self.in_features) if self.bits == 4 else self.qweight
        expanded_scales = self.scales.repeat_interleave(self.group_size, dim=1)
        expanded_scales = expanded_scales[:, : self.in_features]
        w_fp = qweight.to(self.scales.dtype) * expanded_scales
        out = x_2d @ w_fp.t()
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
        bits: int,
    ) -> "GPTQLinear":
        q = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            group_size=group_size,
            bits=bits,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        if bits == 4:
            q.qweight.copy_(_pack_int4(qweight.to(torch.int8)))
        else:
            q.qweight.copy_(qweight.to(torch.int8))
        q.scales.copy_(scales.to(q.scales.dtype))
        if linear.bias is not None:
            q.bias.data.copy_(linear.bias.data)
        return q


# --------------------------------------------------------------------------
# GPTQ quantization core
# --------------------------------------------------------------------------

def _quantize_linear_gptq(
    linear: nn.Linear,
    hessian: torch.Tensor,
    bits: int,
    group_size: int,
    percdamp: float = 0.01,
) -> "GPTQLinear":
    """Apply the GPTQ update to one linear layer.

    hessian: [in_features, in_features], accumulated E[x x^T] over calibration.
    Returns a GPTQLinear with the same weight shape but int8-quantized rows.
    """
    if bits not in (4, 8):
        raise ValueError(f"only 4-bit and 8-bit GPTQ are implemented, got {bits}")
    W = linear.weight.data.clone().float()           # [O, I]
    O, I = W.shape
    H = hessian.clone().float()

    # Damping: prevents singular H. percdamp of mean diag.
    diag_mean = torch.mean(torch.diag(H))
    damp = percdamp * diag_mean
    H[torch.arange(I), torch.arange(I)] += damp

    # Dead columns: if a column of H is zero (feature never activated),
    # freeze the corresponding weight column to zero to avoid blowups.
    dead = torch.diag(H) == 0
    if dead.any():
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

    # Cholesky factorization of H^{-1}. GPTQ uses the upper-tri Cholesky
    # of H^{-1} as the update direction scaling. We compute it by Cholesky
    # of H, inverse, then Cholesky of that inverse.
    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
    except Exception:
        # Numerical fallback: eye regularization.
        H += torch.eye(I, device=H.device) * (diag_mean * 0.1)
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
    Linv = torch.linalg.cholesky(Hinv, upper=True)    # upper triangular

    # Per-group symmetric quantization.
    qmax = 2 ** (bits - 1) - 1                        # 7 for int4, 127 for int8
    num_groups = (I + group_size - 1) // group_size
    scales = torch.zeros((O, num_groups), device=W.device, dtype=torch.float32)
    qW = torch.zeros_like(W)

    # Process columns left-to-right in chunks of group_size, applying the
    # GPTQ error-compensation update within each chunk.
    for g in range(num_groups):
        g_start = g * group_size
        g_end = min(g_start + group_size, I)

        # Scale per output channel, set once per group from the max abs of
        # the current weight slice.
        w_block = W[:, g_start:g_end]                 # [O, gs]
        absmax = w_block.abs().amax(dim=1).clamp(min=1e-9)
        scale = absmax / qmax                          # [O]
        scales[:, g] = scale

        for j in range(g_start, g_end):
            w = W[:, j]                                # [O]
            q = torch.clamp(torch.round(w / scale), -qmax, qmax)
            qW[:, j] = q
            w_q = q * scale
            err = (w - w_q) / Linv[j, j]               # [O]

            # Push error into remaining (not-yet-quantized) columns.
            if j + 1 < I:
                W[:, j + 1:] -= err.unsqueeze(1) * Linv[j, j + 1:].unsqueeze(0)

    qweight = qW.to(torch.int8)
    return GPTQLinear.from_linear(linear, qweight, scales, group_size, bits)


# --------------------------------------------------------------------------
# Calibration + driver
# --------------------------------------------------------------------------

class _HessianHook:
    """Accumulates E[x x^T] over calls to a linear layer during calibration."""

    def __init__(self, in_features: int, device):
        self.H = torch.zeros((in_features, in_features), device=device, dtype=torch.float32)
        self.n = 0

    def __call__(self, module, inputs, output):
        x = inputs[0]
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        x = x.float()
        # Running average of (x^T x) / batch.
        n_new = x.shape[0]
        self.H = self.H * (self.n / (self.n + n_new)) + (x.t() @ x) / (self.n + n_new)
        self.n += n_new


def _iter_target_linears(model: nn.Module):
    """Yield (parent_module, attr_name, linear) for every linear we want to quantize.

    We quantize attention Q/K/V/O and MLP gate/up/down — the large bulk of
    Llama's params. We SKIP embeddings (kept fp16 for perplexity) and the
    lm_head (same reason; it's tied to embeddings anyway in Llama-3.2).
    """
    for name, module in model.named_modules():
        if not hasattr(module, "self_attn"):
            continue
        if "layers." not in name:
            continue
        # module is a LlamaDecoderLayer.
        attn = module.self_attn
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(attn, attr, None)
            if isinstance(lin, nn.Linear):
                yield attn, attr, lin
        mlp = module.mlp
        for attr in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(mlp, attr, None)
            if isinstance(lin, nn.Linear):
                yield mlp, attr, lin


DEFAULT_CALIB_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "The stock market closed higher today on news of the central bank's decision.",
    "To be, or not to be, that is the question.",
    "import numpy as np\nx = np.arange(10)\nprint(x.mean())",
    "Once upon a time, in a land far, far away, there lived a young princess.",
    "The mitochondrion is the powerhouse of the cell.",
]


@torch.inference_mode()
def apply_gptq_quantization(
    model: nn.Module,
    bits: int = 8,
    group_size: int = 128,
    calibration_prompts: Optional[List[str]] = None,
    tokenizer=None,
) -> None:
    """In-place: replace eligible linear layers with GPTQLinear.

    For a research-grade run we calibrate on a small fixed set of prompts.
    For stronger results, pass ~128 samples from WikiText-2 or C4 via
    `calibration_prompts`.
    """
    if calibration_prompts is None:
        calibration_prompts = DEFAULT_CALIB_PROMPTS

    # We register hooks on every target linear, run the calibration data,
    # then pop hooks and do the actual quantization pass. This is O(layers)
    # in memory for Hessians. For 8B models this is expensive; use a small
    # calibration set or quantize one module group at a time for production.
    targets = list(_iter_target_linears(model))
    hooks = {}
    handles = []
    for parent, attr, lin in targets:
        hk = _HessianHook(lin.in_features, lin.weight.device)
        h = lin.register_forward_hook(hk)
        hooks[(id(parent), attr)] = hk
        handles.append(h)

    if tokenizer is None:
        from transformers import AutoTokenizer
        # The model itself doesn't store a tokenizer reference. In practice
        # apply_gptq_quantization is called from LLMEngine, which has the
        # tokenizer — we'll pass it in from there. This fallback is just
        # for standalone testing.
        raise RuntimeError(
            "apply_gptq_quantization requires a tokenizer; "
            "call it from LLMEngine or pass tokenizer= explicitly."
        )

    device = next(model.parameters()).device
    model.eval()
    try:
        for prompt in calibration_prompts:
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            _ = model(ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    # Quantize and replace.
    for parent, attr, lin in targets:
        hk = hooks[(id(parent), attr)]
        q = _quantize_linear_gptq(lin, hk.H, bits=bits, group_size=group_size)
        setattr(parent, attr, q)
        # Free original fp16 weights.
        del lin
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
