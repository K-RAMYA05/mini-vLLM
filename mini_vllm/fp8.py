"""FP8 helpers used by the Hopper-only paths."""
from __future__ import annotations

import torch


FP8_DTYPE_NAME = "float8_e4m3fn"


def get_fp8_dtype():
    dtype = getattr(torch, FP8_DTYPE_NAME, None)
    if dtype is None:
        raise RuntimeError(
            "This PyTorch build does not expose torch.float8_e4m3fn; "
            "FP8 weights/KV require a recent Hopper-capable build."
        )
    return dtype


def is_fp8_dtype(dtype) -> bool:
    fp8_dtype = getattr(torch, FP8_DTYPE_NAME, None)
    return fp8_dtype is not None and dtype == fp8_dtype


def has_hopper_fp8(device: torch.device | str) -> bool:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    if getattr(torch, FP8_DTYPE_NAME, None) is None:
        return False
    major, _minor = torch.cuda.get_device_capability(device)
    return major >= 9


def require_hopper_fp8(device: torch.device | str, what: str = "FP8") -> None:
    if not has_hopper_fp8(device):
        raise RuntimeError(
            f"{what} requires an SM90+ Hopper GPU with torch.{FP8_DTYPE_NAME} support "
            f"(for example H100 / H200)."
        )
