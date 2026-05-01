import pytest
import torch
import torch.nn as nn

from mini_vllm.block_manager import BlockTable, KVCache
from mini_vllm.quant.fp8 import FP8Linear, _quantize_linear_fp8


def _has_fp8():
    return hasattr(torch, "float8_e4m3fn")


@pytest.mark.skipif(not _has_fp8(), reason="PyTorch build does not expose float8_e4m3fn")
def test_quantize_linear_fp8_returns_fp8linear():
    linear = nn.Linear(10, 4, bias=True)
    qlinear = _quantize_linear_fp8(linear, group_size=4)

    assert isinstance(qlinear, FP8Linear)
    assert qlinear.qweight.dtype == torch.float8_e4m3fn

    x = torch.randn(2, 10, dtype=linear.weight.dtype)
    out = qlinear(x)
    assert out.shape == (2, 4)


@pytest.mark.skipif(not _has_fp8(), reason="PyTorch build does not expose float8_e4m3fn")
def test_fp8_kv_cache_round_trip_with_tolerance():
    cache = KVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=4,
        block_size=4,
        dtype=torch.float32,
        device="cpu",
        kv_cache_dtype="fp8",
    )
    bt = BlockTable()
    bt.append(cache.allocator.allocate())
    bt.append(cache.allocator.allocate())

    torch.manual_seed(0)
    keys = torch.randn(8, 2, 8)
    values = torch.randn(8, 2, 8)

    cache.write_prefill(0, bt, start_pos=0, keys=keys, values=values)
    assert cache.key_cache.dtype == torch.float8_e4m3fn
    assert cache.value_cache.dtype == torch.float8_e4m3fn
    assert cache.key_scales is not None

    rk, rv = cache.read_tokens(0, bt, end_pos=8)
    torch.testing.assert_close(rk, keys, atol=0.15, rtol=0.15)
    torch.testing.assert_close(rv, values, atol=0.15, rtol=0.15)
