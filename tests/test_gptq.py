import torch
import torch.nn as nn

from mini_vllm.quant.gptq import GPTQLinear, _pack_int4, _quantize_linear_gptq, _unpack_int4


def test_gptq_linear_handles_non_divisible_group_size():
    linear = nn.Linear(10, 3, bias=True)
    qweight = torch.randint(-7, 8, (3, 10), dtype=torch.int8)
    scales = torch.rand(3, 3, dtype=linear.weight.dtype)
    qlinear = GPTQLinear.from_linear(linear, qweight, scales, group_size=4, bits=4)

    x = torch.randn(2, 10)
    out = qlinear(x)

    expanded_scales = scales.repeat_interleave(4, dim=1)[:, :10]
    expected_weight = qweight.to(scales.dtype) * expanded_scales
    expected = x @ expected_weight.t() + linear.bias
    torch.testing.assert_close(out, expected)
    assert qlinear.qweight.dtype == torch.uint8
    assert qlinear.qweight.shape == (3, 5)


def test_int4_pack_round_trips_signed_nibbles():
    qweight = torch.tensor([[-8, -7, -1, 0, 1, 6, 7]], dtype=torch.int8)
    packed = _pack_int4(qweight)
    unpacked = _unpack_int4(packed, qweight.shape[1])
    torch.testing.assert_close(unpacked, qweight)


def test_quantize_linear_gptq_supports_4bit_and_8bit():
    torch.manual_seed(0)
    linear = nn.Linear(10, 4, bias=False)
    hessian = torch.eye(10)

    q4 = _quantize_linear_gptq(linear, hessian, bits=4, group_size=4)
    q8 = _quantize_linear_gptq(linear, hessian, bits=8, group_size=4)

    assert q4.bits == 4
    assert q8.bits == 8
    q4_unpacked = _unpack_int4(q4.qweight, q4.in_features)
    assert int(q4_unpacked.min()) >= -7
    assert int(q4_unpacked.max()) <= 7
    assert int(q8.qweight.min()) >= -127
    assert int(q8.qweight.max()) <= 127
