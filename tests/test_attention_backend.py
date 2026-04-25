import pytest
import torch

from mini_vllm.attention import _attention_prefill
from mini_vllm.config import EngineConfig


def test_engine_config_accepts_flash_attn_backend():
    cfg = EngineConfig(prefill_attention_backend="flash_attn")
    assert cfg.prefill_attention_backend == "flash_attn"


def test_engine_config_rejects_unknown_backend():
    with pytest.raises(ValueError, match="prefill_attention_backend"):
        EngineConfig(prefill_attention_backend="not_a_backend")


def test_flash_backend_routes_to_flash2_on_supported_gpu(monkeypatch):
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    expected = torch.randn(1, 2, 4, 8)

    monkeypatch.setattr("mini_vllm.attention._should_prefer_flash2", lambda _q: True)
    monkeypatch.setattr(
        "mini_vllm.attention._flash2_attention",
        lambda _q, _k, _v, _scale: expected,
    )

    out = _attention_prefill(q, k, v, scale=0.5, backend="flash")
    assert out is expected


def test_flash_attn_backend_requires_cuda_tensors():
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        _attention_prefill(q, k, v, scale=0.5, backend="flash_attn")
