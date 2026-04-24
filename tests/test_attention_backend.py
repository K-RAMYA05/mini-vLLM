import pytest
import torch

from mini_vllm.attention import _attention_prefill
from mini_vllm.config import EngineConfig


def test_engine_config_accepts_flash3_backend():
    cfg = EngineConfig(prefill_attention_backend="flash3")
    assert cfg.prefill_attention_backend == "flash3"


def test_engine_config_rejects_unknown_backend():
    with pytest.raises(ValueError, match="prefill_attention_backend"):
        EngineConfig(prefill_attention_backend="not_a_backend")


def test_flash_backend_prefers_flash3_on_supported_hopper(monkeypatch):
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    expected = torch.randn(1, 2, 4, 8)

    monkeypatch.setattr("mini_vllm.attention._should_prefer_flash3", lambda _q: True)
    monkeypatch.setattr(
        "mini_vllm.attention._flash3_attention",
        lambda _q, _k, _v, _scale: expected,
    )

    out = _attention_prefill(q, k, v, scale=0.5, backend="flash")
    assert out is expected


def test_flash3_backend_requires_cuda_tensors():
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        _attention_prefill(q, k, v, scale=0.5, backend="flash3")
