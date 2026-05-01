import pytest
import torch

from mini_vllm.attention import PagedAttention, _attention_prefill
from mini_vllm.attention_metadata import AttentionMetadata, PrefillSeqInfo
from mini_vllm.block_manager import BlockTable, KVCache
from mini_vllm.config import EngineConfig


def test_engine_config_accepts_flash_family_backends():
    for backend in ("flash_attn", "flash2", "flash3"):
        cfg = EngineConfig(prefill_attention_backend=backend)
        assert cfg.prefill_attention_backend == backend


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


def test_flash_backend_routes_to_flash3_on_hopper(monkeypatch):
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    expected = torch.randn(1, 2, 4, 8)

    monkeypatch.setattr("mini_vllm.attention._is_hopper_or_newer", lambda _q: True)
    monkeypatch.setattr(
        "mini_vllm.attention._flash3_attention",
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


def test_flash3_backend_requires_cuda_tensors():
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        _attention_prefill(q, k, v, scale=0.5, backend="flash3")


def test_prefill_packs_equal_length_no_prefix_sequences(monkeypatch):
    class DummyRoPE:
        def __call__(self, value_states, position_ids):
            seq_len = position_ids.shape[-1]
            head_dim = value_states.shape[-1]
            cos = torch.ones((1, seq_len, head_dim), dtype=value_states.dtype, device=value_states.device)
            sin = torch.zeros((1, seq_len, head_dim), dtype=value_states.dtype, device=value_states.device)
            return cos, sin

    calls = []

    def fake_prefill(q, k, v, scale, backend, sliding_window=0):
        calls.append((q.shape, k.shape, v.shape, backend, sliding_window))
        return q

    monkeypatch.setattr("mini_vllm.attention._attention_prefill", fake_prefill)

    hidden_size = 4
    attn = PagedAttention(
        hidden_size=hidden_size,
        num_heads=1,
        num_kv_heads=1,
        head_dim=hidden_size,
        rope=DummyRoPE(),
        q_proj=torch.nn.Linear(hidden_size, hidden_size, bias=False),
        k_proj=torch.nn.Linear(hidden_size, hidden_size, bias=False),
        v_proj=torch.nn.Linear(hidden_size, hidden_size, bias=False),
        o_proj=torch.nn.Linear(hidden_size, hidden_size, bias=False),
        layer_idx=0,
        use_triton=False,
        prefill_backend="math",
    )
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
        proj.weight.data.copy_(torch.eye(hidden_size))

    cache = KVCache(
        num_layers=1,
        num_kv_heads=1,
        head_dim=hidden_size,
        num_blocks=4,
        block_size=4,
        dtype=torch.float32,
        device="cpu",
    )
    bt0 = BlockTable([cache.allocator.allocate()])
    bt1 = BlockTable([cache.allocator.allocate()])
    metadata = AttentionMetadata(
        num_prefill_tokens=6,
        num_decode_seqs=0,
        prefill_seq_infos=[
            PrefillSeqInfo(block_table=bt0, token_range=(0, 3), start_pos=0),
            PrefillSeqInfo(block_table=bt1, token_range=(3, 6), start_pos=0),
        ],
        decode_seq_infos=[],
        decode_block_tables=torch.empty((0, 0), dtype=torch.int32),
        decode_context_lens=torch.empty((0,), dtype=torch.int32),
        sliding_window=0,
    )
    hidden_states = torch.arange(24, dtype=torch.float32).view(6, hidden_size)
    position_ids = torch.arange(6, dtype=torch.long)

    out = attn(hidden_states, position_ids=position_ids, kv_cache=cache, attn_metadata=metadata)

    assert len(calls) == 1
    assert calls[0][0] == (2, 1, 3, hidden_size)
    torch.testing.assert_close(out, hidden_states)


def test_prefill_packs_equal_length_prefix_cached_sequences(monkeypatch):
    class DummyRoPE:
        def __call__(self, value_states, position_ids):
            seq_len = position_ids.shape[-1]
            head_dim = value_states.shape[-1]
            cos = torch.ones((1, seq_len, head_dim), dtype=value_states.dtype, device=value_states.device)
            sin = torch.zeros((1, seq_len, head_dim), dtype=value_states.dtype, device=value_states.device)
            return cos, sin

    calls = []

    def fake_prefill_with_prefix(q, k, v, prefix_len, scale, sliding_window=0):
        calls.append((q.shape, k.shape, v.shape, prefix_len, sliding_window))
        return q

    monkeypatch.setattr("mini_vllm.attention._attention_prefill_with_prefix", fake_prefill_with_prefix)

    hidden_size = 4
    attn = PagedAttention(
        hidden_size=hidden_size,
        num_heads=1,
        num_kv_heads=1,
        head_dim=hidden_size,
        rope=DummyRoPE(),
        q_proj=torch.nn.Linear(hidden_size, hidden_size, bias=False),
        k_proj=torch.nn.Linear(hidden_size, hidden_size, bias=False),
        v_proj=torch.nn.Linear(hidden_size, hidden_size, bias=False),
        o_proj=torch.nn.Linear(hidden_size, hidden_size, bias=False),
        layer_idx=0,
        use_triton=False,
        prefill_backend="math",
    )
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
        proj.weight.data.copy_(torch.eye(hidden_size))

    cache = KVCache(
        num_layers=1,
        num_kv_heads=1,
        head_dim=hidden_size,
        num_blocks=6,
        block_size=4,
        dtype=torch.float32,
        device="cpu",
    )
    bt0 = BlockTable([cache.allocator.allocate()])
    bt1 = BlockTable([cache.allocator.allocate()])
    prefix = torch.arange(8, dtype=torch.float32).view(2, 1, hidden_size)
    cache.write_prefill(0, bt0, 0, prefix, prefix)
    cache.write_prefill(0, bt1, 0, prefix + 10, prefix + 10)
    metadata = AttentionMetadata(
        num_prefill_tokens=4,
        num_decode_seqs=0,
        prefill_seq_infos=[
            PrefillSeqInfo(block_table=bt0, token_range=(0, 2), start_pos=2),
            PrefillSeqInfo(block_table=bt1, token_range=(2, 4), start_pos=2),
        ],
        decode_seq_infos=[],
        decode_block_tables=torch.empty((0, 0), dtype=torch.int32),
        decode_context_lens=torch.empty((0,), dtype=torch.int32),
        sliding_window=0,
    )
    hidden_states = torch.arange(16, dtype=torch.float32).view(4, hidden_size)
    position_ids = torch.arange(2, 6, dtype=torch.long)

    out = attn(hidden_states, position_ids=position_ids, kv_cache=cache, attn_metadata=metadata)

    assert len(calls) == 1
    assert calls[0][0] == (2, 1, 2, hidden_size)
    assert calls[0][1] == (2, 1, 4, hidden_size)
    assert calls[0][3] == 2
    torch.testing.assert_close(out, hidden_states)
