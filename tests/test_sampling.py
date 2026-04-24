"""Tests for SamplingParams validation and sequence stop logic."""
import pytest

from mini_vllm.sampling import SamplingParams
from mini_vllm.sequence import Sequence, SequenceStatus


def test_greedy_means_temperature_zero():
    sp = SamplingParams(temperature=0.0)
    assert sp.greedy
    sp = SamplingParams(temperature=0.7)
    assert not sp.greedy


def test_rejects_invalid_params():
    with pytest.raises(ValueError):
        SamplingParams(temperature=-1.0)
    with pytest.raises(ValueError):
        SamplingParams(top_p=1.5)
    with pytest.raises(ValueError):
        SamplingParams(top_p=0.0)
    with pytest.raises(ValueError):
        SamplingParams(top_k=0)


def test_sequence_stops_on_length():
    seq = Sequence(prompt="x", prompt_token_ids=[1], sampling_params=SamplingParams(max_tokens=3))
    seq.output_token_ids = [10, 11, 12]
    assert seq.check_stop(eos_token_id=None)
    assert seq.status == SequenceStatus.FINISHED_LENGTH


def test_sequence_stops_on_eos():
    seq = Sequence(prompt="x", prompt_token_ids=[1], sampling_params=SamplingParams(max_tokens=10))
    seq.output_token_ids = [10, 11, 99]
    assert seq.check_stop(eos_token_id=99)
    assert seq.status == SequenceStatus.FINISHED_STOPPED


def test_sequence_stops_on_custom_stop_token():
    seq = Sequence(
        prompt="x", prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=10, stop_token_ids=[42]),
    )
    seq.output_token_ids = [10, 42]
    assert seq.check_stop(eos_token_id=None)
    assert seq.finish_reason == "stop"
