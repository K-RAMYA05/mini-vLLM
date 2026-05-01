"""Draft-model distillation pipeline.

Initializes a small pretrained student (e.g. meta-llama/Llama-3.2-1B) and
distills it against a larger teacher's top-k logits via KL + CE loss.

The student must share the teacher's tokenizer (zero-cost when both are from
the Llama-3 family — same 128k vocab) so it slots straight into
SpeculativeExecutor.
"""
from mini_vllm.distill.distill_loss import distillation_loss

__all__ = ["distillation_loss"]
