"""Draft-model distillation pipeline.

Produces a smaller draft model from Llama-3.1-8B by:
  1. Layer-pruning the target to keep a shortlist of its layers.
  2. Distilling the pruned model via KL-divergence on the target's output
     distribution, optionally blended with ground-truth CE.

The draft shares the target's tokenizer (zero-cost, by construction) and
its architectural conventions (RoPE dims, RMSNorm, GQA config) so it slots
straight into SpeculativeExecutor.
"""
from mini_vllm.distill.prune import prune_llama_to_n_layers
from mini_vllm.distill.distill_loss import distillation_loss

__all__ = ["prune_llama_to_n_layers", "distillation_loss"]
