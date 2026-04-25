"""JAX / Pallas backend for mini_vllm.

Status: design + skeleton only. See DESIGN.md.

Public surface (planned, not yet implemented):
    paged_attention_jax(query, key_cache, value_cache,
                        block_tables, context_lens, scale)

Imports here are deferred so that environments without JAX installed
don't fail at top-level import of mini_vllm.
"""

__all__ = ["paged_attention_jax"]


def paged_attention_jax(*args, **kwargs):
    """Deferred entry — implemented in paged_attention.py once JAX/Pallas
    are wired up. Raises RuntimeError on call until the kernel lands."""
    from .paged_attention import paged_attention_jax as _impl
    return _impl(*args, **kwargs)
