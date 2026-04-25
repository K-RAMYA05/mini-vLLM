"""Backend implementations for the paged-attention decode kernel.

Currently shipped:
    mini_vllm.kernels.paged_attention   (Triton, CUDA)

Planned:
    mini_vllm.backends.jax              (Pallas, JAX-CUDA + JAX-TPU)

See mini_vllm/backends/jax/DESIGN.md.
"""
