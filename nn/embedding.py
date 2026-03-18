"""
Embedding module.
Uses: embedding-lookup.glsl
"""

import os

import numpy as np

from ._helpers import (
    _create_param_wrapper,
    _get_param_array,
)
from .module import Module


class Embedding(Module):
    """
    Embedding layer
    Uses: embedding-lookup.glsl
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        """Initialize the instance."""

        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize embeddings (normal distribution)
        embedding_data = np.random.normal(0, 1, (num_embeddings, embedding_dim)).astype(np.float32)
        self.weight = _create_param_wrapper(embedding_data)

        # Register parameter
        self.register_parameter("weight", self.weight)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using embedding-lookup.glsl"""
        backend = self._get_backend()
        weight = _get_param_array(self.weight)

        gpu_lookup_enabled = os.getenv("GRILLY_EMBEDDING_GPU_LOOKUP", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        if (
            gpu_lookup_enabled
            and hasattr(backend, "learning")
            and hasattr(backend.learning, "embedding_lookup")
        ):
            try:
                return backend.learning.embedding_lookup(
                    x,
                    weight,
                    return_gpu_tensor=self._return_gpu_tensor,
                )
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        if isinstance(x, np.ndarray):
            return weight[x.astype(np.int32)]
        return weight[x]

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for Embedding.

        Args:
            grad_output: Gradient w.r.t. output (batch, seq_len, embedding_dim)
            x: Input token IDs (batch, seq_len)

        Returns:
            grad_input: Gradient w.r.t. input (usually None for embedding indices)
        """
        if x is None:
            raise ValueError("Input token IDs x are required for Embedding backward pass")

        # Embedding backward: accumulate gradients into weight matrix
        if self.weight is not None:
            backend = self._get_backend()

            # Try GPU-accelerated backward if available
            if hasattr(backend, "learning") and hasattr(backend.learning, "embedding_backward"):
                try:
                    grad_weight = backend.learning.embedding_backward(
                        grad_output, x, self.num_embeddings, self.embedding_dim
                    )

                    # Store gradients in parameter
                    if not hasattr(self.weight, "grad") or self.weight.grad is None:
                        self.weight.grad = grad_weight
                    else:
                        self.weight.grad += grad_weight

                    # No gradient w.r.t. input (token IDs are discrete)
                    return None
                except Exception:
                    pass  # Fall back to CPU

            # CPU fallback: accumulate gradients for each token
            grad_weight = np.zeros_like(_get_param_array(self.weight))

            x_flat = x.flatten()
            grad_flat = grad_output.reshape(-1, grad_output.shape[-1])

            for i, token_id in enumerate(x_flat):
                if 0 <= token_id < self.num_embeddings:
                    grad_weight[int(token_id)] += grad_flat[i]

            if not hasattr(self.weight, "grad") or self.weight.grad is None:
                self.weight.grad = grad_weight
            else:
                self.weight.grad += grad_weight

        # No gradient w.r.t. input (token IDs are discrete)
        return None

    def __repr__(self):
        """Return a debug representation."""

        return (
            f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"
        )
