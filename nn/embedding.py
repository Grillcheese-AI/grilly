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

    def forward(self, x):
        """Forward pass using embedding-lookup.glsl.

        Autograd: when called through ``Module.__call__`` with a LongTensor
        input (standard pattern for token ids), the output is wrapped in a
        ``Variable`` whose ``GradFn`` calls ``self.backward(grad_output, ids)``
        on loss.backward(). The GradFn has an empty inputs list because token
        ids are discrete and don't receive gradients — we only use the
        closure to populate ``self.weight.grad`` via the existing
        ``self.backward``.

        Mirrors ``nn.Linear.forward``'s autograd wiring.
        """
        try:
            from grilly.nn.autograd import GradFn as _GradFn
            from grilly.nn.autograd import Variable as _Variable
            from grilly.nn.autograd import _grad_enabled
        except ImportError:
            _Variable = None  # type: ignore[assignment]
            _GradFn = None  # type: ignore[assignment]
            _grad_enabled = False

        weight = _get_param_array(self.weight)

        # Extract the raw token-id ndarray (caller may pass a LongTensor
        # subclass or plain ndarray). Keep the original around so the
        # backward closure has access to the indices.
        ids_data = np.asarray(x)

        # Numpy fancy-index lookup. The legacy ``backend.learning.embedding_lookup``
        # expects a (batch, seq) shape and adds a leading batch dim for 1D
        # inputs, which breaks downstream ops that don't expect the extra
        # axis. Fancy indexing preserves the exact input shape with a
        # trailing embedding dim, matches what PyTorch's Embedding does,
        # and is plenty fast for the sizes we care about (LUT bandwidth
        # bound — even at 8192 vocab × 384 dim it's under 1 ms).
        result = weight[ids_data.astype(np.int32)]

        # ---- Autograd wiring ----
        # Always wrap the output in a Variable with a GradFn when autograd is
        # enabled — Embedding's input is discrete (token ids), so there's no
        # upstream Variable to chain to, but we still need the GradFn so that
        # ``loss.backward()`` can flow back here and update ``weight.grad``.
        if (
            _GradFn is not None
            and _grad_enabled
            and isinstance(result, np.ndarray)
            and not isinstance(result, _Variable)
        ):
            def backward_fn(grad_output):
                # self.backward populates self.weight.grad in-place.
                # Returns None for the input gradient (token ids are discrete).
                self.backward(np.asarray(grad_output), ids_data)
                return ()  # no upstream inputs

            grad_fn = _GradFn("Embedding", backward_fn, [])
            return _Variable(np.asarray(result), requires_grad=True, grad_fn=grad_fn)

        return result

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
