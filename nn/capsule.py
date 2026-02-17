"""
Capsule Operations

Uses: capsule-project.glsl, semantic-encoder.glsl, dg-sparse-expand.glsl

Reference: grilly/backend/capsule_transformer.py
"""

import numpy as np

from .module import Module


class CapsuleProject(Module):
    """
    Capsule Projection layer - Project embeddings to capsule space.

    Uses: capsule-project.glsl

    Projects: 384D → 32D cognitive vectors

    Reference: grilly/backend/capsule_transformer.py
    """

    def __init__(self, in_dim: int = 384, out_dim: int = 32):
        """
        Initialize CapsuleProject layer.

        Args:
            in_dim: Input dimension (default: 384)
            out_dim: Output capsule dimension (default: 32)
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Capsule projection weights
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        weight_data = np.random.uniform(-limit, limit, (out_dim, in_dim)).astype(np.float32)
        bias_data = np.zeros(out_dim, dtype=np.float32)

        # Register as parameters (will be converted to Parameter if available)
        self.register_parameter("weight", weight_data)
        self.register_parameter("bias", bias_data)

        # Store direct references for backward compatibility
        self.weight = self._parameters["weight"]
        self.bias = self._parameters["bias"]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass - project to capsule space.

        Args:
            x: Input embeddings (batch, in_dim) or (batch, seq_len, in_dim)

        Returns:
            Capsule vectors (batch, out_dim) or (batch, seq_len, out_dim)
        """
        backend = self._get_backend()

        # Store input for backward pass
        self._cached_input = x

        # Try GPU shader if available
        if hasattr(backend, "shaders") and "capsule-project" in backend.shaders:
            try:
                # Use linear backend for now (capsule-project is similar to linear)
                # Full capsule-project shader would include cognitive features injection
                return backend.linear(x, self.weight, self.bias)
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        return x @ self.weight.T + self.bias

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for CapsuleProject.

        Args:
            grad_output: Gradient w.r.t. output (batch, out_dim) or (batch, seq_len, out_dim)
            x: Input from forward pass (optional, uses cached if not provided)

        Returns:
            grad_input: Gradient w.r.t. input
        """
        if x is None:
            x = getattr(self, "_cached_input", None)
        if x is None:
            raise ValueError("Input x is required for backward pass")

        # Extract parameter data
        if hasattr(self.weight, "data"):
            weight_data = self.weight.data
        elif isinstance(self.weight, np.ndarray):
            weight_data = self.weight
        else:
            weight_data = np.asarray(self.weight, dtype=np.float32)

        if hasattr(self.bias, "data"):
            pass
        elif isinstance(self.bias, np.ndarray):
            pass
        else:
            np.asarray(self.bias, dtype=np.float32)

        # Handle batched input
        if grad_output.ndim == 3:
            # (batch, seq_len, out_dim)
            batch_size, seq_len, out_dim = grad_output.shape
            grad_output_flat = grad_output.reshape(-1, out_dim)
            x_flat = x.reshape(-1, x.shape[-1])
        else:
            # (batch, out_dim)
            grad_output_flat = grad_output
            x_flat = x

        # Gradient w.r.t. weight: grad_output^T @ x
        grad_weight = grad_output_flat.T @ x_flat  # (out_dim, in_dim)

        # Gradient w.r.t. bias: sum over batch dimension
        grad_bias = np.sum(grad_output_flat, axis=0)  # (out_dim,)

        # Gradient w.r.t. input: grad_output @ weight
        grad_input = grad_output_flat @ weight_data  # (batch, in_dim)

        # Accumulate gradients
        # Handle both Parameter objects and numpy arrays
        if hasattr(self.weight, "grad"):
            if self.weight.grad is None:
                self.weight.grad = grad_weight
            else:
                self.weight.grad += grad_weight
        else:
            # Create grad attribute if it doesn't exist
            if not hasattr(self.weight, "__dict__"):
                # numpy array - need to use _parameters dict
                if "weight" in self._parameters:
                    param = self._parameters["weight"]
                    if hasattr(param, "grad"):
                        if param.grad is None:
                            param.grad = grad_weight
                        else:
                            param.grad += grad_weight
                    else:
                        param.grad = grad_weight
            else:
                self.weight.grad = grad_weight

        if hasattr(self.bias, "grad"):
            if self.bias.grad is None:
                self.bias.grad = grad_bias
            else:
                self.bias.grad += grad_bias
        else:
            # Create grad attribute if it doesn't exist
            if not hasattr(self.bias, "__dict__"):
                # numpy array - need to use _parameters dict
                if "bias" in self._parameters:
                    param = self._parameters["bias"]
                    if hasattr(param, "grad"):
                        if param.grad is None:
                            param.grad = grad_bias
                        else:
                            param.grad += grad_bias
                    else:
                        param.grad = grad_bias
            else:
                self.bias.grad = grad_bias

        # Reshape grad_input if needed
        if grad_output.ndim == 3:
            grad_input = grad_input.reshape(batch_size, seq_len, -1)

        return grad_input

    def __repr__(self):
        """Return a debug representation."""

        return f"CapsuleProject(in_dim={self.in_dim}, out_dim={self.out_dim})"


class SemanticEncoder(Module):
    """
    Semantic Encoder layer.

    Uses: semantic-encoder.glsl
    """

    def __init__(self, in_dim: int, out_dim: int):
        """
        Initialize SemanticEncoder layer.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        limit = np.sqrt(6.0 / (in_dim + out_dim))
        weight_data = np.random.uniform(-limit, limit, (out_dim, in_dim)).astype(np.float32)
        bias_data = np.zeros(out_dim, dtype=np.float32)

        # Register as parameters
        self.register_parameter("weight", weight_data)
        self.register_parameter("bias", bias_data)

        # Store direct references
        self.weight = self._parameters["weight"]
        self.bias = self._parameters["bias"]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass - semantic encoding.

        Args:
            x: Input (batch, in_dim) or (batch, seq_len, in_dim)

        Returns:
            Encoded output (batch, out_dim) or (batch, seq_len, out_dim)
        """
        backend = self._get_backend()

        # Try GPU shader if available
        if hasattr(backend, "shaders") and "semantic-encoder" in backend.shaders:
            try:
                # Use linear backend for now (semantic-encoder is similar to linear)
                return backend.linear(x, self.weight, self.bias)
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        return x @ self.weight.T + self.bias

    def __repr__(self):
        """Return a debug representation."""

        return f"SemanticEncoder(in_dim={self.in_dim}, out_dim={self.out_dim})"


class DentateGyrus(Module):
    """
    Dentate Gyrus layer - Pattern separation via sparse expansion.

    Uses: dg-sparse-expand.glsl

    Expands: 32D → 128D with 2% sparsity

    Reference: grilly/backend/capsule_transformer.py DentateGyrus
    """

    def __init__(self, in_dim: int = 32, out_dim: int = 128, sparsity: float = 0.02):
        """
        Initialize DentateGyrus layer.

        Args:
            in_dim: Input dimension (default: 32)
            out_dim: Output dimension (default: 128)
            sparsity: Sparsity level (default: 0.02 = 2%)
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sparsity = sparsity

        # Sparse expansion matrix (only 2% of connections active)
        # Initialize with sparse pattern
        num_connections = int(in_dim * out_dim * sparsity)
        weight_data = np.zeros((out_dim, in_dim), dtype=np.float32)

        # Randomly activate sparse connections
        indices = np.random.choice(in_dim * out_dim, num_connections, replace=False)
        row_indices = indices // in_dim
        col_indices = indices % in_dim

        limit = np.sqrt(6.0 / (in_dim + out_dim))
        weight_data[row_indices, col_indices] = np.random.uniform(
            -limit, limit, num_connections
        ).astype(np.float32)

        # Register as parameter
        self.register_parameter("weight", weight_data)

        # Store direct reference for backward compatibility
        self.weight = self._parameters["weight"]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass - sparse expansion.

        Args:
            x: Input capsule vectors (batch, in_dim) or (batch, seq_len, in_dim)

        Returns:
            Sparse expanded vectors (batch, out_dim) or (batch, seq_len, out_dim)
        """
        backend = self._get_backend()

        # Store input and activations for backward pass
        self._cached_input = x

        # Extract parameter data (convert to numpy array)
        if hasattr(self.weight, "data"):
            weight_data = np.asarray(self.weight.data, dtype=np.float32)
        elif isinstance(self.weight, np.ndarray):
            weight_data = np.asarray(self.weight, dtype=np.float32)
        else:
            weight_data = np.asarray(self.weight, dtype=np.float32)

        # Try GPU shader if available
        if hasattr(backend, "shaders") and "dg-sparse-expand" in backend.shaders:
            try:
                # backend.linear computes x @ W.T + b, and weight_data is (out_dim, in_dim),
                # so pass weight_data directly to get activations = x @ weight_data.T
                activations = backend.linear(x, weight_data, None)

                # Apply top-k sparsification (CPU for now, would be in shader)
                k = max(1, int(self.out_dim * self.sparsity))
                if activations.ndim == 2:
                    # (batch, out_dim)
                    sparse = np.zeros_like(activations)
                    top_k_indices = []
                    for i in range(activations.shape[0]):
                        top_k_idx = np.argsort(np.abs(activations[i]))[-k:]
                        top_k_indices.append(top_k_idx)
                        sparse[i, top_k_idx] = activations[i, top_k_idx]
                        # Normalize
                        norm = np.linalg.norm(sparse[i])
                        if norm > 1e-8:
                            sparse[i] /= norm
                    self._cached_top_k = top_k_indices
                    self._cached_activations = activations
                    return sparse
                else:
                    # (out_dim,)
                    top_k_idx = np.argsort(np.abs(activations))[-k:]
                    sparse = np.zeros_like(activations)
                    sparse[top_k_idx] = activations[top_k_idx]
                    norm = np.linalg.norm(sparse)
                    if norm > 1e-8:
                        sparse /= norm
                    self._cached_top_k = top_k_idx
                    self._cached_activations = activations
                    return sparse
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback with top-k sparsification
        activations = x @ weight_data.T
        k = max(1, int(self.out_dim * self.sparsity))

        if activations.ndim == 2:
            # (batch, out_dim)
            sparse = np.zeros_like(activations)
            top_k_indices = []
            for i in range(activations.shape[0]):
                top_k_idx = np.argsort(np.abs(activations[i]))[-k:]
                top_k_indices.append(top_k_idx)
                sparse[i, top_k_idx] = activations[i, top_k_idx]
                norm = np.linalg.norm(sparse[i])
                if norm > 1e-8:
                    sparse[i] /= norm
            self._cached_top_k = top_k_indices
            self._cached_activations = activations
            return sparse
        else:
            # (out_dim,)
            top_k_idx = np.argsort(np.abs(activations))[-k:]
            sparse = np.zeros_like(activations)
            sparse[top_k_idx] = activations[top_k_idx]
            norm = np.linalg.norm(sparse)
            if norm > 1e-8:
                sparse /= norm
            self._cached_top_k = top_k_idx
            self._cached_activations = activations
            return sparse

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for DentateGyrus.

        Args:
            grad_output: Gradient w.r.t. output (batch, out_dim)
            x: Input from forward pass (optional, uses cached if not provided)

        Returns:
            grad_input: Gradient w.r.t. input
        """
        if x is None:
            x = getattr(self, "_cached_input", None)
        if x is None:
            raise ValueError("Input x is required for backward pass")

        top_k_indices = getattr(self, "_cached_top_k", None)
        activations = getattr(self, "_cached_activations", None)

        if top_k_indices is None or activations is None:
            # Fallback: recompute
            activations = x @ self.weight.T
            k = max(1, int(self.out_dim * self.sparsity))
            if activations.ndim == 2:
                top_k_indices = [
                    np.argsort(np.abs(activations[i]))[-k:] for i in range(activations.shape[0])
                ]
            else:
                top_k_indices = np.argsort(np.abs(activations))[-k:]

        # Extract parameter data (convert to numpy array)
        if hasattr(self.weight, "data"):
            weight_data = np.asarray(self.weight.data, dtype=np.float32)
        elif isinstance(self.weight, np.ndarray):
            weight_data = np.asarray(self.weight, dtype=np.float32)
        else:
            weight_data = np.asarray(self.weight, dtype=np.float32)

        # Create mask for sparse gradients (only top-k positions get gradients)
        if grad_output.ndim == 2:
            # (batch, out_dim)
            grad_sparse = np.zeros_like(grad_output)
            for i in range(grad_output.shape[0]):
                if isinstance(top_k_indices, list):
                    grad_sparse[i, top_k_indices[i]] = grad_output[i, top_k_indices[i]]
                else:
                    grad_sparse[i, top_k_indices] = grad_output[i, top_k_indices]
        else:
            # (out_dim,)
            grad_sparse = np.zeros_like(grad_output)
            grad_sparse[top_k_indices] = grad_output[top_k_indices]

        # Gradient w.r.t. input: grad_sparse @ weight
        if grad_output.ndim == 2:
            grad_input = grad_sparse @ weight_data
        else:
            grad_input = grad_sparse @ weight_data

        # Gradient w.r.t. weight: grad_sparse^T @ x
        if grad_output.ndim == 2:
            x_flat = x.reshape(-1, x.shape[-1]) if x.ndim == 3 else x
            grad_weight = grad_sparse.T @ x_flat
        else:
            grad_weight = np.outer(grad_sparse, x)

        # Accumulate gradients (same pattern as CapsuleProject)
        if hasattr(self.weight, "grad"):
            if self.weight.grad is None:
                self.weight.grad = grad_weight
            else:
                self.weight.grad += grad_weight
        else:
            # Create grad attribute if it doesn't exist
            if not hasattr(self.weight, "__dict__"):
                # numpy array - need to use _parameters dict
                if "weight" in self._parameters:
                    param = self._parameters["weight"]
                    if hasattr(param, "grad"):
                        if param.grad is None:
                            param.grad = grad_weight
                        else:
                            param.grad += grad_weight
                    else:
                        param.grad = grad_weight
            else:
                self.weight.grad = grad_weight

        return grad_input

    def __repr__(self):
        """Return a debug representation."""

        return (
            f"DentateGyrus(in_dim={self.in_dim}, out_dim={self.out_dim}, sparsity={self.sparsity})"
        )
