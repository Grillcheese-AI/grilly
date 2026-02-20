"""
Normalization layers (PyTorch-like).
GPU-accelerated batch normalization using Vulkan shaders.
"""

import numpy as np

from .module import Module
from .parameter import Parameter


class BatchNorm2d(Module):
    """
    Batch Normalization for 2D inputs (matches torch.nn.BatchNorm2d API)

    Normalizes inputs across batch and spatial dimensions per channel.
    Uses Welford's algorithm for numerically stable variance computation.

    Shaders:
    - batchnorm2d-forward.glsl: Forward pass with running stats update
    - batchnorm2d-backward.glsl: Backward pass with gamma/beta gradients

    Performance: Efficient GPU implementation with workgroup size 256
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        """
        Initialize Batch Normalization layer.

        Args:
            num_features: Number of channels (C from N,C,H,W)
            eps: Value added for numerical stability (default: 1e-5)
            momentum: Running mean/var momentum (default: 0.1)
            affine: If True, learnable gamma/beta parameters (default: True)
            track_running_stats: If True, track running mean/var (default: True)

        Shape:
            Input: (N, C, H, W)
            Output: (N, C, H, W)

        Examples:
            >>> bn = BatchNorm2d(64)
            >>> x = np.random.randn(32, 64, 56, 56).astype(np.float32)
            >>> output = bn(x)
            >>> output.shape
            (32, 64, 56, 56)
        """
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Learnable parameters (if affine=True)
        if self.affine:
            self.weight = Parameter(
                np.ones(num_features, dtype=np.float32), requires_grad=True
            )  # gamma
            self.bias = Parameter(
                np.zeros(num_features, dtype=np.float32), requires_grad=True
            )  # beta
            self.register_parameter("weight", self.weight)
            self.register_parameter("bias", self.bias)
        else:
            self.weight = None
            self.bias = None

        # Running statistics (not trainable)
        if self.track_running_stats:
            self.running_mean = np.zeros(num_features, dtype=np.float32)
            self.running_var = np.ones(num_features, dtype=np.float32)
            self.num_batches_tracked = 0
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

        # Cache for backward pass
        self._cache_input = None
        self._cache_batch_mean = None
        self._cache_batch_var = None

    def forward(self, x) -> np.ndarray:
        """
        Forward pass using batchnorm2d-forward.glsl

        Args:
            x: Input tensor (batch, channels, height, width) — numpy or VulkanTensor

        Returns:
            Normalized output (batch, channels, height, width)
        """
        from ..utils.tensor_conversion import VulkanTensor

        is_vt = isinstance(x, VulkanTensor)

        # Validate input
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.ndim}D")
        if x.shape[1] != self.num_features:
            raise ValueError(f"Expected {self.num_features} channels, got {x.shape[1]}")

        batch_size, num_features, height, width = x.shape

        # Cache input for backward pass (download if needed)
        if self._grad_enabled and self.training:
            x_np = x.numpy() if is_vt else x
            self._cache_input = x_np.copy()

        backend = self._get_backend()

        # Check if shader is available
        if "batchnorm2d-forward" not in backend.shaders:
            if is_vt:
                x = x.numpy()
            return self._batchnorm2d_cpu(x)

        # Get parameters
        gamma = (
            np.asarray(self.weight, dtype=np.float32)
            if self.affine
            else np.ones(num_features, dtype=np.float32)
        )
        beta = (
            np.asarray(self.bias, dtype=np.float32)
            if self.affine
            else np.zeros(num_features, dtype=np.float32)
        )

        # Initialize buffers for batch statistics
        batch_mean = np.zeros(num_features, dtype=np.float32)
        batch_var = np.zeros(num_features, dtype=np.float32)

        # Call backend batchnorm2d
        output = backend.normalization.batchnorm2d_forward(
            x,
            gamma,
            beta,
            self.running_mean if self.track_running_stats else None,
            self.running_var if self.track_running_stats else None,
            batch_mean,
            batch_var,
            eps=self.eps,
            momentum=self.momentum,
            training=self.training,
            affine=self.affine,
            return_gpu_tensor=self._return_gpu_tensor,
        )

        # Cache batch statistics for backward
        if self.training:
            self._cache_batch_mean = batch_mean
            self._cache_batch_var = batch_var

            # Update num_batches_tracked
            if self.track_running_stats:
                self.num_batches_tracked += 1

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass using batchnorm2d-backward.glsl

        Computes gradients w.r.t. input, weight (gamma), and bias (beta).

        Args:
            grad_output: Gradient w.r.t. output (batch, channels, height, width)

        Returns:
            grad_input: Gradient w.r.t. input (batch, channels, height, width)
        """
        if not self._grad_enabled or not self.training:
            return None

        if self._cache_input is None:
            raise RuntimeError("backward() called but no cached input from forward pass")

        backend = self._get_backend()

        # Check if shader is available
        if "batchnorm2d-backward" not in backend.shaders:
            return self._batchnorm2d_backward_cpu(grad_output)

        gamma = (
            np.asarray(self.weight, dtype=np.float32)
            if self.affine
            else np.ones(self.num_features, dtype=np.float32)
        )

        # Compute gradients
        grad_input, grad_gamma, grad_beta = backend.normalization.batchnorm2d_backward(
            grad_output,
            self._cache_input,
            self._cache_batch_mean,
            self._cache_batch_var,
            gamma,
            eps=self.eps,
            affine=self.affine,
        )

        # Store gradients in parameters
        if self.affine:
            if self.weight.grad is None:
                self.weight.grad = grad_gamma
            else:
                self.weight.grad += grad_gamma

            if self.bias.grad is None:
                self.bias.grad = grad_beta
            else:
                self.bias.grad += grad_beta

        return grad_input

    def _batchnorm2d_cpu(self, x: np.ndarray) -> np.ndarray:
        """CPU fallback for forward pass"""
        batch_size, num_features, height, width = x.shape

        if self.training or not self.track_running_stats:
            # Compute batch statistics
            mean = np.mean(x, axis=(0, 2, 3), keepdims=False)  # (num_features,)
            var = np.var(x, axis=(0, 2, 3), keepdims=False)  # (num_features,)

            # Update running statistics
            if self.track_running_stats and self.training:
                self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var

            # Cache for backward
            if self.training:
                self._cache_batch_mean = mean
                self._cache_batch_var = var
        else:
            # Use running statistics
            mean = self.running_mean
            var = self.running_var

        # Normalize
        mean_expanded = mean[np.newaxis, :, np.newaxis, np.newaxis]
        var_expanded = var[np.newaxis, :, np.newaxis, np.newaxis]
        x_normalized = (x - mean_expanded) / np.sqrt(var_expanded + self.eps)

        # Affine transformation
        if self.affine:
            gamma_expanded = np.asarray(self.weight)[np.newaxis, :, np.newaxis, np.newaxis]
            beta_expanded = np.asarray(self.bias)[np.newaxis, :, np.newaxis, np.newaxis]
            output = gamma_expanded * x_normalized + beta_expanded
        else:
            output = x_normalized

        return output.astype(np.float32)

    def _batchnorm2d_backward_cpu(self, grad_output: np.ndarray) -> np.ndarray:
        """CPU fallback for backward pass"""
        batch_size, num_features, height, width = grad_output.shape
        n = batch_size * height * width

        mean = self._cache_batch_mean
        var = self._cache_batch_var
        x = self._cache_input

        gamma = np.asarray(self.weight) if self.affine else np.ones(num_features, dtype=np.float32)

        # Compute gradients
        mean_expanded = mean[np.newaxis, :, np.newaxis, np.newaxis]
        var_expanded = var[np.newaxis, :, np.newaxis, np.newaxis]
        inv_std = 1.0 / np.sqrt(var_expanded + self.eps)

        x_normalized = (x - mean_expanded) * inv_std

        # Gradient w.r.t. gamma and beta
        if self.affine:
            grad_gamma = np.sum(grad_output * x_normalized, axis=(0, 2, 3))
            grad_beta = np.sum(grad_output, axis=(0, 2, 3))

            if self.weight.grad is None:
                self.weight.grad = grad_gamma
            else:
                self.weight.grad += grad_gamma

            if self.bias.grad is None:
                self.bias.grad = grad_beta
            else:
                self.bias.grad += grad_beta

        # Gradient w.r.t. input
        grad_output_sum = np.sum(grad_output, axis=(0, 2, 3), keepdims=True)
        grad_output_dot = np.sum(grad_output * x_normalized, axis=(0, 2, 3), keepdims=True)

        gamma_expanded = gamma[np.newaxis, :, np.newaxis, np.newaxis]
        grad_input = (
            gamma_expanded
            * inv_std
            / n
            * (n * grad_output - grad_output_sum - x_normalized * grad_output_dot)
        )

        return grad_input.astype(np.float32)

    def extra_repr(self) -> str:
        """String representation for debugging"""
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats}"

    def __repr__(self):
        """Return a debug representation."""

        return f"BatchNorm2d({self.extra_repr()})"


class BatchNorm1d(Module):
    """
    Batch Normalization for 1D inputs (matches torch.nn.BatchNorm1d API)

    For sequences: (N, C, L) or features: (N, C)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        """
        Initialize 1D Batch Normalization.

        Args:
            num_features: Number of features/channels
            eps: Numerical stability term
            momentum: Running stats momentum
            affine: Learnable parameters
            track_running_stats: Track running statistics

        Shape:
            Input: (N, C) or (N, C, L)
            Output: Same shape as input
        """
        super().__init__()

        # Use 2D implementation internally
        self.bn2d = BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Share parameters
        self.weight = self.bn2d.weight
        self.bias = self.bn2d.bias
        self.running_mean = self.bn2d.running_mean
        self.running_var = self.bn2d.running_var

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Args:
            x: Input tensor (N, C) or (N, C, L)

        Returns:
            Normalized output
        """
        if x.ndim == 2:
            # (N, C) -> (N, C, 1, 1)
            x_4d = x[:, :, np.newaxis, np.newaxis]
            out_4d = self.bn2d(x_4d)
            return out_4d.squeeze((2, 3))
        elif x.ndim == 3:
            # (N, C, L) -> (N, C, 1, L)
            x_4d = x[:, :, np.newaxis, :]
            out_4d = self.bn2d(x_4d)
            return out_4d.squeeze(2)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass"""
        if grad_output.ndim == 2:
            grad_4d = grad_output[:, :, np.newaxis, np.newaxis]
            grad_input_4d = self.bn2d.backward(grad_4d)
            return grad_input_4d.squeeze((2, 3)) if grad_input_4d is not None else None
        elif grad_output.ndim == 3:
            grad_4d = grad_output[:, :, np.newaxis, :]
            grad_input_4d = self.bn2d.backward(grad_4d)
            return grad_input_4d.squeeze(2) if grad_input_4d is not None else None

    def __repr__(self):
        """Return a debug representation."""

        return f"BatchNorm1d({self.bn2d.extra_repr()})"
