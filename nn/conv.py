"""Convolutional layers for the Grilly neural network API."""

import numpy as np

from .module import Module
from .parameter import Parameter


def _pair(x):
    """Convert single value to pair (h, w)"""
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


class Conv2d(Module):
    """
    2D Convolution Layer (matches torch.nn.Conv2d API)

    Uses Vulkan shaders:
    - conv2d-forward.glsl: Forward pass
    - conv2d-backward-input.glsl: Gradient w.r.t. input
    - conv2d-backward-weight.glsl: Gradient w.r.t. weights/bias

    Performance: >50x speedup vs CPU on AMD RX 6750 XT
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        """Initialize a 2D convolution layer compatible with torch.nn.Conv2d."""
        super().__init__()

        if padding_mode != "zeros":
            raise NotImplementedError(
                f"Only 'zeros' padding mode is supported, got '{padding_mode}'"
            )

        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by groups ({groups})"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode

        # Initialize weights using Kaiming/He initialization
        # For Conv2d: fan_in = in_channels/groups * kernel_h * kernel_w
        kernel_h, kernel_w = self.kernel_size
        in_channels_per_group = in_channels // groups
        fan_in = in_channels_per_group * kernel_h * kernel_w

        # Kaiming uniform initialization (same as PyTorch default)
        bound = np.sqrt(1.0 / fan_in)
        weight_data = np.random.uniform(
            -bound, bound, (out_channels, in_channels_per_group, kernel_h, kernel_w)
        ).astype(np.float32)

        self.weight = Parameter(weight_data, requires_grad=True)
        self.register_parameter("weight", self.weight)

        if bias:
            bias_data = np.random.uniform(-bound, bound, (out_channels,)).astype(np.float32)
            self.bias = Parameter(bias_data, requires_grad=True)
            self.register_parameter("bias", self.bias)
        else:
            self.bias = None

        # Cache for backward pass
        self._cache_input = None

    def forward(self, x) -> np.ndarray:
        """
        Forward pass using conv2d-forward.glsl

        Args:
            x: Input tensor (batch, in_channels, height, width) — numpy or VulkanTensor

        Returns:
            Output tensor (batch, out_channels, out_h, out_w)
        """
        from ..utils.tensor_conversion import VulkanTensor

        is_vt = isinstance(x, VulkanTensor)

        # Validate input shape
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.ndim}D")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")

        # Cache input for backward pass (download if needed)
        if self._grad_enabled:
            x_np = x.numpy() if is_vt else x
            self._cache_input = x_np.copy()

        backend = self._get_backend()
        weight = self.weight.data if hasattr(self.weight, "data") else np.asarray(self.weight)
        bias = (
            (self.bias.data if hasattr(self.bias, "data") else np.asarray(self.bias))
            if self.bias is not None
            else None
        )

        # Call backend conv2d operation
        return backend.conv.conv2d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            return_gpu_tensor=self._return_gpu_tensor,
        )

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass using conv2d backward shaders.

        Computes gradients and stores them in self.weight.grad and self.bias.grad.

        Args:
            grad_output: Gradient w.r.t. output (batch, out_channels, out_h, out_w)

        Returns:
            grad_input: Gradient w.r.t. input (batch, in_channels, in_h, in_w)
        """
        if not self._grad_enabled:
            return None

        if self._cache_input is None:
            raise RuntimeError("backward() called but no cached input from forward pass")

        backend = self._get_backend()
        weight = self.weight.data if hasattr(self.weight, "data") else np.asarray(self.weight)

        # Compute gradient w.r.t. input (for backprop to previous layers)
        grad_input = backend.conv.conv2d_backward_input(
            grad_output,
            weight,
            self._cache_input.shape,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # Compute gradient w.r.t. weights and bias (for parameter updates)
        grad_weight, grad_bias = backend.conv.conv2d_backward_weight(
            grad_output,
            self._cache_input,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            has_bias=(self.bias is not None),
        )

        # Store gradients in parameters
        if self.weight.grad is None:
            self.weight.grad = grad_weight
        else:
            self.weight.grad += grad_weight

        if self.bias is not None:
            if self.bias.grad is None:
                self.bias.grad = grad_bias
            else:
                self.bias.grad += grad_bias

        return grad_input

    def extra_repr(self) -> str:
        """String representation for debugging"""
        s = f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}"
        if self.stride != (1, 1):
            s += f", stride={self.stride}"
        if self.padding != (0, 0):
            s += f", padding={self.padding}"
        if self.dilation != (1, 1):
            s += f", dilation={self.dilation}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"
        return s

    def __repr__(self):
        """Return a debug representation."""

        return f"Conv2d({self.extra_repr()})"


class Conv1d(Module):
    """
    1D Convolution Layer (matches torch.nn.Conv1d API)

    Implemented as a wrapper around Conv2d with height=1.
    For dedicated 1D kernels, create dedicated conv1d shader variants.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """Initialize a 1D convolution layer compatible with torch.nn.Conv1d."""
        super().__init__()

        # Use Conv2d with height=1
        self.conv2d = Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            dilation=(1, dilation),
            groups=groups,
            bias=bias,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Share parameters with internal Conv2d
        self.weight = self.conv2d.weight
        self.bias = self.conv2d.bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Args:
            x: Input tensor (batch, in_channels, length)

        Returns:
            Output tensor (batch, out_channels, out_length)
        """
        # Add height dimension: (N, C, L) -> (N, C, 1, L)
        x_2d = x[:, :, np.newaxis, :]

        # Apply Conv2d
        out_2d = self.conv2d(x_2d)

        # Remove height dimension: (N, C, 1, L) -> (N, C, L)
        return out_2d.squeeze(2)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass

        Args:
            grad_output: Gradient w.r.t. output (batch, out_channels, out_length)

        Returns:
            grad_input: Gradient w.r.t. input (batch, in_channels, length)
        """
        # Add height dimension
        grad_output_2d = grad_output[:, :, np.newaxis, :]

        # Backward through Conv2d
        grad_input_2d = self.conv2d.backward(grad_output_2d)

        # Remove height dimension
        return grad_input_2d.squeeze(2) if grad_input_2d is not None else None

    def __repr__(self):
        """Return a debug representation."""

        s = f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}"
        if self.stride != 1:
            s += f", stride={self.stride}"
        if self.padding != 0:
            s += f", padding={self.padding}"
        if self.dilation != 1:
            s += f", dilation={self.dilation}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"
        return f"Conv1d({s})"
