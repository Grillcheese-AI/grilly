"""
Pooling layers for neural networks.
"""

import numpy as np

from .module import Module


def get_compute():
    """Run get compute."""

    pass


class MaxPool2d(Module):
    """
    2D Max Pooling layer (matches torch.nn.MaxPool2d).

    Applies max pooling over spatial dimensions, selecting the maximum value
    in each pooling window and tracking indices for backward pass.

    Args:
        kernel_size: Size of pooling window (int or tuple)
        stride: Stride of pooling (defaults to kernel_size)
        padding: Zero-padding added to both sides (default: 0)
        dilation: Spacing between pooling elements (default: 1)
        return_indices: If True, return indices of max values (default: False)

    Shape:
        Input: (N, C, H_in, W_in)
        Output: (N, C, H_out, W_out)

    Example:
        >>> pool = MaxPool2d(kernel_size=2, stride=2)
        >>> x = np.random.randn(1, 3, 8, 8).astype(np.float32)
        >>> output = pool(x)
        >>> output.shape
        (1, 3, 4, 4)
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False):
        """Initialize the instance."""

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

        self._cached_indices = None
        self._cached_input_shape = None

    def forward(self, x):
        """Forward pass. Accepts numpy array or VulkanTensor."""
        from ..utils.tensor_conversion import VulkanTensor

        is_vt = isinstance(x, VulkanTensor)
        if not is_vt:
            x = np.asarray(x, dtype=np.float32)
        self._cached_input_shape = x.shape

        compute = self._get_backend()
        output, indices = compute.pooling.maxpool2d(
            x, self.kernel_size, self.stride, self.padding, self.dilation,
            return_gpu_tensor=self._return_gpu_tensor,
        )

        self._cached_indices = indices

        if self.return_indices:
            return output, indices
        return output

    def backward(self, grad_output):
        """Backward pass."""
        if self._cached_indices is None or self._cached_input_shape is None:
            raise RuntimeError("Must call forward before backward")

        compute = self._get_backend()
        grad_input = compute.pooling.maxpool2d_backward(
            grad_output, self._cached_indices, self._cached_input_shape
        )

        return grad_input

    def extra_repr(self):
        """String representation."""
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}"
        )


class AvgPool2d(Module):
    """
    2D Average Pooling layer (matches torch.nn.AvgPool2d).

    Applies average pooling over spatial dimensions, computing the mean
    of values in each pooling window.

    Args:
        kernel_size: Size of pooling window (int or tuple)
        stride: Stride of pooling (defaults to kernel_size)
        padding: Zero-padding added to both sides (default: 0)
        count_include_pad: Whether to include padding in average (default: True)

    Shape:
        Input: (N, C, H_in, W_in)
        Output: (N, C, H_out, W_out)

    Example:
        >>> pool = AvgPool2d(kernel_size=2, stride=2)
        >>> x = np.random.randn(1, 3, 8, 8).astype(np.float32)
        >>> output = pool(x)
        >>> output.shape
        (1, 3, 4, 4)
    """

    def __init__(self, kernel_size, stride=None, padding=0, count_include_pad=True):
        """Initialize the instance."""

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.count_include_pad = count_include_pad

        self._cached_input_shape = None

    def forward(self, x):
        """Forward pass."""
        x = np.asarray(x, dtype=np.float32)
        self._cached_input_shape = x.shape

        compute = self._get_backend()
        output = compute.pooling.avgpool2d(
            x, self.kernel_size, self.stride, self.padding, self.count_include_pad
        )

        return output

    def backward(self, grad_output):
        """Backward pass."""
        if self._cached_input_shape is None:
            raise RuntimeError("Must call forward before backward")

        compute = self._get_backend()
        grad_input = compute.pooling.avgpool2d_backward(
            grad_output,
            self._cached_input_shape,
            self.kernel_size,
            self.stride,
            self.padding,
            self.count_include_pad,
        )

        return grad_input

    def extra_repr(self):
        """String representation."""
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, count_include_pad={self.count_include_pad}"
        )


class AdaptiveMaxPool2d(Module):
    """
    2D Adaptive Max Pooling (matches torch.nn.AdaptiveMaxPool2d).

    Applies max pooling with dynamically computed kernel/stride to produce
    a fixed output size regardless of input dimensions.

    Args:
        output_size: Target output size (int or tuple)

    Shape:
        Input: (N, C, H_in, W_in)
        Output: (N, C, H_out, W_out)

    Example:
        >>> pool = AdaptiveMaxPool2d(output_size=(7, 7))
        >>> x = np.random.randn(1, 512, 14, 14).astype(np.float32)
        >>> output = pool(x)
        >>> output.shape
        (1, 512, 7, 7)
    """

    def __init__(self, output_size):
        """Initialize the instance."""

        super().__init__()
        self.output_size = (
            output_size if isinstance(output_size, tuple) else (output_size, output_size)
        )

    def forward(self, x):
        """Forward pass."""
        x = np.asarray(x, dtype=np.float32)
        _, _, h_in, w_in = x.shape
        h_out, w_out = self.output_size

        # Compute adaptive kernel and stride
        stride_h = h_in // h_out
        stride_w = w_in // w_out
        kernel_h = h_in - (h_out - 1) * stride_h
        kernel_w = w_in - (w_out - 1) * stride_w

        # Use MaxPool2d with computed parameters
        pool = MaxPool2d(kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w), padding=0)
        return pool(x)

    def backward(self, grad_output):
        """Backward pass (not implemented - use MaxPool2d directly for training)."""
        raise NotImplementedError(
            "AdaptiveMaxPool2d backward not implemented. Use MaxPool2d for training."
        )

    def extra_repr(self):
        """String representation."""
        return f"output_size={self.output_size}"


class AdaptiveAvgPool2d(Module):
    """
    2D Adaptive Average Pooling (matches torch.nn.AdaptiveAvgPool2d).

    Applies average pooling with dynamically computed kernel/stride to produce
    a fixed output size regardless of input dimensions.

    Args:
        output_size: Target output size (int or tuple)

    Shape:
        Input: (N, C, H_in, W_in)
        Output: (N, C, H_out, W_out)

    Example:
        >>> pool = AdaptiveAvgPool2d(output_size=(1, 1))
        >>> x = np.random.randn(1, 512, 7, 7).astype(np.float32)
        >>> output = pool(x)  # Global average pooling
        >>> output.shape
        (1, 512, 1, 1)
    """

    def __init__(self, output_size):
        """Initialize the instance."""

        super().__init__()
        self.output_size = (
            output_size if isinstance(output_size, tuple) else (output_size, output_size)
        )

    def forward(self, x):
        """Forward pass."""
        x = np.asarray(x, dtype=np.float32)
        _, _, h_in, w_in = x.shape
        h_out, w_out = self.output_size

        # Compute adaptive kernel and stride
        stride_h = h_in // h_out
        stride_w = w_in // w_out
        kernel_h = h_in - (h_out - 1) * stride_h
        kernel_w = w_in - (w_out - 1) * stride_w

        # Use AvgPool2d with computed parameters
        pool = AvgPool2d(kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w), padding=0)
        return pool(x)

    def backward(self, grad_output):
        """Backward pass (not implemented - use AvgPool2d directly for training)."""
        raise NotImplementedError(
            "AdaptiveAvgPool2d backward not implemented. Use AvgPool2d for training."
        )

    def extra_repr(self):
        """String representation."""
        return f"output_size={self.output_size}"
