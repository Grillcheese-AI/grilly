"""
Convolutional operations for Vulkan backend.
GPU-accelerated 2D convolutions with backward pass support.

Performance hierarchy:
1. Vulkan GPU shader (fastest)
2. Numba JIT (fast CPU fallback)
3. Pure numpy (baseline fallback)
"""

import numpy as np
import struct
from typing import Optional, Tuple, List
from .base import VULKAN_AVAILABLE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

if VULKAN_AVAILABLE:
    from vulkan import *

# Try to import buffer pool for GPU buffer reuse
try:
    from .buffer_pool import get_buffer_pool, PooledBuffer, VMABuffer, VMABufferPool, BufferPool
    BUFFER_POOL_AVAILABLE = True
except ImportError:
    BUFFER_POOL_AVAILABLE = False
    get_buffer_pool = None
    PooledBuffer = None
    VMABuffer = None
    VMABufferPool = None
    BufferPool = None


class _DirectBuffer:
    """Wrapper for direct buffer allocation when pool is unavailable"""
    __slots__ = ('handle', 'memory', 'size')

    def __init__(self, handle, memory, size):
        self.handle = handle
        self.memory = memory
        self.size = size

    def release(self):
        """No-op for compatibility - must call destroy explicitly"""
        pass

    def destroy(self, device):
        """Destroy the buffer"""
        if self.handle:
            vkDestroyBuffer(device, self.handle, None)
            self.handle = None
        if self.memory:
            vkFreeMemory(device, self.memory, None)
            self.memory = None


class VulkanConv:
    """Convolutional operations: Conv2d forward and backward passes"""

    def __init__(self, core, pipelines, shaders):
        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders
        self._pool = None  # Lazy initialization

    @property
    def buffer_pool(self):
        """Get or initialize the buffer pool (per-instance pool)"""
        if self._pool is None and BUFFER_POOL_AVAILABLE:
            try:
                self._pool = BufferPool(self.core)
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Buffer pool init failed: {e}")
                pass
        return self._pool

    def _acquire_buffer(self, size: int, usage: int = None) -> 'PooledBuffer':
        """Acquire a buffer from the pool or create directly if pool unavailable."""
        if usage is None:
            usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

        pool = self.buffer_pool
        if pool is not None:
            return pool.acquire(size, usage)
        else:
            handle, memory = self.core._create_buffer(size, usage)
            return _DirectBuffer(handle, memory, size)

    def _release_buffers(self, buffers: List):
        """Release multiple buffers back to pool or destroy directly"""
        for buf in buffers:
            if VMABuffer is not None and isinstance(buf, VMABuffer):
                buf.release()
            elif PooledBuffer is not None and isinstance(buf, PooledBuffer):
                buf.release()
            elif isinstance(buf, _DirectBuffer):
                buf.destroy(self.core.device)
            elif isinstance(buf, tuple) and len(buf) == 2:
                handle, memory = buf
                vkDestroyBuffer(self.core.device, handle, None)
                vkFreeMemory(self.core.device, memory, None)

    def _is_vma_buffer(self, buf) -> bool:
        """Check if buffer is a VMA-allocated buffer"""
        return VMABuffer is not None and isinstance(buf, VMABuffer)

    def _upload_buffer(self, buf, data: np.ndarray):
        """Upload data to buffer, handling VMA and direct buffers"""
        if self._is_vma_buffer(buf):
            pool = self.buffer_pool
            if pool is not None and isinstance(pool, VMABufferPool):
                pool.upload_data(buf, data)
                return
        self.core._upload_buffer(buf.handle, buf.memory, data)

    def _download_buffer(self, buf, size: int, dtype=np.float32) -> np.ndarray:
        """Download data from buffer, handling VMA and direct buffers"""
        if self._is_vma_buffer(buf):
            pool = self.buffer_pool
            if pool is not None and isinstance(pool, VMABufferPool):
                return pool.download_data(buf, size, dtype)
        return self.core._download_buffer(buf.memory, size, dtype)

    def _get_buffer_handle(self, buf):
        """Get Vulkan-compatible buffer handle"""
        if self._is_vma_buffer(buf):
            return buf.get_vulkan_handle()
        return buf.handle

    def conv2d(
        self,
        input_data: np.ndarray,  # (batch, in_channels, height, width)
        weight: np.ndarray,      # (out_channels, in_channels/groups, kernel_h, kernel_w)
        bias: Optional[np.ndarray] = None,  # (out_channels,)
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1
    ) -> np.ndarray:
        """
        2D Convolution forward pass.

        Args:
            input_data: Input tensor (batch, in_channels, height, width)
            weight: Convolution kernel (out_channels, in_channels/groups, kernel_h, kernel_w)
            bias: Optional bias (out_channels,)
            stride: Convolution stride (h, w)
            padding: Zero-padding (h, w)
            dilation: Kernel dilation (h, w)
            groups: Number of blocked connections

        Returns:
            Output tensor (batch, out_channels, out_h, out_w)
        """
        # Check if shader is available
        if 'conv2d-forward' not in self.shaders:
            return self._conv2d_cpu(input_data, weight, bias, stride, padding, dilation, groups)

        # Ensure float32 and convert to numpy if needed
        input_data = np.asarray(input_data, dtype=np.float32)
        weight = np.asarray(weight, dtype=np.float32)
        if bias is not None:
            bias = np.asarray(bias, dtype=np.float32)

        # Extract dimensions
        batch_size, in_channels, in_height, in_width = input_data.shape
        out_channels, channels_per_group, kernel_h, kernel_w = weight.shape
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        # Calculate output dimensions
        out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

        output_size = batch_size * out_channels * out_height * out_width * 4  # BYTES (float32)

        # Allocate buffers
        buf_input = self._acquire_buffer(input_data.nbytes)
        buf_weight = self._acquire_buffer(weight.nbytes)
        buf_bias = self._acquire_buffer((bias.nbytes if bias is not None else 4))  # Dummy buffer if no bias
        buf_output = self._acquire_buffer(output_size)

        try:
            # Upload data
            self._upload_buffer(buf_input, input_data.flatten())
            self._upload_buffer(buf_weight, weight.flatten())
            if bias is not None:
                self._upload_buffer(buf_bias, bias.flatten())
            else:
                self._upload_buffer(buf_bias, np.zeros(1, dtype=np.float32))

            # Create pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                'conv2d-forward', 4, push_constant_size=68
            )

            # Get buffer handles
            in_handle = self._get_buffer_handle(buf_input)
            weight_handle = self._get_buffer_handle(buf_weight)
            bias_handle = self._get_buffer_handle(buf_bias)
            out_handle = self._get_buffer_handle(buf_output)

            # Get cached descriptor set
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                'conv2d-forward',
                [
                    (in_handle, input_data.nbytes),
                    (weight_handle, weight.nbytes),
                    (bias_handle, (bias.nbytes if bias is not None else 4)),
                    (out_handle, output_size * 4)
                ]
            )

            # Pack push constants
            push_data = struct.pack(
                '17I',
                batch_size, in_channels, in_height, in_width,
                out_channels, out_height, out_width,
                kernel_h, kernel_w,
                stride_h, stride_w,
                padding_h, padding_w,
                dilation_h, dilation_w,
                groups,
                1 if bias is not None else 0
            )

            # Dispatch compute shader
            group_count_x = (out_width + 7) // 8
            group_count_y = (out_height + 7) // 8
            group_count_z = batch_size

            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set,
                group_count_x, push_data, group_count_y, group_count_z
            )

            # Download result
            output_flat = self._download_buffer(buf_output, output_size, np.float32)
            num_elements = batch_size * out_channels * out_height * out_width
            return output_flat.reshape(batch_size, out_channels, out_height, out_width)

        finally:
            self._release_buffers([buf_input, buf_weight, buf_bias, buf_output])

    def conv2d_backward_input(
        self,
        grad_output: np.ndarray,  # (batch, out_channels, out_h, out_w)
        weight: np.ndarray,       # (out_channels, in_channels/groups, kernel_h, kernel_w)
        input_shape: Tuple[int, int, int, int],  # (batch, in_channels, in_h, in_w)
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1
    ) -> np.ndarray:
        """
        2D Convolution backward pass - gradient w.r.t. input.

        Args:
            grad_output: Gradient w.r.t. output (batch, out_channels, out_h, out_w)
            weight: Convolution kernel (out_channels, in_channels/groups, kernel_h, kernel_w)
            input_shape: Shape of original input (batch, in_channels, in_h, in_w)
            stride: Convolution stride (h, w)
            padding: Zero-padding (h, w)
            dilation: Kernel dilation (h, w)
            groups: Number of blocked connections

        Returns:
            Gradient w.r.t. input (batch, in_channels, in_h, in_w)
        """
        # Check if shader is available
        if 'conv2d-backward-input' not in self.shaders:
            return self._conv2d_backward_input_cpu(grad_output, weight, input_shape, stride, padding, dilation, groups)

        # Ensure float32 and convert to numpy if needed
        grad_output = np.asarray(grad_output, dtype=np.float32)
        weight = np.asarray(weight, dtype=np.float32)

        # Extract dimensions
        batch_size, out_channels, out_height, out_width = grad_output.shape
        in_batch, in_channels, in_height, in_width = input_shape
        _, channels_per_group, kernel_h, kernel_w = weight.shape
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        grad_input_size = batch_size * in_channels * in_height * in_width * 4  # BYTES

        # Allocate buffers
        buf_grad_output = self._acquire_buffer(grad_output.nbytes)
        buf_weight = self._acquire_buffer(weight.nbytes)
        buf_grad_input = self._acquire_buffer(grad_input_size)

        try:
            # Upload data
            self._upload_buffer(buf_grad_output, grad_output.flatten())
            self._upload_buffer(buf_weight, weight.flatten())
            # Initialize grad_input to zero
            self._upload_buffer(buf_grad_input, np.zeros(grad_input_size, dtype=np.float32))

            # Create pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                'conv2d-backward-input', 3, push_constant_size=64
            )

            # Get buffer handles
            grad_out_handle = self._get_buffer_handle(buf_grad_output)
            weight_handle = self._get_buffer_handle(buf_weight)
            grad_in_handle = self._get_buffer_handle(buf_grad_input)

            # Get cached descriptor set
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                'conv2d-backward-input',
                [
                    (grad_out_handle, grad_output.nbytes),
                    (weight_handle, weight.nbytes),
                    (grad_in_handle, grad_input_size)
                ]
            )

            # Pack push constants
            push_data = struct.pack(
                '16I',
                batch_size, in_channels, in_height, in_width,
                out_channels, out_height, out_width,
                kernel_h, kernel_w,
                stride_h, stride_w,
                padding_h, padding_w,
                dilation_h, dilation_w,
                groups
            )

            # Dispatch compute shader (Z includes both batch and channels)
            group_count_x = (in_width + 7) // 8
            group_count_y = (in_height + 7) // 8
            group_count_z = batch_size * in_channels

            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set,
                group_count_x, push_data, group_count_y, group_count_z
            )

            # Download result
            grad_input_flat = self._download_buffer(buf_grad_input, grad_input_size, np.float32)
            return grad_input_flat.reshape(batch_size, in_channels, in_height, in_width)

        finally:
            self._release_buffers([buf_grad_output, buf_weight, buf_grad_input])

    def conv2d_backward_weight(
        self,
        grad_output: np.ndarray,  # (batch, out_channels, out_h, out_w)
        input_data: np.ndarray,   # (batch, in_channels, in_h, in_w)
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        has_bias: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        2D Convolution backward pass - gradient w.r.t. weights and bias.

        Args:
            grad_output: Gradient w.r.t. output (batch, out_channels, out_h, out_w)
            input_data: Original input (batch, in_channels, in_h, in_w)
            kernel_size: Kernel dimensions (h, w)
            stride: Convolution stride (h, w)
            padding: Zero-padding (h, w)
            dilation: Kernel dilation (h, w)
            groups: Number of blocked connections
            has_bias: Whether to compute bias gradient

        Returns:
            (grad_weight, grad_bias) where:
            - grad_weight: (out_channels, in_channels/groups, kernel_h, kernel_w)
            - grad_bias: (out_channels,) or None
        """
        # Check if shader is available
        if 'conv2d-backward-weight' not in self.shaders:
            return self._conv2d_backward_weight_cpu(grad_output, input_data, kernel_size, stride, padding, dilation, groups, has_bias)

        # Ensure float32 and convert to numpy if needed
        grad_output = np.asarray(grad_output, dtype=np.float32)
        input_data = np.asarray(input_data, dtype=np.float32)

        # Extract dimensions
        batch_size, out_channels, out_height, out_width = grad_output.shape
        _, in_channels, in_height, in_width = input_data.shape
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        in_channels_per_group = in_channels // groups
        grad_weight_size = out_channels * in_channels_per_group * kernel_h * kernel_w * 4  # BYTES

        # Allocate buffers
        buf_grad_output = self._acquire_buffer(grad_output.nbytes)
        buf_input = self._acquire_buffer(input_data.nbytes)
        buf_grad_weight = self._acquire_buffer(grad_weight_size)
        buf_grad_bias = self._acquire_buffer(out_channels * 4 if has_bias else 4)

        try:
            # Upload data
            self._upload_buffer(buf_grad_output, grad_output.flatten())
            self._upload_buffer(buf_input, input_data.flatten())
            # Initialize gradients to zero
            num_weight_elements = out_channels * in_channels_per_group * kernel_h * kernel_w
            self._upload_buffer(buf_grad_weight, np.zeros(num_weight_elements, dtype=np.float32))
            if has_bias:
                self._upload_buffer(buf_grad_bias, np.zeros(out_channels, dtype=np.float32))

            # Create pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                'conv2d-backward-weight', 4, push_constant_size=68
            )

            # Get buffer handles
            grad_out_handle = self._get_buffer_handle(buf_grad_output)
            input_handle = self._get_buffer_handle(buf_input)
            grad_weight_handle = self._get_buffer_handle(buf_grad_weight)
            grad_bias_handle = self._get_buffer_handle(buf_grad_bias)

            # Get cached descriptor set
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                'conv2d-backward-weight',
                [
                    (grad_out_handle, grad_output.nbytes),
                    (input_handle, input_data.nbytes),
                    (grad_weight_handle, grad_weight_size),
                    (grad_bias_handle, (out_channels * 4 if has_bias else 4))
                ]
            )

            # Pack push constants
            push_data = struct.pack(
                '17I',
                batch_size, in_channels, in_height, in_width,
                out_channels, out_height, out_width,
                kernel_h, kernel_w,
                stride_h, stride_w,
                padding_h, padding_w,
                dilation_h, dilation_w,
                groups,
                1 if has_bias else 0
            )

            # Dispatch compute shader
            total_weight_spatial = in_channels_per_group * kernel_h * kernel_w
            group_count_x = (total_weight_spatial + 15) // 16
            group_count_y = (out_channels + 15) // 16
            group_count_z = 1

            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set,
                group_count_x, push_data, group_count_y, group_count_z
            )

            # Download results
            grad_weight_flat = self._download_buffer(buf_grad_weight, grad_weight_size, np.float32)
            grad_weight = grad_weight_flat.reshape(out_channels, in_channels_per_group, kernel_h, kernel_w)

            grad_bias = None
            if has_bias:
                grad_bias = self._download_buffer(buf_grad_bias, out_channels * 4, np.float32)

            return grad_weight, grad_bias

        finally:
            self._release_buffers([buf_grad_output, buf_input, buf_grad_weight, buf_grad_bias])

    # CPU fallbacks (using numpy)
    def _conv2d_cpu(self, input_data, weight, bias, stride, padding, dilation, groups):
        """CPU fallback for conv2d forward pass"""
        batch_size, in_channels, in_height, in_width = input_data.shape
        out_channels, channels_per_group, kernel_h, kernel_w = weight.shape
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        # Calculate output dimensions
        out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

        # Pad input
        if padding_h > 0 or padding_w > 0:
            input_padded = np.pad(
                input_data,
                ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)),
                mode='constant'
            )
        else:
            input_padded = input_data

        output = np.zeros((batch_size, out_channels, out_height, out_width), dtype=np.float32)

        # Naive convolution
        for b in range(batch_size):
            for oc in range(out_channels):
                group = oc // (out_channels // groups)
                ic_start = group * channels_per_group
                for oh in range(out_height):
                    for ow in range(out_width):
                        val = 0.0
                        for ic in range(channels_per_group):
                            for kh in range(kernel_h):
                                for kw in range(kernel_w):
                                    ih = oh * stride_h + kh * dilation_h
                                    iw = ow * stride_w + kw * dilation_w
                                    val += input_padded[b, ic_start + ic, ih, iw] * weight[oc, ic, kh, kw]
                        if bias is not None:
                            val += bias[oc]
                        output[b, oc, oh, ow] = val

        return output

    def _conv2d_backward_input_cpu(self, grad_output, weight, input_shape, stride, padding, dilation, groups):
        """CPU fallback for conv2d backward input"""
        batch_size, in_channels, in_height, in_width = input_shape
        _, out_channels, out_height, out_width = grad_output.shape
        _, channels_per_group, kernel_h, kernel_w = weight.shape
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        grad_input = np.zeros(input_shape, dtype=np.float32)

        # Naive backward pass
        for b in range(batch_size):
            for ic in range(in_channels):
                group = ic // channels_per_group
                oc_start = group * (out_channels // groups)
                oc_end = oc_start + (out_channels // groups)
                for ih in range(in_height):
                    for iw in range(in_width):
                        val = 0.0
                        for oc in range(oc_start, oc_end):
                            for kh in range(kernel_h):
                                for kw in range(kernel_w):
                                    oh_num = ih + padding_h - kh * dilation_h
                                    ow_num = iw + padding_w - kw * dilation_w
                                    if oh_num % stride_h == 0 and ow_num % stride_w == 0:
                                        oh = oh_num // stride_h
                                        ow = ow_num // stride_w
                                        if 0 <= oh < out_height and 0 <= ow < out_width:
                                            val += grad_output[b, oc, oh, ow] * weight[oc, ic % channels_per_group, kh, kw]
                        grad_input[b, ic, ih, iw] = val

        return grad_input

    def _conv2d_backward_weight_cpu(self, grad_output, input_data, kernel_size, stride, padding, dilation, groups, has_bias):
        """CPU fallback for conv2d backward weight"""
        batch_size, out_channels, out_height, out_width = grad_output.shape
        _, in_channels, in_height, in_width = input_data.shape
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        # Pad input
        if padding_h > 0 or padding_w > 0:
            input_padded = np.pad(
                input_data,
                ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)),
                mode='constant'
            )
        else:
            input_padded = input_data

        in_channels_per_group = in_channels // groups
        grad_weight = np.zeros((out_channels, in_channels_per_group, kernel_h, kernel_w), dtype=np.float32)

        # Naive backward pass
        for oc in range(out_channels):
            group = oc // (out_channels // groups)
            ic_start = group * in_channels_per_group
            for ic in range(in_channels_per_group):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        val = 0.0
                        for b in range(batch_size):
                            for oh in range(out_height):
                                for ow in range(out_width):
                                    ih = oh * stride_h + kh * dilation_h
                                    iw = ow * stride_w + kw * dilation_w
                                    val += grad_output[b, oc, oh, ow] * input_padded[b, ic_start + ic, ih, iw]
                        grad_weight[oc, ic, kh, kw] = val

        grad_bias = None
        if has_bias:
            grad_bias = np.sum(grad_output, axis=(0, 2, 3))

        return grad_weight, grad_bias
