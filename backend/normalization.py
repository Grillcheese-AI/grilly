"""
Normalization operations for Vulkan backend.
GPU-accelerated batch normalization with forward/backward passes.
"""

import numpy as np
import struct
from typing import Optional, Tuple
from .base import VULKAN_AVAILABLE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

if VULKAN_AVAILABLE:
    from vulkan import *

try:
    from .buffer_pool import BufferPool, PooledBuffer, VMABuffer, VMABufferPool
    BUFFER_POOL_AVAILABLE = True
except ImportError:
    BUFFER_POOL_AVAILABLE = False


class _DirectBuffer:
    __slots__ = ('handle', 'memory', 'size')
    def __init__(self, handle, memory, size):
        self.handle, self.memory, self.size = handle, memory, size
    def release(self): pass
    def destroy(self, device):
        if self.handle:
            vkDestroyBuffer(device, self.handle, None)
            self.handle = None
        if self.memory:
            vkFreeMemory(device, self.memory, None)
            self.memory = None


class VulkanNormalization:
    """Normalization operations: BatchNorm2d forward/backward"""

    def __init__(self, core, pipelines, shaders):
        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders
        self._pool = None

    @property
    def buffer_pool(self):
        if self._pool is None and BUFFER_POOL_AVAILABLE:
            try:
                self._pool = BufferPool(self.core)
            except Exception:
                pass
        return self._pool

    def _acquire_buffer(self, size: int):
        pool = self.buffer_pool
        if pool:
            return pool.acquire(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        handle, memory = self.core._create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        return _DirectBuffer(handle, memory, size)

    def _release_buffers(self, buffers):
        for buf in buffers:
            if isinstance(buf, (PooledBuffer, VMABuffer)):
                buf.release()
            elif isinstance(buf, _DirectBuffer):
                buf.destroy(self.core.device)

    def _upload_buffer(self, buf, data):
        if isinstance(buf, VMABuffer) and self.buffer_pool and isinstance(self.buffer_pool, VMABufferPool):
            self.buffer_pool.upload_data(buf, data)
        else:
            self.core._upload_buffer(buf.handle, buf.memory, data)

    def _download_buffer(self, buf, size, dtype=np.float32):
        if isinstance(buf, VMABuffer) and self.buffer_pool and isinstance(self.buffer_pool, VMABufferPool):
            return self.buffer_pool.download_data(buf, size, dtype)
        return self.core._download_buffer(buf.memory, size, dtype)

    def _get_buffer_handle(self, buf):
        return buf.get_vulkan_handle() if isinstance(buf, VMABuffer) else buf.handle

    def batchnorm2d_forward(
        self,
        input_data: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        running_mean: Optional[np.ndarray],
        running_var: Optional[np.ndarray],
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        eps: float = 1e-5,
        momentum: float = 0.1,
        training: bool = True,
        affine: bool = True
    ) -> np.ndarray:
        """BatchNorm2d forward pass using shader"""
        if 'batchnorm2d-forward' not in self.shaders:
            return self._batchnorm2d_forward_cpu(input_data, gamma, beta, running_mean, running_var, batch_mean, batch_var, eps, momentum, training, affine)

        input_data = np.asarray(input_data, dtype=np.float32)
        gamma = np.asarray(gamma, dtype=np.float32)
        beta = np.asarray(beta, dtype=np.float32)

        batch_size, num_features, height, width = input_data.shape
        output_size = batch_size * num_features * height * width * 4

        # Allocate buffers
        buf_input = self._acquire_buffer(input_data.nbytes)
        buf_output = self._acquire_buffer(output_size)
        buf_gamma = self._acquire_buffer(gamma.nbytes)
        buf_beta = self._acquire_buffer(beta.nbytes)
        buf_running_mean = self._acquire_buffer(num_features * 4)
        buf_running_var = self._acquire_buffer(num_features * 4)
        buf_batch_mean = self._acquire_buffer(num_features * 4)
        buf_batch_var = self._acquire_buffer(num_features * 4)

        try:
            # Upload data
            self._upload_buffer(buf_input, input_data.flatten())
            self._upload_buffer(buf_gamma, gamma)
            self._upload_buffer(buf_beta, beta)
            self._upload_buffer(buf_running_mean, running_mean if running_mean is not None else np.zeros(num_features, dtype=np.float32))
            self._upload_buffer(buf_running_var, running_var if running_var is not None else np.ones(num_features, dtype=np.float32))
            self._upload_buffer(buf_batch_mean, np.zeros(num_features, dtype=np.float32))
            self._upload_buffer(buf_batch_var, np.zeros(num_features, dtype=np.float32))

            # Create pipeline
            pipeline, pipeline_layout, _ = self.pipelines.get_or_create_pipeline('batchnorm2d-forward', 8, push_constant_size=32)

            # Get handles
            handles = [self._get_buffer_handle(b) for b in [buf_input, buf_output, buf_gamma, buf_beta, buf_running_mean, buf_running_var, buf_batch_mean, buf_batch_var]]

            # Descriptor set
            descriptor_set = self.pipelines.get_cached_descriptor_set('batchnorm2d-forward', [
                (handles[0], input_data.nbytes), (handles[1], output_size),
                (handles[2], gamma.nbytes), (handles[3], beta.nbytes),
                (handles[4], num_features * 4), (handles[5], num_features * 4),
                (handles[6], num_features * 4), (handles[7], num_features * 4)
            ])

            # Push constants (8 values: batch, channels, h, w, eps, momentum, training, affine)
            push_data = struct.pack('IIIIffII', batch_size, num_features, height, width, eps, momentum, 1 if training else 0, 1 if affine else 0)

            # Dispatch
            group_count = (num_features + 255) // 256
            self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, group_count, push_data)

            # Download
            output = self._download_buffer(buf_output, output_size, np.float32).reshape(batch_size, num_features, height, width)

            # Download batch stats if training
            if training and running_mean is not None:
                batch_mean[:] = self._download_buffer(buf_batch_mean, num_features * 4, np.float32)
                batch_var[:] = self._download_buffer(buf_batch_var, num_features * 4, np.float32)
                # Update running stats from buffer
                running_mean[:] = self._download_buffer(buf_running_mean, num_features * 4, np.float32)
                running_var[:] = self._download_buffer(buf_running_var, num_features * 4, np.float32)

            return output

        finally:
            self._release_buffers([buf_input, buf_output, buf_gamma, buf_beta, buf_running_mean, buf_running_var, buf_batch_mean, buf_batch_var])

    def batchnorm2d_backward(
        self,
        grad_output: np.ndarray,
        input_data: np.ndarray,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        gamma: np.ndarray,
        eps: float = 1e-5,
        affine: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """BatchNorm2d backward pass"""
        if 'batchnorm2d-backward' not in self.shaders:
            return self._batchnorm2d_backward_cpu(grad_output, input_data, batch_mean, batch_var, gamma, eps, affine)

        grad_output = np.asarray(grad_output, dtype=np.float32)
        input_data = np.asarray(input_data, dtype=np.float32)

        batch_size, num_features, height, width = input_data.shape
        grad_input_size = batch_size * num_features * height * width * 4

        # Allocate buffers
        bufs = [
            self._acquire_buffer(grad_output.nbytes),  # grad_output
            self._acquire_buffer(input_data.nbytes),   # input
            self._acquire_buffer(grad_input_size),     # grad_input
            self._acquire_buffer(num_features * 4),    # batch_mean
            self._acquire_buffer(num_features * 4),    # batch_var
            self._acquire_buffer(gamma.nbytes),        # gamma
            self._acquire_buffer(num_features * 4),    # grad_gamma
            self._acquire_buffer(num_features * 4)     # grad_beta
        ]

        try:
            # Upload
            self._upload_buffer(bufs[0], grad_output.flatten())
            self._upload_buffer(bufs[1], input_data.flatten())
            self._upload_buffer(bufs[2], np.zeros(batch_size * num_features * height * width, dtype=np.float32))
            self._upload_buffer(bufs[3], batch_mean)
            self._upload_buffer(bufs[4], batch_var)
            self._upload_buffer(bufs[5], gamma)
            self._upload_buffer(bufs[6], np.zeros(num_features, dtype=np.float32))
            self._upload_buffer(bufs[7], np.zeros(num_features, dtype=np.float32))

            # Pipeline
            pipeline, pipeline_layout, _ = self.pipelines.get_or_create_pipeline('batchnorm2d-backward', 8, push_constant_size=24)

            # Descriptor set
            handles = [self._get_buffer_handle(b) for b in bufs]
            descriptor_set = self.pipelines.get_cached_descriptor_set('batchnorm2d-backward', [
                (handles[0], grad_output.nbytes), (handles[1], input_data.nbytes),
                (handles[2], grad_input_size), (handles[3], num_features * 4),
                (handles[4], num_features * 4), (handles[5], gamma.nbytes),
                (handles[6], num_features * 4), (handles[7], num_features * 4)
            ])

            # Push constants
            push_data = struct.pack('IIIIfI', batch_size, num_features, height, width, eps, 1 if affine else 0)

            # Dispatch
            self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, (num_features + 255) // 256, push_data)

            # Download
            grad_input = self._download_buffer(bufs[2], grad_input_size, np.float32).reshape(batch_size, num_features, height, width)
            grad_gamma = self._download_buffer(bufs[6], num_features * 4, np.float32) if affine else None
            grad_beta = self._download_buffer(bufs[7], num_features * 4, np.float32) if affine else None

            return grad_input, grad_gamma, grad_beta

        finally:
            self._release_buffers(bufs)

    def _batchnorm2d_forward_cpu(self, input_data, gamma, beta, running_mean, running_var, batch_mean, batch_var, eps, momentum, training, affine):
        """CPU fallback"""
        if training:
            mean = np.mean(input_data, axis=(0, 2, 3))
            var = np.var(input_data, axis=(0, 2, 3))
            batch_mean[:] = mean
            batch_var[:] = var
            if running_mean is not None:
                running_mean[:] = momentum * mean + (1 - momentum) * running_mean
                running_var[:] = momentum * var + (1 - momentum) * running_var
        else:
            mean = running_mean
            var = running_var

        mean_exp = mean[None, :, None, None]
        var_exp = var[None, :, None, None]
        normalized = (input_data - mean_exp) / np.sqrt(var_exp + eps)

        if affine:
            gamma_exp = gamma[None, :, None, None]
            beta_exp = beta[None, :, None, None]
            return (gamma_exp * normalized + beta_exp).astype(np.float32)
        return normalized.astype(np.float32)

    def _batchnorm2d_backward_cpu(self, grad_output, input_data, batch_mean, batch_var, gamma, eps, affine):
        """CPU fallback"""
        batch_size, num_features, height, width = input_data.shape
        n = batch_size * height * width

        mean_exp = batch_mean[None, :, None, None]
        var_exp = batch_var[None, :, None, None]
        inv_std = 1.0 / np.sqrt(var_exp + eps)
        x_norm = (input_data - mean_exp) * inv_std

        grad_gamma = np.sum(grad_output * x_norm, axis=(0, 2, 3)) if affine else None
        grad_beta = np.sum(grad_output, axis=(0, 2, 3)) if affine else None

        grad_sum = np.sum(grad_output, axis=(0, 2, 3), keepdims=True)
        grad_dot = np.sum(grad_output * x_norm, axis=(0, 2, 3), keepdims=True)

        gamma_exp = gamma[None, :, None, None]
        grad_input = gamma_exp * inv_std / n * (n * grad_output - grad_sum - x_norm * grad_dot)

        return grad_input.astype(np.float32), grad_gamma, grad_beta
