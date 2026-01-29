"""
Feedforward Neural Network operations for Vulkan backend.
GPU-accelerated FNN operations: activations, layer normalization, linear layers, dropout.

Performance hierarchy:
1. Vulkan GPU shader (fastest)
2. Numba JIT (fast CPU fallback)
3. Pure numpy (baseline fallback)
"""

import numpy as np
import struct
from typing import Optional, List
from .base import VULKAN_AVAILABLE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

if VULKAN_AVAILABLE:
    from vulkan import *

# Try to import numba-accelerated operations for CPU fallback
try:
    from ..utils.numba_ops import (
        layernorm as numba_layernorm,
        softmax as numba_softmax,
        linear as numba_linear,
        gelu as numba_gelu,
        silu as numba_silu,
        relu as numba_relu,
        gcu as numba_gcu,
        roswish as numba_roswish,
        swiglu as numba_swiglu,
        NUMBA_AVAILABLE,
    )
except ImportError:
    NUMBA_AVAILABLE = False
    numba_layernorm = None
    numba_softmax = None
    numba_linear = None
    numba_gelu = None
    numba_silu = None
    numba_relu = None
    numba_gcu = None
    numba_roswish = None
    numba_swiglu = None

# Import buffer pool for GPU buffer reuse
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


class VulkanFNN:
    """FNN operations: activations, layer normalization, linear layers, dropout"""

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
                # Use per-instance pool instead of global pool
                # This avoids issues with stale device references across tests
                self._pool = BufferPool(self.core)
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Buffer pool init failed: {e}")
                pass  # Pool initialization failed, will use direct allocation
        return self._pool
    
    def gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        GEMM: C = A @ B
        A: (M, K), B: (K, N) -> C: (M, N)
        Uses gemm-mnk.glsl
        """
        if 'gemm-mnk' not in self.shaders:
            # CPU fallback
            return A @ B

        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        M, K = A.shape
        K2, N = B.shape
        assert K == K2

        # Bytes
        A_bytes = M * K * 4
        B_bytes = K * N * 4
        C_bytes = M * N * 4

        # Allocate buffers
        buf_A = self.core._acquire_buffer(A_bytes)
        buf_B = self.core._acquire_buffer(B_bytes)
        buf_C = self.core._acquire_buffer(C_bytes)

        try:
            self.core._upload_buffer(buf_A, A.flatten())
            self.core._upload_buffer(buf_B, B.flatten())

            pipeline, layout, _ = self.pipelines.get_or_create_pipeline(
                'gemm_mnk', 3, push_constant_size=12
            )

            A_handle = self.core._get_buffer_handle(buf_A)
            B_handle = self.core._get_buffer_handle(buf_B)
            C_handle = self.core._get_buffer_handle(buf_C)

            desc = self.pipelines.get_cached_descriptor_set(
                'gemm-mnk',
                [
                    (A_handle, A_bytes),
                    (B_handle, B_bytes),
                    (C_handle, C_bytes),
                ]
            )

            push = struct.pack('3I', M, K, N)

            group_x = (N + 15) // 16
            group_y = (M + 15) // 16
            group_z = 1

            self.core._dispatch_compute(
                pipeline, layout, desc,
                group_x, push, group_y, group_z
            )

            C_flat = self.core._download_buffer(buf_C, C_bytes, np.float32)
            return C_flat.reshape(M, N)

        finally:
            self.core._release_buffers([buf_A, buf_B, buf_C])

    def _acquire_buffer(self, size: int, usage: int = None) -> 'PooledBuffer':
        """
        Acquire a buffer from the pool or create directly if pool unavailable.

        Returns:
            PooledBuffer if pool available, or tuple (handle, memory) otherwise
        """
        if usage is None:
            usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

        pool = self.buffer_pool
        if pool is not None:
            return pool.acquire(size, usage)
        else:
            # Fallback to direct allocation (returns tuple)
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
                # Legacy tuple (handle, memory)
                handle, memory = buf
                vkDestroyBuffer(self.core.device, handle, None)
                vkFreeMemory(self.core.device, memory, None)

    def _is_vma_buffer(self, buf) -> bool:
        """Check if buffer is a VMA-allocated buffer"""
        return VMABuffer is not None and isinstance(buf, VMABuffer)

    def _upload_buffer(self, buf, data: np.ndarray):
        """Upload data to buffer, handling VMA and direct buffers appropriately"""
        if self._is_vma_buffer(buf):
            # Use VMA's memory mapping for VMA buffers
            pool = self.buffer_pool
            if pool is not None and isinstance(pool, VMABufferPool):
                pool.upload_data(buf, data)
                return
        # Direct buffer or PooledBuffer - use core's upload
        self.core._upload_buffer(buf.handle, buf.memory, data)

    def _download_buffer(self, buf, size: int, dtype=np.float32) -> np.ndarray:
        """Download data from buffer, handling VMA and direct buffers appropriately"""
        if self._is_vma_buffer(buf):
            # Use VMA's memory mapping for VMA buffers
            pool = self.buffer_pool
            if pool is not None and isinstance(pool, VMABufferPool):
                return pool.download_data(buf, size, dtype)
        # Direct buffer or PooledBuffer - use core's download
        return self.core._download_buffer(buf.memory, size, dtype)

    def _get_buffer_handle(self, buf):
        """Get Vulkan-compatible buffer handle"""
        if self._is_vma_buffer(buf):
            return buf.get_vulkan_handle()
        return buf.handle

    def activation_relu(self, input_data):
        """Apply ReLU activation: max(0, x)"""
        # Check if shader is available
        if 'activation-relu' not in self.shaders:
            # CPU fallback (numba if available)
            if NUMBA_AVAILABLE and numba_relu is not None:
                return numba_relu(input_data.astype(np.float32))
            return np.maximum(0, input_data).astype(np.float32)

        data = input_data.astype(np.float32).flatten()
        total_elements = len(data)

        # Acquire buffers from pool
        buf_in = self._acquire_buffer(data.nbytes)
        buf_out = self._acquire_buffer(data.nbytes)

        # Upload data (uses VMA memory mapping for VMA buffers)
        self._upload_buffer(buf_in, data)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-relu', 2, push_constant_size=4
        )

        # Get buffer handles (converts VMA handles to vulkan-compatible handles)
        in_handle = self._get_buffer_handle(buf_in)
        out_handle = self._get_buffer_handle(buf_out)

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-relu',
            [(in_handle, data.nbytes), (out_handle, data.nbytes)]
        )

        # Pack push constants
        push_constants = struct.pack('I', total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )

        # Download results (uses VMA memory mapping for VMA buffers)
        result = self._download_buffer(buf_out, data.nbytes, np.float32)
        result = result[:total_elements]

        # Release buffers back to pool
        self._release_buffers([buf_in, buf_out])

        return result.reshape(input_data.shape) if input_data.ndim > 1 else result
    
    def activation_gelu(self, input_data):
        """Apply GELU activation"""
        # Check if shader is available
        if 'activation-gelu' not in self.shaders:
            # CPU fallback (numba if available)
            if NUMBA_AVAILABLE and numba_gelu is not None:
                return numba_gelu(input_data.astype(np.float32))
            sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
            coeff = 0.044715
            return 0.5 * input_data * (1 + np.tanh(sqrt_2_over_pi * (input_data + coeff * input_data ** 3)))

        data = input_data.astype(np.float32).flatten()
        total_elements = len(data)

        # Acquire buffers from pool
        buf_in = self._acquire_buffer(data.nbytes)
        buf_out = self._acquire_buffer(data.nbytes)

        # Upload data
        self._upload_buffer(buf_in, data)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-gelu', 2, push_constant_size=4
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-gelu',
            [(self._get_buffer_handle(buf_in), data.nbytes), (self._get_buffer_handle(buf_out), data.nbytes)]
        )

        # Pack push constants
        push_constants = struct.pack('I', total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )

        # Download results
        result = self._download_buffer(buf_out, data.nbytes, np.float32)
        result = result[:total_elements]

        # Check for NaN/Inf and fallback to CPU if needed
        if np.isnan(result).any() or np.isinf(result).any():
            # CPU fallback
            sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
            coeff = 0.044715
            result = 0.5 * input_data * (1 + np.tanh(sqrt_2_over_pi * (input_data + coeff * input_data ** 3)))
            result = result.astype(np.float32).flatten()

        # Release buffers back to pool
        self._release_buffers([buf_in, buf_out])

        return result.reshape(input_data.shape) if input_data.ndim > 1 else result
    
    def activation_silu(self, input_data):
        """Apply SiLU (Swish) activation: x * sigmoid(x)"""
        # Check if shader is available
        if 'activation-silu' not in self.shaders:
            # CPU fallback (numba if available)
            if NUMBA_AVAILABLE and numba_silu is not None:
                return numba_silu(input_data.astype(np.float32))
            return input_data / (1.0 + np.exp(-input_data))

        data = input_data.astype(np.float32).flatten()
        total_elements = len(data)

        # Acquire buffers from pool
        buf_in = self._acquire_buffer(data.nbytes)
        buf_out = self._acquire_buffer(data.nbytes)

        # Upload data
        self._upload_buffer(buf_in, data)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-silu', 2, push_constant_size=4
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-silu',
            [(self._get_buffer_handle(buf_in), data.nbytes), (self._get_buffer_handle(buf_out), data.nbytes)]
        )

        # Pack push constants
        push_constants = struct.pack('I', total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )

        # Download results
        result = self._download_buffer(buf_out, data.nbytes, np.float32)
        result = result[:total_elements]

        # Release buffers back to pool
        self._release_buffers([buf_in, buf_out])

        return result.reshape(input_data.shape) if input_data.ndim > 1 else result

    def activation_gcu(self, input_data):
        """Apply GCU (Growing Cosine Unit) activation: x * cos(x)"""
        # Check if shader is available
        if 'activation-gcu' not in self.shaders:
            # CPU fallback (numba if available)
            if NUMBA_AVAILABLE and numba_gcu is not None:
                return numba_gcu(input_data.astype(np.float32))
            return input_data * np.cos(input_data)

        data = input_data.astype(np.float32).flatten()
        total_elements = len(data)

        # Acquire buffers from pool
        buf_in = self._acquire_buffer(data.nbytes)
        buf_out = self._acquire_buffer(data.nbytes)

        # Upload data
        self._upload_buffer(buf_in, data)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-gcu', 2, push_constant_size=4
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-gcu',
            [(self._get_buffer_handle(buf_in), data.nbytes), (self._get_buffer_handle(buf_out), data.nbytes)]
        )

        # Pack push constants
        push_constants = struct.pack('I', total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )

        # Download results
        result = self._download_buffer(buf_out, data.nbytes, np.float32)
        result = result[:total_elements]

        # Release buffers back to pool
        self._release_buffers([buf_in, buf_out])

        return result.reshape(input_data.shape) if input_data.ndim > 1 else result

    def activation_roswish(self, input_data, alpha=1.0, beta=1.0):
        """
        Apply RoSwish activation: (x + α) * sigmoid(β * x) - 0.5 * α

        Args:
            input_data: Input array
            alpha: Rotation parameter (learnable, default 1.0)
            beta: Gating parameter (learnable, default 1.0)
        """
        # Check if shader is available
        if 'activation-roswish' not in self.shaders:
            # CPU fallback (numba if available)
            if NUMBA_AVAILABLE and numba_roswish is not None:
                return numba_roswish(input_data.astype(np.float32), alpha, beta)
            sigmoid_bx = 1.0 / (1.0 + np.exp(-beta * input_data))
            return (input_data + alpha) * sigmoid_bx - 0.5 * alpha

        data = input_data.astype(np.float32).flatten()
        total_elements = len(data)

        # Acquire buffers from pool
        buf_in = self._acquire_buffer(data.nbytes)
        buf_out = self._acquire_buffer(data.nbytes)

        # Upload data
        self._upload_buffer(buf_in, data)

        # Get or create pipeline (12 bytes push constants: uint + 2 floats)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-roswish', 2, push_constant_size=12
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-roswish',
            [(self._get_buffer_handle(buf_in), data.nbytes), (self._get_buffer_handle(buf_out), data.nbytes)]
        )

        # Pack push constants: total_elements (uint32), alpha (float32), beta (float32)
        push_constants = struct.pack('Iff', total_elements, float(alpha), float(beta))

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )

        # Download results
        result = self._download_buffer(buf_out, data.nbytes, np.float32)
        result = result[:total_elements]

        # Release buffers back to pool
        self._release_buffers([buf_in, buf_out])

        return result.reshape(input_data.shape) if input_data.ndim > 1 else result

    def activation_swiglu(self, input_data):
        """
        Apply SwiGLU (Swish-Gated Linear Unit) activation: x1 * silu(x2)

        Input is split along the last dimension into [x1, x2].
        Output = x1 * silu(x2) where silu(x) = x * sigmoid(x)

        Args:
            input_data: Input array of shape (..., 2*hidden_dim)

        Returns:
            Output array of shape (..., hidden_dim)
        """
        # Check if shader is available
        if 'activation-swiglu' not in self.shaders:
            # CPU fallback (numba if available)
            if NUMBA_AVAILABLE and numba_swiglu is not None:
                return numba_swiglu(input_data.astype(np.float32))
            # Pure numpy fallback
            hidden_dim = input_data.shape[-1] // 2
            x1 = input_data[..., :hidden_dim]
            x2 = input_data[..., hidden_dim:]
            sigmoid_x2 = 1.0 / (1.0 + np.exp(-x2))
            silu_x2 = x2 * sigmoid_x2
            return x1 * silu_x2

        original_shape = input_data.shape
        data = input_data.astype(np.float32).reshape(-1, original_shape[-1])
        batch_size = data.shape[0]
        input_dim = data.shape[1]
        hidden_dim = input_dim // 2
        output_elements = batch_size * hidden_dim

        data_flat = data.flatten()

        # Acquire buffers from pool
        buf_in = self._acquire_buffer(data_flat.nbytes)
        buf_out = self._acquire_buffer(output_elements * 4)  # float32 = 4 bytes

        # Upload data
        self._upload_buffer(buf_in, data_flat)

        # Get or create pipeline (8 bytes push constants: 2 uints)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-swiglu', 2, push_constant_size=8
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-swiglu',
            [(self._get_buffer_handle(buf_in), data_flat.nbytes),
             (self._get_buffer_handle(buf_out), output_elements * 4)]
        )

        # Pack push constants: output_elements (uint32), hidden_dim (uint32)
        push_constants = struct.pack('II', output_elements, hidden_dim)

        # Dispatch
        workgroups = (output_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )

        # Download results
        result = self._download_buffer(buf_out, output_elements * 4, np.float32)
        result = result[:output_elements]

        # Release buffers back to pool
        self._release_buffers([buf_in, buf_out])

        # Reshape to match expected output shape
        output_shape = original_shape[:-1] + (hidden_dim,)
        return result.reshape(output_shape)

    def activation_softmax(self, input_data, axis=-1):
        """
        Apply softmax activation: exp(x) / sum(exp(x))

        Args:
            input_data: Input array
            axis: Axis along which to compute softmax (default: -1)

        Returns:
            Softmax probabilities
        """
        # Check if shader is available
        if 'activation-softmax' not in self.shaders:
            # CPU fallback (numba if available)
            if NUMBA_AVAILABLE and numba_softmax is not None:
                return numba_softmax(input_data.astype(np.float32))
            exp_x = np.exp(input_data - np.max(input_data, axis=axis, keepdims=True))
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

        data = input_data.astype(np.float32)
        original_shape = data.shape

        # Handle different input shapes - shader expects (batch, seq_len, features)
        if data.ndim == 1:
            batch_size, seq_len, features = 1, 1, len(data)
            data = data.reshape(1, 1, -1)
        elif data.ndim == 2:
            batch_size, seq_len, features = data.shape[0], 1, data.shape[1]
            data = data.reshape(data.shape[0], 1, -1)
        else:
            batch_size, seq_len, features = data.shape

        data_flat = data.flatten()

        # Acquire buffers from pool - shader needs 4 buffers: input, output, max_vals, sum_exp
        buf_in = self._acquire_buffer(data_flat.nbytes)
        buf_out = self._acquire_buffer(data_flat.nbytes)
        buf_max = self._acquire_buffer(batch_size * seq_len * 4)
        buf_sum = self._acquire_buffer(batch_size * seq_len * 4)

        # Upload data
        self._upload_buffer(buf_in, data_flat)

        # Get or create pipeline - 4 buffers, 24 bytes push constants (5 uints + padding)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-softmax', 4, push_constant_size=24
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-softmax',
            [
                (self._get_buffer_handle(buf_in), data_flat.nbytes),
                (self._get_buffer_handle(buf_out), data_flat.nbytes),
                (self._get_buffer_handle(buf_max), batch_size * seq_len * 4),
                (self._get_buffer_handle(buf_sum), batch_size * seq_len * 4)
            ]
        )

        # Pass 1: Compute max for numerical stability
        push_constants = struct.pack('IIIII', batch_size, seq_len, features, 0, features)
        workgroups = ((batch_size * seq_len) + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )

        # Pass 2: Compute sum of exponentials
        push_constants = struct.pack('IIIII', batch_size, seq_len, features, 1, features)
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )

        # Pass 3: Normalize
        push_constants = struct.pack('IIIII', batch_size, seq_len, features, 2, features)
        workgroups = (len(data_flat) + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )

        # Download results
        result = self._download_buffer(buf_out, data_flat.nbytes, np.float32)
        result = result[:len(data_flat)].reshape(original_shape)

        # Release buffers back to pool
        self._release_buffers([buf_in, buf_out, buf_max, buf_sum])

        return result
    
    def xavier_init(self, input_dim: int, output_dim: int, seed: int = 42) -> np.ndarray:
        """
        GPU-accelerated Xavier initialization

        Generates weights from normal distribution scaled by sqrt(2.0 / input_dim)
        Uses shader: fnn-xavier-init.glsl

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            seed: Random seed for reproducibility

        Returns:
            Weight matrix (output_dim, input_dim) with Xavier initialization
        """
        # Check if shader is available
        if 'fnn-xavier-init' not in self.shaders:
            # CPU fallback
            scale = np.sqrt(2.0 / input_dim)
            return np.random.default_rng(seed).normal(0, scale, (output_dim, input_dim)).astype(np.float32)

        scale = np.sqrt(2.0 / input_dim)
        weights_flat = np.zeros(input_dim * output_dim, dtype=np.float32)

        # Acquire buffer from pool
        buf_weights = self._acquire_buffer(weights_flat.nbytes)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fnn-xavier-init', 1, push_constant_size=16
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'fnn-xavier-init',
            [(self._get_buffer_handle(buf_weights), weights_flat.nbytes)]
        )

        # Pack push constants: input_dim, output_dim, scale, seed
        push_constants = struct.pack('IIfI', input_dim, output_dim, scale, seed)

        # Dispatch: 2D workgroups (one thread per weight)
        workgroups_x = (input_dim + 15) // 16
        workgroups_y = (output_dim + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y, 1
        )

        # Download results
        result = self._download_buffer(buf_weights, weights_flat.nbytes, np.float32)
        result = result[:input_dim * output_dim]

        # Release buffer back to pool
        self._release_buffers([buf_weights])

        return result.reshape(output_dim, input_dim)
    
    def activation_gelu_backward(self, grad_output, input_data):
        """
        GPU-accelerated GELU backward pass

        Args:
            grad_output: Gradient from next layer (same shape as input_data)
            input_data: Input to GELU (for computing derivative)

        Returns:
            Gradient w.r.t. input
        """
        grad_out = grad_output.astype(np.float32).flatten()
        input_flat = input_data.astype(np.float32).flatten()
        total_elements = len(input_flat)

        if len(grad_out) != total_elements:
            raise ValueError(f"grad_output size {len(grad_out)} != input_data size {total_elements}")

        # Check if shader is available
        if 'activation-gelu-backward' not in self.shaders:
            # CPU fallback (vectorized)
            sqrt_2_over_pi = 0.7978845608028654
            coeff = 0.044715
            x = input_flat
            x_cubed = x * x * x
            z = sqrt_2_over_pi * (x + coeff * x_cubed)
            tanh_z = np.tanh(z)
            sech_sq = 1.0 / (np.cosh(z) ** 2)
            dz_dx = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x)
            gelu_grad = 0.5 * (1.0 + tanh_z + x * sech_sq * dz_dx)
            grad_in = grad_out * gelu_grad
            return grad_in.reshape(input_data.shape)

        # Acquire buffers from pool
        buf_grad_out = self._acquire_buffer(grad_out.nbytes)
        buf_input = self._acquire_buffer(input_flat.nbytes)
        buf_grad_in = self._acquire_buffer(input_flat.nbytes)

        # Upload data
        self._upload_buffer(buf_grad_out, grad_out)
        self._upload_buffer(buf_input, input_flat)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-gelu-backward', 3, push_constant_size=4
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-gelu-backward',
            [
                (self._get_buffer_handle(buf_grad_out), grad_out.nbytes),
                (self._get_buffer_handle(buf_input), input_flat.nbytes),
                (self._get_buffer_handle(buf_grad_in), input_flat.nbytes)
            ]
        )

        # Pack push constants
        push_constants = struct.pack('I', total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )

        # Download results
        result = self._download_buffer(buf_grad_in, input_flat.nbytes, np.float32)
        result = result[:total_elements].reshape(input_data.shape)

        # Release buffers back to pool
        self._release_buffers([buf_grad_out, buf_input, buf_grad_in])

        return result

    def activation_relu_backward(self, grad_output, input_data):
        """
        GPU-accelerated ReLU backward pass

        Args:
            grad_output: Gradient from next layer (same shape as input_data)
            input_data: Input to ReLU (for computing derivative)

        Returns:
            Gradient w.r.t. input
        """
        grad_out = grad_output.astype(np.float32).flatten()
        input_flat = input_data.astype(np.float32).flatten()
        total_elements = len(input_flat)

        if len(grad_out) != total_elements:
            raise ValueError(f"grad_output size {len(grad_out)} != input_data size {total_elements}")

        # Check if shader is available
        if 'activation-relu-backward' not in self.shaders:
            # CPU fallback
            relu_grad = (input_flat > 0.0).astype(np.float32)
            grad_in = grad_out * relu_grad
            return grad_in.reshape(input_data.shape)

        # Acquire buffers from pool
        buf_grad_out = self._acquire_buffer(grad_out.nbytes)
        buf_input = self._acquire_buffer(input_flat.nbytes)
        buf_grad_in = self._acquire_buffer(input_flat.nbytes)

        # Upload data
        self._upload_buffer(buf_grad_out, grad_out)
        self._upload_buffer(buf_input, input_flat)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-relu-backward', 3, push_constant_size=4
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-relu-backward',
            [
                (self._get_buffer_handle(buf_grad_out), grad_out.nbytes),
                (self._get_buffer_handle(buf_input), input_flat.nbytes),
                (self._get_buffer_handle(buf_grad_in), input_flat.nbytes)
            ]
        )

        # Pack push constants
        push_constants = struct.pack('I', total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )

        # Download results
        result = self._download_buffer(buf_grad_in, input_flat.nbytes, np.float32)
        result = result[:total_elements].reshape(input_data.shape)

        # Release buffers back to pool
        self._release_buffers([buf_grad_out, buf_input, buf_grad_in])

        return result

    def activation_silu_backward(self, grad_output, input_data):
        """
        GPU-accelerated SiLU (Swish) backward pass

        SiLU(x) = x * sigmoid(x)
        d/dx SiLU(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

        Args:
            grad_output: Gradient from next layer (same shape as input_data)
            input_data: Input to SiLU (for computing derivative)

        Returns:
            Gradient w.r.t. input
        """
        grad_out = grad_output.astype(np.float32).flatten()
        input_flat = input_data.astype(np.float32).flatten()
        total_elements = len(input_flat)

        if len(grad_out) != total_elements:
            raise ValueError(f"grad_output size {len(grad_out)} != input_data size {total_elements}")

        # Check if shader is available
        if 'activation-silu-backward' not in self.shaders:
            # CPU fallback
            x = input_flat
            sigmoid_x = 1.0 / (1.0 + np.exp(-x))
            silu_grad = sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))
            grad_in = grad_out * silu_grad
            return grad_in.reshape(input_data.shape)

        # Acquire buffers from pool
        buf_grad_out = self._acquire_buffer(grad_out.nbytes)
        buf_input = self._acquire_buffer(input_flat.nbytes)
        buf_grad_in = self._acquire_buffer(input_flat.nbytes)

        # Upload data
        self._upload_buffer(buf_grad_out, grad_out)
        self._upload_buffer(buf_input, input_flat)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-silu-backward', 3, push_constant_size=4
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-silu-backward',
            [
                (self._get_buffer_handle(buf_grad_out), grad_out.nbytes),
                (self._get_buffer_handle(buf_input), input_flat.nbytes),
                (self._get_buffer_handle(buf_grad_in), input_flat.nbytes)
            ]
        )

        # Pack push constants
        push_constants = struct.pack('I', total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )

        # Download results
        result = self._download_buffer(buf_grad_in, input_flat.nbytes, np.float32)
        result = result[:total_elements].reshape(input_data.shape)

        # Release buffers back to pool
        self._release_buffers([buf_grad_out, buf_input, buf_grad_in])

        return result

    def activation_gcu_backward(self, grad_output, input_data):
        """
        GPU-accelerated GCU (Growing Cosine Unit) backward pass

        GCU(x) = x * cos(x)
        d/dx GCU(x) = cos(x) - x * sin(x)

        Args:
            grad_output: Gradient from next layer (same shape as input_data)
            input_data: Input to GCU (for computing derivative)

        Returns:
            Gradient w.r.t. input
        """
        grad_out = grad_output.astype(np.float32).flatten()
        input_flat = input_data.astype(np.float32).flatten()
        total_elements = len(input_flat)

        if len(grad_out) != total_elements:
            raise ValueError(f"grad_output size {len(grad_out)} != input_data size {total_elements}")

        # Check if shader is available
        if 'activation-gcu-backward' not in self.shaders:
            # CPU fallback
            x = input_flat
            gcu_grad = np.cos(x) - x * np.sin(x)
            grad_in = grad_out * gcu_grad
            return grad_in.reshape(input_data.shape)

        # Acquire buffers from pool
        buf_grad_out = self._acquire_buffer(grad_out.nbytes)
        buf_input = self._acquire_buffer(input_flat.nbytes)
        buf_grad_in = self._acquire_buffer(input_flat.nbytes)

        # Upload data
        self._upload_buffer(buf_grad_out, grad_out)
        self._upload_buffer(buf_input, input_flat)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-gcu-backward', 3, push_constant_size=4
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-gcu-backward',
            [
                (self._get_buffer_handle(buf_grad_out), grad_out.nbytes),
                (self._get_buffer_handle(buf_input), input_flat.nbytes),
                (self._get_buffer_handle(buf_grad_in), input_flat.nbytes)
            ]
        )

        # Pack push constants
        push_constants = struct.pack('I', total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )

        # Download results
        result = self._download_buffer(buf_grad_in, input_flat.nbytes, np.float32)
        result = result[:total_elements].reshape(input_data.shape)

        # Release buffers back to pool
        self._release_buffers([buf_grad_out, buf_input, buf_grad_in])

        return result

    def activation_roswish_backward(self, grad_output, input_data, alpha=1.0, beta=1.0):
        """
        GPU-accelerated RoSwish backward pass

        RoSwish(x) = (x + α) * sigmoid(β * x) - 0.5 * α
        d/dx RoSwish = sigmoid(β*x) + β*(x + α)*sigmoid(β*x)*(1 - sigmoid(β*x))

        Args:
            grad_output: Gradient from next layer (same shape as input_data)
            input_data: Input to RoSwish (for computing derivative)
            alpha: Rotation parameter
            beta: Gating parameter

        Returns:
            Gradient w.r.t. input
        """
        grad_out = grad_output.astype(np.float32).flatten()
        input_flat = input_data.astype(np.float32).flatten()
        total_elements = len(input_flat)

        if len(grad_out) != total_elements:
            raise ValueError(f"grad_output size {len(grad_out)} != input_data size {total_elements}")

        # Check if shader is available
        if 'activation-roswish-backward' not in self.shaders:
            # CPU fallback
            x = input_flat
            beta_x = beta * x
            # Numerically stable sigmoid
            sigmoid_bx = np.where(beta_x >= 0,
                                  1.0 / (1.0 + np.exp(-beta_x)),
                                  np.exp(beta_x) / (1.0 + np.exp(beta_x)))
            roswish_grad = sigmoid_bx + beta * (x + alpha) * sigmoid_bx * (1.0 - sigmoid_bx)
            grad_in = grad_out * roswish_grad
            return grad_in.reshape(input_data.shape)

        # Acquire buffers from pool
        buf_grad_out = self._acquire_buffer(grad_out.nbytes)
        buf_input = self._acquire_buffer(input_flat.nbytes)
        buf_grad_in = self._acquire_buffer(input_flat.nbytes)

        # Upload data
        self._upload_buffer(buf_grad_out, grad_out)
        self._upload_buffer(buf_input, input_flat)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-roswish-backward', 3, push_constant_size=12
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-roswish-backward',
            [
                (self._get_buffer_handle(buf_grad_out), grad_out.nbytes),
                (self._get_buffer_handle(buf_input), input_flat.nbytes),
                (self._get_buffer_handle(buf_grad_in), input_flat.nbytes)
            ]
        )

        # Pack push constants
        push_constants = struct.pack('Iff', total_elements, float(alpha), float(beta))

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )

        # Download results
        result = self._download_buffer(buf_grad_in, input_flat.nbytes, np.float32)
        result = result[:total_elements].reshape(input_data.shape)

        # Release buffers back to pool
        self._release_buffers([buf_grad_out, buf_input, buf_grad_in])

        return result

    def activation_swiglu_backward(self, grad_output, input_data):
        """
        GPU-accelerated SwiGLU backward pass

        Forward: output = x1 * silu(x2) where input = [x1, x2]
        d/dx1 = silu(x2)
        d/dx2 = x1 * d/dx2(silu(x2))

        Args:
            grad_output: Gradient from next layer (shape: batch * hidden_dim)
            input_data: Input to SwiGLU (shape: batch * 2*hidden_dim)

        Returns:
            Gradient w.r.t. input (shape: batch * 2*hidden_dim)
        """
        original_shape = input_data.shape
        grad_out_shape = grad_output.shape

        # Reshape for processing
        input_flat = input_data.astype(np.float32).reshape(-1, original_shape[-1])
        grad_out_flat = grad_output.astype(np.float32).reshape(-1, grad_out_shape[-1])

        batch_size = input_flat.shape[0]
        input_dim = input_flat.shape[1]
        hidden_dim = input_dim // 2
        output_elements = batch_size * hidden_dim

        if grad_out_flat.shape[0] != batch_size or grad_out_flat.shape[1] != hidden_dim:
            raise ValueError(f"grad_output shape mismatch: expected ({batch_size}, {hidden_dim}), got {grad_out_flat.shape}")

        # Check if shader is available
        if 'activation-swiglu-backward' not in self.shaders:
            # CPU fallback
            x1 = input_flat[:, :hidden_dim]
            x2 = input_flat[:, hidden_dim:]

            # Compute sigmoid(x2) numerically stable
            sigmoid_x2 = np.where(x2 >= 0,
                                  1.0 / (1.0 + np.exp(-x2)),
                                  np.exp(x2) / (1.0 + np.exp(x2)))
            silu_x2 = x2 * sigmoid_x2

            # Gradients
            grad_x1 = grad_out_flat * silu_x2
            silu_derivative = sigmoid_x2 * (1.0 + x2 * (1.0 - sigmoid_x2))
            grad_x2 = grad_out_flat * x1 * silu_derivative

            # Concatenate gradients
            grad_in = np.concatenate([grad_x1, grad_x2], axis=-1)
            return grad_in.reshape(original_shape)

        # Flatten for GPU processing
        input_data_flat = input_flat.flatten()
        grad_out_1d = grad_out_flat.flatten()

        # Acquire buffers from pool
        buf_grad_out = self._acquire_buffer(grad_out_1d.nbytes)
        buf_input = self._acquire_buffer(input_data_flat.nbytes)
        buf_grad_in = self._acquire_buffer(input_data_flat.nbytes)

        # Upload data
        self._upload_buffer(buf_grad_out, grad_out_1d)
        self._upload_buffer(buf_input, input_data_flat)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-swiglu-backward', 3, push_constant_size=8
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-swiglu-backward',
            [
                (self._get_buffer_handle(buf_grad_out), grad_out_1d.nbytes),
                (self._get_buffer_handle(buf_input), input_data_flat.nbytes),
                (self._get_buffer_handle(buf_grad_in), input_data_flat.nbytes)
            ]
        )

        # Pack push constants
        push_constants = struct.pack('II', output_elements, hidden_dim)

        # Dispatch
        workgroups = (output_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )

        # Download results
        result = self._download_buffer(buf_grad_in, input_data_flat.nbytes, np.float32)
        result = result[:len(input_data_flat)].reshape(original_shape)

        # Release buffers back to pool
        self._release_buffers([buf_grad_out, buf_input, buf_grad_in])

        return result

    def cross_entropy_backward(self, logits, targets):
        """
        GPU-accelerated cross-entropy backward pass (combined with softmax)

        Computes gradient of cross-entropy loss w.r.t. logits directly:
        grad = softmax(logits) - one_hot(targets)

        This is more numerically stable than computing softmax and
        cross-entropy gradients separately.

        Args:
            logits: Raw logits (batch_size, num_classes)
            targets: Target class indices (batch_size,) as integers or floats

        Returns:
            Gradient w.r.t. logits (batch_size, num_classes)
        """
        logits = logits.astype(np.float32)
        original_shape = logits.shape

        if logits.ndim == 1:
            batch_size, num_classes = 1, logits.shape[0]
            logits = logits.reshape(1, -1)
        else:
            batch_size, num_classes = logits.shape

        targets = np.asarray(targets).astype(np.float32).flatten()

        # Check if shader is available
        if 'cross-entropy-backward' not in self.shaders:
            # CPU fallback: softmax - one_hot
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Create one-hot encoding
            one_hot = np.zeros_like(softmax)
            for i in range(batch_size):
                target_idx = int(targets[i])
                if 0 <= target_idx < num_classes:
                    one_hot[i, target_idx] = 1.0

            grad = softmax - one_hot
            return grad.reshape(original_shape)

        # Flatten logits
        logits_flat = logits.flatten()
        grad_size = batch_size * num_classes * 4

        # Acquire buffers from pool
        buf_logits = self._acquire_buffer(logits_flat.nbytes)
        buf_targets = self._acquire_buffer(targets.nbytes)
        buf_grad = self._acquire_buffer(grad_size)

        # Upload data
        self._upload_buffer(buf_logits, logits_flat)
        self._upload_buffer(buf_targets, targets)

        # Get or create pipeline (3 buffers, push constants: 2 uints = 8 bytes)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'cross-entropy-backward', 3, push_constant_size=8
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'cross-entropy-backward',
            [
                (self._get_buffer_handle(buf_logits), logits_flat.nbytes),
                (self._get_buffer_handle(buf_targets), targets.nbytes),
                (self._get_buffer_handle(buf_grad), grad_size)
            ]
        )

        # Pack push constants: batch_size, num_classes
        push_constants = struct.pack('II', batch_size, num_classes)

        # Dispatch: one workgroup per batch element
        workgroups = batch_size
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )

        # Download results
        result = self._download_buffer(buf_grad, grad_size, np.float32)
        result = result[:batch_size * num_classes].reshape(original_shape)

        # Release buffers back to pool
        self._release_buffers([buf_logits, buf_targets, buf_grad])

        return result

    # ------------------------------------------------------------------
    # Layer normalization (GPU accelerated with 3-pass shader)
    # ------------------------------------------------------------------
    def layernorm(self, x: np.ndarray, gamma: np.ndarray = None, beta: np.ndarray = None, eps: float = 1e-5) -> np.ndarray:
        """
        GPU-accelerated LayerNorm using fnn-layernorm.glsl shader.
        Normalizes across the last dimension (features).

        3-pass algorithm:
        - Pass 0: Compute mean along feature dimension
        - Pass 1: Compute variance
        - Pass 2: Normalize and apply affine transformation
        """
        original_shape = x.shape
        features = original_shape[-1]

        # Default gamma/beta if not provided
        if gamma is None:
            gamma = np.ones(features, dtype=np.float32)
        if beta is None:
            beta = np.zeros(features, dtype=np.float32)

        # Check if shader is available
        if 'fnn-layernorm' not in self.shaders:
            # CPU fallback (numba-accelerated if available)
            if NUMBA_AVAILABLE and numba_layernorm is not None:
                return numba_layernorm(x, gamma, beta, eps)
            else:
                mean = x.mean(axis=-1, keepdims=True)
                var = x.var(axis=-1, keepdims=True)
                normalized = (x - mean) / np.sqrt(var + eps)
                return normalized * gamma + beta

        # Handle different input shapes: (features,) or (batch, features) or (batch, seq, features)
        if len(original_shape) == 1:
            batch_size, seq_len = 1, 1
            x = x.reshape(1, 1, features)
        elif len(original_shape) == 2:
            batch_size, features = original_shape
            seq_len = 1
            x = x.reshape(batch_size, 1, features)
        else:
            batch_size, seq_len, features = original_shape

        # Flatten to 1D arrays
        x_flat = x.astype(np.float32).flatten()
        gamma_flat = gamma.astype(np.float32).flatten()
        beta_flat = beta.astype(np.float32).flatten()

        total_positions = batch_size * seq_len
        total_elements = batch_size * seq_len * features

        # Acquire buffers from pool
        buf_input = self._acquire_buffer(x_flat.nbytes)
        buf_output = self._acquire_buffer(x_flat.nbytes)
        buf_gamma = self._acquire_buffer(gamma_flat.nbytes)
        buf_beta = self._acquire_buffer(beta_flat.nbytes)
        buf_mean = self._acquire_buffer(total_positions * 4)
        buf_var = self._acquire_buffer(total_positions * 4)

        # Upload data
        self._upload_buffer(buf_input, x_flat)
        self._upload_buffer(buf_gamma, gamma_flat)
        self._upload_buffer(buf_beta, beta_flat)

        # Get or create pipeline (6 buffers, push constants: 4 uints + 1 float = 20 bytes)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fnn-layernorm', 6, push_constant_size=20
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'fnn-layernorm',
            [
                (self._get_buffer_handle(buf_input), x_flat.nbytes),
                (self._get_buffer_handle(buf_output), x_flat.nbytes),
                (self._get_buffer_handle(buf_gamma), gamma_flat.nbytes),
                (self._get_buffer_handle(buf_beta), beta_flat.nbytes),
                (self._get_buffer_handle(buf_mean), total_positions * 4),
                (self._get_buffer_handle(buf_var), total_positions * 4),
            ]
        )

        # Run 3 passes
        for pass_type in range(3):
            # Push constants: batch_size, seq_len, features, eps, pass_type
            push_constants = struct.pack('IIIfI', batch_size, seq_len, features, eps, pass_type)

            # Workgroups depend on pass type
            if pass_type < 2:
                # Passes 0 and 1: one thread per position (batch * seq)
                workgroups = (total_positions + 255) // 256
            else:
                # Pass 2: one thread per element
                workgroups = (total_elements + 255) // 256

            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set,
                workgroups, push_constants
            )

        # Download result
        result = self._download_buffer(buf_output, x_flat.nbytes, np.float32)

        # Release buffers back to pool
        self._release_buffers([buf_input, buf_output, buf_gamma, buf_beta, buf_mean, buf_var])

        return result.reshape(original_shape)

    # ------------------------------------------------------------------
    # Linear projection (GPU accelerated)
    # ------------------------------------------------------------------
    def linear(self, x: np.ndarray, weights: np.ndarray, bias: Optional[np.ndarray] = None) -> np.ndarray:
        """
        GPU-accelerated linear projection using fnn-linear.glsl shader.
        output = x @ W^T + b

        Args:
            x: Input tensor (batch_seq, input_dim) or (batch, seq, input_dim)
            weights: Weight matrix (output_dim, input_dim)
            bias: Optional bias vector (output_dim,)

        Returns:
            Output tensor with same batch dimensions, last dim = output_dim
        """
        # Check if shader is available
        if 'fnn-linear' not in self.shaders:
            # CPU fallback (numba if available)
            if NUMBA_AVAILABLE and numba_linear is not None:
                return numba_linear(x.astype(np.float32), weights.astype(np.float32),
                                   bias.astype(np.float32) if bias is not None else None)
            out = np.matmul(x, weights.T)
            if bias is not None:
                out = out + bias
            return out

        original_shape = x.shape
        output_dim, input_dim = weights.shape

        # Reshape to 2D: (batch_seq, input_dim)
        if len(original_shape) > 2:
            batch_seq = int(np.prod(original_shape[:-1]))
            x_2d = x.reshape(batch_seq, input_dim)
        else:
            batch_seq = original_shape[0]
            x_2d = x

        # Flatten arrays
        x_flat = x_2d.astype(np.float32).flatten()
        w_flat = weights.astype(np.float32).flatten()

        # Output size
        output_size = batch_seq * output_dim * 4  # float32

        # Acquire buffers from pool
        buf_input = self._acquire_buffer(x_flat.nbytes)
        buf_weights = self._acquire_buffer(w_flat.nbytes)
        buf_output = self._acquire_buffer(output_size)

        # Handle bias
        has_bias = 1 if bias is not None else 0
        if bias is not None:
            bias_flat = bias.astype(np.float32).flatten()
            buf_bias = self._acquire_buffer(bias_flat.nbytes)
            self._upload_buffer(buf_bias, bias_flat)
        else:
            # Create dummy bias buffer (shader expects 4 buffers)
            buf_bias = self._acquire_buffer(4)
            bias_flat = None

        # Upload data
        self._upload_buffer(buf_input, x_flat)
        self._upload_buffer(buf_weights, w_flat)

        # Get or create pipeline (4 buffers, push constants: 4 uints = 16 bytes)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fnn-linear', 4, push_constant_size=16
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'fnn-linear',
            [
                (self._get_buffer_handle(buf_input), x_flat.nbytes),
                (self._get_buffer_handle(buf_weights), w_flat.nbytes),
                (self._get_buffer_handle(buf_bias), bias_flat.nbytes if bias_flat is not None else 4),
                (self._get_buffer_handle(buf_output), output_size),
            ]
        )

        # Push constants: batch_seq, input_dim, output_dim, has_bias
        push_constants = struct.pack('IIII', batch_seq, input_dim, output_dim, has_bias)

        # 2D dispatch: rows = batch_seq, cols = output_dim
        # Shader uses 16x16 workgroups
        workgroups_x = (output_dim + 15) // 16
        workgroups_y = (batch_seq + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y
        )

        # Download result
        result = self._download_buffer(buf_output, output_size, np.float32)

        # Release buffers back to pool
        self._release_buffers([buf_input, buf_weights, buf_bias, buf_output])

        # Reshape output to match input batch dimensions
        if len(original_shape) > 2:
            output_shape = original_shape[:-1] + (output_dim,)
            return result.reshape(output_shape)
        else:
            return result.reshape(batch_seq, output_dim)
    
    # ------------------------------------------------------------------
    # Linear backward pass
    # ------------------------------------------------------------------
    def linear_backward(
        self,
        grad_output: np.ndarray,
        x: np.ndarray,
        weights: np.ndarray,
        bias: Optional[np.ndarray] = None
    ) -> tuple:
        """
        Backward pass for linear layer using GEMM.

        Computes:
        - grad_input = grad_output @ weights        # (batch, in_features)
        - grad_weight = grad_output.T @ x           # (out_features, in_features)
        - grad_bias = sum(grad_output, axis=0)      # (out_features,)

        Args:
            grad_output: Gradient w.r.t. output (batch, out_features)
            x: Input (batch, in_features)
            weights: Weight matrix (out_features, in_features)
            bias: Optional bias (out_features,)

        Returns:
            (grad_input, grad_weight, grad_bias)
        """
        # Ensure arrays are float32
        grad_output = np.asarray(grad_output, dtype=np.float32)
        x = np.asarray(x, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)

        # Handle both 2D and 3D inputs
        grad_output_shape = grad_output.shape
        x_shape = x.shape

        # Flatten to 2D for GEMM
        if grad_output.ndim == 3:
            batch, seq, out_features = grad_output.shape
            grad_output_2d = grad_output.reshape(batch * seq, out_features)
            x_2d = x.reshape(batch * seq, x.shape[-1])
            in_features = x.shape[-1]
        else:
            grad_output_2d = grad_output
            x_2d = x
            batch, out_features = grad_output.shape
            _, in_features = x.shape

        # Decide whether to use GEMM or fallback shader/CPU
        # Use GEMM for larger problems (same heuristic as forward)
        use_gemm = (
            'gemm_mnk' in self.shaders and
            batch * in_features >= 4096
        )

        if use_gemm:
            # ============ GEMM-based backward ============

            # 1) grad_input = grad_output @ weights
            #    (batch*seq, out_features) @ (out_features, in_features) = (batch*seq, in_features)
            grad_input_2d = self.gemm(grad_output_2d, weights)

            # 2) grad_weight = grad_output.T @ x
            #    (out_features, batch*seq) @ (batch*seq, in_features) = (out_features, in_features)
            grad_weight = self.gemm(grad_output_2d.T.copy(), x_2d)

            # 3) grad_bias = sum over batch dimension
            grad_bias = np.sum(grad_output_2d, axis=0, dtype=np.float32) if bias is not None else None

            # Reshape grad_input back to original shape
            if grad_output.ndim == 3:
                grad_input = grad_input_2d.reshape(grad_output_shape[0], grad_output_shape[1], -1)
            else:
                grad_input = grad_input_2d

            return grad_input, grad_weight, grad_bias

        # ============ Fallback: use fnn-linear-backward shader or CPU ============
        if 'fnn-linear-backward' not in self.shaders:
            # CPU fallback (using 2D arrays)
            grad_input_2d = grad_output_2d @ weights  # (batch*seq, in_features)
            grad_weight = grad_output_2d.T @ x_2d  # (out_features, in_features)
            grad_bias = np.sum(grad_output_2d, axis=0) if bias is not None else None

            # Reshape grad_input back to original shape
            if grad_output.ndim == 3:
                grad_input = grad_input_2d.reshape(grad_output_shape[0], grad_output_shape[1], -1)
            else:
                grad_input = grad_input_2d

            return grad_input.astype(np.float32), grad_weight.astype(np.float32), grad_bias

        # GPU shader implementation (using 2D arrays)
        batch_seq, output_dim = grad_output_2d.shape
        _, input_dim = x_2d.shape

        # Flatten 2D arrays for shader
        grad_out_flat = grad_output_2d.astype(np.float32).flatten()
        x_flat = x_2d.astype(np.float32).flatten()
        w_flat = weights.astype(np.float32).flatten()

        # Output buffers sizes
        grad_input_size = batch_seq * input_dim * 4
        grad_weight_size = output_dim * input_dim * 4
        grad_bias_size = output_dim * 4

        # Acquire buffers from pool
        buf_grad_out = self._acquire_buffer(grad_out_flat.nbytes)
        buf_x = self._acquire_buffer(x_flat.nbytes)
        buf_w = self._acquire_buffer(w_flat.nbytes)
        buf_grad_in = self._acquire_buffer(grad_input_size)
        buf_grad_w = self._acquire_buffer(grad_weight_size)

        buffers_list = [buf_grad_out, buf_x, buf_w, buf_grad_in, buf_grad_w]
        buffers = [
            (self._get_buffer_handle(buf_grad_out), grad_out_flat.nbytes),
            (self._get_buffer_handle(buf_x), x_flat.nbytes),
            (self._get_buffer_handle(buf_w), w_flat.nbytes),
            (self._get_buffer_handle(buf_grad_in), grad_input_size),
            (self._get_buffer_handle(buf_grad_w), grad_weight_size),
        ]

        if bias is not None:
            buf_grad_b = self._acquire_buffer(grad_bias_size)
            buffers_list.append(buf_grad_b)
            buffers.append((self._get_buffer_handle(buf_grad_b), grad_bias_size))
        else:
            buf_grad_b = None

        # Upload data
        self._upload_buffer(buf_grad_out, grad_out_flat)
        self._upload_buffer(buf_x, x_flat)
        self._upload_buffer(buf_w, w_flat)
        # Initialize output buffers to zero
        self._upload_buffer(buf_grad_in, np.zeros(batch_seq * input_dim, dtype=np.float32))
        self._upload_buffer(buf_grad_w, np.zeros(output_dim * input_dim, dtype=np.float32))
        if bias is not None:
            self._upload_buffer(buf_grad_b, np.zeros(output_dim, dtype=np.float32))

        # Get or create pipeline
        num_bindings = 6 if bias is not None else 5
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fnn-linear-backward', num_bindings, push_constant_size=16
        )

        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(desc_layout, buffers)

        # Pass 0: Compute grad_input
        push_constants = struct.pack('IIII', batch_seq, input_dim, output_dim, 0)
        workgroups_x = (input_dim + 15) // 16
        workgroups_y = (batch_seq + 15) // 16
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y
        )

        # Pass 1: Compute grad_weight
        push_constants = struct.pack('IIII', batch_seq, input_dim, output_dim, 1)
        workgroups_x = (input_dim + 15) // 16
        workgroups_y = (output_dim + 15) // 16
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y
        )

        # Pass 2: Compute grad_bias (if bias exists)
        if bias is not None:
            push_constants = struct.pack('IIII', batch_seq, input_dim, output_dim, 2)
            workgroups = (output_dim + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set,
                workgroups, push_constants
            )

        # Download results
        grad_input_flat = self._download_buffer(buf_grad_in, grad_input_size, np.float32)
        grad_weight_flat = self._download_buffer(buf_grad_w, grad_weight_size, np.float32)
        if bias is not None:
            grad_bias_flat = self._download_buffer(buf_grad_b, grad_bias_size, np.float32)
        else:
            grad_bias_flat = None

        # Reshape
        grad_input_2d = grad_input_flat[:batch_seq * input_dim].reshape(batch_seq, input_dim)
        grad_weight = grad_weight_flat[:output_dim * input_dim].reshape(output_dim, input_dim)
        grad_bias = grad_bias_flat[:output_dim] if grad_bias_flat is not None else None

        # Reshape grad_input back to original shape
        if grad_output.ndim == 3:
            grad_input = grad_input_2d.reshape(grad_output_shape[0], grad_output_shape[1], -1)
        else:
            grad_input = grad_input_2d

        # Free descriptor set and release buffers
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        self._release_buffers(buffers_list)

        return grad_input, grad_weight, grad_bias

    # ------------------------------------------------------------------
    # LayerNorm backward pass (GPU accelerated)
    # ------------------------------------------------------------------
    def layernorm_backward(
        self,
        grad_output: np.ndarray,
        x: np.ndarray,
        gamma: np.ndarray,
        mean: np.ndarray = None,
        var: np.ndarray = None,
        eps: float = 1e-5
    ) -> tuple:
        """
        GPU-accelerated LayerNorm backward pass using fnn-layernorm-backward.glsl.

        Args:
            grad_output: Gradient w.r.t. output (same shape as input)
            x: Original input tensor
            gamma: Scale parameter
            mean: Mean from forward pass (if not provided, will be computed)
            var: Variance from forward pass (if not provided, will be computed)
            eps: Epsilon for numerical stability

        Returns:
            (grad_input, grad_gamma, grad_beta)
        """
        original_shape = x.shape
        features = original_shape[-1]

        # Compute mean/var if not provided
        if mean is None:
            mean = x.mean(axis=-1, keepdims=True)
        if var is None:
            var = x.var(axis=-1, keepdims=True)

        # Check if shader is available
        if 'fnn-layernorm-backward' not in self.shaders:
            # CPU fallback
            std = np.sqrt(var + eps)
            x_norm = (x - mean) / std

            # Gradients w.r.t. gamma and beta
            grad_gamma = np.sum(grad_output * x_norm, axis=tuple(range(len(original_shape) - 1)))
            grad_beta = np.sum(grad_output, axis=tuple(range(len(original_shape) - 1)))

            # Gradient w.r.t. input
            N = features
            dx_norm = grad_output * gamma

            dvar = np.sum(dx_norm * (x - mean) * (-0.5) * (var + eps) ** (-1.5), axis=-1, keepdims=True)
            dmean = np.sum(dx_norm * (-1.0 / std), axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (x - mean), axis=-1, keepdims=True)

            grad_input = dx_norm / std + dvar * 2.0 * (x - mean) / N + dmean / N

            return grad_input.astype(np.float32), grad_gamma.astype(np.float32), grad_beta.astype(np.float32)

        # Handle different input shapes
        if len(original_shape) == 1:
            batch_size, seq_len = 1, 1
            x = x.reshape(1, 1, features)
            grad_output = grad_output.reshape(1, 1, features)
        elif len(original_shape) == 2:
            batch_size, features = original_shape
            seq_len = 1
            x = x.reshape(batch_size, 1, features)
            grad_output = grad_output.reshape(batch_size, 1, features)
        else:
            batch_size, seq_len, features = original_shape

        # Flatten arrays
        grad_out_flat = grad_output.astype(np.float32).flatten()
        x_flat = x.astype(np.float32).flatten()
        gamma_flat = gamma.astype(np.float32).flatten()
        mean_flat = mean.astype(np.float32).flatten()
        var_flat = var.astype(np.float32).flatten()

        total_positions = batch_size * seq_len
        total_elements = batch_size * seq_len * features

        # Acquire buffers from pool
        buf_grad_out = self._acquire_buffer(grad_out_flat.nbytes)
        buf_input = self._acquire_buffer(x_flat.nbytes)
        buf_gamma = self._acquire_buffer(gamma_flat.nbytes)
        buf_mean = self._acquire_buffer(mean_flat.nbytes)
        buf_var = self._acquire_buffer(var_flat.nbytes)
        buf_grad_in = self._acquire_buffer(x_flat.nbytes)
        buf_grad_gamma = self._acquire_buffer(gamma_flat.nbytes)
        buf_grad_beta = self._acquire_buffer(gamma_flat.nbytes)

        buffers_list = [buf_grad_out, buf_input, buf_gamma, buf_mean, buf_var, buf_grad_in, buf_grad_gamma, buf_grad_beta]

        # Upload data
        self._upload_buffer(buf_grad_out, grad_out_flat)
        self._upload_buffer(buf_input, x_flat)
        self._upload_buffer(buf_gamma, gamma_flat)
        self._upload_buffer(buf_mean, mean_flat)
        self._upload_buffer(buf_var, var_flat)

        # Initialize grad buffers to zero
        self._upload_buffer(buf_grad_in, np.zeros(total_elements, dtype=np.float32))
        self._upload_buffer(buf_grad_gamma, np.zeros(features, dtype=np.float32))
        self._upload_buffer(buf_grad_beta, np.zeros(features, dtype=np.float32))

        # Get or create pipeline (8 buffers, push constants: 4 uints + 1 float = 20 bytes)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fnn-layernorm-backward', 8, push_constant_size=20
        )

        buffers = [
            (self._get_buffer_handle(buf_grad_out), grad_out_flat.nbytes),
            (self._get_buffer_handle(buf_input), x_flat.nbytes),
            (self._get_buffer_handle(buf_gamma), gamma_flat.nbytes),
            (self._get_buffer_handle(buf_mean), mean_flat.nbytes),
            (self._get_buffer_handle(buf_var), var_flat.nbytes),
            (self._get_buffer_handle(buf_grad_in), x_flat.nbytes),
            (self._get_buffer_handle(buf_grad_gamma), gamma_flat.nbytes),
            (self._get_buffer_handle(buf_grad_beta), gamma_flat.nbytes),
        ]

        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(desc_layout, buffers)

        # Pass 0: Compute intermediate sums
        push_constants = struct.pack('IIIfI', batch_size, seq_len, features, eps, 0)
        workgroups = (total_positions + 255) // 256
        self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)

        # Pass 1: Compute grad_input
        push_constants = struct.pack('IIIfI', batch_size, seq_len, features, eps, 1)
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)

        # Pass 2: Compute grad_gamma and grad_beta
        push_constants = struct.pack('IIIfI', batch_size, seq_len, features, eps, 2)
        workgroups = (features + 255) // 256
        self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)

        # Download results
        grad_input = self._download_buffer(buf_grad_in, x_flat.nbytes, np.float32)
        grad_gamma = self._download_buffer(buf_grad_gamma, gamma_flat.nbytes, np.float32)
        grad_beta = self._download_buffer(buf_grad_beta, gamma_flat.nbytes, np.float32)

        # Free descriptor set and release buffers
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        self._release_buffers(buffers_list)

        return grad_input.reshape(original_shape), grad_gamma[:features], grad_beta[:features]

    # ------------------------------------------------------------------
    # Softmax backward pass (GPU accelerated)
    # ------------------------------------------------------------------
    def softmax_backward(
        self,
        grad_output: np.ndarray,
        softmax_output: np.ndarray,
        dim: int = -1
    ) -> np.ndarray:
        """
        GPU-accelerated softmax backward pass using activation-softmax-backward.glsl.

        Args:
            grad_output: Gradient w.r.t. softmax output
            softmax_output: Output from forward softmax pass
            dim: Dimension along which softmax was applied

        Returns:
            Gradient w.r.t. input (pre-softmax logits)
        """
        original_shape = grad_output.shape

        # Check if shader is available
        if 'activation-softmax-backward' not in self.shaders:
            # CPU fallback: grad_input = s * (grad_output - sum(grad_output * s, dim))
            sum_term = np.sum(grad_output * softmax_output, axis=dim, keepdims=True)
            grad_input = softmax_output * (grad_output - sum_term)
            return grad_input.astype(np.float32)

        # Handle different input shapes
        if len(original_shape) == 1:
            batch_size, seq_len, num_classes = 1, 1, original_shape[0]
        elif len(original_shape) == 2:
            batch_size, num_classes = original_shape
            seq_len = 1
        else:
            # Assume (batch, seq, classes) or flatten all but last dim
            num_classes = original_shape[-1]
            batch_size = int(np.prod(original_shape[:-1]))
            seq_len = 1

        # Flatten arrays
        grad_out_flat = grad_output.astype(np.float32).flatten()
        softmax_flat = softmax_output.astype(np.float32).flatten()

        total_rows = batch_size * seq_len

        # Acquire buffers from pool
        buf_grad_out = self._acquire_buffer(grad_out_flat.nbytes)
        buf_softmax = self._acquire_buffer(softmax_flat.nbytes)
        buf_grad_in = self._acquire_buffer(grad_out_flat.nbytes)

        # Upload data
        self._upload_buffer(buf_grad_out, grad_out_flat)
        self._upload_buffer(buf_softmax, softmax_flat)

        # Get or create pipeline (3 buffers, push constants: 3 uints = 12 bytes)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-softmax-backward', 3, push_constant_size=12
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-softmax-backward',
            [
                (self._get_buffer_handle(buf_grad_out), grad_out_flat.nbytes),
                (self._get_buffer_handle(buf_softmax), softmax_flat.nbytes),
                (self._get_buffer_handle(buf_grad_in), grad_out_flat.nbytes),
            ]
        )

        # Push constants: batch_size, seq_len, num_classes
        push_constants = struct.pack('III', batch_size, seq_len, num_classes)

        # Dispatch: one thread per row
        workgroups = (total_rows + 255) // 256
        self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)

        # Download results
        grad_input = self._download_buffer(buf_grad_in, grad_out_flat.nbytes, np.float32)

        # Release buffers back to pool
        self._release_buffers([buf_grad_out, buf_softmax, buf_grad_in])

        return grad_input.reshape(original_shape)

    # ------------------------------------------------------------------
    # Dropout (CPU fallback)
    # ------------------------------------------------------------------
    def dropout(self, x: np.ndarray, dropout_prob: float = 0.1, is_training: bool = True, seed: Optional[int] = None) -> np.ndarray:
        """
        Simple dropout implementation for test coverage. Scales activations to
        keep expected value consistent during training.
        """
        if not is_training or dropout_prob <= 0:
            return x
        rng = np.random.default_rng(seed)
        mask = rng.random(x.shape, dtype=x.dtype) >= dropout_prob
        scale = 1.0 / (1.0 - dropout_prob)
        return x * mask * scale
    
    # ------------------------------------------------------------------
    # Residual connection
    # ------------------------------------------------------------------
    def residual(self, x: np.ndarray, module_output: np.ndarray) -> np.ndarray:
        """
        Residual connection: output = x + module_output

        Uses: fnn-residual.glsl

        Args:
            x: Input tensor
            module_output: Output from module

        Returns:
            x + module_output
        """
        # Check if shader is available
        if 'fnn-residual' not in self.shaders:
            # CPU fallback
            return (x + module_output).astype(np.float32)

        # GPU implementation
        x_flat = x.astype(np.float32).flatten()
        module_flat = module_output.astype(np.float32).flatten()
        total_elements = len(x_flat)

        # Acquire buffers from pool
        buf_x = self._acquire_buffer(x_flat.nbytes)
        buf_module = self._acquire_buffer(module_flat.nbytes)
        buf_out = self._acquire_buffer(x_flat.nbytes)

        # Upload data
        self._upload_buffer(buf_x, x_flat)
        self._upload_buffer(buf_module, module_flat)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fnn-residual', 3, push_constant_size=4
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'fnn-residual',
            [
                (self._get_buffer_handle(buf_x), x_flat.nbytes),
                (self._get_buffer_handle(buf_module), module_flat.nbytes),
                (self._get_buffer_handle(buf_out), x_flat.nbytes)
            ]
        )

        # Pack push constants
        push_constants = struct.pack('I', total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )

        # Download results
        result = self._download_buffer(buf_out, x_flat.nbytes, np.float32)
        result = result[:total_elements].reshape(x.shape)

        # Release buffers back to pool
        self._release_buffers([buf_x, buf_module, buf_out])

        return result

    # ==================================================================
    # FUSED OPERATIONS
    # ==================================================================
    # These combine common operation pairs into single GPU dispatches
    # to reduce memory bandwidth and kernel launch overhead.

    def fused_linear_gelu(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        bias: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fused Linear + GELU: GELU(x @ W.T + b)

        Uses: fused-linear-gelu.glsl

        Common in Transformer FFN (first layer).

        Args:
            x: Input tensor (..., input_dim)
            weights: Weight matrix (output_dim, input_dim)
            bias: Optional bias (output_dim,)

        Returns:
            GELU(Linear(x))
        """
        if 'fused-linear-gelu' not in self.shaders:
            # Fallback to separate operations
            linear_out = self.linear(x, weights, bias)
            return self.activation_gelu(linear_out)

        # GPU implementation
        original_shape = x.shape
        x = x.astype(np.float32)
        input_dim = x.shape[-1]
        output_dim = weights.shape[0]

        # Flatten batch dimensions
        if x.ndim > 2:
            batch_seq = int(np.prod(x.shape[:-1]))
            x_flat = x.reshape(-1, input_dim).flatten()
        else:
            batch_seq = x.shape[0] if x.ndim == 2 else 1
            x_flat = x.flatten()

        w_flat = weights.astype(np.float32).flatten()
        output_size = batch_seq * output_dim * 4

        # Handle bias
        if bias is not None:
            b_flat = bias.astype(np.float32).flatten()
            has_bias = 1
        else:
            b_flat = np.zeros(output_dim, dtype=np.float32)
            has_bias = 0

        # Acquire buffers
        buf_input = self._acquire_buffer(x_flat.nbytes)
        buf_weights = self._acquire_buffer(w_flat.nbytes)
        buf_bias = self._acquire_buffer(b_flat.nbytes)
        buf_output = self._acquire_buffer(output_size)

        # Upload data
        self._upload_buffer(buf_input, x_flat)
        self._upload_buffer(buf_weights, w_flat)
        self._upload_buffer(buf_bias, b_flat)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fused-linear-gelu', 4, push_constant_size=16
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'fused-linear-gelu',
            [
                (self._get_buffer_handle(buf_input), x_flat.nbytes),
                (self._get_buffer_handle(buf_weights), w_flat.nbytes),
                (self._get_buffer_handle(buf_bias), b_flat.nbytes),
                (self._get_buffer_handle(buf_output), output_size)
            ]
        )

        # Pack push constants: batch_seq, input_dim, output_dim, has_bias
        push_constants = struct.pack('IIII', batch_seq, input_dim, output_dim, has_bias)

        # Dispatch (2D: rows = batch_seq, cols = output_dim)
        workgroups_x = (output_dim + 15) // 16
        workgroups_y = (batch_seq + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y
        )

        # Download result
        result = self._download_buffer(buf_output, output_size, np.float32)

        # Release buffers
        self._release_buffers([buf_input, buf_weights, buf_bias, buf_output])

        # Reshape output
        if len(original_shape) > 2:
            output_shape = original_shape[:-1] + (output_dim,)
            return result.reshape(output_shape)
        else:
            return result.reshape(batch_seq, output_dim)

    def fused_linear_relu(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        bias: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fused Linear + ReLU: ReLU(x @ W.T + b)

        Uses: fused-linear-relu.glsl

        Args:
            x: Input tensor (..., input_dim)
            weights: Weight matrix (output_dim, input_dim)
            bias: Optional bias (output_dim,)

        Returns:
            ReLU(Linear(x))
        """
        if 'fused-linear-relu' not in self.shaders:
            # Fallback to separate operations
            linear_out = self.linear(x, weights, bias)
            return self.activation_relu(linear_out)

        # GPU implementation (same structure as fused_linear_gelu)
        original_shape = x.shape
        x = x.astype(np.float32)
        input_dim = x.shape[-1]
        output_dim = weights.shape[0]

        if x.ndim > 2:
            batch_seq = int(np.prod(x.shape[:-1]))
            x_flat = x.reshape(-1, input_dim).flatten()
        else:
            batch_seq = x.shape[0] if x.ndim == 2 else 1
            x_flat = x.flatten()

        w_flat = weights.astype(np.float32).flatten()
        output_size = batch_seq * output_dim * 4

        if bias is not None:
            b_flat = bias.astype(np.float32).flatten()
            has_bias = 1
        else:
            b_flat = np.zeros(output_dim, dtype=np.float32)
            has_bias = 0

        buf_input = self._acquire_buffer(x_flat.nbytes)
        buf_weights = self._acquire_buffer(w_flat.nbytes)
        buf_bias = self._acquire_buffer(b_flat.nbytes)
        buf_output = self._acquire_buffer(output_size)

        self._upload_buffer(buf_input, x_flat)
        self._upload_buffer(buf_weights, w_flat)
        self._upload_buffer(buf_bias, b_flat)

        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fused-linear-relu', 4, push_constant_size=16
        )

        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'fused-linear-relu',
            [
                (self._get_buffer_handle(buf_input), x_flat.nbytes),
                (self._get_buffer_handle(buf_weights), w_flat.nbytes),
                (self._get_buffer_handle(buf_bias), b_flat.nbytes),
                (self._get_buffer_handle(buf_output), output_size)
            ]
        )

        push_constants = struct.pack('IIII', batch_seq, input_dim, output_dim, has_bias)

        workgroups_x = (output_dim + 15) // 16
        workgroups_y = (batch_seq + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y
        )

        result = self._download_buffer(buf_output, output_size, np.float32)
        self._release_buffers([buf_input, buf_weights, buf_bias, buf_output])

        if len(original_shape) > 2:
            output_shape = original_shape[:-1] + (output_dim,)
            return result.reshape(output_shape)
        else:
            return result.reshape(batch_seq, output_dim)

    def fused_linear_silu(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        bias: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fused Linear + SiLU: SiLU(x @ W.T + b)

        Uses: fused-linear-silu.glsl

        Common in LLaMA, Mistral FFN layers.

        Args:
            x: Input tensor (..., input_dim)
            weights: Weight matrix (output_dim, input_dim)
            bias: Optional bias (output_dim,)

        Returns:
            SiLU(Linear(x))
        """
        if 'fused-linear-silu' not in self.shaders:
            # Fallback to separate operations
            linear_out = self.linear(x, weights, bias)
            return self.activation_silu(linear_out)

        # GPU implementation
        original_shape = x.shape
        x = x.astype(np.float32)
        input_dim = x.shape[-1]
        output_dim = weights.shape[0]

        if x.ndim > 2:
            batch_seq = int(np.prod(x.shape[:-1]))
            x_flat = x.reshape(-1, input_dim).flatten()
        else:
            batch_seq = x.shape[0] if x.ndim == 2 else 1
            x_flat = x.flatten()

        w_flat = weights.astype(np.float32).flatten()
        output_size = batch_seq * output_dim * 4

        if bias is not None:
            b_flat = bias.astype(np.float32).flatten()
            has_bias = 1
        else:
            b_flat = np.zeros(output_dim, dtype=np.float32)
            has_bias = 0

        buf_input = self._acquire_buffer(x_flat.nbytes)
        buf_weights = self._acquire_buffer(w_flat.nbytes)
        buf_bias = self._acquire_buffer(b_flat.nbytes)
        buf_output = self._acquire_buffer(output_size)

        self._upload_buffer(buf_input, x_flat)
        self._upload_buffer(buf_weights, w_flat)
        self._upload_buffer(buf_bias, b_flat)

        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fused-linear-silu', 4, push_constant_size=16
        )

        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'fused-linear-silu',
            [
                (self._get_buffer_handle(buf_input), x_flat.nbytes),
                (self._get_buffer_handle(buf_weights), w_flat.nbytes),
                (self._get_buffer_handle(buf_bias), b_flat.nbytes),
                (self._get_buffer_handle(buf_output), output_size)
            ]
        )

        push_constants = struct.pack('IIII', batch_seq, input_dim, output_dim, has_bias)

        workgroups_x = (output_dim + 15) // 16
        workgroups_y = (batch_seq + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y
        )

        result = self._download_buffer(buf_output, output_size, np.float32)
        self._release_buffers([buf_input, buf_weights, buf_bias, buf_output])

        if len(original_shape) > 2:
            output_shape = original_shape[:-1] + (output_dim,)
            return result.reshape(output_shape)
        else:
            return result.reshape(batch_seq, output_dim)



    def fused_linear_gcu(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        bias: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fused Linear + GCU: GCU(x @ W.T + b)
        Uses: fused-linear-gcu.glsl
        """
        if 'fused-linear-gcu' not in self.shaders:
            linear_out = self.linear(x, weights, bias)
            return self.activation_gcu(linear_out)

        original_shape = x.shape
        x = x.astype(np.float32)
        input_dim = x.shape[-1]
        output_dim = weights.shape[0]

        if x.ndim > 2:
            batch_seq = int(np.prod(x.shape[:-1]))
            x_flat = x.reshape(-1, input_dim).flatten()
        else:
            batch_seq = x.shape[0] if x.ndim == 2 else 1
            x_flat = x.flatten()

        w_flat = weights.astype(np.float32).flatten()
        output_size = batch_seq * output_dim * 4

        if bias is not None:
            b_flat = bias.astype(np.float32).flatten()
            has_bias = 1
        else:
            b_flat = np.zeros(output_dim, dtype=np.float32)
            has_bias = 0

        buf_input = self._acquire_buffer(x_flat.nbytes)
        buf_weights = self._acquire_buffer(w_flat.nbytes)
        buf_bias = self._acquire_buffer(b_flat.nbytes)
        buf_output = self._acquire_buffer(output_size)

        self._upload_buffer(buf_input, x_flat)
        self._upload_buffer(buf_weights, w_flat)
        self._upload_buffer(buf_bias, b_flat)

        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fused-linear-gcu', 4, push_constant_size=16
        )

        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'fused-linear-gcu',
            [
                (self._get_buffer_handle(buf_input), x_flat.nbytes),
                (self._get_buffer_handle(buf_weights), w_flat.nbytes),
                (self._get_buffer_handle(buf_bias), b_flat.nbytes),
                (self._get_buffer_handle(buf_output), output_size)
            ]
        )

        push_constants = struct.pack('IIII', batch_seq, input_dim, output_dim, has_bias)

        workgroups_x = (output_dim + 15) // 16
        workgroups_y = (batch_seq + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y
        )

        result = self._download_buffer(buf_output, output_size, np.float32)
        self._release_buffers([buf_input, buf_weights, buf_bias, buf_output])

        if len(original_shape) > 2:
            output_shape = original_shape[:-1] + (output_dim,)
            return result.reshape(output_shape)
        else:
            return result.reshape(batch_seq, output_dim)

    def fused_linear_roswish(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        bias: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        beta: float = 1.0
    ) -> np.ndarray:
        """
        Fused Linear + RoSwish: RoSwish(x @ W.T + b)
        Uses: fused-linear-roswish.glsl
        """
        if 'fused-linear-roswish' not in self.shaders:
            linear_out = self.linear(x, weights, bias)
            return self.activation_roswish(linear_out, alpha=alpha, beta=beta)

        original_shape = x.shape
        x = x.astype(np.float32)
        input_dim = x.shape[-1]
        output_dim = weights.shape[0]

        if x.ndim > 2:
            batch_seq = int(np.prod(x.shape[:-1]))
            x_flat = x.reshape(-1, input_dim).flatten()
        else:
            batch_seq = x.shape[0] if x.ndim == 2 else 1
            x_flat = x.flatten()

        w_flat = weights.astype(np.float32).flatten()
        output_size = batch_seq * output_dim * 4

        if bias is not None:
            b_flat = bias.astype(np.float32).flatten()
            has_bias = 1
        else:
            b_flat = np.zeros(output_dim, dtype=np.float32)
            has_bias = 0

        buf_input = self._acquire_buffer(x_flat.nbytes)
        buf_weights = self._acquire_buffer(w_flat.nbytes)
        buf_bias = self._acquire_buffer(b_flat.nbytes)
        buf_output = self._acquire_buffer(output_size)

        self._upload_buffer(buf_input, x_flat)
        self._upload_buffer(buf_weights, w_flat)
        self._upload_buffer(buf_bias, b_flat)

        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fused-linear-roswish', 4, push_constant_size=24
        )

        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'fused-linear-roswish',
            [
                (self._get_buffer_handle(buf_input), x_flat.nbytes),
                (self._get_buffer_handle(buf_weights), w_flat.nbytes),
                (self._get_buffer_handle(buf_bias), b_flat.nbytes),
                (self._get_buffer_handle(buf_output), output_size)
            ]
        )

        push_constants = struct.pack('IIIIff', batch_seq, input_dim, output_dim, has_bias, alpha, beta)

        workgroups_x = (output_dim + 15) // 16
        workgroups_y = (batch_seq + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y
        )

        result = self._download_buffer(buf_output, output_size, np.float32)
        self._release_buffers([buf_input, buf_weights, buf_bias, buf_output])

        if len(original_shape) > 2:
            output_shape = original_shape[:-1] + (output_dim,)
            return result.reshape(output_shape)
        else:
            return result.reshape(batch_seq, output_dim)
