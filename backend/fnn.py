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
from typing import Optional
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


class VulkanFNN:
    """FNN operations: activations, layer normalization, linear layers, dropout"""
    
    def __init__(self, core, pipelines, shaders):
        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders
    
    def activation_relu(self, input_data):
        """Apply ReLU activation: max(0, x)"""
        data = input_data.astype(np.float32).flatten()
        total_elements = len(data)
        
        # Create buffers
        buf_in, mem_in = self.core._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_in, mem_in, data)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-relu', 2, push_constant_size=4
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-relu',
            [(buf_in, data.nbytes), (buf_out, data.nbytes)]
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
        result = self.core._download_buffer(mem_out, data.nbytes, dtype=np.float32)
        result = result[:total_elements]
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_in, None)
        vkDestroyBuffer(self.core.device, buf_out, None)
        vkFreeMemory(self.core.device, mem_in, None)
        vkFreeMemory(self.core.device, mem_out, None)
        
        return result.reshape(input_data.shape) if input_data.ndim > 1 else result
    
    def activation_gelu(self, input_data):
        """Apply GELU activation"""
        # Check if shader is available
        if 'activation-gelu' not in self.shaders:
            # CPU fallback
            sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
            coeff = 0.044715
            return 0.5 * input_data * (1 + np.tanh(sqrt_2_over_pi * (input_data + coeff * input_data ** 3)))
        
        data = input_data.astype(np.float32).flatten()
        total_elements = len(data)
        
        # Create buffers
        buf_in, mem_in = self.core._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_in, mem_in, data)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-gelu', 2, push_constant_size=4
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-gelu',
            [(buf_in, data.nbytes), (buf_out, data.nbytes)]
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
        result = self.core._download_buffer(mem_out, data.nbytes, dtype=np.float32)
        result = result[:total_elements]
        
        # Check for NaN/Inf and fallback to CPU if needed
        if np.isnan(result).any() or np.isinf(result).any():
            # CPU fallback
            sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
            coeff = 0.044715
            result = 0.5 * input_data * (1 + np.tanh(sqrt_2_over_pi * (input_data + coeff * input_data ** 3)))
            result = result.astype(np.float32).flatten()
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_in, None)
        vkDestroyBuffer(self.core.device, buf_out, None)
        vkFreeMemory(self.core.device, mem_in, None)
        vkFreeMemory(self.core.device, mem_out, None)
        
        return result.reshape(input_data.shape) if input_data.ndim > 1 else result
    
    def activation_silu(self, input_data):
        """Apply SiLU (Swish) activation: x * sigmoid(x)"""
        data = input_data.astype(np.float32).flatten()
        total_elements = len(data)
        
        # Create buffers
        buf_in, mem_in = self.core._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_in, mem_in, data)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-silu', 2, push_constant_size=4
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-silu',
            [(buf_in, data.nbytes), (buf_out, data.nbytes)]
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
        result = self.core._download_buffer(mem_out, data.nbytes, dtype=np.float32)
        result = result[:total_elements]
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_in, None)
        vkDestroyBuffer(self.core.device, buf_out, None)
        vkFreeMemory(self.core.device, mem_in, None)
        vkFreeMemory(self.core.device, mem_out, None)
        
        return result.reshape(input_data.shape) if input_data.ndim > 1 else result
    
    def activation_softmax(self, input_data, axis=-1):
        """
        Apply softmax activation: exp(x) / sum(exp(x))
        
        Args:
            input_data: Input array
            axis: Axis along which to compute softmax (default: -1)
        
        Returns:
            Softmax probabilities
        """
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
        
        # Create buffers - shader needs 4 buffers: input, output, max_vals, sum_exp
        buf_in, mem_in = self.core._create_buffer(data_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(data_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_max, mem_max = self.core._create_buffer(batch_size * seq_len * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_sum, mem_sum = self.core._create_buffer(batch_size * seq_len * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_in, mem_in, data_flat)
        
        # Get or create pipeline - 4 buffers, 24 bytes push constants (5 uints + padding)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-softmax', 4, push_constant_size=24
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-softmax',
            [
                (buf_in, data_flat.nbytes),
                (buf_out, data_flat.nbytes),
                (buf_max, batch_size * seq_len * 4),
                (buf_sum, batch_size * seq_len * 4)
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
        result = self.core._download_buffer(mem_out, data_flat.nbytes, dtype=np.float32)
        result = result[:len(data_flat)].reshape(original_shape)
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_in, None)
        vkDestroyBuffer(self.core.device, buf_out, None)
        vkDestroyBuffer(self.core.device, buf_max, None)
        vkDestroyBuffer(self.core.device, buf_sum, None)
        vkFreeMemory(self.core.device, mem_in, None)
        vkFreeMemory(self.core.device, mem_out, None)
        vkFreeMemory(self.core.device, mem_max, None)
        vkFreeMemory(self.core.device, mem_sum, None)
        
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
        scale = np.sqrt(2.0 / input_dim)
        weights_flat = np.zeros(input_dim * output_dim, dtype=np.float32)
        
        # Create output buffer
        buf_weights, mem_weights = self.core._create_buffer(weights_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Check if shader is available
        if 'fnn-xavier-init' not in self.core.shaders:
            raise RuntimeError("fnn-xavier-init shader not compiled. Run: glslc shaders/fnn-xavier-init.glsl -o shaders/spv/fnn-xavier-init.spv")
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fnn-xavier-init', 1, push_constant_size=16
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'fnn-xavier-init',
            [(buf_weights, weights_flat.nbytes)]
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
        result = self.core._download_buffer(mem_weights, weights_flat.nbytes, dtype=np.float32)
        result = result[:input_dim * output_dim]
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_weights, None)
        vkFreeMemory(self.core.device, mem_weights, None)
        
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
        
        # Create buffers
        buf_grad_out, mem_grad_out = self.core._create_buffer(grad_out.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_input, mem_input = self.core._create_buffer(input_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_grad_in, mem_grad_in = self.core._create_buffer(input_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_grad_out, mem_grad_out, grad_out)
        self.core._upload_buffer(buf_input, mem_input, input_flat)
        
        # Check if shader is available
        if 'activation-gelu-backward' not in self.shaders:
            # CPU fallback
            sqrt_2_over_pi = 0.7978845608028654
            coeff = 0.044715
            grad_in = np.zeros_like(input_flat)
            for i in range(total_elements):
                x = input_flat[i]
                x_cubed = x * x * x
                z = sqrt_2_over_pi * (x + coeff * x_cubed)
                tanh_z = np.tanh(z)
                sech_z = 1.0 / np.cosh(z)
                sech_sq = sech_z * sech_z
                dz_dx = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x)
                gelu_grad = 0.5 * (1.0 + tanh_z + x * sech_sq * dz_dx)
                grad_in[i] = grad_out[i] * gelu_grad
            return grad_in.reshape(input_data.shape)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-gelu-backward', 3, push_constant_size=4
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-gelu-backward',
            [
                (buf_grad_out, grad_out.nbytes),
                (buf_input, input_flat.nbytes),
                (buf_grad_in, input_flat.nbytes)
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
        result = self.core._download_buffer(mem_grad_in, input_flat.nbytes, dtype=np.float32)
        result = result[:total_elements].reshape(input_data.shape)
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_grad_out, None)
        vkDestroyBuffer(self.core.device, buf_input, None)
        vkDestroyBuffer(self.core.device, buf_grad_in, None)
        vkFreeMemory(self.core.device, mem_grad_out, None)
        vkFreeMemory(self.core.device, mem_input, None)
        vkFreeMemory(self.core.device, mem_grad_in, None)
        
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

        # Create buffers
        buf_input, mem_input = self.core._create_buffer(x_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_output, mem_output = self.core._create_buffer(x_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_gamma, mem_gamma = self.core._create_buffer(gamma_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_beta, mem_beta = self.core._create_buffer(beta_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_mean, mem_mean = self.core._create_buffer(total_positions * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_var, mem_var = self.core._create_buffer(total_positions * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)

        # Upload data
        self.core._upload_buffer(buf_input, mem_input, x_flat)
        self.core._upload_buffer(buf_gamma, mem_gamma, gamma_flat)
        self.core._upload_buffer(buf_beta, mem_beta, beta_flat)

        # Get or create pipeline (6 buffers, push constants: 4 uints + 1 float = 20 bytes)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fnn-layernorm', 6, push_constant_size=20
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'fnn-layernorm',
            [
                (buf_input, x_flat.nbytes),
                (buf_output, x_flat.nbytes),
                (buf_gamma, gamma_flat.nbytes),
                (buf_beta, beta_flat.nbytes),
                (buf_mean, total_positions * 4),
                (buf_var, total_positions * 4),
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
        result = self.core._download_buffer(mem_output, x_flat.nbytes, dtype=np.float32)

        # Cleanup
        vkDestroyBuffer(self.core.device, buf_input, None)
        vkDestroyBuffer(self.core.device, buf_output, None)
        vkDestroyBuffer(self.core.device, buf_gamma, None)
        vkDestroyBuffer(self.core.device, buf_beta, None)
        vkDestroyBuffer(self.core.device, buf_mean, None)
        vkDestroyBuffer(self.core.device, buf_var, None)
        vkFreeMemory(self.core.device, mem_input, None)
        vkFreeMemory(self.core.device, mem_output, None)
        vkFreeMemory(self.core.device, mem_gamma, None)
        vkFreeMemory(self.core.device, mem_beta, None)
        vkFreeMemory(self.core.device, mem_mean, None)
        vkFreeMemory(self.core.device, mem_var, None)

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
            # CPU fallback
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

        # Create buffers
        buf_input, mem_input = self.core._create_buffer(x_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_weights, mem_weights = self.core._create_buffer(w_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_output, mem_output = self.core._create_buffer(output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)

        # Handle bias
        has_bias = 1 if bias is not None else 0
        if bias is not None:
            bias_flat = bias.astype(np.float32).flatten()
            buf_bias, mem_bias = self.core._create_buffer(bias_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            self.core._upload_buffer(buf_bias, mem_bias, bias_flat)
        else:
            # Create dummy bias buffer (shader expects 4 buffers)
            buf_bias, mem_bias = self.core._create_buffer(4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)

        # Upload data
        self.core._upload_buffer(buf_input, mem_input, x_flat)
        self.core._upload_buffer(buf_weights, mem_weights, w_flat)

        # Get or create pipeline (4 buffers, push constants: 4 uints = 16 bytes)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fnn-linear', 4, push_constant_size=16
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'fnn-linear',
            [
                (buf_input, x_flat.nbytes),
                (buf_weights, w_flat.nbytes),
                (buf_bias, bias_flat.nbytes if bias is not None else 4),
                (buf_output, output_size),
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
        result = self.core._download_buffer(mem_output, output_size, dtype=np.float32)

        # Cleanup
        vkDestroyBuffer(self.core.device, buf_input, None)
        vkDestroyBuffer(self.core.device, buf_weights, None)
        vkDestroyBuffer(self.core.device, buf_bias, None)
        vkDestroyBuffer(self.core.device, buf_output, None)
        vkFreeMemory(self.core.device, mem_input, None)
        vkFreeMemory(self.core.device, mem_weights, None)
        vkFreeMemory(self.core.device, mem_bias, None)
        vkFreeMemory(self.core.device, mem_output, None)

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
        Backward pass for linear layer using fnn-linear-backward.glsl
        
        Args:
            grad_output: Gradient w.r.t. output (batch, out_features)
            x: Input (batch, in_features)
            weights: Weight matrix (out_features, in_features)
            bias: Optional bias (out_features,)
        
        Returns:
            (grad_input, grad_weight, grad_bias)
        """
        # Check if shader is available
        if 'fnn-linear-backward' not in self.shaders:
            # CPU fallback
            grad_input = grad_output @ weights  # (batch, in_features)
            grad_weight = grad_output.T @ x  # (out_features, in_features)
            grad_bias = np.sum(grad_output, axis=0) if bias is not None else None
            return grad_input, grad_weight, grad_bias
        
        # GPU implementation
        batch_seq, output_dim = grad_output.shape
        _, input_dim = x.shape
        
        # Flatten arrays
        grad_out_flat = grad_output.astype(np.float32).flatten()
        x_flat = x.astype(np.float32).flatten()
        w_flat = weights.astype(np.float32).flatten()
        
        # Output buffers
        grad_input_flat = np.zeros(batch_seq * input_dim, dtype=np.float32)
        grad_weight_flat = np.zeros(output_dim * input_dim, dtype=np.float32)
        grad_bias_flat = np.zeros(output_dim, dtype=np.float32) if bias is not None else None
        
        # Create buffers
        buf_grad_out, mem_grad_out = self.core._create_buffer(grad_out_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_x, mem_x = self.core._create_buffer(x_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_w, mem_w = self.core._create_buffer(w_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_grad_in, mem_grad_in = self.core._create_buffer(grad_input_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_grad_w, mem_grad_w = self.core._create_buffer(grad_weight_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        buffers = [
            (buf_grad_out, grad_out_flat.nbytes),
            (buf_x, x_flat.nbytes),
            (buf_w, w_flat.nbytes),
            (buf_grad_in, grad_input_flat.nbytes),
            (buf_grad_w, grad_weight_flat.nbytes),
        ]
        
        if bias is not None:
            buf_grad_b, mem_grad_b = self.core._create_buffer(grad_bias_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            buffers.append((buf_grad_b, grad_bias_flat.nbytes))
        else:
            buf_grad_b = None
            mem_grad_b = None
        
        # Upload data
        self.core._upload_buffer(buf_grad_out, mem_grad_out, grad_out_flat)
        self.core._upload_buffer(buf_x, mem_x, x_flat)
        self.core._upload_buffer(buf_w, mem_w, w_flat)
        self.core._upload_buffer(buf_grad_in, mem_grad_in, grad_input_flat)
        self.core._upload_buffer(buf_grad_w, mem_grad_w, grad_weight_flat)
        if bias is not None:
            self.core._upload_buffer(buf_grad_b, mem_grad_b, grad_bias_flat)
        
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
        grad_input_flat = self.core._download_buffer(mem_grad_in, grad_input_flat.nbytes, dtype=np.float32)
        grad_weight_flat = self.core._download_buffer(mem_grad_w, grad_weight_flat.nbytes, dtype=np.float32)
        if bias is not None:
            grad_bias_flat = self.core._download_buffer(mem_grad_b, grad_bias_flat.nbytes, dtype=np.float32)
        else:
            grad_bias_flat = None
        
        # Reshape
        grad_input = grad_input_flat[:batch_seq * input_dim].reshape(batch_seq, input_dim)
        grad_weight = grad_weight_flat[:output_dim * input_dim].reshape(output_dim, input_dim)
        grad_bias = grad_bias_flat[:output_dim] if grad_bias_flat is not None else None
        
        # Cleanup
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        vkDestroyBuffer(self.core.device, buf_grad_out, None)
        vkDestroyBuffer(self.core.device, buf_x, None)
        vkDestroyBuffer(self.core.device, buf_w, None)
        vkDestroyBuffer(self.core.device, buf_grad_in, None)
        vkDestroyBuffer(self.core.device, buf_grad_w, None)
        if buf_grad_b is not None:
            vkDestroyBuffer(self.core.device, buf_grad_b, None)
        vkFreeMemory(self.core.device, mem_grad_out, None)
        vkFreeMemory(self.core.device, mem_x, None)
        vkFreeMemory(self.core.device, mem_w, None)
        vkFreeMemory(self.core.device, mem_grad_in, None)
        vkFreeMemory(self.core.device, mem_grad_w, None)
        if mem_grad_b is not None:
            vkFreeMemory(self.core.device, mem_grad_b, None)
        
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

        # Create buffers
        buf_grad_out, mem_grad_out = self.core._create_buffer(grad_out_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_input, mem_input = self.core._create_buffer(x_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_gamma, mem_gamma = self.core._create_buffer(gamma_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_mean, mem_mean = self.core._create_buffer(mean_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_var, mem_var = self.core._create_buffer(var_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_grad_in, mem_grad_in = self.core._create_buffer(x_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_grad_gamma, mem_grad_gamma = self.core._create_buffer(gamma_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_grad_beta, mem_grad_beta = self.core._create_buffer(gamma_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)

        # Upload data
        self.core._upload_buffer(buf_grad_out, mem_grad_out, grad_out_flat)
        self.core._upload_buffer(buf_input, mem_input, x_flat)
        self.core._upload_buffer(buf_gamma, mem_gamma, gamma_flat)
        self.core._upload_buffer(buf_mean, mem_mean, mean_flat)
        self.core._upload_buffer(buf_var, mem_var, var_flat)

        # Initialize grad buffers to zero
        self.core._upload_buffer(buf_grad_in, mem_grad_in, np.zeros(total_elements, dtype=np.float32))
        self.core._upload_buffer(buf_grad_gamma, mem_grad_gamma, np.zeros(features, dtype=np.float32))
        self.core._upload_buffer(buf_grad_beta, mem_grad_beta, np.zeros(features, dtype=np.float32))

        # Get or create pipeline (8 buffers, push constants: 4 uints + 1 float = 20 bytes)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fnn-layernorm-backward', 8, push_constant_size=20
        )

        buffers = [
            (buf_grad_out, grad_out_flat.nbytes),
            (buf_input, x_flat.nbytes),
            (buf_gamma, gamma_flat.nbytes),
            (buf_mean, mean_flat.nbytes),
            (buf_var, var_flat.nbytes),
            (buf_grad_in, x_flat.nbytes),
            (buf_grad_gamma, gamma_flat.nbytes),
            (buf_grad_beta, gamma_flat.nbytes),
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
        grad_input = self.core._download_buffer(mem_grad_in, x_flat.nbytes, dtype=np.float32)
        grad_gamma = self.core._download_buffer(mem_grad_gamma, gamma_flat.nbytes, dtype=np.float32)
        grad_beta = self.core._download_buffer(mem_grad_beta, gamma_flat.nbytes, dtype=np.float32)

        # Cleanup
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        for buf in [buf_grad_out, buf_input, buf_gamma, buf_mean, buf_var, buf_grad_in, buf_grad_gamma, buf_grad_beta]:
            vkDestroyBuffer(self.core.device, buf, None)
        for mem in [mem_grad_out, mem_input, mem_gamma, mem_mean, mem_var, mem_grad_in, mem_grad_gamma, mem_grad_beta]:
            vkFreeMemory(self.core.device, mem, None)

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

        # Create buffers
        buf_grad_out, mem_grad_out = self.core._create_buffer(grad_out_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_softmax, mem_softmax = self.core._create_buffer(softmax_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_grad_in, mem_grad_in = self.core._create_buffer(grad_out_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)

        # Upload data
        self.core._upload_buffer(buf_grad_out, mem_grad_out, grad_out_flat)
        self.core._upload_buffer(buf_softmax, mem_softmax, softmax_flat)

        # Get or create pipeline (3 buffers, push constants: 3 uints = 12 bytes)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-softmax-backward', 3, push_constant_size=12
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-softmax-backward',
            [
                (buf_grad_out, grad_out_flat.nbytes),
                (buf_softmax, softmax_flat.nbytes),
                (buf_grad_in, grad_out_flat.nbytes),
            ]
        )

        # Push constants: batch_size, seq_len, num_classes
        push_constants = struct.pack('III', batch_size, seq_len, num_classes)

        # Dispatch: one thread per row
        workgroups = (total_rows + 255) // 256
        self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)

        # Download results
        grad_input = self.core._download_buffer(mem_grad_in, grad_out_flat.nbytes, dtype=np.float32)

        # Cleanup
        vkDestroyBuffer(self.core.device, buf_grad_out, None)
        vkDestroyBuffer(self.core.device, buf_softmax, None)
        vkDestroyBuffer(self.core.device, buf_grad_in, None)
        vkFreeMemory(self.core.device, mem_grad_out, None)
        vkFreeMemory(self.core.device, mem_softmax, None)
        vkFreeMemory(self.core.device, mem_grad_in, None)

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
            return x + module_output
        
        # GPU implementation
        x_flat = x.astype(np.float32).flatten()
        module_flat = module_output.astype(np.float32).flatten()
        total_elements = len(x_flat)
        
        # Create buffers
        buf_x, mem_x = self.core._create_buffer(x_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_module, mem_module = self.core._create_buffer(module_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(x_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_x, mem_x, x_flat)
        self.core._upload_buffer(buf_module, mem_module, module_flat)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fnn-residual', 3, push_constant_size=4
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'fnn-residual',
            [
                (buf_x, x_flat.nbytes),
                (buf_module, module_flat.nbytes),
                (buf_out, x_flat.nbytes)
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
        result = self.core._download_buffer(mem_out, x_flat.nbytes, dtype=np.float32)
        result = result[:total_elements].reshape(x.shape)
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_x, None)
        vkDestroyBuffer(self.core.device, buf_module, None)
        vkDestroyBuffer(self.core.device, buf_out, None)
        vkFreeMemory(self.core.device, mem_x, None)
        vkFreeMemory(self.core.device, mem_module, None)
        vkFreeMemory(self.core.device, mem_out, None)
        
        return result

