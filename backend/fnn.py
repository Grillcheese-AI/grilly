"""
Feedforward Neural Network operations for Vulkan backend.
GPU-accelerated FNN operations: activations, layer normalization, linear layers, dropout.

Performance hierarchy:
1. Vulkan GPU shader (fastest)
2. Numba JIT (fast CPU fallback)
3. Pure numpy (baseline fallback)
"""

import struct

import numpy as np

from .base import VULKAN_AVAILABLE, BufferMixin

if VULKAN_AVAILABLE:
    from vulkan import *

# Try to import numba-accelerated operations for CPU fallback
try:
    from ..utils.numba_ops import (
        NUMBA_AVAILABLE,
    )
    from ..utils.numba_ops import (
        gcu as numba_gcu,
    )
    from ..utils.numba_ops import (
        gelu as numba_gelu,
    )
    from ..utils.numba_ops import (
        layernorm as numba_layernorm,
    )
    from ..utils.numba_ops import (
        linear as numba_linear,
    )
    from ..utils.numba_ops import (
        relu as numba_relu,
    )
    from ..utils.numba_ops import (
        roswish as numba_roswish,
    )
    from ..utils.numba_ops import (
        silu as numba_silu,
    )
    from ..utils.numba_ops import (
        softmax as numba_softmax,
    )
    from ..utils.numba_ops import (
        swiglu as numba_swiglu,
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


class VulkanFNN(BufferMixin):
    """FNN operations: activations, layer normalization, linear layers, dropout"""

    def __init__(self, core, pipelines, shaders):
        """Initialize the instance."""

        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders
        self._pool = None  # Lazy initialization

    def gemm(self, A, B, return_gpu_tensor=False, cache_B=False, force_fp32=False):
        """
        GEMM: C = A @ B
        A: (M, K), B: (K, N) -> C: (M, N)

        Shader priority:
        1. gemm_coopmat — cooperative matrix (fp16 in, fp32 out, hardware WMMA)
        2. gemm_tiled   — tiled fp32 (4x4 register blocking)
        3. gemm_mnk     — basic fp32

        Args:
            A: Left matrix, numpy array or VulkanTensor
            B: Right matrix, numpy array or VulkanTensor
            return_gpu_tensor: If True, return VulkanTensor (stays on GPU)
            cache_B: If True, keep B (weight matrix) GPU-resident across calls
            force_fp32: If True, skip cooperative matrix and use fp32 shader
        """
        import os as _os

        from ..utils.tensor_conversion import VulkanTensor

        # Select shader: prefer cooperative matrix → tiled → basic
        use_coopmat = (
            not force_fp32
            and "gemm_coopmat" in self.shaders
            and self.core.has_cooperative_matrix
            and self.core.has_float16
            and _os.environ.get("GRILLY_DISABLE_COOPMAT", "0") != "1"
        )
        use_tiled = "gemm_tiled" in self.shaders
        use_basic = "gemm_mnk" in self.shaders

        if not use_coopmat and not use_tiled and not use_basic:
            A_np = A.numpy() if isinstance(A, VulkanTensor) else np.asarray(A, dtype=np.float32)
            B_np = B.numpy() if isinstance(B, VulkanTensor) else np.asarray(B, dtype=np.float32)
            return A_np @ B_np  # CPU fallback

        # Get shapes without forcing download
        M, K = A.shape
        K2, N = B.shape
        assert K == K2

        if use_coopmat:
            return self._gemm_coopmat(A, B, M, K, N, return_gpu_tensor, cache_B)
        else:
            return self._gemm_scalar(A, B, M, K, N, return_gpu_tensor, cache_B,
                                     use_tiled)

    def _gemm_coopmat(self, A, B, M, K, N, return_gpu_tensor, cache_B):
        """Cooperative matrix GEMM: fp16 inputs, fp32 accumulation."""
        from ..utils.tensor_conversion import VulkanTensor

        # Pad dimensions for cooperative matrix workgroup tiling:
        # M, K → multiple of 16 (single 16x16 tile per subgroup row)
        # N → multiple of 64 (4 subgroups × 16 cols per workgroup)
        M_pad = (M + 15) & ~15
        K_pad = (K + 15) & ~15
        N_pad = (N + 63) & ~63

        # Convert to fp16 and pad
        A_np = A.numpy() if isinstance(A, VulkanTensor) else np.asarray(A, dtype=np.float32)
        B_np = B.numpy() if isinstance(B, VulkanTensor) else np.asarray(B, dtype=np.float32)

        if M_pad != M or K_pad != K:
            A_f16 = np.zeros((M_pad, K_pad), dtype=np.float16)
            A_f16[:M, :K] = A_np
        else:
            A_f16 = A_np.astype(np.float16)

        if K_pad != K or N_pad != N:
            B_f16 = np.zeros((K_pad, N_pad), dtype=np.float16)
            B_f16[:K, :N] = B_np
        else:
            B_f16 = B_np.astype(np.float16)

        A_bytes = M_pad * K_pad * 2  # fp16
        B_bytes = K_pad * N_pad * 2
        C_bytes = M_pad * N_pad * 4  # fp32 output

        buf_A = self._acquire_buffer(A_bytes)
        self._upload_buffer_raw(buf_A, A_f16)
        release_A = True

        buf_B = self._acquire_buffer(B_bytes)
        self._upload_buffer_raw(buf_B, B_f16)
        release_B = True

        buf_C = self._acquire_buffer(C_bytes)

        try:
            shader_name = "gemm_coopmat"
            # 4 subgroups per workgroup: 16 rows × 64 cols per workgroup
            group_x = (N_pad + 63) // 64
            group_y = (M_pad + 15) // 16

            pipeline, layout, _ = self.pipelines.get_or_create_pipeline(
                shader_name, 3, push_constant_size=12
            )

            A_handle = self._get_buffer_handle(buf_A)
            B_handle = self._get_buffer_handle(buf_B)
            C_handle = self._get_buffer_handle(buf_C)

            desc = self.pipelines.get_cached_descriptor_set(
                shader_name,
                [
                    (A_handle, A_bytes),
                    (B_handle, B_bytes),
                    (C_handle, C_bytes),
                ],
            )

            push = struct.pack("3I", M_pad, K_pad, N_pad)
            self.core._dispatch_compute(pipeline, layout, desc, group_x, push, group_y, 1)

            if return_gpu_tensor:
                # TODO: trim padding on download for gpu tensors
                return self._wrap_output_tensor(buf_C, (M_pad, N_pad))
            else:
                C_flat = self._download_buffer(buf_C, C_bytes, np.float32)
                C_full = C_flat.reshape(M_pad, N_pad)
                return C_full[:M, :N]  # Trim padding

        finally:
            if release_A:
                self._release_buffer(buf_A)
            if release_B:
                self._release_buffer(buf_B)
            if not return_gpu_tensor:
                self._release_buffer(buf_C)

    def _gemm_scalar(self, A, B, M, K, N, return_gpu_tensor, cache_B, use_tiled):
        """Scalar tiled/basic GEMM: fp32 inputs and outputs."""
        A_bytes = M * K * 4
        B_bytes = K * N * 4
        C_bytes = M * N * 4

        use_device_local = return_gpu_tensor and hasattr(self, "_acquire_device_local_buffer")

        buf_A, release_A = self._prepare_input(A, size=A_bytes)
        if cache_B and isinstance(B, np.ndarray):
            if use_device_local:
                buf_B, release_B = self._get_or_upload_weight_device_local(B)
            else:
                buf_B, release_B = self._get_or_upload_weight(B)
        else:
            buf_B, release_B = self._prepare_input(B, size=B_bytes)

        if use_device_local:
            buf_C = self._acquire_device_local_buffer(C_bytes)
        else:
            buf_C = self._acquire_buffer(C_bytes)

        try:
            if use_tiled:
                shader_name = "gemm_tiled"
                group_x = (N + 63) // 64
                group_y = (M + 63) // 64
            else:
                shader_name = "gemm_mnk"
                group_x = (N + 15) // 16
                group_y = (M + 15) // 16

            pipeline, layout, _ = self.pipelines.get_or_create_pipeline(
                shader_name, 3, push_constant_size=12
            )

            A_handle = self._get_buffer_handle(buf_A)
            B_handle = self._get_buffer_handle(buf_B)
            C_handle = self._get_buffer_handle(buf_C)

            desc = self.pipelines.get_cached_descriptor_set(
                shader_name,
                [
                    (A_handle, A_bytes),
                    (B_handle, B_bytes),
                    (C_handle, C_bytes),
                ],
            )

            push = struct.pack("3I", M, K, N)
            self.core._dispatch_compute(pipeline, layout, desc, group_x, push, group_y, 1)

            if return_gpu_tensor:
                if use_device_local and getattr(buf_C, "is_device_local", False):
                    return self._wrap_output_tensor_device_local(buf_C, (M, N))
                return self._wrap_output_tensor(buf_C, (M, N))
            else:
                C_flat = self._download_buffer(buf_C, C_bytes, np.float32)
                return C_flat.reshape(M, N)

        finally:
            if release_A:
                self._release_buffer(buf_A)
            if release_B:
                self._release_buffer(buf_B)
            if not return_gpu_tensor:
                self._release_buffer(buf_C)

    def activation_relu(self, input_data, return_gpu_tensor=False):
        """Apply ReLU activation: max(0, x)"""
        from ..utils.tensor_conversion import VulkanTensor

        is_vt = isinstance(input_data, VulkanTensor)

        # Check if shader is available
        if "activation-relu" not in self.shaders:
            # CPU fallback (numba if available)
            data_np = input_data.numpy() if is_vt else np.asarray(input_data, dtype=np.float32)
            if NUMBA_AVAILABLE and numba_relu is not None:
                return numba_relu(data_np.astype(np.float32))
            return np.maximum(0, data_np).astype(np.float32)

        original_shape = input_data.shape
        total_elements = int(np.prod(original_shape))
        data_nbytes = total_elements * 4

        use_device_local = return_gpu_tensor and hasattr(self, "_acquire_device_local_buffer")

        buf_in, release_in = self._prepare_input(input_data, size=data_nbytes)
        if use_device_local:
            buf_out = self._acquire_device_local_buffer(data_nbytes)
        else:
            buf_out = self._acquire_buffer(data_nbytes)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            "activation-relu", 2, push_constant_size=4
        )

        # Get buffer handles (converts VMA handles to vulkan-compatible handles)
        in_handle = self._get_buffer_handle(buf_in)
        out_handle = self._get_buffer_handle(buf_out)

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "activation-relu", [(in_handle, data_nbytes), (out_handle, data_nbytes)]
        )

        # Pack push constants
        push_constants = struct.pack("I", total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
        )

        if return_gpu_tensor:
            if release_in:
                self._release_buffer(buf_in)
            if use_device_local and getattr(buf_out, "is_device_local", False):
                return self._wrap_output_tensor_device_local(buf_out, original_shape)
            return self._wrap_output_tensor(buf_out, original_shape)
        else:
            # Download results (uses VMA memory mapping for VMA buffers)
            result = self._download_buffer(buf_out, data_nbytes, np.float32)
            result = result[:total_elements]

            # Release buffers back to pool
            if release_in:
                self._release_buffer(buf_in)
            self._release_buffer(buf_out)

            return result.reshape(original_shape) if len(original_shape) > 1 else result

    def activation_gelu(self, input_data, return_gpu_tensor=False):
        """Apply GELU activation"""
        from ..utils.tensor_conversion import VulkanTensor

        is_vt = isinstance(input_data, VulkanTensor)

        # Check if shader is available
        if "activation-gelu" not in self.shaders:
            # CPU fallback (numba if available)
            data_np = input_data.numpy() if is_vt else np.asarray(input_data, dtype=np.float32)
            if NUMBA_AVAILABLE and numba_gelu is not None:
                return numba_gelu(data_np.astype(np.float32))
            sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
            coeff = 0.044715
            return 0.5 * data_np * (1 + np.tanh(sqrt_2_over_pi * (data_np + coeff * data_np**3)))

        original_shape = input_data.shape
        total_elements = int(np.prod(original_shape))
        data_nbytes = total_elements * 4

        use_device_local = return_gpu_tensor and hasattr(self, "_acquire_device_local_buffer")

        buf_in, release_in = self._prepare_input(input_data, size=data_nbytes)
        if use_device_local:
            buf_out = self._acquire_device_local_buffer(data_nbytes)
        else:
            buf_out = self._acquire_buffer(data_nbytes)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            "activation-gelu", 2, push_constant_size=4
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "activation-gelu",
            [
                (self._get_buffer_handle(buf_in), data_nbytes),
                (self._get_buffer_handle(buf_out), data_nbytes),
            ],
        )

        # Pack push constants
        push_constants = struct.pack("I", total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
        )

        if return_gpu_tensor:
            if release_in:
                self._release_buffer(buf_in)
            if use_device_local and getattr(buf_out, "is_device_local", False):
                return self._wrap_output_tensor_device_local(buf_out, original_shape)
            return self._wrap_output_tensor(buf_out, original_shape)
        else:
            # Download results
            result = self._download_buffer(buf_out, data_nbytes, np.float32)
            result = result[:total_elements]

            # Check for NaN/Inf and fallback to CPU if needed
            if np.isnan(result).any() or np.isinf(result).any():
                # CPU fallback
                data_np = input_data.numpy() if is_vt else np.asarray(input_data, dtype=np.float32)
                sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
                coeff = 0.044715
                result = (
                    0.5 * data_np * (1 + np.tanh(sqrt_2_over_pi * (data_np + coeff * data_np**3)))
                )
                result = result.astype(np.float32).flatten()

            # Release buffers back to pool
            if release_in:
                self._release_buffer(buf_in)
            self._release_buffer(buf_out)

            return result.reshape(original_shape) if len(original_shape) > 1 else result

    def activation_silu(self, input_data, return_gpu_tensor=False):
        """Apply SiLU (Swish) activation: x * sigmoid(x)"""
        from ..utils.tensor_conversion import VulkanTensor

        is_vt = isinstance(input_data, VulkanTensor)

        # Check if shader is available
        if "activation-silu" not in self.shaders:
            # CPU fallback (numba if available)
            data_np = input_data.numpy() if is_vt else np.asarray(input_data, dtype=np.float32)
            if NUMBA_AVAILABLE and numba_silu is not None:
                return numba_silu(data_np.astype(np.float32))
            return data_np / (1.0 + np.exp(-data_np))

        original_shape = input_data.shape
        total_elements = int(np.prod(original_shape))
        data_nbytes = total_elements * 4

        use_device_local = return_gpu_tensor and hasattr(self, "_acquire_device_local_buffer")

        buf_in, release_in = self._prepare_input(input_data, size=data_nbytes)
        if use_device_local:
            buf_out = self._acquire_device_local_buffer(data_nbytes)
        else:
            buf_out = self._acquire_buffer(data_nbytes)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            "activation-silu", 2, push_constant_size=4
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "activation-silu",
            [
                (self._get_buffer_handle(buf_in), data_nbytes),
                (self._get_buffer_handle(buf_out), data_nbytes),
            ],
        )

        # Pack push constants
        push_constants = struct.pack("I", total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
        )

        if return_gpu_tensor:
            if release_in:
                self._release_buffer(buf_in)
            if use_device_local and getattr(buf_out, "is_device_local", False):
                return self._wrap_output_tensor_device_local(buf_out, original_shape)
            return self._wrap_output_tensor(buf_out, original_shape)
        else:
            # Download results
            result = self._download_buffer(buf_out, data_nbytes, np.float32)
            result = result[:total_elements]

            # Release buffers back to pool
            if release_in:
                self._release_buffer(buf_in)
            self._release_buffer(buf_out)

            return result.reshape(original_shape) if len(original_shape) > 1 else result

    def activation_tanh(self, input_data, return_gpu_tensor=False):
        """Apply tanh activation: tanh(x)

        Uses: activation-tanh.glsl
        """
        from ..utils.tensor_conversion import VulkanTensor

        is_vt = isinstance(input_data, VulkanTensor)

        if "activation-tanh" not in self.shaders:
            data_np = input_data.numpy() if is_vt else np.asarray(input_data, dtype=np.float32)
            return np.tanh(data_np).astype(np.float32)

        original_shape = input_data.shape
        total_elements = int(np.prod(original_shape))
        data_nbytes = total_elements * 4

        use_device_local = return_gpu_tensor and hasattr(self, "_acquire_device_local_buffer")

        buf_in, release_in = self._prepare_input(input_data, size=data_nbytes)
        if use_device_local:
            buf_out = self._acquire_device_local_buffer(data_nbytes)
        else:
            buf_out = self._acquire_buffer(data_nbytes)

        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            "activation-tanh", 2, push_constant_size=4
        )

        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "activation-tanh",
            [
                (self._get_buffer_handle(buf_in), data_nbytes),
                (self._get_buffer_handle(buf_out), data_nbytes),
            ],
        )

        push_constants = struct.pack("I", total_elements)

        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
        )

        if return_gpu_tensor:
            if release_in:
                self._release_buffer(buf_in)
            if use_device_local and getattr(buf_out, "is_device_local", False):
                return self._wrap_output_tensor_device_local(buf_out, original_shape)
            return self._wrap_output_tensor(buf_out, original_shape)
        else:
            result = self._download_buffer(buf_out, data_nbytes, np.float32)
            result = result[:total_elements]

            if release_in:
                self._release_buffer(buf_in)
            self._release_buffer(buf_out)

            return result.reshape(original_shape) if len(original_shape) > 1 else result

    def activation_tanh_backward(self, grad_output, tanh_output):
        """Backward pass for tanh: grad_input = grad_output * (1 - tanh_output^2)

        Uses: activation-tanh-backward.glsl

        Args:
            grad_output: Gradient from next layer
            tanh_output: Cached tanh output from forward pass
        """
        from ..utils.tensor_conversion import VulkanTensor

        is_vt_grad = isinstance(grad_output, VulkanTensor)
        is_vt_tanh = isinstance(tanh_output, VulkanTensor)

        if "activation-tanh-backward" not in self.shaders:
            g = grad_output.numpy() if is_vt_grad else np.asarray(grad_output, dtype=np.float32)
            t = tanh_output.numpy() if is_vt_tanh else np.asarray(tanh_output, dtype=np.float32)
            return (g * (1.0 - t * t)).astype(np.float32)

        original_shape = grad_output.shape
        total_elements = int(np.prod(original_shape))
        data_nbytes = total_elements * 4

        buf_grad, release_grad = self._prepare_input(grad_output, size=data_nbytes)
        buf_tanh, release_tanh = self._prepare_input(tanh_output, size=data_nbytes)
        buf_out = self._acquire_buffer(data_nbytes)

        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            "activation-tanh-backward", 3, push_constant_size=4
        )

        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "activation-tanh-backward",
            [
                (self._get_buffer_handle(buf_grad), data_nbytes),
                (self._get_buffer_handle(buf_tanh), data_nbytes),
                (self._get_buffer_handle(buf_out), data_nbytes),
            ],
        )

        push_constants = struct.pack("I", total_elements)

        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
        )

        result = self._download_buffer(buf_out, data_nbytes, np.float32)
        result = result[:total_elements]

        if release_grad:
            self._release_buffer(buf_grad)
        if release_tanh:
            self._release_buffer(buf_tanh)
        self._release_buffer(buf_out)

        return result.reshape(original_shape) if len(original_shape) > 1 else result

    def activation_gcu(self, input_data):
        """Apply GCU (Growing Cosine Unit) activation: x * cos(x)"""
        # Check if shader is available
        if "activation-gcu" not in self.shaders:
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
            "activation-gcu", 2, push_constant_size=4
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "activation-gcu",
            [
                (self._get_buffer_handle(buf_in), data.nbytes),
                (self._get_buffer_handle(buf_out), data.nbytes),
            ],
        )

        # Pack push constants
        push_constants = struct.pack("I", total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
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
        if "activation-roswish" not in self.shaders:
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
            "activation-roswish", 2, push_constant_size=12
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "activation-roswish",
            [
                (self._get_buffer_handle(buf_in), data.nbytes),
                (self._get_buffer_handle(buf_out), data.nbytes),
            ],
        )

        # Pack push constants: total_elements (uint32), alpha (float32), beta (float32)
        push_constants = struct.pack("Iff", total_elements, float(alpha), float(beta))

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
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
        if "activation-swiglu" not in self.shaders:
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
            "activation-swiglu", 2, push_constant_size=8
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "activation-swiglu",
            [
                (self._get_buffer_handle(buf_in), data_flat.nbytes),
                (self._get_buffer_handle(buf_out), output_elements * 4),
            ],
        )

        # Pack push constants: output_elements (uint32), hidden_dim (uint32)
        push_constants = struct.pack("II", output_elements, hidden_dim)

        # Dispatch
        workgroups = (output_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
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
        if "activation-softmax" not in self.shaders:
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
            "activation-softmax", 4, push_constant_size=24
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "activation-softmax",
            [
                (self._get_buffer_handle(buf_in), data_flat.nbytes),
                (self._get_buffer_handle(buf_out), data_flat.nbytes),
                (self._get_buffer_handle(buf_max), batch_size * seq_len * 4),
                (self._get_buffer_handle(buf_sum), batch_size * seq_len * 4),
            ],
        )

        # Pass 1: Compute max for numerical stability
        push_constants = struct.pack("IIIII", batch_size, seq_len, features, 0, features)
        workgroups = ((batch_size * seq_len) + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
        )

        # Pass 2: Compute sum of exponentials
        push_constants = struct.pack("IIIII", batch_size, seq_len, features, 1, features)
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
        )

        # Pass 3: Normalize
        push_constants = struct.pack("IIIII", batch_size, seq_len, features, 2, features)
        workgroups = (len(data_flat) + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
        )

        # Download results
        result = self._download_buffer(buf_out, data_flat.nbytes, np.float32)
        result = result[: len(data_flat)].reshape(original_shape)

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
        if "fnn-xavier-init" not in self.shaders:
            # CPU fallback
            scale = np.sqrt(2.0 / input_dim)
            return (
                np.random.default_rng(seed)
                .normal(0, scale, (output_dim, input_dim))
                .astype(np.float32)
            )

        scale = np.sqrt(2.0 / input_dim)
        weights_flat = np.zeros(input_dim * output_dim, dtype=np.float32)

        # Acquire buffer from pool
        buf_weights = self._acquire_buffer(weights_flat.nbytes)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            "fnn-xavier-init", 1, push_constant_size=16
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "fnn-xavier-init", [(self._get_buffer_handle(buf_weights), weights_flat.nbytes)]
        )

        # Pack push constants: input_dim, output_dim, scale, seed
        push_constants = struct.pack("IIfI", input_dim, output_dim, scale, seed)

        # Dispatch: 2D workgroups (one thread per weight)
        workgroups_x = (input_dim + 15) // 16
        workgroups_y = (output_dim + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups_x, push_constants, workgroups_y, 1
        )

        # Download results
        result = self._download_buffer(buf_weights, weights_flat.nbytes, np.float32)
        result = result[: input_dim * output_dim]

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
            raise ValueError(
                f"grad_output size {len(grad_out)} != input_data size {total_elements}"
            )

        # Check if shader is available
        if "activation-gelu-backward" not in self.shaders:
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
            "activation-gelu-backward", 3, push_constant_size=4
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "activation-gelu-backward",
            [
                (self._get_buffer_handle(buf_grad_out), grad_out.nbytes),
                (self._get_buffer_handle(buf_input), input_flat.nbytes),
                (self._get_buffer_handle(buf_grad_in), input_flat.nbytes),
            ],
        )

        # Pack push constants
        push_constants = struct.pack("I", total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
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
            raise ValueError(
                f"grad_output size {len(grad_out)} != input_data size {total_elements}"
            )

        # Check if shader is available
        if "activation-relu-backward" not in self.shaders:
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
            "activation-relu-backward", 3, push_constant_size=4
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "activation-relu-backward",
            [
                (self._get_buffer_handle(buf_grad_out), grad_out.nbytes),
                (self._get_buffer_handle(buf_input), input_flat.nbytes),
                (self._get_buffer_handle(buf_grad_in), input_flat.nbytes),
            ],
        )

        # Pack push constants
        push_constants = struct.pack("I", total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
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
            raise ValueError(
                f"grad_output size {len(grad_out)} != input_data size {total_elements}"
            )

        # Check if shader is available
        if "activation-silu-backward" not in self.shaders:
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
            "activation-silu-backward", 3, push_constant_size=4
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "activation-silu-backward",
            [
                (self._get_buffer_handle(buf_grad_out), grad_out.nbytes),
                (self._get_buffer_handle(buf_input), input_flat.nbytes),
                (self._get_buffer_handle(buf_grad_in), input_flat.nbytes),
            ],
        )

        # Pack push constants
        push_constants = struct.pack("I", total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
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
            raise ValueError(
                f"grad_output size {len(grad_out)} != input_data size {total_elements}"
            )

        # Check if shader is available
        if "activation-gcu-backward" not in self.shaders:
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
            "activation-gcu-backward", 3, push_constant_size=4
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "activation-gcu-backward",
            [
                (self._get_buffer_handle(buf_grad_out), grad_out.nbytes),
                (self._get_buffer_handle(buf_input), input_flat.nbytes),
                (self._get_buffer_handle(buf_grad_in), input_flat.nbytes),
            ],
        )

        # Pack push constants
        push_constants = struct.pack("I", total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
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
            raise ValueError(
                f"grad_output size {len(grad_out)} != input_data size {total_elements}"
            )

        # Check if shader is available
        if "activation-roswish-backward" not in self.shaders:
            # CPU fallback
            x = input_flat
            beta_x = beta * x
            # Numerically stable sigmoid
            sigmoid_bx = np.where(
                beta_x >= 0, 1.0 / (1.0 + np.exp(-beta_x)), np.exp(beta_x) / (1.0 + np.exp(beta_x))
            )
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
            "activation-roswish-backward", 3, push_constant_size=12
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "activation-roswish-backward",
            [
                (self._get_buffer_handle(buf_grad_out), grad_out.nbytes),
                (self._get_buffer_handle(buf_input), input_flat.nbytes),
                (self._get_buffer_handle(buf_grad_in), input_flat.nbytes),
            ],
        )

        # Pack push constants
        push_constants = struct.pack("Iff", total_elements, float(alpha), float(beta))

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
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
            raise ValueError(
                f"grad_output shape mismatch: expected ({batch_size}, {hidden_dim}), got {grad_out_flat.shape}"
            )

        # Check if shader is available
        if "activation-swiglu-backward" not in self.shaders:
            # CPU fallback
            x1 = input_flat[:, :hidden_dim]
            x2 = input_flat[:, hidden_dim:]

            # Compute sigmoid(x2) numerically stable
            sigmoid_x2 = np.where(
                x2 >= 0, 1.0 / (1.0 + np.exp(-x2)), np.exp(x2) / (1.0 + np.exp(x2))
            )
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
            "activation-swiglu-backward", 3, push_constant_size=8
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "activation-swiglu-backward",
            [
                (self._get_buffer_handle(buf_grad_out), grad_out_1d.nbytes),
                (self._get_buffer_handle(buf_input), input_data_flat.nbytes),
                (self._get_buffer_handle(buf_grad_in), input_data_flat.nbytes),
            ],
        )

        # Pack push constants
        push_constants = struct.pack("II", output_elements, hidden_dim)

        # Dispatch
        workgroups = (output_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
        )

        # Download results
        result = self._download_buffer(buf_grad_in, input_data_flat.nbytes, np.float32)
        result = result[: len(input_data_flat)].reshape(original_shape)

        # Release buffers back to pool
        self._release_buffers([buf_grad_out, buf_input, buf_grad_in])

        return result

    def cross_entropy_backward(self, logits, targets, *, return_gpu_tensor=False):
        """
        GPU-accelerated cross-entropy backward pass (combined with softmax)

        Computes gradient of cross-entropy loss w.r.t. logits directly:
        grad = softmax(logits) - one_hot(targets)

        This is more numerically stable than computing softmax and
        cross-entropy gradients separately.

        Args:
            logits: Raw logits (batch_size, num_classes) or (B, S, num_classes).
                    May be VulkanTensor (zero-copy on GPU).
            targets: Target class indices as integers or floats.
            return_gpu_tensor: If True, return VulkanTensor (stays on GPU)

        Returns:
            Gradient w.r.t. logits, same shape as input logits.
        """
        from ..utils.tensor_conversion import VulkanTensor

        is_vt = isinstance(logits, VulkanTensor)

        if not is_vt:
            logits = np.asarray(logits, dtype=np.float32)
        original_shape = logits.shape

        if logits.ndim == 1:
            batch_size, num_classes = 1, logits.shape[0]
        elif logits.ndim == 3:
            b, s, num_classes = logits.shape
            batch_size = b * s
        else:
            batch_size, num_classes = logits.shape

        targets = np.asarray(targets).astype(np.float32).flatten()

        # Check if shader is available
        if "cross-entropy-backward" not in self.shaders:
            # CPU fallback: softmax - one_hot
            logits_np = logits.numpy() if is_vt else logits
            if logits_np.ndim == 1:
                logits_np = logits_np.reshape(1, -1)
            elif logits_np.ndim == 3:
                logits_np = logits_np.reshape(-1, logits_np.shape[-1])
            logits_max = np.max(logits_np, axis=1, keepdims=True)
            exp_logits = np.exp(np.clip(logits_np - logits_max, -60.0, 60.0))
            softmax = exp_logits / np.maximum(np.sum(exp_logits, axis=1, keepdims=True), 1e-12)

            one_hot = np.zeros_like(softmax)
            for i in range(batch_size):
                target_idx = int(targets[i])
                if 0 <= target_idx < num_classes:
                    one_hot[i, target_idx] = 1.0

            grad = softmax - one_hot
            return grad.reshape(original_shape)

        logits_nbytes = int(batch_size * num_classes * 4)
        grad_size = logits_nbytes

        # Accept VulkanTensor logits zero-copy via _prepare_input
        if is_vt:
            buf_logits, release_logits = self._prepare_input(logits, size=logits_nbytes)
        else:
            logits_flat = logits.flatten() if logits.ndim > 1 else logits.reshape(1, -1).flatten()
            buf_logits = self._acquire_buffer(logits_flat.nbytes)
            self._upload_buffer(buf_logits, logits_flat)
            release_logits = True

        buf_targets = self._acquire_buffer(targets.nbytes)
        buf_grad = self._acquire_buffer(grad_size)

        self._upload_buffer(buf_targets, targets)

        # Get or create pipeline (3 buffers, push constants: 2 uints = 8 bytes)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            "cross-entropy-backward", 3, push_constant_size=8
        )

        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "cross-entropy-backward",
            [
                (self._get_buffer_handle(buf_logits), logits_nbytes),
                (self._get_buffer_handle(buf_targets), targets.nbytes),
                (self._get_buffer_handle(buf_grad), grad_size),
            ],
        )

        push_constants = struct.pack("II", batch_size, num_classes)

        workgroups = batch_size
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
        )

        if return_gpu_tensor:
            result = self._wrap_output_tensor(buf_grad, (batch_size, num_classes))
            release_bufs = [buf_targets]
            if release_logits:
                release_bufs.append(buf_logits)
            self._release_buffers(release_bufs)
            return result

        result = self._download_buffer(buf_grad, grad_size, np.float32)
        result = result[: batch_size * num_classes].reshape(original_shape)

        release_bufs = [buf_targets, buf_grad]
        if release_logits:
            release_bufs.append(buf_logits)
        self._release_buffers(release_bufs)

        return result

    def cross_entropy_loss(self, logits, targets, label_smoothing=0.0, reduction="mean"):
        """
        GPU-accelerated cross-entropy loss using loss-cross-entropy.glsl.

        Args:
            logits: (batch, num_classes) or (batch, seq_len, num_classes).
                    May be VulkanTensor (zero-copy on GPU).
            targets: (batch,) or (batch, seq_len) target class indices
            label_smoothing: Optional smoothing factor in [0, 1)
            reduction: "mean", "sum", or "none"

        Returns:
            Scalar float for mean/sum, or per-position loss array for "none".
        """
        from ..utils.tensor_conversion import VulkanTensor

        is_vt = isinstance(logits, VulkanTensor)
        if not is_vt:
            logits = np.asarray(logits, dtype=np.float32)
        original_ndim = logits.ndim
        if original_ndim == 2:
            batch_size, vocab_size = logits.shape
            seq_len = 1
        elif original_ndim == 3:
            batch_size, seq_len, vocab_size = logits.shape
        else:
            raise ValueError("cross_entropy_loss expects logits shape (B,V) or (B,S,V)")

        if batch_size <= 0 or seq_len <= 0 or vocab_size <= 0:
            if reduction == "none":
                return np.zeros((batch_size, seq_len), dtype=np.float32)
            return 0.0

        tgt = np.asarray(targets)
        if tgt.ndim == 1:
            if tgt.shape[0] != batch_size:
                raise ValueError("targets shape mismatch for 2D logits")
            targets_2d = tgt.reshape(batch_size, 1)
        elif tgt.ndim == 2:
            if tgt.shape[0] != batch_size or tgt.shape[1] != seq_len:
                raise ValueError("targets shape mismatch for 3D logits")
            targets_2d = tgt
        else:
            raise ValueError("targets must be 1D or 2D")
        targets_u32 = np.asarray(np.clip(targets_2d, 0, max(0, vocab_size - 1)), dtype=np.uint32)

        if "loss-cross-entropy" not in self.shaders:
            # CPU fallback
            logits_np = logits.numpy() if is_vt else logits
            logits_3d = logits_np.reshape(batch_size, seq_len, vocab_size) if original_ndim != 3 else logits_np
            row_logits = logits_3d.reshape(-1, vocab_size).astype(np.float32, copy=False)
            row_targets = targets_u32.reshape(-1).astype(np.int64, copy=False)
            row_max = np.max(row_logits, axis=1, keepdims=True)
            shifted = np.clip(row_logits - row_max, -60.0, 60.0)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                lse = row_max.reshape(-1) + np.log(np.maximum(np.sum(np.exp(shifted), axis=1), 1e-12))
            tlog = row_logits[np.arange(row_logits.shape[0]), row_targets]
            losses = (lse - tlog).astype(np.float32, copy=False).reshape(batch_size, seq_len)
            if reduction == "none":
                return losses if original_ndim == 3 else losses.reshape(batch_size)
            if reduction == "sum":
                return float(np.sum(losses))
            return float(np.mean(losses))

        total_positions = int(batch_size * seq_len)
        logits_nbytes = int(batch_size * seq_len * vocab_size * 4)
        losses_nbytes = int(total_positions * 4)

        # Accept VulkanTensor logits zero-copy via _prepare_input
        if is_vt:
            buf_logits, release_logits = self._prepare_input(logits, size=logits_nbytes)
        else:
            logits_3d = logits.reshape(batch_size, seq_len, vocab_size) if original_ndim != 3 else logits
            logits_flat = np.ascontiguousarray(logits_3d, dtype=np.float32).reshape(-1)
            buf_logits = self._acquire_buffer(logits_flat.nbytes)
            self._upload_buffer(buf_logits, logits_flat)
            release_logits = True

        targets_flat = np.ascontiguousarray(targets_u32, dtype=np.uint32).reshape(-1)
        targets_nbytes = int(targets_flat.nbytes)

        buf_targets = self._acquire_buffer(targets_nbytes)
        buf_losses = self._acquire_buffer(losses_nbytes)
        buf_max = self._acquire_buffer(losses_nbytes)
        buf_sum = self._acquire_buffer(losses_nbytes)

        descriptor_set = None
        try:
            self._upload_buffer(buf_targets, targets_flat)

            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "loss-cross-entropy", 5, push_constant_size=20
            )
            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_logits), logits_nbytes),
                    (self._get_buffer_handle(buf_targets), targets_nbytes),
                    (self._get_buffer_handle(buf_losses), losses_nbytes),
                    (self._get_buffer_handle(buf_max), losses_nbytes),
                    (self._get_buffer_handle(buf_sum), losses_nbytes),
                ],
            )

            workgroups = (total_positions + 255) // 256
            ls = float(np.clip(float(label_smoothing), 0.0, 0.999))
            for pass_type in (0, 1, 2):
                push_constants = struct.pack(
                    "IIIIf",
                    int(batch_size),
                    int(seq_len),
                    int(vocab_size),
                    int(pass_type),
                    ls,
                )
                self.core._dispatch_compute(
                    pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
                )

            losses = self._download_buffer(buf_losses, losses_nbytes, np.float32).reshape(
                batch_size, seq_len
            )
            if reduction == "none":
                return losses if original_ndim == 3 else losses.reshape(batch_size)
            if reduction == "sum":
                return float(np.sum(losses, dtype=np.float64))
            return float(np.mean(losses, dtype=np.float64))
        finally:
            if descriptor_set is not None:
                try:
                    vkFreeDescriptorSets(
                        self.core.device,
                        self.core.descriptor_pool,
                        1,
                        [descriptor_set],
                    )
                except Exception:
                    pass
            release_bufs = [buf_targets, buf_losses, buf_max, buf_sum]
            if release_logits:
                release_bufs.append(buf_logits)
            self._release_buffers(release_bufs)

    # ------------------------------------------------------------------
    # Layer normalization (GPU accelerated with 3-pass shader)
    # ------------------------------------------------------------------
    def layernorm(
        self,
        x: np.ndarray,
        gamma: np.ndarray = None,
        beta: np.ndarray = None,
        eps: float = 1e-5,
        return_gpu_tensor: bool = False,
    ) -> np.ndarray:
        """
        GPU-accelerated LayerNorm using fnn-layernorm.glsl shader.
        Normalizes across the last dimension (features).

        3-pass algorithm:
        - Pass 0: Compute mean along feature dimension
        - Pass 1: Compute variance
        - Pass 2: Normalize and apply affine transformation
        """
        from ..utils.tensor_conversion import VulkanTensor

        is_vt = isinstance(x, VulkanTensor)
        original_shape = x.shape
        features = original_shape[-1]

        # Default gamma/beta if not provided
        if gamma is None:
            gamma = np.ones(features, dtype=np.float32)
        if beta is None:
            beta = np.zeros(features, dtype=np.float32)

        # Check if shader is available
        if "fnn-layernorm" not in self.shaders:
            # CPU fallback (numba-accelerated if available)
            x_np = x.numpy() if is_vt else np.asarray(x, dtype=np.float32)
            if NUMBA_AVAILABLE and numba_layernorm is not None:
                return numba_layernorm(x_np, gamma, beta, eps)
            else:
                mean = x_np.mean(axis=-1, keepdims=True)
                var = x_np.var(axis=-1, keepdims=True)
                normalized = (x_np - mean) / np.sqrt(var + eps)
                return normalized * gamma + beta

        # Handle different input shapes: (features,) or (batch, features) or (batch, seq, features)
        if len(original_shape) == 1:
            batch_size, seq_len = 1, 1
        elif len(original_shape) == 2:
            batch_size, features = original_shape
            seq_len = 1
        else:
            batch_size, seq_len, features = original_shape

        x_nbytes = int(batch_size * seq_len * features * 4)
        if is_vt:
            buf_input, release_input = self._prepare_input(x, size=x_nbytes)
        else:
            x_np = np.asarray(x, dtype=np.float32)
            if len(original_shape) == 1:
                x_np = x_np.reshape(1, 1, features)
            elif len(original_shape) == 2:
                x_np = x_np.reshape(batch_size, 1, features)
            x_flat = x_np.astype(np.float32).reshape(-1)
            x_nbytes = x_flat.nbytes
            buf_input = self._acquire_buffer(x_nbytes)
            self._upload_buffer(buf_input, x_flat)
            release_input = True

        gamma_flat = gamma.astype(np.float32).flatten()
        beta_flat = beta.astype(np.float32).flatten()

        total_positions = batch_size * seq_len
        total_elements = batch_size * seq_len * features

        # Acquire output buffer — DEVICE_LOCAL when returning GPU tensor
        use_device_local = return_gpu_tensor and hasattr(self, "_acquire_device_local_buffer")
        if use_device_local:
            buf_output = self._acquire_device_local_buffer(x_nbytes)
        else:
            buf_output = self._acquire_buffer(x_nbytes)
        buf_gamma = self._acquire_buffer(gamma_flat.nbytes)
        buf_beta = self._acquire_buffer(beta_flat.nbytes)
        buf_mean = self._acquire_buffer(total_positions * 4)
        buf_var = self._acquire_buffer(total_positions * 4)

        # Upload data
        self._upload_buffer(buf_gamma, gamma_flat)
        self._upload_buffer(buf_beta, beta_flat)

        # Get or create pipeline (6 buffers, push constants: 4 uints + 1 float = 20 bytes)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            "fnn-layernorm", 6, push_constant_size=20
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "fnn-layernorm",
            [
                (self._get_buffer_handle(buf_input), x_nbytes),
                (self._get_buffer_handle(buf_output), x_nbytes),
                (self._get_buffer_handle(buf_gamma), gamma_flat.nbytes),
                (self._get_buffer_handle(buf_beta), beta_flat.nbytes),
                (self._get_buffer_handle(buf_mean), total_positions * 4),
                (self._get_buffer_handle(buf_var), total_positions * 4),
            ],
        )

        # Run 3 passes — batch into one command buffer when possible
        use_batched = return_gpu_tensor and hasattr(self.core, "record_commands")
        if use_batched:
            with self.core.record_commands() as rec:
                for pass_type in range(3):
                    push_constants = struct.pack("IIIfI", batch_size, seq_len, features, eps, pass_type)
                    if pass_type < 2:
                        workgroups = (total_positions + 255) // 256
                    else:
                        workgroups = (total_elements + 255) // 256
                    rec.dispatch(pipeline, pipeline_layout, descriptor_set, (workgroups, 1, 1), push_constants)
                    if pass_type < 2:
                        rec.barrier()
        else:
            for pass_type in range(3):
                push_constants = struct.pack("IIIfI", batch_size, seq_len, features, eps, pass_type)
                if pass_type < 2:
                    workgroups = (total_positions + 255) // 256
                else:
                    workgroups = (total_elements + 255) // 256
                self.core._dispatch_compute(
                    pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
                )

        if return_gpu_tensor:
            if release_input:
                self._release_buffer(buf_input)
            self._release_buffers([buf_gamma, buf_beta, buf_mean, buf_var])
            if use_device_local and getattr(buf_output, "is_device_local", False):
                return self._wrap_output_tensor_device_local(buf_output, original_shape)
            return self._wrap_output_tensor(buf_output, original_shape)

        # Download result
        result = self._download_buffer(buf_output, x_nbytes, np.float32)

        # Release buffers back to pool
        if release_input:
            self._release_buffer(buf_input)
        self._release_buffers([buf_output, buf_gamma, buf_beta, buf_mean, buf_var])

        return result.reshape(original_shape)

    # ------------------------------------------------------------------
    # Linear projection (GPU accelerated)
    # ------------------------------------------------------------------
    def linear(self, x, weights, bias=None, return_gpu_tensor=False):
        """
        GPU-accelerated linear projection using fnn-linear.glsl shader.
        output = x @ W^T + b

        Args:
            x: Input tensor (batch_seq, input_dim) or (batch, seq, input_dim).
                Can be numpy array or VulkanTensor.
            weights: Weight matrix (output_dim, input_dim)
            bias: Optional bias vector (output_dim,)
            return_gpu_tensor: If True, return VulkanTensor (stays on GPU)

        Returns:
            Output tensor with same batch dimensions, last dim = output_dim
        """
        from ..utils.tensor_conversion import VulkanTensor

        is_vt = isinstance(x, VulkanTensor)

        # Check if shader is available
        if "fnn-linear" not in self.shaders:
            # CPU fallback (numba if available)
            x_np = x.numpy() if is_vt else np.asarray(x, dtype=np.float32)
            if NUMBA_AVAILABLE and numba_linear is not None:
                return numba_linear(
                    x_np.astype(np.float32),
                    weights.astype(np.float32),
                    bias.astype(np.float32) if bias is not None else None,
                )
            out = np.matmul(x_np, weights.T)
            if bias is not None:
                out = out + bias
            return out

        # Get shape info without forcing download for VulkanTensor
        original_shape = x.shape
        output_dim, input_dim = weights.shape

        # Reshape to 2D: (batch_seq, input_dim)
        if len(original_shape) > 2:
            batch_seq = int(np.prod(original_shape[:-1]))
        else:
            batch_seq = original_shape[0]

        # Prepare input data
        if is_vt:
            # Zero-copy: use existing GPU buffer
            input_nbytes = batch_seq * input_dim * 4
            buf_input, release_input = self._prepare_input(x, size=input_nbytes)
        else:
            x_np = np.asarray(x, dtype=np.float32)
            if len(original_shape) > 2:
                x_2d = x_np.reshape(batch_seq, input_dim)
            else:
                x_2d = x_np
            x_flat = x_2d.astype(np.float32).flatten()
            input_nbytes = x_flat.nbytes
            buf_input = self._acquire_buffer(input_nbytes)
            self._upload_buffer(buf_input, x_flat)
            release_input = True

        # Flatten weights
        w_np = np.ascontiguousarray(weights, dtype=np.float32)
        w_nbytes = int(np.prod(weights.shape)) * 4

        # Output size
        output_size = batch_seq * output_dim * 4  # float32

        # Use DEVICE_LOCAL for weights/output when returning GPU tensor
        use_device_local = return_gpu_tensor and hasattr(self, "_acquire_device_local_buffer")

        if use_device_local:
            buf_weights, release_weights = self._get_or_upload_weight_device_local(w_np)
            buf_output = self._acquire_device_local_buffer(output_size)
        else:
            buf_weights, release_weights = self._get_or_upload_weight(w_np)
            buf_output = self._acquire_buffer(output_size)

        # Handle bias
        has_bias = 1 if bias is not None else 0
        if bias is not None:
            bias_np = np.ascontiguousarray(bias, dtype=np.float32)
            buf_bias, release_bias = self._get_or_upload_weight(bias_np)
            bias_flat = bias_np.flatten()
        else:
            # Create dummy bias buffer (shader expects 4 buffers)
            buf_bias = self._acquire_buffer(4)
            release_bias = True
            bias_flat = None

        # Get or create pipeline (4 buffers, push constants: 4 uints = 16 bytes)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            "fnn-linear", 4, push_constant_size=16
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "fnn-linear",
            [
                (self._get_buffer_handle(buf_input), input_nbytes),
                (self._get_buffer_handle(buf_weights), w_nbytes),
                (
                    self._get_buffer_handle(buf_bias),
                    bias_flat.nbytes if bias_flat is not None else 4,
                ),
                (self._get_buffer_handle(buf_output), output_size),
            ],
        )

        # Push constants: batch_seq, input_dim, output_dim, has_bias
        push_constants = struct.pack("IIII", batch_seq, input_dim, output_dim, has_bias)

        # 2D dispatch: rows = batch_seq, cols = output_dim
        # Shader uses 16x16 workgroups
        workgroups_x = (output_dim + 15) // 16
        workgroups_y = (batch_seq + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups_x, push_constants, workgroups_y
        )

        # Compute output shape
        if len(original_shape) > 2:
            output_shape = original_shape[:-1] + (output_dim,)
        else:
            output_shape = (batch_seq, output_dim)

        if return_gpu_tensor:
            if use_device_local and getattr(buf_output, "is_device_local", False):
                result = self._wrap_output_tensor_device_local(buf_output, output_shape)
            else:
                result = self._wrap_output_tensor(buf_output, output_shape)
            # Release only owned buffers (cache-owned buffers are not released)
            if release_input:
                self._release_buffer(buf_input)
            if release_weights:
                self._release_buffer(buf_weights)
            if release_bias:
                self._release_buffer(buf_bias)
            return result
        else:
            # Download result
            result = self._download_buffer(buf_output, output_size, np.float32)

            # Release only owned buffers (cache-owned buffers are not released)
            if release_input:
                self._release_buffer(buf_input)
            if release_weights:
                self._release_buffer(buf_weights)
            if release_bias:
                self._release_buffer(buf_bias)
            self._release_buffer(buf_output)

            return result.reshape(output_shape)

    # ------------------------------------------------------------------
    # Linear backward pass
    # ------------------------------------------------------------------
    def linear_backward(
        self,
        grad_output: np.ndarray,
        x: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray | None = None,
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
        use_gemm = "gemm_mnk" in self.shaders and batch * in_features >= 4096

        if use_gemm:
            # ============ GEMM-based backward ============

            # 1) grad_input = grad_output @ weights
            #    (batch*seq, out_features) @ (out_features, in_features) = (batch*seq, in_features)
            grad_input_2d = self.gemm(grad_output_2d, weights, force_fp32=True)

            # 2) grad_weight = grad_output.T @ x
            #    (out_features, batch*seq) @ (batch*seq, in_features) = (out_features, in_features)
            grad_weight = self.gemm(grad_output_2d.T.copy(), x_2d, force_fp32=True)

            # 3) grad_bias = sum over batch dimension
            grad_bias = (
                np.sum(grad_output_2d, axis=0, dtype=np.float32) if bias is not None else None
            )

            # Reshape grad_input back to original shape
            if grad_output.ndim == 3:
                grad_input = grad_input_2d.reshape(grad_output_shape[0], grad_output_shape[1], -1)
            else:
                grad_input = grad_input_2d

            return grad_input, grad_weight, grad_bias

        # ============ Fallback: use fnn-linear-backward shader or CPU ============
        if "fnn-linear-backward" not in self.shaders:
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

        # Always 6 bindings - shader declares binding 5 for grad_bias
        if bias is not None:
            buf_grad_b = self._acquire_buffer(grad_bias_size)
        else:
            buf_grad_b = self._acquire_buffer(4)  # dummy buffer for binding 5
        buffers_list.append(buf_grad_b)
        buffers.append(
            (self._get_buffer_handle(buf_grad_b), grad_bias_size if bias is not None else 4)
        )

        # Upload data
        self._upload_buffer(buf_grad_out, grad_out_flat)
        self._upload_buffer(buf_x, x_flat)
        self._upload_buffer(buf_w, w_flat)
        # Initialize output buffers to zero
        self._upload_buffer(buf_grad_in, np.zeros(batch_seq * input_dim, dtype=np.float32))
        self._upload_buffer(buf_grad_w, np.zeros(output_dim * input_dim, dtype=np.float32))
        if bias is not None:
            self._upload_buffer(buf_grad_b, np.zeros(output_dim, dtype=np.float32))

        # Get or create pipeline (always 6 bindings to match shader)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            "fnn-linear-backward", 6, push_constant_size=16
        )

        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(desc_layout, buffers)

        try:
            # Pass 0: Compute grad_input
            push_constants = struct.pack("IIII", batch_seq, input_dim, output_dim, 0)
            workgroups_x = (input_dim + 15) // 16
            workgroups_y = (batch_seq + 15) // 16
            self.core._dispatch_compute(
                pipeline,
                pipeline_layout,
                descriptor_set,
                workgroups_x,
                push_constants,
                workgroups_y,
            )

            # Pass 1: Compute grad_weight
            push_constants = struct.pack("IIII", batch_seq, input_dim, output_dim, 1)
            workgroups_x = (input_dim + 15) // 16
            workgroups_y = (output_dim + 15) // 16
            self.core._dispatch_compute(
                pipeline,
                pipeline_layout,
                descriptor_set,
                workgroups_x,
                push_constants,
                workgroups_y,
            )

            # Pass 2: Compute grad_bias (if bias exists)
            if bias is not None:
                push_constants = struct.pack("IIII", batch_seq, input_dim, output_dim, 2)
                workgroups = (output_dim + 255) // 256
                self.core._dispatch_compute(
                    pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
                )

            # Download results
            grad_input_flat = self._download_buffer(buf_grad_in, grad_input_size, np.float32)
            grad_weight_flat = self._download_buffer(buf_grad_w, grad_weight_size, np.float32)
            if bias is not None:
                grad_bias_flat = self._download_buffer(buf_grad_b, grad_bias_size, np.float32)
            else:
                grad_bias_flat = None

            # Reshape
            grad_input_2d = grad_input_flat[: batch_seq * input_dim].reshape(batch_seq, input_dim)
            grad_weight = grad_weight_flat[: output_dim * input_dim].reshape(output_dim, input_dim)
            grad_bias = grad_bias_flat[:output_dim] if grad_bias_flat is not None else None

            # Reshape grad_input back to original shape
            if grad_output.ndim == 3:
                grad_input = grad_input_2d.reshape(grad_output_shape[0], grad_output_shape[1], -1)
            else:
                grad_input = grad_input_2d

            return grad_input, grad_weight, grad_bias
        finally:
            vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
            self._release_buffers(buffers_list)

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
        eps: float = 1e-5,
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
        if "fnn-layernorm-backward" not in self.shaders:
            # CPU fallback
            std = np.sqrt(var + eps)
            x_norm = (x - mean) / std

            # Gradients w.r.t. gamma and beta
            grad_gamma = np.sum(grad_output * x_norm, axis=tuple(range(len(original_shape) - 1)))
            grad_beta = np.sum(grad_output, axis=tuple(range(len(original_shape) - 1)))

            # Gradient w.r.t. input
            N = features
            dx_norm = grad_output * gamma

            dvar = np.sum(
                dx_norm * (x - mean) * (-0.5) * (var + eps) ** (-1.5), axis=-1, keepdims=True
            )
            dmean = np.sum(dx_norm * (-1.0 / std), axis=-1, keepdims=True) + dvar * np.mean(
                -2.0 * (x - mean), axis=-1, keepdims=True
            )

            grad_input = dx_norm / std + dvar * 2.0 * (x - mean) / N + dmean / N

            return (
                grad_input.astype(np.float32),
                grad_gamma.astype(np.float32),
                grad_beta.astype(np.float32),
            )

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

        buffers_list = [
            buf_grad_out,
            buf_input,
            buf_gamma,
            buf_mean,
            buf_var,
            buf_grad_in,
            buf_grad_gamma,
            buf_grad_beta,
        ]

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
            "fnn-layernorm-backward", 8, push_constant_size=20
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
        push_constants = struct.pack("IIIfI", batch_size, seq_len, features, eps, 0)
        workgroups = (total_positions + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
        )

        # Pass 1: Compute grad_input
        push_constants = struct.pack("IIIfI", batch_size, seq_len, features, eps, 1)
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
        )

        # Pass 2: Compute grad_gamma and grad_beta
        push_constants = struct.pack("IIIfI", batch_size, seq_len, features, eps, 2)
        workgroups = (features + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
        )

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
        self, grad_output: np.ndarray, softmax_output: np.ndarray, dim: int = -1
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
        if "activation-softmax-backward" not in self.shaders:
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
            "activation-softmax-backward", 3, push_constant_size=12
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "activation-softmax-backward",
            [
                (self._get_buffer_handle(buf_grad_out), grad_out_flat.nbytes),
                (self._get_buffer_handle(buf_softmax), softmax_flat.nbytes),
                (self._get_buffer_handle(buf_grad_in), grad_out_flat.nbytes),
            ],
        )

        # Push constants: batch_size, seq_len, num_classes
        push_constants = struct.pack("III", batch_size, seq_len, num_classes)

        # Dispatch: one thread per row
        workgroups = (total_rows + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
        )

        # Download results
        grad_input = self._download_buffer(buf_grad_in, grad_out_flat.nbytes, np.float32)

        # Release buffers back to pool
        self._release_buffers([buf_grad_out, buf_softmax, buf_grad_in])

        return grad_input.reshape(original_shape)

    # ------------------------------------------------------------------
    # Dropout (CPU fallback)
    # ------------------------------------------------------------------
    def dropout(
        self,
        x: np.ndarray,
        dropout_prob: float = 0.1,
        is_training: bool = True,
        seed: int | None = None,
    ) -> np.ndarray:
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
    def residual(self, x: np.ndarray, module_output: np.ndarray, return_gpu_tensor: bool = False) -> np.ndarray:
        """
        Residual connection: output = x + module_output

        Uses: fnn-residual.glsl

        Args:
            x: Input tensor
            module_output: Output from module

        Returns:
            x + module_output
        """
        from ..utils.tensor_conversion import VulkanTensor

        is_vt_x = isinstance(x, VulkanTensor)
        is_vt_m = isinstance(module_output, VulkanTensor)
        x_shape = tuple(int(d) for d in x.shape)
        total_elements = int(np.prod(x_shape))
        x_nbytes = int(total_elements * 4)

        # Check if shader is available
        if "fnn-residual" not in self.shaders:
            # CPU fallback
            x_np = x.numpy() if is_vt_x else np.asarray(x, dtype=np.float32)
            m_np = module_output.numpy() if is_vt_m else np.asarray(module_output, dtype=np.float32)
            return (x_np + m_np).astype(np.float32)

        # GPU implementation
        if is_vt_x:
            buf_x, release_x = self._prepare_input(x, size=x_nbytes)
        else:
            x_np = np.ascontiguousarray(np.asarray(x, dtype=np.float32)).reshape(-1)
            buf_x = self._acquire_buffer(x_nbytes)
            self._upload_buffer(buf_x, x_np)
            release_x = True

        if is_vt_m:
            buf_module, release_module = self._prepare_input(module_output, size=x_nbytes)
        else:
            module_np = np.ascontiguousarray(np.asarray(module_output, dtype=np.float32)).reshape(-1)
            buf_module = self._acquire_buffer(x_nbytes)
            self._upload_buffer(buf_module, module_np)
            release_module = True

        # Acquire buffers from pool
        buf_out = self._acquire_buffer(x_nbytes)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            "fnn-residual", 3, push_constant_size=4
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "fnn-residual",
            [
                (self._get_buffer_handle(buf_x), x_nbytes),
                (self._get_buffer_handle(buf_module), x_nbytes),
                (self._get_buffer_handle(buf_out), x_nbytes),
            ],
        )

        # Pack push constants
        push_constants = struct.pack("I", total_elements)

        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
        )

        if return_gpu_tensor:
            # Caller takes ownership of output pooled buffer through VulkanTensor wrapper.
            if release_x:
                self._release_buffer(buf_x)
            if release_module:
                self._release_buffer(buf_module)
            return self._wrap_output_tensor(buf_out, x_shape)

        # Download results
        result = self._download_buffer(buf_out, x_nbytes, np.float32)
        result = result[:total_elements].reshape(x_shape)

        # Release buffers back to pool
        if release_x:
            self._release_buffer(buf_x)
        if release_module:
            self._release_buffer(buf_module)
        self._release_buffer(buf_out)

        return result

    # ==================================================================
    # FUSED OPERATIONS
    # ==================================================================
    # These combine common operation pairs into single GPU dispatches
    # to reduce memory bandwidth and kernel launch overhead.

    def fused_linear_gelu(
        self,
        x,
        weights: np.ndarray,
        bias: np.ndarray | None = None,
        return_gpu_tensor=False,
    ) -> np.ndarray:
        """
        Fused Linear + GELU: GELU(x @ W.T + b)

        Uses: fused-linear-gelu.glsl

        Common in Transformer FFN (first layer).

        Args:
            x: Input tensor (..., input_dim), numpy array or VulkanTensor
            weights: Weight matrix (output_dim, input_dim)
            bias: Optional bias (output_dim,)
            return_gpu_tensor: If True, return VulkanTensor (stays on GPU)

        Returns:
            GELU(Linear(x))
        """
        if "fused-linear-gelu" not in self.shaders:
            # Fallback to separate operations
            linear_out = self.linear(x, weights, bias, return_gpu_tensor=return_gpu_tensor)
            return self.activation_gelu(linear_out, return_gpu_tensor=return_gpu_tensor)

        # GPU implementation
        original_shape = x.shape
        input_dim = x.shape[-1]
        output_dim = weights.shape[0]

        # Flatten batch dimensions
        if len(original_shape) > 2:
            batch_seq = int(np.prod(original_shape[:-1]))
        else:
            batch_seq = original_shape[0] if len(original_shape) == 2 else 1

        input_nbytes = batch_seq * input_dim * 4
        output_size = batch_seq * output_dim * 4

        # Prepare input (zero-copy for VulkanTensor)
        buf_input, release_input = self._prepare_input(x, size=input_nbytes)

        # Use weight cache for weights and bias
        w_np = np.ascontiguousarray(weights, dtype=np.float32)
        w_nbytes = int(np.prod(weights.shape)) * 4
        buf_weights, release_weights = self._get_or_upload_weight(w_np)

        if bias is not None:
            bias_np = np.ascontiguousarray(bias, dtype=np.float32)
            buf_bias, release_bias = self._get_or_upload_weight(bias_np)
            b_nbytes = bias_np.size * 4
            has_bias = 1
        else:
            b_flat = np.zeros(output_dim, dtype=np.float32)
            buf_bias = self._acquire_buffer(b_flat.nbytes)
            self._upload_buffer(buf_bias, b_flat)
            b_nbytes = b_flat.nbytes
            release_bias = True
            has_bias = 0

        buf_output = self._acquire_buffer(output_size)

        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            "fused-linear-gelu", 4, push_constant_size=16
        )

        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "fused-linear-gelu",
            [
                (self._get_buffer_handle(buf_input), input_nbytes),
                (self._get_buffer_handle(buf_weights), w_nbytes),
                (self._get_buffer_handle(buf_bias), b_nbytes),
                (self._get_buffer_handle(buf_output), output_size),
            ],
        )

        # Pack push constants: batch_seq, input_dim, output_dim, has_bias
        push_constants = struct.pack("IIII", batch_seq, input_dim, output_dim, has_bias)

        # Dispatch (2D: rows = batch_seq, cols = output_dim)
        workgroups_x = (output_dim + 15) // 16
        workgroups_y = (batch_seq + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups_x, push_constants, workgroups_y
        )

        # Compute output shape
        if len(original_shape) > 2:
            output_shape = original_shape[:-1] + (output_dim,)
        else:
            output_shape = (batch_seq, output_dim)

        if return_gpu_tensor:
            result = self._wrap_output_tensor(buf_output, output_shape)
            if release_input:
                self._release_buffer(buf_input)
            if release_weights:
                self._release_buffer(buf_weights)
            if release_bias:
                self._release_buffer(buf_bias)
            return result
        else:
            result = self._download_buffer(buf_output, output_size, np.float32)
            if release_input:
                self._release_buffer(buf_input)
            if release_weights:
                self._release_buffer(buf_weights)
            if release_bias:
                self._release_buffer(buf_bias)
            self._release_buffer(buf_output)
            return result.reshape(output_shape)

    def fused_linear_relu(
        self,
        x,
        weights: np.ndarray,
        bias: np.ndarray | None = None,
        return_gpu_tensor=False,
    ) -> np.ndarray:
        """
        Fused Linear + ReLU: ReLU(x @ W.T + b)

        Uses: fused-linear-relu.glsl

        Args:
            x: Input tensor (..., input_dim), numpy array or VulkanTensor
            weights: Weight matrix (output_dim, input_dim)
            bias: Optional bias (output_dim,)
            return_gpu_tensor: If True, return VulkanTensor (stays on GPU)

        Returns:
            ReLU(Linear(x))
        """
        if "fused-linear-relu" not in self.shaders:
            linear_out = self.linear(x, weights, bias, return_gpu_tensor=return_gpu_tensor)
            return self.activation_relu(linear_out, return_gpu_tensor=return_gpu_tensor)

        original_shape = x.shape
        input_dim = x.shape[-1]
        output_dim = weights.shape[0]

        if len(original_shape) > 2:
            batch_seq = int(np.prod(original_shape[:-1]))
        else:
            batch_seq = original_shape[0] if len(original_shape) == 2 else 1

        input_nbytes = batch_seq * input_dim * 4
        output_size = batch_seq * output_dim * 4

        buf_input, release_input = self._prepare_input(x, size=input_nbytes)

        w_np = np.ascontiguousarray(weights, dtype=np.float32)
        w_nbytes = int(np.prod(weights.shape)) * 4
        buf_weights, release_weights = self._get_or_upload_weight(w_np)

        if bias is not None:
            bias_np = np.ascontiguousarray(bias, dtype=np.float32)
            buf_bias, release_bias = self._get_or_upload_weight(bias_np)
            b_nbytes = bias_np.size * 4
            has_bias = 1
        else:
            b_flat = np.zeros(output_dim, dtype=np.float32)
            buf_bias = self._acquire_buffer(b_flat.nbytes)
            self._upload_buffer(buf_bias, b_flat)
            b_nbytes = b_flat.nbytes
            release_bias = True
            has_bias = 0

        buf_output = self._acquire_buffer(output_size)

        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            "fused-linear-relu", 4, push_constant_size=16
        )

        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "fused-linear-relu",
            [
                (self._get_buffer_handle(buf_input), input_nbytes),
                (self._get_buffer_handle(buf_weights), w_nbytes),
                (self._get_buffer_handle(buf_bias), b_nbytes),
                (self._get_buffer_handle(buf_output), output_size),
            ],
        )

        push_constants = struct.pack("IIII", batch_seq, input_dim, output_dim, has_bias)

        workgroups_x = (output_dim + 15) // 16
        workgroups_y = (batch_seq + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups_x, push_constants, workgroups_y
        )

        if len(original_shape) > 2:
            output_shape = original_shape[:-1] + (output_dim,)
        else:
            output_shape = (batch_seq, output_dim)

        if return_gpu_tensor:
            result = self._wrap_output_tensor(buf_output, output_shape)
            if release_input:
                self._release_buffer(buf_input)
            if release_weights:
                self._release_buffer(buf_weights)
            if release_bias:
                self._release_buffer(buf_bias)
            return result
        else:
            result = self._download_buffer(buf_output, output_size, np.float32)
            if release_input:
                self._release_buffer(buf_input)
            if release_weights:
                self._release_buffer(buf_weights)
            if release_bias:
                self._release_buffer(buf_bias)
            self._release_buffer(buf_output)
            return result.reshape(output_shape)

    def fused_linear_silu(
        self,
        x,
        weights: np.ndarray,
        bias: np.ndarray | None = None,
        return_gpu_tensor=False,
    ) -> np.ndarray:
        """
        Fused Linear + SiLU: SiLU(x @ W.T + b)

        Uses: fused-linear-silu.glsl

        Common in LLaMA, Mistral FFN layers.

        Args:
            x: Input tensor (..., input_dim), numpy array or VulkanTensor
            weights: Weight matrix (output_dim, input_dim)
            bias: Optional bias (output_dim,)
            return_gpu_tensor: If True, return VulkanTensor (stays on GPU)

        Returns:
            SiLU(Linear(x))
        """
        if "fused-linear-silu" not in self.shaders:
            linear_out = self.linear(x, weights, bias, return_gpu_tensor=return_gpu_tensor)
            return self.activation_silu(linear_out, return_gpu_tensor=return_gpu_tensor)

        original_shape = x.shape
        input_dim = x.shape[-1]
        output_dim = weights.shape[0]

        if len(original_shape) > 2:
            batch_seq = int(np.prod(original_shape[:-1]))
        else:
            batch_seq = original_shape[0] if len(original_shape) == 2 else 1

        input_nbytes = batch_seq * input_dim * 4
        output_size = batch_seq * output_dim * 4

        buf_input, release_input = self._prepare_input(x, size=input_nbytes)

        w_np = np.ascontiguousarray(weights, dtype=np.float32)
        w_nbytes = int(np.prod(weights.shape)) * 4
        buf_weights, release_weights = self._get_or_upload_weight(w_np)

        if bias is not None:
            bias_np = np.ascontiguousarray(bias, dtype=np.float32)
            buf_bias, release_bias = self._get_or_upload_weight(bias_np)
            b_nbytes = bias_np.size * 4
            has_bias = 1
        else:
            b_flat = np.zeros(output_dim, dtype=np.float32)
            buf_bias = self._acquire_buffer(b_flat.nbytes)
            self._upload_buffer(buf_bias, b_flat)
            b_nbytes = b_flat.nbytes
            release_bias = True
            has_bias = 0

        buf_output = self._acquire_buffer(output_size)

        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            "fused-linear-silu", 4, push_constant_size=16
        )

        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "fused-linear-silu",
            [
                (self._get_buffer_handle(buf_input), input_nbytes),
                (self._get_buffer_handle(buf_weights), w_nbytes),
                (self._get_buffer_handle(buf_bias), b_nbytes),
                (self._get_buffer_handle(buf_output), output_size),
            ],
        )

        push_constants = struct.pack("IIII", batch_seq, input_dim, output_dim, has_bias)

        workgroups_x = (output_dim + 15) // 16
        workgroups_y = (batch_seq + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups_x, push_constants, workgroups_y
        )

        if len(original_shape) > 2:
            output_shape = original_shape[:-1] + (output_dim,)
        else:
            output_shape = (batch_seq, output_dim)

        if return_gpu_tensor:
            result = self._wrap_output_tensor(buf_output, output_shape)
            if release_input:
                self._release_buffer(buf_input)
            if release_weights:
                self._release_buffer(buf_weights)
            if release_bias:
                self._release_buffer(buf_bias)
            return result
        else:
            result = self._download_buffer(buf_output, output_size, np.float32)
            if release_input:
                self._release_buffer(buf_input)
            if release_weights:
                self._release_buffer(buf_weights)
            if release_bias:
                self._release_buffer(buf_bias)
            self._release_buffer(buf_output)
            return result.reshape(output_shape)

    def fused_linear_gcu(
        self, x: np.ndarray, weights: np.ndarray, bias: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Fused Linear + GCU: GCU(x @ W.T + b)
        Uses: fused-linear-gcu.glsl
        """
        if "fused-linear-gcu" not in self.shaders:
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
            "fused-linear-gcu", 4, push_constant_size=16
        )

        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "fused-linear-gcu",
            [
                (self._get_buffer_handle(buf_input), x_flat.nbytes),
                (self._get_buffer_handle(buf_weights), w_flat.nbytes),
                (self._get_buffer_handle(buf_bias), b_flat.nbytes),
                (self._get_buffer_handle(buf_output), output_size),
            ],
        )

        push_constants = struct.pack("IIII", batch_seq, input_dim, output_dim, has_bias)

        workgroups_x = (output_dim + 15) // 16
        workgroups_y = (batch_seq + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups_x, push_constants, workgroups_y
        )

        result = self._download_buffer(buf_output, output_size, np.float32)
        self._release_buffers([buf_input, buf_weights, buf_bias, buf_output])

        if len(original_shape) > 2:
            output_shape = original_shape[:-1] + (output_dim,)
            return result.reshape(output_shape)
        else:
            return result.reshape(batch_seq, output_dim)

    def fused_linear_tanh(
        self,
        x,
        weights: np.ndarray,
        bias: np.ndarray | None = None,
        return_gpu_tensor=False,
    ) -> np.ndarray:
        """
        Fused Linear + Tanh: tanh(x @ W.T + b)

        Uses: fused-linear-tanh.glsl

        Used by VSA Reasoning Head to project hidden states to bipolar VSA space.

        Args:
            x: Input tensor (..., input_dim), numpy array or VulkanTensor
            weights: Weight matrix (output_dim, input_dim)
            bias: Optional bias (output_dim,)
            return_gpu_tensor: If True, return VulkanTensor (stays on GPU)

        Returns:
            tanh(Linear(x))
        """
        if "fused-linear-tanh" not in self.shaders:
            linear_out = self.linear(x, weights, bias, return_gpu_tensor=return_gpu_tensor)
            return self.activation_tanh(linear_out, return_gpu_tensor=return_gpu_tensor)

        original_shape = x.shape
        input_dim = x.shape[-1]
        output_dim = weights.shape[0]

        if len(original_shape) > 2:
            batch_seq = int(np.prod(original_shape[:-1]))
        else:
            batch_seq = original_shape[0] if len(original_shape) == 2 else 1

        input_nbytes = batch_seq * input_dim * 4
        output_size = batch_seq * output_dim * 4

        buf_input, release_input = self._prepare_input(x, size=input_nbytes)

        w_np = np.ascontiguousarray(weights, dtype=np.float32)
        buf_weights, release_weights = self._get_or_upload_weight(w_np)

        if bias is not None:
            bias_np = np.ascontiguousarray(bias, dtype=np.float32)
            buf_bias, release_bias = self._get_or_upload_weight(bias_np)
            b_nbytes = bias_np.size * 4
            has_bias = 1
        else:
            b_flat = np.zeros(output_dim, dtype=np.float32)
            buf_bias = self._acquire_buffer(b_flat.nbytes)
            self._upload_buffer(buf_bias, b_flat)
            b_nbytes = b_flat.nbytes
            release_bias = True
            has_bias = 0

        buf_output = self._acquire_buffer(output_size)

        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            "fused-linear-tanh", 4, push_constant_size=16
        )

        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "fused-linear-tanh",
            [
                (self._get_buffer_handle(buf_input), input_nbytes),
                (self._get_buffer_handle(buf_weights), w_np.size * 4),
                (self._get_buffer_handle(buf_bias), b_nbytes),
                (self._get_buffer_handle(buf_output), output_size),
            ],
        )

        push_constants = struct.pack("IIII", batch_seq, input_dim, output_dim, has_bias)

        workgroups_x = (output_dim + 15) // 16
        workgroups_y = (batch_seq + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups_x, push_constants, workgroups_y
        )

        if len(original_shape) > 2:
            output_shape = original_shape[:-1] + (output_dim,)
        else:
            output_shape = (batch_seq, output_dim)

        if return_gpu_tensor:
            result = self._wrap_output_tensor(buf_output, output_shape)
            if release_input:
                self._release_buffer(buf_input)
            if release_weights:
                self._release_buffer(buf_weights)
            if release_bias:
                self._release_buffer(buf_bias)
            return result
        else:
            result = self._download_buffer(buf_output, output_size, np.float32)
            if release_input:
                self._release_buffer(buf_input)
            if release_weights:
                self._release_buffer(buf_weights)
            if release_bias:
                self._release_buffer(buf_bias)
            self._release_buffer(buf_output)
            return result.reshape(output_shape)

    def fused_linear_roswish(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray | None = None,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> np.ndarray:
        """
        Fused Linear + RoSwish: RoSwish(x @ W.T + b)
        Uses: fused-linear-roswish.glsl
        """
        if "fused-linear-roswish" not in self.shaders:
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
            "fused-linear-roswish", 4, push_constant_size=24
        )

        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "fused-linear-roswish",
            [
                (self._get_buffer_handle(buf_input), x_flat.nbytes),
                (self._get_buffer_handle(buf_weights), w_flat.nbytes),
                (self._get_buffer_handle(buf_bias), b_flat.nbytes),
                (self._get_buffer_handle(buf_output), output_size),
            ],
        )

        push_constants = struct.pack(
            "IIIIff", batch_seq, input_dim, output_dim, has_bias, alpha, beta
        )

        workgroups_x = (output_dim + 15) // 16
        workgroups_y = (batch_seq + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups_x, push_constants, workgroups_y
        )

        result = self._download_buffer(buf_output, output_size, np.float32)
        self._release_buffers([buf_input, buf_weights, buf_bias, buf_output])

        if len(original_shape) > 2:
            output_shape = original_shape[:-1] + (output_dim,)
            return result.reshape(output_shape)
        else:
            return result.reshape(batch_seq, output_dim)

    # ------------------------------------------------------------------
    # RMSNorm (GPU accelerated)
    # ------------------------------------------------------------------
    def rms_norm(
        self,
        x: np.ndarray,
        weight: np.ndarray = None,
        eps: float = 1e-5,
        return_gpu_tensor: bool = False,
    ) -> np.ndarray:
        """
        GPU-accelerated RMSNorm using rms-norm.glsl shader.

        RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight

        Unlike LayerNorm, RMSNorm has no mean subtraction and no bias.
        Used by Llama, Gemma, and other modern architectures.

        2-pass algorithm:
        - Pass 0: Compute mean(x^2) along feature dimension
        - Pass 1: Normalize and scale by weight

        Args:
            x: Input tensor (..., features). Can be numpy array or VulkanTensor.
            weight: Scale parameter (features,). Defaults to ones.
            eps: Small constant for numerical stability (default: 1e-5)
            return_gpu_tensor: If True, return VulkanTensor (stays on GPU)

        Returns:
            RMSNorm-normalized tensor (same shape as input)
        """
        from ..utils.tensor_conversion import VulkanTensor

        is_vt = isinstance(x, VulkanTensor)
        original_shape = x.shape
        features = original_shape[-1]

        if weight is None:
            weight = np.ones(features, dtype=np.float32)

        # CPU fallback if shader unavailable
        if "rms-norm" not in self.shaders:
            x_np = x.numpy() if is_vt else np.asarray(x, dtype=np.float32)
            mean_sq = np.mean(x_np ** 2, axis=-1, keepdims=True)
            normed = x_np * (1.0 / np.sqrt(mean_sq + eps))
            return normed * weight

        # Determine batch_size and seq_len from shape
        if len(original_shape) == 1:
            batch_size, seq_len = 1, 1
        elif len(original_shape) == 2:
            batch_size = original_shape[0]
            seq_len = 1
        else:
            batch_size = int(np.prod(original_shape[:-2])) if len(original_shape) > 3 else original_shape[0]
            seq_len = original_shape[-2] if len(original_shape) >= 3 else 1

        x_nbytes = int(batch_size * seq_len * features * 4)

        if is_vt:
            buf_input, release_input = self._prepare_input(x, size=x_nbytes)
        else:
            x_np = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
            x_nbytes = x_np.nbytes
            buf_input = self._acquire_buffer(x_nbytes)
            self._upload_buffer(buf_input, x_np)
            release_input = True

        weight_flat = np.ascontiguousarray(weight, dtype=np.float32).flatten()
        total_positions = batch_size * seq_len
        total_elements = batch_size * seq_len * features

        use_device_local = return_gpu_tensor and hasattr(self, "_acquire_device_local_buffer")
        if use_device_local:
            buf_output = self._acquire_device_local_buffer(x_nbytes)
        else:
            buf_output = self._acquire_buffer(x_nbytes)
        buf_weight = self._acquire_buffer(weight_flat.nbytes)
        buf_rms = self._acquire_buffer(total_positions * 4)

        self._upload_buffer(buf_weight, weight_flat)

        pipeline, pipeline_layout, _ = self.pipelines.get_or_create_pipeline(
            "rms-norm", 4, push_constant_size=20
        )

        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "rms-norm",
            [
                (self._get_buffer_handle(buf_input), x_nbytes),
                (self._get_buffer_handle(buf_output), x_nbytes),
                (self._get_buffer_handle(buf_weight), weight_flat.nbytes),
                (self._get_buffer_handle(buf_rms), total_positions * 4),
            ],
        )

        # Run 2 passes
        use_batched = return_gpu_tensor and hasattr(self.core, "record_commands")
        if use_batched:
            with self.core.record_commands() as rec:
                for pass_type in range(2):
                    push_constants = struct.pack("IIIfI", batch_size, seq_len, features, eps, pass_type)
                    if pass_type == 0:
                        workgroups = (total_positions + 255) // 256
                    else:
                        workgroups = (total_elements + 255) // 256
                    rec.dispatch(pipeline, pipeline_layout, descriptor_set, (workgroups, 1, 1), push_constants)
                    if pass_type == 0:
                        rec.barrier()
        else:
            for pass_type in range(2):
                push_constants = struct.pack("IIIfI", batch_size, seq_len, features, eps, pass_type)
                if pass_type == 0:
                    workgroups = (total_positions + 255) // 256
                else:
                    workgroups = (total_elements + 255) // 256
                self.core._dispatch_compute(
                    pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
                )

        if return_gpu_tensor:
            if release_input:
                self._release_buffer(buf_input)
            self._release_buffers([buf_weight, buf_rms])
            if use_device_local and getattr(buf_output, "is_device_local", False):
                return self._wrap_output_tensor_device_local(buf_output, original_shape)
            return self._wrap_output_tensor(buf_output, original_shape)

        result = self._download_buffer(buf_output, x_nbytes, np.float32)
        if release_input:
            self._release_buffer(buf_input)
        self._release_buffers([buf_output, buf_weight, buf_rms])

        return result.reshape(original_shape)

    # ------------------------------------------------------------------
    # Fused SwiGLU FFN (GPU accelerated)
    # ------------------------------------------------------------------
    def swiglu_fused(
        self,
        x: np.ndarray,
        gate_weights: np.ndarray,
        up_weights: np.ndarray,
        return_gpu_tensor: bool = False,
    ) -> np.ndarray:
        """
        Fused SwiGLU: output = SiLU(x @ gate_proj.T) * (x @ up_proj.T)

        Fuses 2 matmuls + SiLU + elementwise multiply into one GPU dispatch.
        Eliminates 2 intermediate buffers vs separate operations.

        Args:
            x: Input tensor (batch_seq, input_dim) or (batch, seq, input_dim)
            gate_weights: Gate projection (intermediate_size, input_dim)
            up_weights: Up projection (intermediate_size, input_dim)
            return_gpu_tensor: If True, return VulkanTensor (stays on GPU)

        Returns:
            SwiGLU output (batch_seq, intermediate_size)
        """
        from ..utils.tensor_conversion import VulkanTensor

        is_vt = isinstance(x, VulkanTensor)
        original_shape = x.shape
        input_dim = original_shape[-1]
        intermediate_size = gate_weights.shape[0]

        if "swiglu-fused" not in self.shaders:
            x_np = x.numpy() if is_vt else np.asarray(x, dtype=np.float32)
            x_2d = x_np.reshape(-1, input_dim)
            gate = x_2d @ gate_weights.T
            up = x_2d @ up_weights.T
            sigmoid_gate = 1.0 / (1.0 + np.exp(-np.clip(gate, -88, 88)))
            silu_gate = gate * sigmoid_gate
            result = silu_gate * up
            if len(original_shape) > 2:
                return result.reshape(original_shape[:-1] + (intermediate_size,))
            return result

        x_np = x.numpy() if is_vt else np.asarray(x, dtype=np.float32)
        x_2d = np.ascontiguousarray(x_np.reshape(-1, input_dim), dtype=np.float32)
        batch_seq = x_2d.shape[0]

        gate_flat = np.ascontiguousarray(gate_weights, dtype=np.float32).flatten()
        up_flat = np.ascontiguousarray(up_weights, dtype=np.float32).flatten()
        x_flat = x_2d.flatten()

        x_bytes = x_flat.nbytes
        gate_bytes = gate_flat.nbytes
        up_bytes = up_flat.nbytes
        out_bytes = int(batch_seq * intermediate_size * 4)

        buf_input = self._acquire_buffer(x_bytes)
        buf_gate = self._acquire_buffer(gate_bytes)
        buf_up = self._acquire_buffer(up_bytes)
        buf_output = self._acquire_buffer(out_bytes)

        self._upload_buffer(buf_input, x_flat)
        self._upload_buffer(buf_gate, gate_flat)
        self._upload_buffer(buf_up, up_flat)

        pipeline, pipeline_layout, _ = self.pipelines.get_or_create_pipeline(
            "swiglu-fused", 4, push_constant_size=12
        )

        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "swiglu-fused",
            [
                (self._get_buffer_handle(buf_input), x_bytes),
                (self._get_buffer_handle(buf_gate), gate_bytes),
                (self._get_buffer_handle(buf_up), up_bytes),
                (self._get_buffer_handle(buf_output), out_bytes),
            ],
        )

        push_constants = struct.pack("III", batch_seq, input_dim, intermediate_size)
        workgroups_x = (intermediate_size + 15) // 16
        workgroups_y = (batch_seq + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups_x, push_constants, workgroups_y
        )

        if return_gpu_tensor:
            self._release_buffers([buf_input, buf_gate, buf_up])
            out_shape = original_shape[:-1] + (intermediate_size,)
            return self._wrap_output_tensor(buf_output, out_shape)

        result = self._download_buffer(buf_output, out_bytes, np.float32)
        self._release_buffers([buf_input, buf_gate, buf_up, buf_output])

        if len(original_shape) > 2:
            return result.reshape(original_shape[:-1] + (intermediate_size,))
        return result.reshape(batch_seq, intermediate_size)

    # ------------------------------------------------------------------
    # INT8 weight-only GEMM (GPU accelerated)
    # ------------------------------------------------------------------
    def gemm_int8(
        self,
        activations: np.ndarray,
        weights_int8: np.ndarray,
        scales: np.ndarray,
        group_size: int = 64,
        return_gpu_tensor: bool = False,
    ) -> np.ndarray:
        """
        INT8 weight-only GEMM with FP32 accumulation.

        Activations are fp32, weights are int8 with per-group fp32 scales.
        Optimal for inference: 2x memory reduction, minimal accuracy loss
        when combined with SmoothQuant calibration.

        Args:
            activations: Input tensor (M, K) fp32
            weights_int8: INT8 weights (N, K) as int8 numpy array
            scales: Per-group scales (N, ceil(K/group_size)) fp32
            group_size: Quantization group size (default: 64)
            return_gpu_tensor: If True, return VulkanTensor (stays on GPU)

        Returns:
            Output tensor (M, N) fp32
        """
        M, K = activations.shape
        N = weights_int8.shape[0]

        if "int8-gemm" not in self.shaders:
            # CPU fallback: dequantize and matmul
            num_groups = (K + group_size - 1) // group_size
            w_fp32 = np.zeros((N, K), dtype=np.float32)
            for g in range(num_groups):
                k_start = g * group_size
                k_end = min(k_start + group_size, K)
                w_fp32[:, k_start:k_end] = weights_int8[:, k_start:k_end].astype(np.float32) * scales[:, g:g+1]
            return activations @ w_fp32.T

        # Pack int8 weights as uint32 (4 per uint32)
        packed_K = (K + 3) // 4
        w_packed = np.zeros((N, packed_K), dtype=np.uint32)
        w_bytes_arr = weights_int8.view(np.uint8)
        for i in range(K):
            pack_idx = i // 4
            pack_offset = i % 4
            w_packed[:, pack_idx] |= w_bytes_arr[:, i].astype(np.uint32) << (pack_offset * 8)

        act_flat = np.ascontiguousarray(activations, dtype=np.float32).flatten()
        w_flat = np.ascontiguousarray(w_packed).flatten()
        s_flat = np.ascontiguousarray(scales, dtype=np.float32).flatten()

        act_nbytes = act_flat.nbytes
        w_nbytes = w_flat.nbytes
        s_nbytes = s_flat.nbytes
        out_bytes = int(M * N * 4)

        buf_act = self._acquire_buffer(act_nbytes)
        buf_w = self._acquire_buffer(w_nbytes)
        buf_s = self._acquire_buffer(s_nbytes)
        buf_out = self._acquire_buffer(out_bytes)

        self._upload_buffer(buf_act, act_flat)
        self._upload_buffer_raw(buf_w, w_flat)
        self._upload_buffer(buf_s, s_flat)

        pipeline, pipeline_layout, _ = self.pipelines.get_or_create_pipeline(
            "int8-gemm", 4, push_constant_size=16
        )

        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "int8-gemm",
            [
                (self._get_buffer_handle(buf_act), act_nbytes),
                (self._get_buffer_handle(buf_w), w_nbytes),
                (self._get_buffer_handle(buf_s), s_nbytes),
                (self._get_buffer_handle(buf_out), out_bytes),
            ],
        )

        push_constants = struct.pack("IIII", M, K, N, group_size)
        workgroups_x = (N + 15) // 16
        workgroups_y = (M + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups_x, push_constants, workgroups_y
        )

        if return_gpu_tensor:
            self._release_buffers([buf_act, buf_w, buf_s])
            return self._wrap_output_tensor(buf_out, (M, N))

        result = self._download_buffer(buf_out, out_bytes, np.float32)
        self._release_buffers([buf_act, buf_w, buf_s, buf_out])

        return result.reshape(M, N)
