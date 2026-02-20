"""
Convolutional operations for Vulkan backend.
GPU-accelerated 2D convolutions with backward pass support.

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


class VulkanConv(BufferMixin):
    """Convolutional operations: Conv2d forward and backward passes"""

    def __init__(self, core, pipelines, shaders):
        """Initialize the instance."""

        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders

    def _conv2d_gemm(
        self,
        input_data: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray | None,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
    ) -> np.ndarray:
        # 1) Extract shapes and check assumptions
        """Execute conv2d gemm."""

        batch_size, in_channels, in_h, in_w = input_data.shape
        out_channels, _, kernel_h, kernel_w = weight.shape
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        # For first version, only support groups=1, dilation=1
        assert groups == 1
        assert dilation_h == 1 and dilation_w == 1

        # 2) Compute output dims
        out_h = (in_h + 2 * padding_h - kernel_h) // stride_h + 1
        out_w = (in_w + 2 * padding_w - kernel_w) // stride_w + 1

        K_dim = in_channels * kernel_h * kernel_w  # rows in im2col
        N_cols = batch_size * out_h * out_w  # columns in im2col and GEMM

        # 3) Prepare data (weights as (M,K) row-major)
        M = out_channels
        A = weight.reshape(M, K_dim).astype(np.float32)  # (M,K)

        # 4) Allocate GPU buffers: input, A, cols, mat_out
        buf_input = self._acquire_buffer(input_data.nbytes)
        buf_A = self._acquire_buffer(A.nbytes)
        buf_cols = self._acquire_buffer(K_dim * N_cols * 4)
        buf_matout = self._acquire_buffer(M * N_cols * 4)

        try:
            # Upload input and weight
            self._upload_buffer(buf_input, input_data.flatten())
            self._upload_buffer(buf_A, A.flatten())

            # --- Step 1: im2col ---
            pipeline_im2col, layout_im2col, _ = self.pipelines.get_or_create_pipeline(
                "convd_im2col",
                2,
                push_constant_size=56,  # adjust size as needed
            )

            in_handle = self._get_buffer_handle(buf_input)
            cols_handle = self._get_buffer_handle(buf_cols)

            desc_im2col = self.pipelines.get_cached_descriptor_set(
                "convd_im2col",
                [
                    (in_handle, input_data.nbytes),
                    (cols_handle, K_dim * N_cols * 4),
                ],
            )

            push_im2col = struct.pack(
                "14I",
                batch_size,
                in_channels,
                in_h,
                in_w,
                out_h,
                out_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
            )

            group_x = (K_dim + 15) // 16
            group_y = (N_cols + 15) // 16
            group_z = 1

            self.core._dispatch_compute(
                pipeline_im2col, layout_im2col, desc_im2col, group_x, push_im2col, group_y, group_z
            )

            # --- Step 2: GEMM (A * cols -> mat_out) ---
            pipeline_gemm, layout_gemm, _ = self.pipelines.get_or_create_pipeline(
                "gemm_mnk", 3, push_constant_size=12
            )

            A_handle = self._get_buffer_handle(buf_A)
            B_handle = cols_handle
            C_handle = self._get_buffer_handle(buf_matout)

            desc_gemm = self.pipelines.get_cached_descriptor_set(
                "gemm_mnk",
                [
                    (A_handle, A.nbytes),
                    (B_handle, K_dim * N_cols * 4),
                    (C_handle, M * N_cols * 4),
                ],
            )

            push_gemm = struct.pack("3I", M, K_dim, N_cols)

            group_x = (N_cols + 15) // 16
            group_y = (M + 15) // 16
            group_z = 1

            self.core._dispatch_compute(
                pipeline_gemm, layout_gemm, desc_gemm, group_x, push_gemm, group_y, group_z
            )

            # --- Step 3: download & reshape + bias on CPU (for now) ---
            mat_flat = self._download_buffer(buf_matout, M * N_cols * 4, np.float32)
            mat = mat_flat.reshape(M, N_cols)

            if bias is not None:
                mat += bias.reshape(M, 1)

            # Reshape to (N, C_out, H_out, W_out)
            out = mat.reshape(out_channels, batch_size, out_h, out_w)
            out = np.transpose(out, (1, 0, 2, 3)).copy()
            return out

        finally:
            self._release_buffers([buf_input, buf_A, buf_cols, buf_matout])

    def conv2d(
        self,
        input_data,  # (batch, in_channels, height, width) — numpy or VulkanTensor
        weight: np.ndarray,  # (out_channels, in_channels/groups, kernel_h, kernel_w)
        bias: np.ndarray | None = None,  # (out_channels,)
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
        return_gpu_tensor: bool = False,
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
        from ..utils.tensor_conversion import VulkanTensor

        is_vt = isinstance(input_data, VulkanTensor)
        if "conv2d-forward" not in self.shaders:
            if is_vt:
                input_data = input_data.numpy()
            return self._conv2d_cpu(input_data, weight, bias, stride, padding, dilation, groups)

        # Ensure float32 — accept VulkanTensor or numpy
        if is_vt:
            input_shape = input_data.shape
        else:
            input_data = np.asarray(input_data, dtype=np.float32)
            input_shape = input_data.shape
        weight = np.asarray(weight, dtype=np.float32)
        if bias is not None:
            bias = np.asarray(bias, dtype=np.float32)

        # Extract dimensions
        batch_size, in_channels, in_height, in_width = input_shape
        out_channels, channels_per_group, kernel_h, kernel_w = weight.shape
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        use_gemm = (
            "convd_im2col" in self.shaders
            and "gemm_mnk" in self.shaders
            and channels_per_group == in_channels  # groups == 1
            and dilation_h == 1
            and dilation_w == 1
        )
        if use_gemm:
            if is_vt:
                input_data = input_data.numpy()
            return self._conv2d_gemm(input_data, weight, bias, stride, padding, dilation, groups)
        # Calculate output dimensions
        out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

        num_elements = batch_size * out_channels * out_height * out_width
        output_bytes = num_elements * 4  # float32
        input_nbytes = batch_size * in_channels * in_height * in_width * 4

        _output_owned = False  # Track whether VulkanTensor owns buf_output
        buf_output = self._acquire_buffer(output_bytes)
        # Allocate / prepare input buffer (zero-copy for VulkanTensor)
        if is_vt:
            buf_input, release_input = self._prepare_input(input_data)
        else:
            buf_input = self._acquire_buffer(input_nbytes)
            release_input = True
        buf_weight = self._acquire_buffer(weight.nbytes)
        buf_bias = self._acquire_buffer(
            bias.nbytes if bias is not None else 4
        )  # Dummy buffer if no bias

        try:
            # Upload data (skip input upload for VulkanTensor — already on GPU)
            if not is_vt:
                self._upload_buffer(buf_input, input_data.flatten())
            self._upload_buffer(buf_weight, weight.flatten())
            if bias is not None:
                self._upload_buffer(buf_bias, bias.flatten())
            else:
                self._upload_buffer(buf_bias, np.zeros(1, dtype=np.float32))

            # Create pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "conv2d-forward", 4, push_constant_size=68
            )

            # Get buffer handles
            in_handle = self._get_buffer_handle(buf_input)
            weight_handle = self._get_buffer_handle(buf_weight)
            bias_handle = self._get_buffer_handle(buf_bias)
            out_handle = self._get_buffer_handle(buf_output)

            # Get cached descriptor set
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                "conv2d-forward",
                [
                    (in_handle, input_nbytes),
                    (weight_handle, weight.nbytes),
                    (bias_handle, (bias.nbytes if bias is not None else 4)),
                    (out_handle, output_bytes),
                ],
            )

            # Pack push constants
            push_data = struct.pack(
                "17I",
                batch_size,
                in_channels,
                in_height,
                in_width,
                out_channels,
                out_height,
                out_width,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
                groups,
                1 if bias is not None else 0,
            )

            # Dispatch compute shader
            group_count_x = (out_width + 7) // 8
            group_count_y = (out_height + 7) // 8
            group_count_z = batch_size * out_channels

            self.core._dispatch_compute(
                pipeline,
                pipeline_layout,
                descriptor_set,
                group_count_x,
                push_data,
                group_count_y,
                group_count_z,
            )

            output_shape = (batch_size, out_channels, out_height, out_width)

            if return_gpu_tensor:
                result = self._wrap_output_tensor(buf_output, output_shape)
                _output_owned = True  # VulkanTensor now owns buf_output
                return result
            else:
                output_flat = self._download_buffer(buf_output, output_bytes, np.float32)
                return output_flat.reshape(output_shape)

        finally:
            # Always release input (unless borrowed from VulkanTensor), weight, bias
            if release_input:
                self._release_buffer(buf_input)
            self._release_buffers([buf_weight, buf_bias])
            # Only release output if NOT wrapped in VulkanTensor
            if not _output_owned:
                self._release_buffer(buf_output)

    def _conv2d_backward_input_gemm(
        self,
        grad_output: np.ndarray,  # (batch, out_channels, out_h, out_w)
        weight: np.ndarray,  # (out_channels, in_channels, kernel_h, kernel_w)
        input_shape: tuple[int, int, int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
    ) -> np.ndarray:
        """
        Conv2d backward input using GEMM + col2im.

        grad_input computation:
        - Reshape weight: (C_out, C_in, kH, kW) -> (C_out, K_dim) where K_dim = C_in*kH*kW
        - Reshape grad_output: (N, C_out, H_out, W_out) -> (C_out, N*H_out*W_out)
        - GEMM: cols = weight.T @ grad_output = (K_dim, N*H_out*W_out)
        - col2im(cols) -> grad_input (N, C_in, H_in, W_in)
        """
        batch_size, in_channels, in_h, in_w = input_shape
        _, out_channels, out_h, out_w = grad_output.shape
        _, in_channels_per_group, kernel_h, kernel_w = weight.shape
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        # Only for groups==1, dilation==1
        assert groups == 1
        assert dilation_h == 1 and dilation_w == 1

        K_dim = in_channels * kernel_h * kernel_w
        N_cols = batch_size * out_h * out_w

        # --- Step 1: Prepare weight and grad_output for GEMM ---
        # weight: (C_out, C_in, kH, kW) -> (C_out, K_dim)
        weight_reshaped = weight.reshape(out_channels, K_dim)

        # grad_output: (N, C_out, H_out, W_out) -> (C_out, N*H_out*W_out)
        grad_out_reshaped = grad_output.transpose(1, 0, 2, 3).reshape(out_channels, N_cols)

        # --- Step 2: GEMM: cols = weight.T @ grad_output ---
        # (K_dim, C_out) @ (C_out, N_cols) = (K_dim, N_cols)
        cols = weight_reshaped.T @ grad_out_reshaped  # (K_dim, N_cols)

        # --- Step 3: col2im to convert cols back to image ---
        buf_cols = self._acquire_buffer(K_dim * N_cols * 4)
        buf_grad_input = self._acquire_buffer(batch_size * in_channels * in_h * in_w * 4)

        try:
            # Upload cols
            self._upload_buffer(buf_cols, cols.flatten())
            # Initialize grad_input to zero
            self._upload_buffer(
                buf_grad_input, np.zeros(batch_size * in_channels * in_h * in_w, dtype=np.float32)
            )

            # col2im shader
            pipeline_col2im, layout_col2im, _ = self.pipelines.get_or_create_pipeline(
                "convd_col2im", 2, push_constant_size=56
            )

            cols_handle = self._get_buffer_handle(buf_cols)
            grad_in_handle = self._get_buffer_handle(buf_grad_input)

            desc_col2im = self.pipelines.get_cached_descriptor_set(
                "convd_col2im",
                [
                    (cols_handle, K_dim * N_cols * 4),
                    (grad_in_handle, batch_size * in_channels * in_h * in_w * 4),
                ],
            )

            push_col2im = struct.pack(
                "14I",
                batch_size,
                in_channels,
                in_h,
                in_w,
                out_h,
                out_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
            )

            group_x = (K_dim + 15) // 16
            group_y = (N_cols + 15) // 16
            self.core._dispatch_compute(
                pipeline_col2im, layout_col2im, desc_col2im, group_x, push_col2im, group_y, 1
            )

            # Download result
            grad_input_flat = self._download_buffer(
                buf_grad_input, batch_size * in_channels * in_h * in_w * 4, np.float32
            )
            return grad_input_flat.reshape(batch_size, in_channels, in_h, in_w)

        finally:
            self._release_buffers([buf_cols, buf_grad_input])

    def conv2d_backward_input(
        self,
        grad_output: np.ndarray,  # (batch, out_channels, out_h, out_w)
        weight: np.ndarray,  # (out_channels, in_channels/groups, kernel_h, kernel_w)
        input_shape: tuple[int, int, int, int],  # (batch, in_channels, in_h, in_w)
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
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
        # Ensure float32
        grad_output = np.asarray(grad_output, dtype=np.float32)
        weight = np.asarray(weight, dtype=np.float32)

        # Try GEMM path first (groups==1, dilation==1)
        use_gemm = (
            "convd_col2im" in self.shaders and groups == 1 and dilation[0] == 1 and dilation[1] == 1
        )
        if use_gemm:
            return self._conv2d_backward_input_gemm(
                grad_output, weight, input_shape, stride, padding, dilation, groups
            )

        # Check if shader is available
        if "conv2d-backward-input" not in self.shaders:
            return self._conv2d_backward_input_cpu(
                grad_output, weight, input_shape, stride, padding, dilation, groups
            )

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

        batch_size * in_channels * in_height * in_width * 4  # BYTES
        num_elements = batch_size * out_channels * out_height * out_width
        output_bytes = num_elements * 4  # float32

        self._acquire_buffer(output_bytes)
        # Number of elements and bytes for grad_input
        num_grad_input_elems = batch_size * in_channels * in_height * in_width
        grad_input_bytes = num_grad_input_elems * 4  # float32

        # Allocate buffers
        buf_grad_output = self._acquire_buffer(grad_output.nbytes)
        buf_weight = self._acquire_buffer(weight.nbytes)
        buf_grad_input = self._acquire_buffer(grad_input_bytes)

        try:
            # Upload data
            self._upload_buffer(buf_grad_output, grad_output.flatten())
            self._upload_buffer(buf_weight, weight.flatten())
            # Initialize grad_input to zero (note: elements, not bytes)
            self._upload_buffer(buf_grad_input, np.zeros(num_grad_input_elems, dtype=np.float32))

            # Create pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "conv2d-backward-input", 3, push_constant_size=64
            )

            # Get buffer handles
            grad_out_handle = self._get_buffer_handle(buf_grad_output)
            weight_handle = self._get_buffer_handle(buf_weight)
            grad_in_handle = self._get_buffer_handle(buf_grad_input)

            # Descriptor set: third size must be bytes, not elements
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                "conv2d-backward-input",
                [
                    (grad_out_handle, grad_output.nbytes),
                    (weight_handle, weight.nbytes),
                    (grad_in_handle, grad_input_bytes),
                ],
            )

            # Pack push constants
            push_data = struct.pack(
                "16I",
                batch_size,
                in_channels,
                in_height,
                in_width,
                out_channels,
                out_height,
                out_width,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
                groups,
            )

            # Dispatch compute shader (Z includes both batch and channels)
            group_count_x = (in_width + 7) // 8
            group_count_y = (in_height + 7) // 8
            group_count_z = batch_size * in_channels

            self.core._dispatch_compute(
                pipeline,
                pipeline_layout,
                descriptor_set,
                group_count_x,
                push_data,
                group_count_y,
                group_count_z,
            )

            # Download result
            grad_input_flat = self._download_buffer(buf_grad_input, grad_input_bytes, np.float32)
            return grad_input_flat.reshape(batch_size, in_channels, in_height, in_width)
        finally:
            self._release_buffers([buf_grad_output, buf_weight, buf_grad_input])

    def _conv2d_backward_weight_gemm(
        self,
        grad_output: np.ndarray,  # (batch, out_channels, out_h, out_w)
        input_data: np.ndarray,  # (batch, in_channels, in_h, in_w)
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
        has_bias: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Conv2d backward weight using im2col + GEMM.

        grad_weight computation:
        - im2col(input) -> cols shape (K_dim, N_cols) where K_dim = C_in*kH*kW, N_cols = N*H_out*W_out
        - grad_output reshaped to (C_out, N*H_out*W_out)
        - grad_weight = grad_output @ cols.T, shape (C_out, K_dim) -> reshape to (C_out, C_in, kH, kW)
        """
        batch_size, in_channels, in_h, in_w = input_data.shape
        _, out_channels, out_h, out_w = grad_output.shape
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        # Only for groups==1, dilation==1
        assert groups == 1
        assert dilation_h == 1 and dilation_w == 1

        K_dim = in_channels * kernel_h * kernel_w
        N_cols = batch_size * out_h * out_w

        # --- Step 1: im2col(input) ---
        buf_input = self._acquire_buffer(input_data.nbytes)
        buf_cols = self._acquire_buffer(K_dim * N_cols * 4)

        try:
            self._upload_buffer(buf_input, input_data.flatten())

            pipeline_im2col, layout_im2col, _ = self.pipelines.get_or_create_pipeline(
                "convd_im2col", 2, push_constant_size=56
            )

            in_handle = self._get_buffer_handle(buf_input)
            cols_handle = self._get_buffer_handle(buf_cols)

            desc_im2col = self.pipelines.get_cached_descriptor_set(
                "convd_im2col", [(in_handle, input_data.nbytes), (cols_handle, K_dim * N_cols * 4)]
            )

            push_im2col = struct.pack(
                "14I",
                batch_size,
                in_channels,
                in_h,
                in_w,
                out_h,
                out_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
            )

            group_x = (K_dim + 15) // 16
            group_y = (N_cols + 15) // 16
            self.core._dispatch_compute(
                pipeline_im2col, layout_im2col, desc_im2col, group_x, push_im2col, group_y, 1
            )

            # Download cols for GEMM (could be done on GPU, but for now download)
            cols_flat = self._download_buffer(buf_cols, K_dim * N_cols * 4, np.float32)
            cols = cols_flat.reshape(K_dim, N_cols)  # (K_dim, N_cols)

            # --- Step 2: Prepare grad_output for GEMM ---
            # grad_output: (N, C_out, H_out, W_out) -> (C_out, N*H_out*W_out)
            grad_out_reshaped = grad_output.transpose(1, 0, 2, 3).reshape(
                out_channels, N_cols
            )  # (C_out, N_cols)

            # --- Step 3: GEMM: grad_weight = grad_out @ cols.T ---
            # (C_out, N_cols) @ (N_cols, K_dim) = (C_out, K_dim)
            grad_weight_flat = grad_out_reshaped @ cols.T  # (C_out, K_dim)

            # Reshape to (C_out, C_in, kH, kW)
            grad_weight = grad_weight_flat.reshape(out_channels, in_channels, kernel_h, kernel_w)

            # --- Step 4: Compute grad_bias ---
            grad_bias = None
            if has_bias:
                # Sum over all spatial positions and batch
                grad_bias = np.sum(grad_output, axis=(0, 2, 3))  # (C_out,)

            return grad_weight.astype(np.float32), grad_bias

        finally:
            self._release_buffers([buf_input, buf_cols])

    def conv2d_backward_weight(
        self,
        grad_output: np.ndarray,  # (batch, out_channels, out_h, out_w)
        input_data: np.ndarray,  # (batch, in_channels, in_h, in_w)
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
        has_bias: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
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
        # Ensure float32
        grad_output = np.asarray(grad_output, dtype=np.float32)
        input_data = np.asarray(input_data, dtype=np.float32)

        # Try GEMM path first (groups==1, dilation==1)
        use_gemm = (
            "convd_im2col" in self.shaders and groups == 1 and dilation[0] == 1 and dilation[1] == 1
        )
        if use_gemm:
            return self._conv2d_backward_weight_gemm(
                grad_output, input_data, kernel_size, stride, padding, dilation, groups, has_bias
            )

        # Check if shader is available
        if "conv2d-backward-weight" not in self.shaders:
            return self._conv2d_backward_weight_cpu(
                grad_output, input_data, kernel_size, stride, padding, dilation, groups, has_bias
            )

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
                "conv2d-backward-weight", 4, push_constant_size=68
            )

            # Get buffer handles
            grad_out_handle = self._get_buffer_handle(buf_grad_output)
            input_handle = self._get_buffer_handle(buf_input)
            grad_weight_handle = self._get_buffer_handle(buf_grad_weight)
            grad_bias_handle = self._get_buffer_handle(buf_grad_bias)

            # Get cached descriptor set
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                "conv2d-backward-weight",
                [
                    (grad_out_handle, grad_output.nbytes),
                    (input_handle, input_data.nbytes),
                    (grad_weight_handle, grad_weight_size),
                    (grad_bias_handle, (out_channels * 4 if has_bias else 4)),
                ],
            )

            # Pack push constants
            push_data = struct.pack(
                "17I",
                batch_size,
                in_channels,
                in_height,
                in_width,
                out_channels,
                out_height,
                out_width,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
                groups,
                1 if has_bias else 0,
            )

            # Dispatch compute shader
            total_weight_spatial = in_channels_per_group * kernel_h * kernel_w
            group_count_x = (total_weight_spatial + 15) // 16
            group_count_y = (out_channels + 15) // 16
            group_count_z = 1

            self.core._dispatch_compute(
                pipeline,
                pipeline_layout,
                descriptor_set,
                group_count_x,
                push_data,
                group_count_y,
                group_count_z,
            )

            # Download results
            grad_weight_flat = self._download_buffer(buf_grad_weight, grad_weight_size, np.float32)
            grad_weight = grad_weight_flat.reshape(
                out_channels, in_channels_per_group, kernel_h, kernel_w
            )

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
                mode="constant",
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
                                    val += (
                                        input_padded[b, ic_start + ic, ih, iw]
                                        * weight[oc, ic, kh, kw]
                                    )
                        if bias is not None:
                            val += bias[oc]
                        output[b, oc, oh, ow] = val

        return output

    def _conv2d_backward_input_cpu(
        self, grad_output, weight, input_shape, stride, padding, dilation, groups
    ):
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
                                            val += (
                                                grad_output[b, oc, oh, ow]
                                                * weight[oc, ic % channels_per_group, kh, kw]
                                            )
                        grad_input[b, ic, ih, iw] = val

        return grad_input

    def _conv2d_backward_weight_cpu(
        self, grad_output, input_data, kernel_size, stride, padding, dilation, groups, has_bias
    ):
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
                mode="constant",
            )
        else:
            input_padded = input_data

        in_channels_per_group = in_channels // groups
        grad_weight = np.zeros(
            (out_channels, in_channels_per_group, kernel_h, kernel_w), dtype=np.float32
        )

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
                                    val += (
                                        grad_output[b, oc, oh, ow]
                                        * input_padded[b, ic_start + ic, ih, iw]
                                    )
                        grad_weight[oc, ic, kh, kw] = val

        grad_bias = None
        if has_bias:
            grad_bias = np.sum(grad_output, axis=(0, 2, 3))

        return grad_weight, grad_bias
