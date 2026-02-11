"""
LoRA (Low-Rank Adaptation) operations for Vulkan backend.

GPU-accelerated LoRA operations:
- Fused forward pass: x @ W^T + scale * (x @ A^T @ B^T)
- Backward pass for A and B gradients

Shaders used:
- lora-forward.glsl: Fused LoRA forward computation
- lora-backward.glsl: Gradient computation for A and B matrices
"""

import struct

import numpy as np

from .base import VULKAN_AVAILABLE, BufferMixin

if VULKAN_AVAILABLE:
    from vulkan import *


class VulkanLoRA(BufferMixin):
    """
    GPU-accelerated LoRA operations using Vulkan compute shaders.

    Provides:
    - lora_forward: Fused base + LoRA forward pass
    - lora_backward: Gradient computation for LoRA A and B matrices

    Example:
        >>> backend = grilly.Compute()
        >>> output = backend.lora.forward(x, W, A, B, scale=0.5)
        >>> grad_A, grad_B = backend.lora.backward(grad_output, x, A, B, h, scale=0.5)
    """

    def __init__(self, core, pipelines, shaders):
        """
        Initialize LoRA backend.

        Args:
            core: VulkanCore instance
            pipelines: Pipeline cache dictionary
            shaders: Shader module dictionary
        """
        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders

    def forward(
        self,
        x: np.ndarray,
        W: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        scale: float = 1.0,
        bias: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Fused LoRA forward pass.

        Computes: output = x @ W^T + scale * (x @ A^T @ B^T) [+ bias]

        Args:
            x: Input tensor of shape (batch, in_features)
            W: Base weight matrix of shape (out_features, in_features)
            A: LoRA A matrix of shape (rank, in_features)
            B: LoRA B matrix of shape (out_features, rank)
            scale: LoRA scaling factor (alpha / rank)
            bias: Optional bias of shape (out_features,)

        Returns:
            Output tensor of shape (batch, out_features)
        """
        # Ensure float32
        x = np.ascontiguousarray(x, dtype=np.float32)
        W = np.ascontiguousarray(W, dtype=np.float32)
        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)

        batch_size = x.shape[0]
        in_features = x.shape[1] if x.ndim > 1 else x.shape[0]
        out_features = W.shape[0]
        rank = A.shape[0]

        # Check if GPU shader available
        if not VULKAN_AVAILABLE or "lora-forward" not in self.shaders:
            return self._forward_cpu(x, W, A, B, scale, bias)

        try:
            return self._forward_gpu(
                x, W, A, B, scale, bias, batch_size, in_features, out_features, rank
            )
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(f"GPU LoRA forward failed: {e}, using CPU fallback")
            return self._forward_cpu(x, W, A, B, scale, bias)

    def _forward_cpu(
        self,
        x: np.ndarray,
        W: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        scale: float,
        bias: np.ndarray | None,
    ) -> np.ndarray:
        """CPU fallback for LoRA forward."""
        # Base output: x @ W^T
        base_output = np.matmul(x, W.T)

        # LoRA output: x @ A^T @ B^T
        h = np.matmul(x, A.T)  # (batch, rank)
        lora_output = np.matmul(h, B.T)  # (batch, out_features)

        # Combine
        output = base_output + scale * lora_output

        if bias is not None:
            output = output + bias

        return output

    def _forward_gpu(
        self,
        x: np.ndarray,
        W: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        scale: float,
        bias: np.ndarray | None,
        batch_size: int,
        in_features: int,
        out_features: int,
        rank: int,
    ) -> np.ndarray:
        """GPU implementation of LoRA forward."""

        # Allocate buffers
        x_size = batch_size * in_features * 4
        W_size = out_features * in_features * 4
        A_size = rank * in_features * 4
        B_size = out_features * rank * 4
        output_size = batch_size * out_features * 4
        intermediate_size = batch_size * rank * 4

        # Create buffers
        x_buf = self.core.create_buffer(x_size)
        W_buf = self.core.create_buffer(W_size)
        A_buf = self.core.create_buffer(A_size)
        B_buf = self.core.create_buffer(B_size)
        output_buf = self.core.create_buffer(output_size)
        intermediate_buf = self.core.create_buffer(intermediate_size)

        try:
            # Upload data
            self.core.upload_to_buffer(x_buf, x.tobytes())
            self.core.upload_to_buffer(W_buf, W.tobytes())
            self.core.upload_to_buffer(A_buf, A.tobytes())
            self.core.upload_to_buffer(B_buf, B.tobytes())

            # Get or create pipeline
            pipeline_key = "lora-forward"
            if pipeline_key not in self.pipelines:
                self.pipelines[pipeline_key] = self.core.create_compute_pipeline(
                    self.shaders["lora-forward"]
                )
            pipeline = self.pipelines[pipeline_key]

            # Phase 0: Compute intermediate h = x @ A^T
            push_constants_0 = struct.pack(
                "IIIIfi",
                batch_size,
                in_features,
                out_features,
                rank,
                scale,
                0,  # phase=0
            )

            self.core.run_compute_shader(
                pipeline,
                [x_buf, W_buf, A_buf, B_buf, output_buf, intermediate_buf],
                push_constants_0,
                (rank + 15) // 16,  # X workgroups
                (batch_size + 15) // 16,  # Y workgroups
                1,  # Z workgroups
            )

            # Phase 1: Compute output = x @ W^T + scale * h @ B^T
            push_constants_1 = struct.pack(
                "IIIIfi",
                batch_size,
                in_features,
                out_features,
                rank,
                scale,
                1,  # phase=1
            )

            self.core.run_compute_shader(
                pipeline,
                [x_buf, W_buf, A_buf, B_buf, output_buf, intermediate_buf],
                push_constants_1,
                (out_features + 15) // 16,
                (batch_size + 15) // 16,
                1,
            )

            # Download result
            output_bytes = self.core.download_from_buffer(output_buf, output_size)
            output = np.frombuffer(output_bytes, dtype=np.float32).reshape(batch_size, out_features)

            if bias is not None:
                output = output + bias

            return output.copy()

        finally:
            # Cleanup buffers
            self.core.destroy_buffer(x_buf)
            self.core.destroy_buffer(W_buf)
            self.core.destroy_buffer(A_buf)
            self.core.destroy_buffer(B_buf)
            self.core.destroy_buffer(output_buf)
            self.core.destroy_buffer(intermediate_buf)

    def backward(
        self,
        grad_output: np.ndarray,
        x: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        h: np.ndarray | None = None,
        scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        LoRA backward pass - compute gradients for A and B.

        Args:
            grad_output: Gradient from upstream, shape (batch, out_features)
            x: Input from forward pass, shape (batch, in_features)
            A: LoRA A matrix, shape (rank, in_features)
            B: LoRA B matrix, shape (out_features, rank)
            h: Optional intermediate h = x @ A^T, shape (batch, rank)
               If not provided, will be recomputed
            scale: LoRA scaling factor (alpha / rank)

        Returns:
            Tuple of (grad_A, grad_B):
            - grad_A: Gradient for A, shape (rank, in_features)
            - grad_B: Gradient for B, shape (out_features, rank)
        """
        # Ensure float32
        grad_output = np.ascontiguousarray(grad_output, dtype=np.float32)
        x = np.ascontiguousarray(x, dtype=np.float32)
        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)

        # Recompute h if not provided
        if h is None:
            h = np.matmul(x, A.T)
        else:
            h = np.ascontiguousarray(h, dtype=np.float32)

        batch_size = x.shape[0]
        in_features = x.shape[1]
        out_features = B.shape[0]
        rank = A.shape[0]

        # Check if GPU shader available
        if not VULKAN_AVAILABLE or "lora-backward" not in self.shaders:
            return self._backward_cpu(grad_output, x, A, B, h, scale)

        try:
            return self._backward_gpu(
                grad_output, x, A, B, h, scale, batch_size, in_features, out_features, rank
            )
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(f"GPU LoRA backward failed: {e}, using CPU fallback")
            return self._backward_cpu(grad_output, x, A, B, h, scale)

    def _backward_cpu(
        self,
        grad_output: np.ndarray,
        x: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        h: np.ndarray,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """CPU fallback for LoRA backward."""
        # grad_B = scale * h^T @ grad_output
        # h has shape (batch, rank), grad_output has shape (batch, out_features)
        # h^T @ grad_output has shape (rank, out_features)
        # But B has shape (out_features, rank), so we need to transpose
        grad_B = scale * np.matmul(h.T, grad_output).T  # (out_features, rank)

        # grad_A = scale * (grad_output @ B)^T @ x
        # grad_output @ B has shape (batch, rank)
        # (grad_output @ B)^T has shape (rank, batch)
        # (grad_output @ B)^T @ x has shape (rank, in_features)
        temp = np.matmul(grad_output, B)  # (batch, rank)
        grad_A = scale * np.matmul(temp.T, x)  # (rank, in_features)

        return grad_A, grad_B

    def _backward_gpu(
        self,
        grad_output: np.ndarray,
        x: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        h: np.ndarray,
        scale: float,
        batch_size: int,
        in_features: int,
        out_features: int,
        rank: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """GPU implementation of LoRA backward."""

        # Allocate buffers
        grad_output_size = batch_size * out_features * 4
        x_size = batch_size * in_features * 4
        A_size = rank * in_features * 4
        B_size = out_features * rank * 4
        h_size = batch_size * rank * 4
        grad_A_size = rank * in_features * 4
        grad_B_size = out_features * rank * 4
        temp_size = batch_size * rank * 4

        # Create buffers
        grad_output_buf = self.core.create_buffer(grad_output_size)
        x_buf = self.core.create_buffer(x_size)
        A_buf = self.core.create_buffer(A_size)
        B_buf = self.core.create_buffer(B_size)
        h_buf = self.core.create_buffer(h_size)
        grad_A_buf = self.core.create_buffer(grad_A_size)
        grad_B_buf = self.core.create_buffer(grad_B_size)
        temp_buf = self.core.create_buffer(temp_size)

        try:
            # Upload data
            self.core.upload_to_buffer(grad_output_buf, grad_output.tobytes())
            self.core.upload_to_buffer(x_buf, x.tobytes())
            self.core.upload_to_buffer(A_buf, A.tobytes())
            self.core.upload_to_buffer(B_buf, B.tobytes())
            self.core.upload_to_buffer(h_buf, h.tobytes())

            # Initialize grad buffers to zero
            zeros_A = np.zeros((rank, in_features), dtype=np.float32)
            zeros_B = np.zeros((out_features, rank), dtype=np.float32)
            self.core.upload_to_buffer(grad_A_buf, zeros_A.tobytes())
            self.core.upload_to_buffer(grad_B_buf, zeros_B.tobytes())

            # Get or create pipeline
            pipeline_key = "lora-backward"
            if pipeline_key not in self.pipelines:
                self.pipelines[pipeline_key] = self.core.create_compute_pipeline(
                    self.shaders["lora-backward"]
                )
            pipeline = self.pipelines[pipeline_key]

            # Phase 0: Compute grad_B
            push_constants_0 = struct.pack(
                "IIIIfi", batch_size, in_features, out_features, rank, scale, 0
            )

            self.core.run_compute_shader(
                pipeline,
                [grad_output_buf, x_buf, A_buf, B_buf, h_buf, grad_A_buf, grad_B_buf, temp_buf],
                push_constants_0,
                (rank + 15) // 16,
                (out_features + 15) // 16,
                1,
            )

            # Phase 1: Compute temp = grad_output @ B
            push_constants_1 = struct.pack(
                "IIIIfi", batch_size, in_features, out_features, rank, scale, 1
            )

            self.core.run_compute_shader(
                pipeline,
                [grad_output_buf, x_buf, A_buf, B_buf, h_buf, grad_A_buf, grad_B_buf, temp_buf],
                push_constants_1,
                (rank + 15) // 16,
                (batch_size + 15) // 16,
                1,
            )

            # Phase 2: Compute grad_A
            push_constants_2 = struct.pack(
                "IIIIfi", batch_size, in_features, out_features, rank, scale, 2
            )

            self.core.run_compute_shader(
                pipeline,
                [grad_output_buf, x_buf, A_buf, B_buf, h_buf, grad_A_buf, grad_B_buf, temp_buf],
                push_constants_2,
                (in_features + 15) // 16,
                (rank + 15) // 16,
                1,
            )

            # Download results
            grad_A_bytes = self.core.download_from_buffer(grad_A_buf, grad_A_size)
            grad_B_bytes = self.core.download_from_buffer(grad_B_buf, grad_B_size)

            grad_A = np.frombuffer(grad_A_bytes, dtype=np.float32).reshape(rank, in_features)
            grad_B = np.frombuffer(grad_B_bytes, dtype=np.float32).reshape(out_features, rank)

            return grad_A.copy(), grad_B.copy()

        finally:
            # Cleanup
            self.core.destroy_buffer(grad_output_buf)
            self.core.destroy_buffer(x_buf)
            self.core.destroy_buffer(A_buf)
            self.core.destroy_buffer(B_buf)
            self.core.destroy_buffer(h_buf)
            self.core.destroy_buffer(grad_A_buf)
            self.core.destroy_buffer(grad_B_buf)
            self.core.destroy_buffer(temp_buf)

    def forward_with_intermediate(
        self,
        x: np.ndarray,
        W: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        scale: float = 1.0,
        bias: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        LoRA forward pass that also returns intermediate h for backward.

        Args:
            Same as forward()

        Returns:
            Tuple of (output, h) where h = x @ A^T
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        A = np.ascontiguousarray(A, dtype=np.float32)

        # Compute intermediate
        h = np.matmul(x, A.T)

        # Compute full forward
        output = self.forward(x, W, A, B, scale, bias)

        return output, h


# Convenience function for creating VulkanLoRA
def create_lora_backend(core, pipelines, shaders) -> VulkanLoRA:
    """
    Create a VulkanLoRA backend instance.

    Args:
        core: VulkanCore instance
        pipelines: Pipeline cache dictionary
        shaders: Shader module dictionary

    Returns:
        VulkanLoRA instance
    """
    return VulkanLoRA(core, pipelines, shaders)
