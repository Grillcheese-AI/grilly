"""
Affect MLP operations for Vulkan backend.
GPU-accelerated affect prediction and training for amygdala.
"""

import struct
import time

import numpy as np

from .base import BufferMixin


class VulkanAffect(BufferMixin):
    """Affect MLP operations: forward pass, backpropagation, Adam optimizer"""

    def __init__(self, core, pipelines, shaders):
        """Initialize the instance."""

        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders
        self._pool = None

    def affect_mlp_forward(
        self,
        embeddings: np.ndarray,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        w3: np.ndarray,
        b3: np.ndarray,
        leaky_slope: float = 0.01,
        apply_output_activation: bool = False,
        dropout_rate: float = 0.0,
    ) -> tuple:
        """
        GPU-accelerated forward pass for 3-layer affect MLP

        Args:
            embeddings: Input embeddings (batch, embedding_dim)
            w1: Layer 1 weights (hidden1_dim, embedding_dim)
            b1: Layer 1 bias (hidden1_dim)
            w2: Layer 2 weights (hidden2_dim, hidden1_dim)
            b2: Layer 2 bias (hidden2_dim)
            w3: Output weights (2, hidden2_dim) - [valence, arousal]
            b3: Output bias (2)
            leaky_slope: LeakyReLU negative slope
            apply_output_activation: Whether to apply tanh/sigmoid to output
            dropout_rate: Dropout rate (0 during inference)

        Returns:
            Tuple of (predictions, hidden1, hidden2)
        """
        batch_size, embedding_dim = embeddings.shape
        hidden1_dim = w1.shape[0]
        hidden2_dim = w2.shape[0]

        # Flatten all arrays
        emb_flat = embeddings.astype(np.float32).flatten()
        w1_flat = w1.astype(np.float32).flatten()
        b1_flat = b1.astype(np.float32).flatten()
        w2_flat = w2.astype(np.float32).flatten()
        b2_flat = b2.astype(np.float32).flatten()
        w3_flat = w3.astype(np.float32).flatten()
        b3_flat = b3.astype(np.float32).flatten()

        # Output buffers
        hidden1_size = batch_size * hidden1_dim * 4
        hidden2_size = batch_size * hidden2_dim * 4
        output_size = batch_size * 2 * 4

        # Create buffers
        buf_emb = self._acquire_buffer(emb_flat.nbytes)
        buf_w1 = self._acquire_buffer(w1_flat.nbytes)
        buf_b1 = self._acquire_buffer(b1_flat.nbytes)
        buf_w2 = self._acquire_buffer(w2_flat.nbytes)
        buf_b2 = self._acquire_buffer(b2_flat.nbytes)
        buf_w3 = self._acquire_buffer(w3_flat.nbytes)
        buf_b3 = self._acquire_buffer(b3_flat.nbytes)
        buf_h1 = self._acquire_buffer(hidden1_size)
        buf_h2 = self._acquire_buffer(hidden2_size)
        buf_out = self._acquire_buffer(output_size)

        try:
            # Upload data
            self._upload_buffer(buf_emb, emb_flat)
            self._upload_buffer(buf_w1, w1_flat)
            self._upload_buffer(buf_b1, b1_flat)
            self._upload_buffer(buf_w2, w2_flat)
            self._upload_buffer(buf_b2, b2_flat)
            self._upload_buffer(buf_w3, w3_flat)
            self._upload_buffer(buf_b3, b3_flat)

            # Check if shader is available
            if "affect-mlp-forward" not in self.shaders:
                raise RuntimeError(
                    "affect-mlp-forward shader not compiled. "
                    "Run: glslc -fshader-stage=compute shaders/affect-mlp-forward.glsl -o shaders/spv/affect-mlp-forward.spv"
                )

            # Get or create pipeline
            num_bindings = 10  # input, w1, b1, w2, b2, w3, b3, hidden1, hidden2, output
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "affect-mlp-forward", num_bindings, push_constant_size=32
            )

            # Create descriptor set
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                "affect-mlp-forward",
                [
                    (self._get_buffer_handle(buf_emb), emb_flat.nbytes),
                    (self._get_buffer_handle(buf_w1), w1_flat.nbytes),
                    (self._get_buffer_handle(buf_b1), b1_flat.nbytes),
                    (self._get_buffer_handle(buf_w2), w2_flat.nbytes),
                    (self._get_buffer_handle(buf_b2), b2_flat.nbytes),
                    (self._get_buffer_handle(buf_w3), w3_flat.nbytes),
                    (self._get_buffer_handle(buf_b3), b3_flat.nbytes),
                    (self._get_buffer_handle(buf_h1), hidden1_size),
                    (self._get_buffer_handle(buf_h2), hidden2_size),
                    (self._get_buffer_handle(buf_out), output_size),
                ],
            )

            # Pack push constants
            seed = int(time.time() * 1000) % (2**31)
            push_constants = struct.pack(
                "IIIIfIfI",
                batch_size,
                embedding_dim,
                hidden1_dim,
                hidden2_dim,
                leaky_slope,
                1 if apply_output_activation else 0,
                dropout_rate,
                seed,
            )

            # Dispatch
            workgroups = (batch_size + 63) // 64
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download results
            predictions = self._download_buffer(buf_out, output_size, np.float32)
            hidden1 = self._download_buffer(buf_h1, hidden1_size, np.float32)
            hidden2 = self._download_buffer(buf_h2, hidden2_size, np.float32)

            predictions = predictions[: batch_size * 2].reshape(batch_size, 2)
            hidden1 = hidden1[: batch_size * hidden1_dim].reshape(batch_size, hidden1_dim)
            hidden2 = hidden2[: batch_size * hidden2_dim].reshape(batch_size, hidden2_dim)

            return predictions, hidden1, hidden2

        finally:
            self._release_buffers(
                [buf_emb, buf_w1, buf_b1, buf_w2, buf_b2, buf_w3, buf_b3, buf_h1, buf_h2, buf_out]
            )

    def affect_adam_update(
        self,
        weights: np.ndarray,
        gradients: np.ndarray,
        moment1: np.ndarray,
        moment2: np.ndarray,
        learning_rate: float,
        timestep: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0001,
    ) -> np.ndarray:
        """
        GPU-accelerated Adam optimizer update

        Args:
            weights: Weights to update (flattened)
            gradients: Gradients (flattened, same size as weights)
            moment1: First moment buffer (flattened)
            moment2: Second moment buffer (flattened)
            learning_rate: Learning rate
            timestep: Current timestep for bias correction
            beta1: First moment decay
            beta2: Second moment decay
            epsilon: Numerical stability
            weight_decay: L2 regularization

        Returns:
            Updated weights
        """
        total_weights = len(weights)

        # Ensure all arrays are same size and float32
        weights = weights.astype(np.float32).flatten()
        gradients = gradients.astype(np.float32).flatten()
        moment1 = moment1.astype(np.float32).flatten()
        moment2 = moment2.astype(np.float32).flatten()

        if len(gradients) != total_weights:
            raise ValueError(f"Gradients size {len(gradients)} != weights size {total_weights}")

        # Create buffers
        buf_grad = self._acquire_buffer(gradients.nbytes)
        buf_weights = self._acquire_buffer(weights.nbytes)
        buf_m1 = self._acquire_buffer(moment1.nbytes)
        buf_m2 = self._acquire_buffer(moment2.nbytes)

        try:
            # Upload data
            self._upload_buffer(buf_grad, gradients)
            self._upload_buffer(buf_weights, weights)
            self._upload_buffer(buf_m1, moment1)
            self._upload_buffer(buf_m2, moment2)

            # Check if shader is available
            if "affect-adam" not in self.shaders:
                raise RuntimeError(
                    "affect-adam shader not compiled. "
                    "Run: glslc -fshader-stage=compute shaders/affect-adam.glsl -o shaders/spv/affect-adam.spv"
                )

            # Get or create pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "affect-adam", 4, push_constant_size=32
            )

            # Create descriptor set
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                "affect-adam",
                [
                    (self._get_buffer_handle(buf_grad), gradients.nbytes),
                    (self._get_buffer_handle(buf_weights), weights.nbytes),
                    (self._get_buffer_handle(buf_m1), moment1.nbytes),
                    (self._get_buffer_handle(buf_m2), moment2.nbytes),
                ],
            )

            # Pack push constants
            push_constants = struct.pack(
                "IfffffI",
                total_weights,
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                timestep,
            )

            # Dispatch
            workgroups = (total_weights + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download updated weights and moments
            updated_weights = self._download_buffer(buf_weights, weights.nbytes, np.float32)
            updated_m1 = self._download_buffer(buf_m1, moment1.nbytes, np.float32)
            updated_m2 = self._download_buffer(buf_m2, moment2.nbytes, np.float32)

            return updated_weights, updated_m1, updated_m2

        finally:
            self._release_buffers([buf_grad, buf_weights, buf_m1, buf_m2])
