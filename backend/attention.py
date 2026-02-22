"""
Attention operations for Vulkan backend.
GPU-accelerated attention mechanisms for transformers.
"""

import logging
import struct

import numpy as np

from .base import VULKAN_AVAILABLE, BufferMixin
from .shader_registry import get_shader

logger = logging.getLogger(__name__)

# Import numba-accelerated CPU fallbacks
try:
    from ..utils.numba_ops import (
        NUMBA_AVAILABLE,
    )
    from ..utils.numba_ops import (
        attention_output as numba_attention_output,
    )
    from ..utils.numba_ops import (
        prosody_modulation as numba_prosody_modulation,
    )
    from ..utils.numba_ops import (
        rope as numba_rope,
    )
except ImportError:
    NUMBA_AVAILABLE = False
    numba_rope = None
    numba_prosody_modulation = None
    numba_attention_output = None

if VULKAN_AVAILABLE:
    from vulkan import *


class VulkanAttention(BufferMixin):
    """Attention operations: scores, mask, output, concat heads"""

    def __init__(self, core, pipelines, shaders, architecture: str = None):
        """Initialize the instance."""

        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders
        self.architecture = architecture  # Model architecture (e.g., 'bert', 'gpt', 't5')
        self._pool = None  # Lazy initialization

    def attention_scores(self, queries, keys, num_heads, head_dim, scale=None):
        """
        Compute attention scores: Q @ K^T / sqrt(head_dim)

        Args:
            queries: Query tensor (batch, seq_len, num_heads, head_dim) or (batch, seq_len, num_heads * head_dim)
            keys: Key tensor (batch, seq_len, num_heads, head_dim) or (batch, seq_len, num_heads * head_dim)
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            scale: Optional scaling factor (default: 1/sqrt(head_dim))

        Returns:
            Attention scores (batch, num_heads, seq_len, seq_len)
        """
        q = queries.astype(np.float32)
        k = keys.astype(np.float32)

        if scale is None:
            scale = 1.0 / np.sqrt(head_dim)

        # Handle flattened head dimension
        if q.ndim == 3:
            batch_size, seq_len, _ = q.shape
            q = q.reshape(batch_size, seq_len, num_heads, head_dim)
            k = k.reshape(batch_size, seq_len, num_heads, head_dim)
        else:
            batch_size, seq_len, num_heads, head_dim = q.shape

        q_flat = q.flatten()
        k_flat = k.flatten()
        scores_size = batch_size * num_heads * seq_len * seq_len * 4

        # Create buffers
        buf_q = self._acquire_buffer(q_flat.nbytes)
        buf_k = self._acquire_buffer(k_flat.nbytes)
        buf_scores = self._acquire_buffer(scores_size)
        # Create dummy V buffer (required by shader)
        buf_v_dummy = self._acquire_buffer(q_flat.nbytes)

        try:
            # Upload data
            self._upload_buffer(buf_q, q_flat)
            self._upload_buffer(buf_k, k_flat)
            self._upload_buffer(buf_v_dummy, q_flat)

            # Get or create pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "attention-scores", 4, push_constant_size=24
            )

            # Get cached descriptor set
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                "attention-scores",
                [
                    (self._get_buffer_handle(buf_q), q_flat.nbytes),
                    (self._get_buffer_handle(buf_k), k_flat.nbytes),
                    (self._get_buffer_handle(buf_v_dummy), q_flat.nbytes),
                    (self._get_buffer_handle(buf_scores), scores_size),
                ],
            )

            # Pack push constants
            push_constants = struct.pack(
                "IIIIfI", batch_size, seq_len, num_heads, head_dim, scale, 0
            )

            # Dispatch
            workgroups_x = (seq_len + 15) // 16
            workgroups_y = ((batch_size * num_heads * seq_len) + 15) // 16
            self.core._dispatch_compute(
                pipeline,
                pipeline_layout,
                descriptor_set,
                workgroups_x,
                push_constants,
                workgroups_y,
            )

            # Download results
            result = self._download_buffer(buf_scores, scores_size, np.float32)
            result = result[: batch_size * num_heads * seq_len * seq_len].reshape(
                batch_size, num_heads, seq_len, seq_len
            )

            return result
        finally:
            self._release_buffers([buf_q, buf_k, buf_v_dummy, buf_scores])

    def attention_mask(self, attention_scores, use_causal=True, mask_value=-1e9, custom_mask=None):
        """
        Apply mask to attention scores

        Args:
            attention_scores: Attention scores (batch, num_heads, seq_len, seq_len)
            use_causal: Whether to apply causal masking (if True, custom_mask is ignored)
            mask_value: Value to use for masked positions
            custom_mask: Optional custom mask (batch, seq_len) - 1.0 = keep, 0.0 = mask out

        Returns:
            Masked attention scores
        """
        scores = attention_scores.astype(np.float32)
        batch_size, num_heads, seq_len, _ = scores.shape

        scores_flat = scores.flatten()

        # Create mask buffer
        if use_causal:
            # Causal mask (seq_len, seq_len) - for backward compatibility
            mask = np.ones((seq_len, seq_len), dtype=np.float32)
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    mask[i, j] = 0.0
            mask_flat = mask.flatten()
        else:
            # Custom mask (batch, seq_len)
            if custom_mask is None:
                # No mask - return scores unchanged
                return scores
            mask_flat = custom_mask.astype(np.float32).flatten()

        # Create buffers
        buf_scores = self._acquire_buffer(scores_flat.nbytes)
        buf_mask = self._acquire_buffer(mask_flat.nbytes)

        try:
            # Upload data
            self._upload_buffer(buf_scores, scores_flat)
            self._upload_buffer(buf_mask, mask_flat)

            # Get or create pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "attention-mask", 2, push_constant_size=20
            )

            # Get cached descriptor set
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                "attention-mask",
                [
                    (self._get_buffer_handle(buf_scores), scores_flat.nbytes),
                    (self._get_buffer_handle(buf_mask), mask_flat.nbytes),
                ],
            )

            # Pack push constants
            push_constants = struct.pack(
                "IIIIf", batch_size, num_heads, seq_len, 1 if use_causal else 0, mask_value
            )

            # Dispatch
            workgroups = (len(scores_flat) + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download results
            result = self._download_buffer(buf_scores, scores_flat.nbytes, np.float32)
            result = result[: len(scores_flat)].reshape(batch_size, num_heads, seq_len, seq_len)

            return result
        finally:
            self._release_buffers([buf_scores, buf_mask])

    def attention_output(self, attention_weights, values, num_heads, head_dim):
        """
        Compute attention output: weights @ values

        Args:
            attention_weights: Attention weights (batch, num_heads, seq_len, seq_len)
            values: Value tensor (batch, seq_len, num_heads, head_dim)
            num_heads: Number of attention heads
            head_dim: Dimension of each head

        Returns:
            Attention output (batch, seq_len, num_heads, head_dim)
        """
        weights = attention_weights.astype(np.float32)
        v = values.astype(np.float32)

        batch_size, num_heads_w, seq_len, _ = weights.shape

        if v.ndim == 3:
            v = v.reshape(batch_size, seq_len, num_heads, head_dim)

        # Try to get architecture-specific shader, fall back to generic
        shader_name = get_shader("attention-output", self.architecture)
        if shader_name is None:
            shader_name = "attention-output"  # Fall back to generic

        # Check if shader is available
        if shader_name not in self.shaders:
            # CPU fallback: weights @ v
            # weights: (batch, num_heads, seq_len, seq_len)
            # v: (batch, seq_len, num_heads, head_dim) -> transpose to (batch, num_heads, seq_len, head_dim)
            v_transposed = v.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)

            if NUMBA_AVAILABLE and numba_attention_output is not None:
                # Use numba-accelerated attention output
                result = numba_attention_output(weights, v_transposed)
            else:
                # Pure numpy fallback
                result = np.einsum("bhqk,bhkd->bhqd", weights, v_transposed)

            # Transpose back to (batch, seq_len, num_heads, head_dim)
            result = result.transpose(0, 2, 1, 3)
            return result

        weights_flat = weights.flatten()
        v_flat = v.flatten()

        output_size = batch_size * seq_len * num_heads * head_dim * 4

        # Create buffers
        buf_weights = self._acquire_buffer(weights_flat.nbytes)
        buf_v = self._acquire_buffer(v_flat.nbytes)
        buf_out = self._acquire_buffer(output_size)

        try:
            # Upload data
            self._upload_buffer(buf_weights, weights_flat)
            self._upload_buffer(buf_v, v_flat)

            # Get or create pipeline (use shader_name from registry)
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                shader_name, 3, push_constant_size=16
            )

            # Get cached descriptor set (use shader_name from registry)
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                shader_name,
                [
                    (self._get_buffer_handle(buf_weights), weights_flat.nbytes),
                    (self._get_buffer_handle(buf_v), v_flat.nbytes),
                    (self._get_buffer_handle(buf_out), output_size),
                ],
            )

            # Pack push constants
            push_constants = struct.pack("IIII", batch_size, seq_len, num_heads, head_dim)

            # Dispatch
            workgroups = ((batch_size * seq_len * num_heads * head_dim) + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download results
            result = self._download_buffer(buf_out, output_size, np.float32)
            result = result[: batch_size * seq_len * num_heads * head_dim].reshape(
                batch_size, seq_len, num_heads, head_dim
            )

            return result
        finally:
            self._release_buffers([buf_weights, buf_v, buf_out])

    def attention_concat_heads(self, attention_output):
        """
        Concatenate attention heads

        Args:
            attention_output: Attention output (batch, seq_len, num_heads, head_dim)

        Returns:
            Concatenated output (batch, seq_len, num_heads * head_dim)
        """
        output = attention_output.astype(np.float32)
        batch_size, seq_len, num_heads, head_dim = output.shape

        output_flat = output.flatten()
        concat_size = batch_size * seq_len * num_heads * head_dim * 4

        # Create buffers
        buf_in = self._acquire_buffer(output_flat.nbytes)
        buf_out = self._acquire_buffer(concat_size)

        try:
            # Upload data
            self._upload_buffer(buf_in, output_flat)

            # Get or create pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "attention-concat-heads", 2, push_constant_size=16
            )

            # Get cached descriptor set
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                "attention-concat-heads",
                [
                    (self._get_buffer_handle(buf_in), output_flat.nbytes),
                    (self._get_buffer_handle(buf_out), concat_size),
                ],
            )

            # Pack push constants
            push_constants = struct.pack("IIII", batch_size, seq_len, num_heads, head_dim)

            # Dispatch
            workgroups = ((batch_size * seq_len * num_heads * head_dim) + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download results
            result = self._download_buffer(buf_out, concat_size, np.float32)
            result = result[: batch_size * seq_len * num_heads * head_dim].reshape(
                batch_size, seq_len, num_heads * head_dim
            )

            return result
        finally:
            self._release_buffers([buf_in, buf_out])

    def flash_attention2(
        self,
        queries,
        keys,
        values,
        num_heads,
        head_dim,
        tile_size_q=64,
        tile_size_k=64,
        scale=None,
        mask=None,
        causal=False,
    ):
        """
        Flash Attention 2: Tiled attention with online softmax

        Processes attention in blocks to reduce memory from O(N²) to O(N).
        Uses online softmax algorithm for numerical stability.
        Optimized for GPU tiling support.

        Args:
            queries: Query tensor (batch, seq_len, num_heads, head_dim) or (batch, seq_len, num_heads * head_dim)
            keys: Key tensor (batch, seq_len, num_heads, head_dim) or (batch, seq_len, num_heads * head_dim)
            values: Value tensor (batch, seq_len, num_heads, head_dim) or (batch, seq_len, num_heads * head_dim)
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            tile_size_q: Tile size for query dimension (default: 64, optimal for most GPUs)
            tile_size_k: Tile size for key dimension (default: 64, optimal for most GPUs)
            scale: Optional scaling factor (default: 1/sqrt(head_dim))
            mask: Optional attention mask (batch, seq_len) - 0.0 = mask out, 1.0 = keep
            causal: Whether to apply causal masking (default: False)

        Returns:
            Attention output (batch, seq_len, num_heads, head_dim)
        """
        q = queries.astype(np.float32)
        k = keys.astype(np.float32)
        v = values.astype(np.float32)

        if scale is None:
            scale = 1.0 / np.sqrt(head_dim)

        # Handle flattened head dimension
        if q.ndim == 3:
            batch_size, seq_len, _ = q.shape
            q = q.reshape(batch_size, seq_len, num_heads, head_dim)
            k = k.reshape(batch_size, seq_len, num_heads, head_dim)
            v = v.reshape(batch_size, seq_len, num_heads, head_dim)
        else:
            batch_size, seq_len, num_heads, head_dim = q.shape

        # Create causal mask if requested
        if causal and mask is None:
            mask = np.ones((batch_size, seq_len), dtype=np.float32)
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    mask[:, j] = 0.0

        # Flatten tensors
        q_flat = q.flatten()
        k_flat = k.flatten()
        v_flat = v.flatten()

        # Calculate number of tiles
        num_tiles_q = (seq_len + tile_size_q - 1) // tile_size_q
        num_tiles_k = (seq_len + tile_size_k - 1) // tile_size_k

        # Temporary buffers for online softmax
        num_q_positions = batch_size * seq_len * num_heads
        running_max_size = num_q_positions * 4  # float32
        running_sum_size = num_q_positions * 4
        output_accum_size = batch_size * seq_len * num_heads * head_dim * 4

        # Create buffers
        buf_q = self._acquire_buffer(q_flat.nbytes)
        buf_k = self._acquire_buffer(k_flat.nbytes)
        buf_v = self._acquire_buffer(v_flat.nbytes)
        buf_running_max = self._acquire_buffer(running_max_size)
        buf_running_sum = self._acquire_buffer(running_sum_size)
        buf_output_accum = self._acquire_buffer(output_accum_size)
        buf_output = self._acquire_buffer(output_accum_size)

        # Handle mask buffer
        buf_mask = None
        mask_flat = None
        if mask is not None:
            mask_flat = mask.astype(np.float32).flatten()
            buf_mask = self._acquire_buffer(mask_flat.nbytes)

        try:
            # Upload Q, K, V
            self._upload_buffer(buf_q, q_flat)
            self._upload_buffer(buf_k, k_flat)
            self._upload_buffer(buf_v, v_flat)

            if buf_mask is not None:
                self._upload_buffer(buf_mask, mask_flat)

            # Check if shader is available
            if "flash-attention2" not in self.shaders:
                raise RuntimeError(
                    "flash-attention2 shader not compiled. "
                    "Run: glslc -fshader-stage=compute shaders/flash-attention2.glsl -o shaders/spv/flash-attention2.spv"
                )

            # Get or create pipeline
            num_bindings = 8  # Q, K, V, mask, output, running_max, running_sum, output_accum
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "flash-attention2",
                num_bindings,
                push_constant_size=44,  # 11 uint/float values
            )

            # Helper for mask descriptor entry
            mask_handle = (
                self._get_buffer_handle(buf_mask)
                if buf_mask is not None
                else self._get_buffer_handle(buf_q)
            )
            mask_size = mask_flat.nbytes if mask_flat is not None else q_flat.nbytes

            # Pass 0: Initialize running max, sum, and accumulator
            descriptor_set_init = self.pipelines.get_cached_descriptor_set(
                "flash-attention2",
                [
                    (self._get_buffer_handle(buf_q), q_flat.nbytes),
                    (self._get_buffer_handle(buf_k), k_flat.nbytes),
                    (self._get_buffer_handle(buf_v), v_flat.nbytes),
                    (mask_handle, mask_size),
                    (self._get_buffer_handle(buf_output), output_accum_size),
                    (self._get_buffer_handle(buf_running_max), running_max_size),
                    (self._get_buffer_handle(buf_running_sum), running_sum_size),
                    (self._get_buffer_handle(buf_output_accum), output_accum_size),
                ],
            )

            push_constants_init = struct.pack(
                "IIIIfIIIII",
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                scale,
                tile_size_q,
                tile_size_k,
                0,  # pass_type = 0 (initialize)
                1 if mask is not None else 0,  # has_mask
                0,
                0,  # q_tile_idx, k_tile_idx (not used in init)
            )

            # Dispatch initialization
            workgroups_init_x = 16
            workgroups_init_y = (num_q_positions + 15) // 16
            self.core._dispatch_compute(
                pipeline,
                pipeline_layout,
                descriptor_set_init,
                workgroups_init_x,
                push_constants_init,
                workgroups_init_y,
            )

            # Pass 1: Process all tiles
            for q_tile in range(num_tiles_q):
                for k_tile in range(num_tiles_k):
                    descriptor_set_tile = self.pipelines.get_cached_descriptor_set(
                        "flash-attention2",
                        [
                            (self._get_buffer_handle(buf_q), q_flat.nbytes),
                            (self._get_buffer_handle(buf_k), k_flat.nbytes),
                            (self._get_buffer_handle(buf_v), v_flat.nbytes),
                            (mask_handle, mask_size),
                            (self._get_buffer_handle(buf_output), output_accum_size),
                            (self._get_buffer_handle(buf_running_max), running_max_size),
                            (self._get_buffer_handle(buf_running_sum), running_sum_size),
                            (self._get_buffer_handle(buf_output_accum), output_accum_size),
                        ],
                    )

                    push_constants_tile = struct.pack(
                        "IIIIfIIIII",
                        batch_size,
                        seq_len,
                        num_heads,
                        head_dim,
                        scale,
                        tile_size_q,
                        tile_size_k,
                        1,  # pass_type = 1 (process tile)
                        1 if mask is not None else 0,  # has_mask
                        q_tile,
                        k_tile,
                    )

                    # Dispatch tile processing
                    workgroups_tile_x = (tile_size_k + 15) // 16
                    workgroups_tile_y = (batch_size * num_heads * tile_size_q + 15) // 16
                    self.core._dispatch_compute(
                        pipeline,
                        pipeline_layout,
                        descriptor_set_tile,
                        workgroups_tile_x,
                        push_constants_tile,
                        workgroups_tile_y,
                    )

            # Pass 2: Finalize output
            descriptor_set_final = self.pipelines.get_cached_descriptor_set(
                "flash-attention2",
                [
                    (self._get_buffer_handle(buf_q), q_flat.nbytes),
                    (self._get_buffer_handle(buf_k), k_flat.nbytes),
                    (self._get_buffer_handle(buf_v), v_flat.nbytes),
                    (mask_handle, mask_size),
                    (self._get_buffer_handle(buf_output), output_accum_size),
                    (self._get_buffer_handle(buf_running_max), running_max_size),
                    (self._get_buffer_handle(buf_running_sum), running_sum_size),
                    (self._get_buffer_handle(buf_output_accum), output_accum_size),
                ],
            )

            push_constants_final = struct.pack(
                "IIIIfIIIII",
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                scale,
                tile_size_q,
                tile_size_k,
                2,  # pass_type = 2 (finalize)
                1 if mask is not None else 0,  # has_mask
                0,
                0,  # q_tile_idx, k_tile_idx (not used in finalize)
            )

            # Dispatch finalization
            workgroups_final_x = (head_dim + 15) // 16
            workgroups_final_y = (num_q_positions + 15) // 16
            self.core._dispatch_compute(
                pipeline,
                pipeline_layout,
                descriptor_set_final,
                workgroups_final_x,
                push_constants_final,
                workgroups_final_y,
            )

            # Download results
            result = self._download_buffer(buf_output, output_accum_size, np.float32)
            result = result[: batch_size * seq_len * num_heads * head_dim].reshape(
                batch_size, seq_len, num_heads, head_dim
            )

            return result
        finally:
            buffers = [
                buf_q,
                buf_k,
                buf_v,
                buf_running_max,
                buf_running_sum,
                buf_output_accum,
                buf_output,
            ]
            if buf_mask is not None:
                buffers.append(buf_mask)
            self._release_buffers(buffers)

    def apply_rope(
        self,
        q_or_k: np.ndarray,
        position_ids: np.ndarray = None,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
    ) -> np.ndarray:
        """
        Apply RoPE (Rotary Position Embeddings) to Q or K tensors.

        Args:
            q_or_k: Query or Key tensor (batch, seq_len, num_heads, head_dim)
            position_ids: Position indices (batch, seq_len). If None, uses [0, 1, 2, ...]
            rope_base: Base frequency for RoPE (default: 10000.0)
            rope_scaling: Scaling factor for extended context (default: 1.0)

        Returns:
            Rotated Q or K tensor (same shape)
        """
        if "rope" not in self.shaders:
            # CPU fallback for RoPE
            logger.debug("RoPE shader not available, using CPU fallback")
            return self._rope_cpu(q_or_k, position_ids, rope_base, rope_scaling)

        qk = q_or_k.astype(np.float32)
        batch_size, seq_len, num_heads, head_dim = qk.shape

        # Create position_ids if not provided
        if position_ids is None:
            position_ids = np.arange(seq_len, dtype=np.int32)
            position_ids = np.tile(position_ids, (batch_size, 1))

        # Flatten for GPU
        qk_flat = qk.flatten()
        total_elements = len(qk_flat)

        # Create buffers
        buf_input = self._acquire_buffer(qk_flat.nbytes)
        buf_output = self._acquire_buffer(qk_flat.nbytes)

        try:
            # Upload data
            self._upload_buffer(buf_input, qk_flat)

            # Get or create pipeline
            # Push constants: batch_size(uint), seq_len(uint), num_heads(uint), head_dim(uint), rope_base(float), use_precomputed(uint), rope_scaling(float)
            # Total: 4*4 + 4 + 4 + 4 = 28 bytes
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "rope", 2, push_constant_size=28
            )

            # Get cached descriptor set
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                "rope",
                [
                    (self._get_buffer_handle(buf_input), qk_flat.nbytes),
                    (self._get_buffer_handle(buf_output), qk_flat.nbytes),
                ],
            )

            # Pack push constants: batch_size, seq_len, num_heads, head_dim, rope_base, use_precomputed, rope_scaling
            push_constants = struct.pack(
                "IIIIfIf",
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                rope_base,
                0,  # use_precomputed = 0 (compute on-the-fly)
                rope_scaling,
            )

            # Dispatch - shader now processes all elements (not just pairs)
            workgroups = (total_elements + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download results
            result = self._download_buffer(buf_output, qk_flat.nbytes, np.float32)
            result = result[:total_elements].reshape(batch_size, seq_len, num_heads, head_dim)

            return result
        finally:
            self._release_buffers([buf_input, buf_output])

    def apply_prosody_modulation(
        self,
        attention_scores: np.ndarray,
        prosody_features: np.ndarray,
        prosody_weights: np.ndarray,
        prosody_strength: float = 0.3,
    ) -> np.ndarray:
        """
        Apply prosody modulation to attention scores.

        Args:
            attention_scores: Attention scores (batch, num_heads, seq_len, seq_len)
            prosody_features: Prosody features (batch, seq_len, prosody_dim)
            prosody_weights: Prosody projection weights (num_heads, prosody_dim)
            prosody_strength: Modulation strength (default: 0.3)

        Returns:
            Modulated attention scores (same shape)
        """
        if "attention-prosody-modulation" not in self.shaders:
            # CPU fallback
            logger.debug("Prosody modulation shader not available, using CPU fallback")
            return self._prosody_modulation_cpu(
                attention_scores, prosody_features, prosody_weights, prosody_strength
            )

        scores = attention_scores.astype(np.float32)
        prosody = prosody_features.astype(np.float32)
        weights = prosody_weights.astype(np.float32)

        batch_size, num_heads, seq_len, _ = scores.shape
        prosody_dim = prosody_features.shape[-1]

        scores_flat = scores.flatten()
        prosody_flat = prosody.flatten()
        weights_flat = weights.flatten()

        # Create buffers
        buf_scores = self._acquire_buffer(scores_flat.nbytes)
        buf_prosody = self._acquire_buffer(prosody_flat.nbytes)
        buf_weights = self._acquire_buffer(weights_flat.nbytes)

        try:
            # Upload data
            self._upload_buffer(buf_scores, scores_flat)
            self._upload_buffer(buf_prosody, prosody_flat)
            self._upload_buffer(buf_weights, weights_flat)

            # Get or create pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "attention-prosody-modulation", 3, push_constant_size=24
            )

            # Get cached descriptor set
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                "attention-prosody-modulation",
                [
                    (self._get_buffer_handle(buf_scores), scores_flat.nbytes),
                    (self._get_buffer_handle(buf_prosody), prosody_flat.nbytes),
                    (self._get_buffer_handle(buf_weights), weights_flat.nbytes),
                ],
            )

            # Pack push constants: batch_size, num_heads, seq_len, prosody_dim, prosody_strength
            push_constants = struct.pack(
                "IIIIf", batch_size, num_heads, seq_len, prosody_dim, prosody_strength
            )

            # Dispatch
            total_scores = batch_size * num_heads * seq_len * seq_len
            workgroups = (total_scores + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download results
            result = self._download_buffer(buf_scores, scores_flat.nbytes, np.float32)
            result = result[:total_scores].reshape(batch_size, num_heads, seq_len, seq_len)

            return result
        finally:
            self._release_buffers([buf_scores, buf_prosody, buf_weights])

    def _prosody_modulation_cpu(
        self,
        attention_scores: np.ndarray,
        prosody_features: np.ndarray,
        prosody_weights: np.ndarray,
        prosody_strength: float,
    ) -> np.ndarray:
        """CPU fallback for prosody modulation (numba-accelerated if available)"""
        if NUMBA_AVAILABLE and numba_prosody_modulation is not None:
            return numba_prosody_modulation(
                attention_scores.astype(np.float32),
                prosody_features.astype(np.float32),
                prosody_weights.astype(np.float32),
                prosody_strength,
            )

        # Pure numpy fallback
        batch_size, num_heads, seq_len, _ = attention_scores.shape

        # prosody_features: (batch, seq_len, prosody_dim)
        # prosody_weights: (num_heads, prosody_dim)
        # Compute: prosody_bias = prosody_features @ prosody_weights.T -> (batch, seq_len, num_heads)
        prosody_bias = np.einsum("bsd,hd->bsh", prosody_features, prosody_weights)

        # Broadcast to attention shape: (batch, num_heads, seq_len, seq_len)
        prosody_bias = prosody_bias.transpose(0, 2, 1)  # (batch, num_heads, seq_len)
        prosody_bias = prosody_bias[:, :, np.newaxis, :]  # (batch, num_heads, 1, seq_len)

        return attention_scores + prosody_strength * prosody_bias

    def _rope_cpu(
        self,
        q_or_k: np.ndarray,
        position_ids: np.ndarray = None,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
    ) -> np.ndarray:
        """
        CPU fallback for RoPE using PyTorch's rotate_half approach (numba-accelerated if available).

        ModernBERT uses: q_embed = (q * cos) + (rotate_half(q) * sin)
        where rotate_half(q) = [-q[head_dim//2:], q[:head_dim//2]]
        """
        batch_size, seq_len, num_heads, head_dim = q_or_k.shape

        if position_ids is None:
            position_ids = np.arange(seq_len, dtype=np.int32)
            position_ids = np.tile(position_ids, (batch_size, 1))

        if NUMBA_AVAILABLE and numba_rope is not None:
            return numba_rope(
                q_or_k.astype(np.float32), position_ids.astype(np.int32), rope_base, rope_scaling
            )

        # Pure numpy fallback
        half_dim = head_dim // 2
        inv_freq = 1.0 / (rope_base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        result = np.empty_like(q_or_k)

        for b in range(batch_size):
            for s in range(seq_len):
                pos = float(position_ids[b, s]) / rope_scaling
                freqs = pos * inv_freq
                cos_vals = np.cos(freqs)
                sin_vals = np.sin(freqs)

                for h in range(num_heads):
                    qk_first = q_or_k[b, s, h, :half_dim]
                    qk_second = q_or_k[b, s, h, half_dim:]

                    result[b, s, h, :half_dim] = qk_first * cos_vals - qk_second * sin_vals
                    result[b, s, h, half_dim:] = qk_second * cos_vals + qk_first * sin_vals

        return result

    # ------------------------------------------------------------------
    # GQA Decode Attention (single-token, reads from KV-cache)
    # ------------------------------------------------------------------
    def gqa_decode_attention(
        self,
        query: np.ndarray,
        k_cache: np.ndarray,
        v_cache: np.ndarray,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        cache_len: int = None,
        scale: float = None,
    ) -> np.ndarray:
        """
        GQA decode attention: single-token query against KV-cache.

        Fuses repeat_kv expansion: maps query heads to KV heads via
        integer division (kv_head = q_head // group_size).

        For Llama 3.2 3B: 24 query heads, 8 KV heads, group_size=3.

        Args:
            query: Query for decode token (batch, 1, num_q_heads, head_dim)
            k_cache: Cached keys (batch, cache_len, num_kv_heads, head_dim)
            v_cache: Cached values (batch, cache_len, num_kv_heads, head_dim)
            num_q_heads: Number of query heads
            num_kv_heads: Number of KV heads (must divide num_q_heads evenly)
            head_dim: Head dimension
            cache_len: Current length of KV cache (if None, inferred from k_cache)
            scale: Attention scale (default: 1/sqrt(head_dim))

        Returns:
            Attention output (batch, 1, num_q_heads, head_dim)
        """
        if scale is None:
            scale = 1.0 / np.sqrt(head_dim)

        q = np.asarray(query, dtype=np.float32)
        k = np.asarray(k_cache, dtype=np.float32)
        v = np.asarray(v_cache, dtype=np.float32)

        batch_size = q.shape[0]
        if cache_len is None:
            cache_len = k.shape[1]

        # CPU fallback if shader unavailable
        if "gqa-attention" not in self.shaders:
            kv_group_size = num_q_heads // num_kv_heads
            q_2d = q.reshape(batch_size, num_q_heads, head_dim)
            output = np.zeros((batch_size, num_q_heads, head_dim), dtype=np.float32)

            for b in range(batch_size):
                for qh in range(num_q_heads):
                    kv_head = qh // kv_group_size
                    scores = np.einsum("d,sd->s", q_2d[b, qh], k[b, :cache_len, kv_head]) * scale
                    scores_max = np.max(scores)
                    exp_scores = np.exp(scores - scores_max)
                    weights = exp_scores / np.sum(exp_scores)
                    output[b, qh] = np.einsum("s,sd->d", weights, v[b, :cache_len, kv_head])

            return output.reshape(batch_size, 1, num_q_heads, head_dim)

        # GPU path
        q_flat = q.reshape(batch_size, num_q_heads, head_dim).flatten()
        k_flat = k[:, :cache_len].flatten()
        v_flat = v[:, :cache_len].flatten()

        q_bytes = q_flat.nbytes
        k_bytes = k_flat.nbytes
        v_bytes = v_flat.nbytes
        out_bytes = int(batch_size * num_q_heads * head_dim * 4)
        score_bytes = int(batch_size * num_q_heads * cache_len * 4)

        buf_q = self._acquire_buffer(q_bytes)
        buf_k = self._acquire_buffer(k_bytes)
        buf_v = self._acquire_buffer(v_bytes)
        buf_out = self._acquire_buffer(out_bytes)
        buf_scores = self._acquire_buffer(score_bytes)

        self._upload_buffer(buf_q, q_flat)
        self._upload_buffer(buf_k, k_flat)
        self._upload_buffer(buf_v, v_flat)

        pipeline, pipeline_layout, _ = self.pipelines.get_or_create_pipeline(
            "gqa-attention", 5, push_constant_size=24
        )

        descriptor_set = self.pipelines.get_cached_descriptor_set(
            "gqa-attention",
            [
                (self._get_buffer_handle(buf_q), q_bytes),
                (self._get_buffer_handle(buf_k), k_bytes),
                (self._get_buffer_handle(buf_v), v_bytes),
                (self._get_buffer_handle(buf_out), out_bytes),
                (self._get_buffer_handle(buf_scores), score_bytes),
            ],
        )

        push_constants = struct.pack(
            "IIIIIf", batch_size, num_q_heads, num_kv_heads, head_dim, cache_len, scale
        )

        total_q = batch_size * num_q_heads
        workgroups_x = (max(cache_len, head_dim) + 15) // 16
        workgroups_y = (total_q + 15) // 16

        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set, workgroups_x, push_constants, workgroups_y
        )

        result = self._download_buffer(buf_out, out_bytes, np.float32)
        self._release_buffers([buf_q, buf_k, buf_v, buf_out, buf_scores])

        return result.reshape(batch_size, 1, num_q_heads, head_dim)
