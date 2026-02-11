"""
Memory operations for Vulkan backend.
GPU-accelerated memory read/write operations for episodic memory.
"""

import struct

import numpy as np

from .base import BufferMixin


class VulkanMemory(BufferMixin):
    """Memory operations: read, write, inject gate"""

    def __init__(self, core, pipelines, shaders, fnn_module=None):
        """Initialize the instance."""

        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders
        self.fnn = fnn_module  # For activation_softmax in memory_read

    def memory_write(
        self,
        new_key,
        new_value,
        memory_keys,
        memory_values,
        write_index,
        write_mode=0,
        blend_factor=0.5,
    ):
        """
        Write key-value pair to memory

        Args:
            new_key: New key to write (key_dim,)
            new_value: New value to write (value_dim,)
            memory_keys: Memory keys buffer (num_memories, key_dim)
            memory_values: Memory values buffer (num_memories, value_dim)
            write_index: Index to write to
            write_mode: 0 = overwrite, 1 = blend
            blend_factor: For blend mode (default: 0.5)

        Returns:
            (updated_memory_keys, updated_memory_values)
        """
        key = new_key.astype(np.float32).flatten()
        value = new_value.astype(np.float32).flatten()
        keys = memory_keys.astype(np.float32).flatten()
        values = memory_values.astype(np.float32).flatten()

        key_dim = len(key)
        value_dim = len(value)
        num_memories, _ = memory_keys.shape

        # Create buffers
        buf_key = self._acquire_buffer(key.nbytes)
        buf_value = self._acquire_buffer(value.nbytes)
        buf_keys = self._acquire_buffer(keys.nbytes)
        buf_values = self._acquire_buffer(values.nbytes)

        try:
            # Upload data
            self._upload_buffer(buf_key, key)
            self._upload_buffer(buf_value, value)
            self._upload_buffer(buf_keys, keys)
            self._upload_buffer(buf_values, values)

            # Get or create pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "memory-write", 4, push_constant_size=24
            )

            # Get cached descriptor set
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                "memory-write",
                [
                    (self._get_buffer_handle(buf_key), key.nbytes),
                    (self._get_buffer_handle(buf_value), value.nbytes),
                    (self._get_buffer_handle(buf_keys), keys.nbytes),
                    (self._get_buffer_handle(buf_values), values.nbytes),
                ],
            )

            # Pack push constants
            push_constants = struct.pack(
                "IIIIIf", num_memories, key_dim, value_dim, write_index, write_mode, blend_factor
            )

            # Dispatch
            max_dim = max(key_dim, value_dim)
            workgroups = (max_dim + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download updated memory
            updated_keys = self._download_buffer(buf_keys, keys.nbytes, np.float32)
            updated_keys = updated_keys[: len(keys)].reshape(num_memories, key_dim)
            updated_values = self._download_buffer(buf_values, values.nbytes, np.float32)
            updated_values = updated_values[: len(values)].reshape(num_memories, value_dim)

            return updated_keys, updated_values
        finally:
            self._release_buffers([buf_key, buf_value, buf_keys, buf_values])

    def memory_read(self, queries, memory_keys, memory_values, temperature=None):
        """
        Retrieve memories using attention mechanism

        Args:
            queries: Query vectors (batch, key_dim)
            memory_keys: Memory keys (num_memories, key_dim)
            memory_values: Memory values (num_memories, value_dim)
            temperature: Temperature for softmax (default: sqrt(key_dim))

        Returns:
            Retrieved values (batch, value_dim)
        """
        q = queries.astype(np.float32)
        keys = memory_keys.astype(np.float32)
        values = memory_values.astype(np.float32)

        batch_size, key_dim = q.shape
        num_memories, _ = keys.shape
        _, value_dim = values.shape

        if temperature is None:
            temperature = np.sqrt(key_dim)

        q_flat = q.flatten()
        keys_flat = keys.flatten()
        values_flat = values.flatten()

        # Create buffers
        buf_q = self._acquire_buffer(q_flat.nbytes)
        buf_keys = self._acquire_buffer(keys_flat.nbytes)
        buf_values = self._acquire_buffer(values_flat.nbytes)
        buf_scores = self._acquire_buffer(batch_size * num_memories * 4)
        buf_out = self._acquire_buffer(batch_size * value_dim * 4)

        try:
            # Upload data
            self._upload_buffer(buf_q, q_flat)
            self._upload_buffer(buf_keys, keys_flat)
            self._upload_buffer(buf_values, values_flat)

            # Get or create pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "memory-read", 5, push_constant_size=24
            )

            # Get cached descriptor set
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                "memory-read",
                [
                    (self._get_buffer_handle(buf_q), q_flat.nbytes),
                    (self._get_buffer_handle(buf_keys), keys_flat.nbytes),
                    (self._get_buffer_handle(buf_values), values_flat.nbytes),
                    (self._get_buffer_handle(buf_out), batch_size * value_dim * 4),
                    (self._get_buffer_handle(buf_scores), batch_size * num_memories * 4),
                ],
            )

            # Pass 1: Compute attention scores
            push_constants = struct.pack(
                "IIIIfI", batch_size, num_memories, key_dim, value_dim, temperature, 0
            )
            workgroups = ((batch_size * num_memories) + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Pass 2: Apply softmax (using FNN module if available, else CPU)
            scores = self._download_buffer(buf_scores, batch_size * num_memories * 4, np.float32)
            scores = scores[: batch_size * num_memories].reshape(batch_size, num_memories)

            if self.fnn is not None:
                scores_softmax = self.fnn.activation_softmax(scores, axis=-1)
            else:
                # CPU fallback
                scores_max = scores.max(axis=-1, keepdims=True)
                scores_exp = np.exp(scores - scores_max)
                scores_softmax = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

            # Upload softmax scores back
            scores_softmax_flat = scores_softmax.flatten()
            self._upload_buffer(buf_scores, scores_softmax_flat)

            # Pass 3: Weighted sum
            push_constants = struct.pack(
                "IIIIfI", batch_size, num_memories, key_dim, value_dim, temperature, 2
            )
            workgroups = ((batch_size * value_dim) + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download results
            result = self._download_buffer(buf_out, batch_size * value_dim * 4, np.float32)
            result = result[: batch_size * value_dim].reshape(batch_size, value_dim)

            return result
        finally:
            self._release_buffers([buf_q, buf_keys, buf_values, buf_scores, buf_out])

    # ------------------------------------------------------------------
    # CPU helper operations used in tests
    # ------------------------------------------------------------------
    def memory_query_pooling(
        self, x: np.ndarray, W_query: np.ndarray, b_query: np.ndarray
    ) -> np.ndarray:
        """
        Pool sequence representations into a single query vector per batch,
        then apply a linear projection.
        """
        pooled = x.mean(axis=1)  # (batch, dim)
        return np.matmul(pooled, W_query.T) + b_query

    def memory_inject_gate(
        self,
        attn_output: np.ndarray,
        memory_context: np.ndarray,
        W_gate: np.ndarray,
        b_gate: np.ndarray,
        W_mem_proj: np.ndarray,
    ) -> np.ndarray:
        """
        Simple gating between attention output and memory context (CPU fallback).
        """
        # Broadcast memory context to sequence length
        mem_proj = np.matmul(memory_context, W_mem_proj.T)  # (batch, dim)
        mem_proj = mem_proj[:, None, :]  # (batch, 1, dim)
        mem_proj = np.broadcast_to(mem_proj, attn_output.shape)

        gate_input = np.concatenate([attn_output, mem_proj], axis=-1)  # (batch, seq, dim*2)
        gate_raw = np.matmul(gate_input, W_gate.T) + b_gate
        gate = 1 / (1 + np.exp(-gate_raw))  # sigmoid

        return gate * mem_proj + (1 - gate) * attn_output
