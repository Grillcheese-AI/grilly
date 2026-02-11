"""
Decoding Layers

Uses: decode-greedy.glsl, decode-sample.glsl
"""

import struct

import numpy as np

from .module import Module
from .modules import Softmax


class GreedyDecoder(Module):
    """
    Greedy Decoder - Greedy decoding.

    Uses: decode-greedy.glsl
    """

    def __init__(self, vocab_size: int):
        """
        Initialize GreedyDecoder.

        Args:
            vocab_size: Vocabulary size
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.softmax = Softmax(dim=-1)
        self._modules["softmax"] = self.softmax

    def forward(self, logits: np.ndarray) -> np.ndarray:
        """
        Forward pass - greedy decoding.

        Args:
            logits: Logits (batch, seq_len, vocab_size) or (batch, vocab_size)

        Returns:
            Decoded token indices (batch, seq_len) or (batch,)
        """
        backend = self._get_backend()

        # Try GPU shader if available
        if hasattr(backend, "shaders") and "decode-greedy" in backend.shaders:
            try:
                # Reshape for shader (expects batch, seq_len, vocab_size)
                original_shape = logits.shape
                if logits.ndim == 2:
                    # (batch, vocab_size) -> (batch, 1, vocab_size)
                    logits = logits[:, None, :]

                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.astype(np.float32).flatten()
                predictions = np.zeros(batch_size * seq_len, dtype=np.uint32)

                # Create buffers
                buf_logits, mem_logits = backend.core._create_buffer(
                    logits_flat.nbytes, backend.core.base.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                )
                buf_pred, mem_pred = backend.core._create_buffer(
                    predictions.nbytes, backend.core.base.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                )

                # Upload
                backend.core._upload_buffer(buf_logits, mem_logits, logits_flat)

                # Get pipeline
                pipeline, layout, desc_layout = backend.pipelines.get_or_create_pipeline(
                    "decode-greedy", 3, push_constant_size=16
                )
                desc_set = backend.pipelines._create_descriptor_set(
                    desc_layout,
                    [
                        (buf_logits, logits_flat.nbytes),
                        (buf_pred, predictions.nbytes),
                        (buf_pred, predictions.nbytes),
                    ],
                )

                # Dispatch
                push_constants = struct.pack("IIII", batch_size, seq_len, vocab_size, 0)
                workgroups = (batch_size * seq_len + 255) // 256
                backend.core._dispatch_compute(
                    pipeline, layout, desc_set, workgroups, push_constants
                )

                # Download
                predictions = backend.core._download_buffer(
                    mem_pred, predictions.nbytes, dtype=np.uint32
                )

                # Cleanup
                from vulkan import vkDestroyBuffer, vkFreeDescriptorSets, vkFreeMemory

                vkFreeDescriptorSets(
                    backend.core.device, backend.core.descriptor_pool, 1, [desc_set]
                )
                vkDestroyBuffer(backend.core.device, buf_logits, None)
                vkDestroyBuffer(backend.core.device, buf_pred, None)
                vkFreeMemory(backend.core.device, mem_logits, None)
                vkFreeMemory(backend.core.device, mem_pred, None)

                # Reshape back
                predictions = predictions[: batch_size * seq_len].reshape(batch_size, seq_len)
                if len(original_shape) == 2:
                    predictions = predictions[:, 0]  # Remove seq dimension

                return predictions.astype(np.int32)
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        probs = self.softmax(logits)
        return np.argmax(probs, axis=-1)

    def __repr__(self):
        """Return a debug representation."""

        return f"GreedyDecoder(vocab_size={self.vocab_size})"


class SampleDecoder(Module):
    """
    Sample Decoder - Sampling-based decoding.

    Uses: decode-sample.glsl
    """

    def __init__(self, vocab_size: int, temperature: float = 1.0):
        """
        Initialize SampleDecoder.

        Args:
            vocab_size: Vocabulary size
            temperature: Sampling temperature (default: 1.0)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.softmax = Softmax(dim=-1)
        self._modules["softmax"] = self.softmax

    def forward(self, logits: np.ndarray) -> np.ndarray:
        """
        Forward pass - sampling-based decoding.

        Args:
            logits: Logits (batch, seq_len, vocab_size) or (batch, vocab_size)

        Returns:
            Sampled token indices (batch, seq_len) or (batch,)
        """
        backend = self._get_backend()

        # Apply temperature and softmax
        scaled_logits = logits / self.temperature
        probs = self.softmax(scaled_logits)

        # Try GPU shader if available
        if hasattr(backend, "shaders") and "decode-sample" in backend.shaders:
            try:
                # Reshape for shader
                original_shape = probs.shape
                if probs.ndim == 2:
                    probs = probs[:, None, :]

                batch_size, seq_len, vocab_size = probs.shape
                probs_flat = probs.astype(np.float32).flatten()
                randoms = np.random.rand(batch_size * seq_len).astype(np.float32)
                samples = np.zeros(batch_size * seq_len, dtype=np.uint32)

                # Create buffers
                buf_probs, mem_probs = backend.core._create_buffer(
                    probs_flat.nbytes, backend.core.base.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                )
                buf_rand, mem_rand = backend.core._create_buffer(
                    randoms.nbytes, backend.core.base.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                )
                buf_samples, mem_samples = backend.core._create_buffer(
                    samples.nbytes, backend.core.base.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                )

                # Upload
                backend.core._upload_buffer(buf_probs, mem_probs, probs_flat)
                backend.core._upload_buffer(buf_rand, mem_rand, randoms)

                # Get pipeline
                pipeline, layout, desc_layout = backend.pipelines.get_or_create_pipeline(
                    "decode-sample", 3, push_constant_size=12
                )
                desc_set = backend.pipelines._create_descriptor_set(
                    desc_layout,
                    [
                        (buf_probs, probs_flat.nbytes),
                        (buf_rand, randoms.nbytes),
                        (buf_samples, samples.nbytes),
                    ],
                )

                # Dispatch
                import struct

                push_constants = struct.pack("III", batch_size, seq_len, vocab_size)
                workgroups = (batch_size * seq_len + 255) // 256
                backend.core._dispatch_compute(
                    pipeline, layout, desc_set, workgroups, push_constants
                )

                # Download
                samples = backend.core._download_buffer(
                    mem_samples, samples.nbytes, dtype=np.uint32
                )

                # Cleanup
                from vulkan import vkDestroyBuffer, vkFreeDescriptorSets, vkFreeMemory

                vkFreeDescriptorSets(
                    backend.core.device, backend.core.descriptor_pool, 1, [desc_set]
                )
                vkDestroyBuffer(backend.core.device, buf_probs, None)
                vkDestroyBuffer(backend.core.device, buf_rand, None)
                vkDestroyBuffer(backend.core.device, buf_samples, None)
                vkFreeMemory(backend.core.device, mem_probs, None)
                vkFreeMemory(backend.core.device, mem_rand, None)
                vkFreeMemory(backend.core.device, mem_samples, None)

                # Reshape back
                samples = samples[: batch_size * seq_len].reshape(batch_size, seq_len)
                if len(original_shape) == 2:
                    samples = samples[:, 0]

                return samples.astype(np.int32)
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        if probs.ndim == 2:
            # (batch, vocab_size)
            return np.array([np.random.choice(self.vocab_size, p=p) for p in probs])
        else:
            # (batch, seq_len, vocab_size)
            batch_size, seq_len, vocab_size = probs.shape
            samples = np.zeros((batch_size, seq_len), dtype=np.int32)
            for b in range(batch_size):
                for s in range(seq_len):
                    samples[b, s] = np.random.choice(vocab_size, p=probs[b, s])
            return samples

    def __repr__(self):
        """Return a debug representation."""

        return f"SampleDecoder(vocab_size={self.vocab_size}, temperature={self.temperature})"
