"""
Memory Layers

Uses: memory-read.glsl, memory-write.glsl, memory-context-aggregate.glsl,
      memory-query-pooling.glsl, memory-inject-concat.glsl, memory-inject-gate.glsl,
      memory-inject-residual.glsl

Reference: ref/core/ca3_memory_store.py, ref/core/memory_store.py
"""

import numpy as np

from .module import Module


class MemoryRead(Module):
    """
    Memory Read layer - Retrieve memories using attention mechanism.

    Uses: memory-read.glsl

    Implements key-value memory retrieval with attention-based similarity.
    """

    def __init__(
        self, key_dim: int, value_dim: int, num_memories: int, temperature: float | None = None
    ):
        """
        Initialize MemoryRead layer.

        Args:
            key_dim: Dimension of memory keys
            value_dim: Dimension of memory values
            num_memories: Number of memory slots
            temperature: Temperature for softmax (default: sqrt(key_dim))
        """
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_memories = num_memories
        self.temperature = temperature if temperature is not None else np.sqrt(key_dim)

        # Initialize memory buffers
        self.memory_keys = np.random.randn(num_memories, key_dim).astype(np.float32) * 0.01
        self.memory_values = np.zeros((num_memories, value_dim), dtype=np.float32)

        self._parameters["memory_keys"] = self.memory_keys
        self._parameters["memory_values"] = self.memory_values

    def forward(self, queries: np.ndarray) -> np.ndarray:
        """
        Forward pass - retrieve memories.

        Args:
            queries: Query vectors (batch, key_dim)

        Returns:
            Retrieved values (batch, value_dim)
        """
        backend = self._get_backend()
        return backend.memory_read(
            queries, self.memory_keys, self.memory_values, temperature=self.temperature
        )

    def __repr__(self):
        """Return a debug representation."""

        return f"MemoryRead(key_dim={self.key_dim}, value_dim={self.value_dim}, num_memories={self.num_memories})"


class MemoryWrite(Module):
    """
    Memory Write layer - Write key-value pairs to memory.

    Uses: memory-write.glsl
    """

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        num_memories: int,
        write_mode: int = 0,
        blend_factor: float = 0.5,
    ):
        """
        Initialize MemoryWrite layer.

        Args:
            key_dim: Dimension of memory keys
            value_dim: Dimension of memory values
            num_memories: Number of memory slots
            write_mode: 0 = overwrite, 1 = blend
            blend_factor: For blend mode (default: 0.5)
        """
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_memories = num_memories
        self.write_mode = write_mode
        self.blend_factor = blend_factor

        # Initialize memory buffers
        self.memory_keys = np.random.randn(num_memories, key_dim).astype(np.float32) * 0.01
        self.memory_values = np.zeros((num_memories, value_dim), dtype=np.float32)
        self.write_index = 0

        self._parameters["memory_keys"] = self.memory_keys
        self._parameters["memory_values"] = self.memory_values
        self._buffers["write_index"] = np.array([self.write_index], dtype=np.int32)

    def forward(
        self, new_key: np.ndarray, new_value: np.ndarray, write_index: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass - write to memory.

        Args:
            new_key: New key to write (key_dim,)
            new_value: New value to write (value_dim,)
            write_index: Index to write to (default: uses internal counter)

        Returns:
            (updated_memory_keys, updated_memory_values)
        """
        if write_index is None:
            write_index = self.write_index
            self.write_index = (self.write_index + 1) % self.num_memories

        backend = self._get_backend()
        self.memory_keys, self.memory_values = backend.memory_write(
            new_key,
            new_value,
            self.memory_keys,
            self.memory_values,
            write_index,
            self.write_mode,
            self.blend_factor,
        )

        return self.memory_keys, self.memory_values

    def __repr__(self):
        """Return a debug representation."""

        return f"MemoryWrite(key_dim={self.key_dim}, value_dim={self.value_dim}, num_memories={self.num_memories})"


class MemoryContextAggregate(Module):
    """
    Memory Context Aggregate layer - Aggregate memory context.

    Uses: memory-context-aggregate.glsl
    """

    def __init__(self, dim: int):
        """
        Initialize MemoryContextAggregate layer.

        Args:
            dim: Dimension of memory vectors
        """
        super().__init__()
        self.dim = dim

    def forward(self, memory_contexts: np.ndarray) -> np.ndarray:
        """
        Forward pass - aggregate memory contexts.

        Args:
            memory_contexts: Memory contexts (batch, num_memories, dim)

        Returns:
            Aggregated context (batch, dim)
        """
        backend = self._get_backend()

        # Try GPU shader if available
        if hasattr(backend, "shaders") and "memory-context-aggregate" in backend.shaders:
            try:
                # GPU memory context aggregation would go here
                # For now, use CPU fallback
                pass
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback (mean pooling)
        return memory_contexts.mean(axis=1)

    def __repr__(self):
        """Return a debug representation."""

        return f"MemoryContextAggregate(dim={self.dim})"


class MemoryQueryPooling(Module):
    """
    Memory Query Pooling layer - Pool sequence into memory queries.

    Uses: memory-query-pooling.glsl
    """

    def __init__(self, in_dim: int, out_dim: int):
        """
        Initialize MemoryQueryPooling layer.

        Args:
            in_dim: Input dimension
            out_dim: Output query dimension
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Query projection weights
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.weight = np.random.uniform(-limit, limit, (out_dim, in_dim)).astype(np.float32)
        self.bias = np.zeros(out_dim, dtype=np.float32)

        self._parameters["weight"] = self.weight
        self._parameters["bias"] = self.bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass - pool sequence and project to queries.

        Args:
            x: Input sequence (batch, seq_len, in_dim)

        Returns:
            Query vectors (batch, out_dim)
        """
        backend = self._get_backend()
        return backend.memory_query_pooling(x, self.weight, self.bias)

    def __repr__(self):
        """Return a debug representation."""

        return f"MemoryQueryPooling(in_dim={self.in_dim}, out_dim={self.out_dim})"


class MemoryInject(Module):
    """
    Base class for memory injection layers.
    """

    def __init__(self, dim: int):
        """
        Initialize MemoryInject layer.

        Args:
            dim: Dimension of vectors
        """
        super().__init__()
        self.dim = dim


class MemoryInjectConcat(MemoryInject):
    """
    Memory Inject layer - Concatenate memory context.

    Uses: memory-inject-concat.glsl
    """

    def __init__(self, dim: int):
        """
        Initialize MemoryInjectConcat layer.

        Args:
            dim: Dimension of vectors
        """
        super().__init__(dim)
        # Projection for concatenated output
        self.proj = Linear(dim * 2, dim)
        self._modules["proj"] = self.proj

    def forward(self, x: np.ndarray, memory_context: np.ndarray) -> np.ndarray:
        """
        Forward pass - concatenate and project.

        Args:
            x: Input (batch, seq_len, dim)
            memory_context: Memory context (batch, dim)

        Returns:
            Output (batch, seq_len, dim)
        """
        backend = self._get_backend()

        # Try GPU shader if available
        if hasattr(backend, "shaders") and "memory-inject-concat" in backend.shaders:
            try:
                # GPU memory injection with concat would go here
                # For now, use CPU fallback
                pass
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        batch_size, seq_len, dim = x.shape
        mem_expanded = memory_context[:, None, :]  # (batch, 1, dim)
        mem_expanded = np.broadcast_to(mem_expanded, (batch_size, seq_len, dim))
        concat = np.concatenate([x, mem_expanded], axis=-1)  # (batch, seq_len, dim*2)
        return self.proj(concat)

    def __repr__(self):
        """Return a debug representation."""

        return f"MemoryInjectConcat(dim={self.dim})"


class MemoryInjectGate(MemoryInject):
    """
    Memory Inject layer - Gated memory injection.

    Uses: memory-inject-gate.glsl
    """

    def __init__(self, dim: int):
        """
        Initialize MemoryInjectGate layer.

        Args:
            dim: Dimension of vectors
        """
        super().__init__(dim)

        # Gate weights
        limit = np.sqrt(6.0 / (dim * 2 + dim))
        self.W_gate = np.random.uniform(-limit, limit, (dim, dim * 2)).astype(np.float32)
        self.b_gate = np.zeros(dim, dtype=np.float32)
        self.W_mem_proj = np.random.uniform(-limit, limit, (dim, dim)).astype(np.float32)

        self._parameters["W_gate"] = self.W_gate
        self._parameters["b_gate"] = self.b_gate
        self._parameters["W_mem_proj"] = self.W_mem_proj

    def forward(self, x: np.ndarray, memory_context: np.ndarray) -> np.ndarray:
        """
        Forward pass - gated memory injection.

        Args:
            x: Input (batch, seq_len, dim)
            memory_context: Memory context (batch, dim)

        Returns:
            Output (batch, seq_len, dim)
        """
        backend = self._get_backend()
        return backend.memory_inject_gate(
            x, memory_context, self.W_gate, self.b_gate, self.W_mem_proj
        )

    def __repr__(self):
        """Return a debug representation."""

        return f"MemoryInjectGate(dim={self.dim})"


class MemoryInjectResidual(MemoryInject):
    """
    Memory Inject layer - Residual memory injection.

    Uses: memory-inject-residual.glsl
    """

    def __init__(self, dim: int):
        """
        Initialize MemoryInjectResidual layer.

        Args:
            dim: Dimension of vectors
        """
        super().__init__(dim)
        # Projection for memory context
        self.mem_proj = Linear(dim, dim)
        self._modules["mem_proj"] = self.mem_proj

    def forward(self, x: np.ndarray, memory_context: np.ndarray) -> np.ndarray:
        """
        Forward pass - residual memory injection.

        Args:
            x: Input (batch, seq_len, dim)
            memory_context: Memory context (batch, dim)

        Returns:
            Output (batch, seq_len, dim)
        """
        backend = self._get_backend()

        # Try GPU shader if available
        if hasattr(backend, "shaders") and "memory-inject-residual" in backend.shaders:
            try:
                # GPU memory injection with residual would go here
                # For now, use CPU fallback
                pass
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        batch_size, seq_len, dim = x.shape
        mem_proj = self.mem_proj(memory_context)  # (batch, dim)
        mem_expanded = mem_proj[:, None, :]  # (batch, 1, dim)
        mem_expanded = np.broadcast_to(mem_expanded, (batch_size, seq_len, dim))
        return x + mem_expanded

    def __repr__(self):
        """Return a debug representation."""

        return f"MemoryInjectResidual(dim={self.dim})"


# Import Linear for use in MemoryInjectConcat
from .modules import Linear
