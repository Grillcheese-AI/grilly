"""
Functional Memory Operations

Uses: memory-read.glsl, memory-write.glsl, memory-context-aggregate.glsl,
      memory-query-pooling.glsl, memory-inject-concat.glsl, memory-inject-gate.glsl,
      memory-inject-residual.glsl
"""

import numpy as np


def _get_backend():
    """Get backend instance"""
    try:
        from ..backend.compute import Compute

        return Compute()
    except Exception:
        return None


def memory_read(
    queries: np.ndarray,
    memory_keys: np.ndarray,
    memory_values: np.ndarray,
    temperature: float | None = None,
) -> np.ndarray:
    """
    Read from key-value memory using attention mechanism.

    Uses: memory-read.glsl

    Args:
        queries: Query vectors (batch, key_dim)
        memory_keys: Memory keys (num_memories, key_dim)
        memory_values: Memory values (num_memories, value_dim)
        temperature: Temperature for softmax (default: sqrt(key_dim))

    Returns:
        Retrieved values (batch, value_dim)
    """
    from grilly import Compute

    backend = Compute()
    if temperature is None:
        temperature = np.sqrt(memory_keys.shape[1])
    return backend.memory_read(queries, memory_keys, memory_values, temperature)


def memory_write(
    new_key: np.ndarray,
    new_value: np.ndarray,
    memory_keys: np.ndarray,
    memory_values: np.ndarray,
    write_index: int,
    write_mode: int = 0,
    blend_factor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Write key-value pair to memory.

    Uses: memory-write.glsl

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
    from grilly import Compute

    backend = Compute()
    return backend.memory_write(
        new_key, new_value, memory_keys, memory_values, write_index, write_mode, blend_factor
    )


def memory_context_aggregate(memory_contexts: np.ndarray) -> np.ndarray:
    """
    Aggregate memory contexts.

    Uses: memory-context-aggregate.glsl

    Args:
        memory_contexts: Memory contexts (batch, num_memories, dim)

    Returns:
        Aggregated context (batch, dim)
    """
    backend = _get_backend()

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


def memory_query_pooling(x: np.ndarray, W_query: np.ndarray, b_query: np.ndarray) -> np.ndarray:
    """
    Pool sequence representations into memory queries.

    Uses: memory-query-pooling.glsl

    Args:
        x: Input sequence (batch, seq_len, in_dim)
        W_query: Query projection weights (out_dim, in_dim)
        b_query: Query bias (out_dim,)

    Returns:
        Query vectors (batch, out_dim)
    """
    from grilly import Compute

    backend = Compute()
    return backend.memory_query_pooling(x, W_query, b_query)


def memory_inject_concat(
    x: np.ndarray,
    memory_context: np.ndarray,
    proj_weight: np.ndarray,
    proj_bias: np.ndarray | None = None,
) -> np.ndarray:
    """
    Inject memory context by concatenation.

    Uses: memory-inject-concat.glsl

    Args:
        x: Input (batch, seq_len, dim)
        memory_context: Memory context (batch, dim)
        proj_weight: Projection weights (dim, dim * 2)
        proj_bias: Optional projection bias (dim,)

    Returns:
        Output (batch, seq_len, dim)
    """
    backend = _get_backend()

    # Try GPU shader if available
    if backend and hasattr(backend, "shaders") and "memory-inject-concat" in backend.shaders:
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
    output = concat @ proj_weight.T
    if proj_bias is not None:
        output = output + proj_bias
    return output


def memory_inject_gate(
    x: np.ndarray,
    memory_context: np.ndarray,
    W_gate: np.ndarray,
    b_gate: np.ndarray,
    W_mem_proj: np.ndarray,
) -> np.ndarray:
    """
    Inject memory context with gating.

    Uses: memory-inject-gate.glsl

    Args:
        x: Input (batch, seq_len, dim)
        memory_context: Memory context (batch, dim)
        W_gate: Gate weights (dim, dim * 2)
        b_gate: Gate bias (dim,)
        W_mem_proj: Memory projection weights (dim, dim)

    Returns:
        Output (batch, seq_len, dim)
    """
    from grilly import Compute

    backend = Compute()
    return backend.memory_inject_gate(x, memory_context, W_gate, b_gate, W_mem_proj)


def memory_inject_residual(
    x: np.ndarray,
    memory_context: np.ndarray,
    mem_proj_weight: np.ndarray,
    mem_proj_bias: np.ndarray | None = None,
) -> np.ndarray:
    """
    Inject memory context with residual connection.

    Uses: memory-inject-residual.glsl

    Args:
        x: Input (batch, seq_len, dim)
        memory_context: Memory context (batch, dim)
        mem_proj_weight: Memory projection weights (dim, dim)
        mem_proj_bias: Optional memory projection bias (dim,)

    Returns:
        Output (batch, seq_len, dim)
    """
    backend = _get_backend()

    # Try GPU shader if available
    if backend and hasattr(backend, "shaders") and "memory-inject-residual" in backend.shaders:
        try:
            # GPU memory injection with residual would go here
            # For now, use CPU fallback
            pass
        except Exception:
            pass  # Fall back to CPU

    # CPU fallback
    batch_size, seq_len, dim = x.shape
    mem_proj = memory_context @ mem_proj_weight.T
    if mem_proj_bias is not None:
        mem_proj = mem_proj + mem_proj_bias
    mem_expanded = mem_proj[:, None, :]  # (batch, 1, dim)
    mem_expanded = np.broadcast_to(mem_expanded, (batch_size, seq_len, dim))
    return x + mem_expanded
