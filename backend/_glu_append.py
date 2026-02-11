"""Core utilities for glu append."""

import struct

import numpy as np

# ==================================================================
# GATED LINEAR UNITS (GLU Variants)
# ==================================================================
# Modern LLM activation patterns: SwiGLU, ReGLU, GeGLU
# Formula: output = (x @ W_gate + b_gate) * Activation(x @ W_up + b_up)


def fused_swiglu(
    self,
    x: np.ndarray,
    W_gate: np.ndarray,
    W_up: np.ndarray,
    b_gate: np.ndarray | None = None,
    b_up: np.ndarray | None = None,
    return_cache: bool = False,
):
    """
    Fused SwiGLU (SiLU-Gated Linear Unit).

    Used in: LLaMA, Mistral, Mixtral FFN layers.

    Formula: output = (x @ W_gate.T + b_gate) * SiLU(x @ W_up.T + b_up)

    Args:
        x: Input tensor (batch_seq, input_dim)
        W_gate: Gate weight matrix (hidden_dim, input_dim)
        W_up: Up projection weight matrix (hidden_dim, input_dim)
        b_gate: Optional gate bias (hidden_dim,)
        b_up: Optional up bias (hidden_dim,)
        return_cache: If True, also return (gate, up) pre-activations for backward

    Returns:
        output: (batch_seq, hidden_dim)
        or (output, gate_cache, up_cache) if return_cache=True
    """
    x = x.astype(np.float32)
    input_dim = x.shape[-1]
    hidden_dim = W_gate.shape[0]

    if x.ndim > 2:
        batch_seq = int(np.prod(x.shape[:-1]))
        x_2d = x.reshape(-1, input_dim)
    else:
        batch_seq = x.shape[0] if x.ndim == 2 else 1
        x_2d = x.reshape(batch_seq, -1)

    # CPU fallback
    if "fused-swiglu" not in self.shaders:
        gate = x_2d @ W_gate.T
        up = x_2d @ W_up.T
        if b_gate is not None:
            gate = gate + b_gate
        if b_up is not None:
            up = up + b_up
        sigmoid_up = 1.0 / (1.0 + np.exp(-up))
        silu_up = up * sigmoid_up
        output = gate * silu_up
        if return_cache:
            return output.astype(np.float32), gate.astype(np.float32), up.astype(np.float32)
        return output.astype(np.float32)

    # GPU implementation
    x_flat = x_2d.flatten()
    w_gate_flat = W_gate.astype(np.float32).flatten()
    w_up_flat = W_up.astype(np.float32).flatten()
    output_size = batch_seq * hidden_dim * 4

    has_bias = 1 if (b_gate is not None and b_up is not None) else 0
    if has_bias:
        b_gate_flat = b_gate.astype(np.float32).flatten()
        b_up_flat = b_up.astype(np.float32).flatten()
    else:
        b_gate_flat = np.zeros(hidden_dim, dtype=np.float32)
        b_up_flat = np.zeros(hidden_dim, dtype=np.float32)

    buf_input = self._acquire_buffer(x_flat.nbytes)
    buf_w_gate = self._acquire_buffer(w_gate_flat.nbytes)
    buf_w_up = self._acquire_buffer(w_up_flat.nbytes)
    buf_b_gate = self._acquire_buffer(b_gate_flat.nbytes)
    buf_b_up = self._acquire_buffer(b_up_flat.nbytes)
    buf_output = self._acquire_buffer(output_size)

    self._upload_buffer(buf_input, x_flat)
    self._upload_buffer(buf_w_gate, w_gate_flat)
    self._upload_buffer(buf_w_up, w_up_flat)
    self._upload_buffer(buf_b_gate, b_gate_flat)
    self._upload_buffer(buf_b_up, b_up_flat)

    pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
        "fused-swiglu", 6, push_constant_size=16
    )

    descriptor_set = self.pipelines.get_cached_descriptor_set(
        "fused-swiglu",
        [
            (self._get_buffer_handle(buf_input), x_flat.nbytes),
            (self._get_buffer_handle(buf_w_gate), w_gate_flat.nbytes),
            (self._get_buffer_handle(buf_w_up), w_up_flat.nbytes),
            (self._get_buffer_handle(buf_b_gate), b_gate_flat.nbytes),
            (self._get_buffer_handle(buf_b_up), b_up_flat.nbytes),
            (self._get_buffer_handle(buf_output), output_size),
        ],
    )

    push_constants = struct.pack("IIII", batch_seq, input_dim, hidden_dim, has_bias)

    workgroups_x = (hidden_dim + 15) // 16
    workgroups_y = (batch_seq + 15) // 16

    self.core._dispatch_compute(
        pipeline, pipeline_layout, descriptor_set, workgroups_x, push_constants, workgroups_y
    )

    result = self._download_buffer(buf_output, output_size, np.float32)
    self._release_buffers([buf_input, buf_w_gate, buf_w_up, buf_b_gate, buf_b_up, buf_output])

    output = result[: batch_seq * hidden_dim].reshape(batch_seq, hidden_dim)

    if return_cache:
        gate = x_2d @ W_gate.T
        up = x_2d @ W_up.T
        if b_gate is not None:
            gate = gate + b_gate
        if b_up is not None:
            up = up + b_up
        return output, gate.astype(np.float32), up.astype(np.float32)

    return output


def fused_reglu(
    self,
    x: np.ndarray,
    W_gate: np.ndarray,
    W_up: np.ndarray,
    b_gate: np.ndarray | None = None,
    b_up: np.ndarray | None = None,
    return_cache: bool = False,
):
    """
    Fused ReGLU (ReLU-Gated Linear Unit).

    10-20x faster than GeGLU, ideal for edge deployment.

    Formula: output = (x @ W_gate.T + b_gate) * ReLU(x @ W_up.T + b_up)
    """
    x = x.astype(np.float32)
    input_dim = x.shape[-1]
    hidden_dim = W_gate.shape[0]

    if x.ndim > 2:
        batch_seq = int(np.prod(x.shape[:-1]))
        x_2d = x.reshape(-1, input_dim)
    else:
        batch_seq = x.shape[0] if x.ndim == 2 else 1
        x_2d = x.reshape(batch_seq, -1)

    if "fused-reglu" not in self.shaders:
        gate = x_2d @ W_gate.T
        up = x_2d @ W_up.T
        if b_gate is not None:
            gate = gate + b_gate
        if b_up is not None:
            up = up + b_up
        output = gate * np.maximum(0, up)
        if return_cache:
            return output.astype(np.float32), gate.astype(np.float32), up.astype(np.float32)
        return output.astype(np.float32)

    x_flat = x_2d.flatten()
    w_gate_flat = W_gate.astype(np.float32).flatten()
    w_up_flat = W_up.astype(np.float32).flatten()
    output_size = batch_seq * hidden_dim * 4

    has_bias = 1 if (b_gate is not None and b_up is not None) else 0
    b_gate_flat = (
        b_gate.astype(np.float32).flatten() if has_bias else np.zeros(hidden_dim, dtype=np.float32)
    )
    b_up_flat = (
        b_up.astype(np.float32).flatten() if has_bias else np.zeros(hidden_dim, dtype=np.float32)
    )

    buf_input = self._acquire_buffer(x_flat.nbytes)
    buf_w_gate = self._acquire_buffer(w_gate_flat.nbytes)
    buf_w_up = self._acquire_buffer(w_up_flat.nbytes)
    buf_b_gate = self._acquire_buffer(b_gate_flat.nbytes)
    buf_b_up = self._acquire_buffer(b_up_flat.nbytes)
    buf_output = self._acquire_buffer(output_size)

    self._upload_buffer(buf_input, x_flat)
    self._upload_buffer(buf_w_gate, w_gate_flat)
    self._upload_buffer(buf_w_up, w_up_flat)
    self._upload_buffer(buf_b_gate, b_gate_flat)
    self._upload_buffer(buf_b_up, b_up_flat)

    pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
        "fused-reglu", 6, push_constant_size=16
    )

    descriptor_set = self.pipelines.get_cached_descriptor_set(
        "fused-reglu",
        [
            (self._get_buffer_handle(buf_input), x_flat.nbytes),
            (self._get_buffer_handle(buf_w_gate), w_gate_flat.nbytes),
            (self._get_buffer_handle(buf_w_up), w_up_flat.nbytes),
            (self._get_buffer_handle(buf_b_gate), b_gate_flat.nbytes),
            (self._get_buffer_handle(buf_b_up), b_up_flat.nbytes),
            (self._get_buffer_handle(buf_output), output_size),
        ],
    )

    push_constants = struct.pack("IIII", batch_seq, input_dim, hidden_dim, has_bias)
    workgroups_x = (hidden_dim + 15) // 16
    workgroups_y = (batch_seq + 15) // 16

    self.core._dispatch_compute(
        pipeline, pipeline_layout, descriptor_set, workgroups_x, push_constants, workgroups_y
    )

    result = self._download_buffer(buf_output, output_size, np.float32)
    self._release_buffers([buf_input, buf_w_gate, buf_w_up, buf_b_gate, buf_b_up, buf_output])

    output = result[: batch_seq * hidden_dim].reshape(batch_seq, hidden_dim)
    if return_cache:
        gate = x_2d @ W_gate.T + (b_gate if b_gate is not None else 0)
        up = x_2d @ W_up.T + (b_up if b_up is not None else 0)
        return output, gate.astype(np.float32), up.astype(np.float32)
    return output


def fused_geglu(
    self,
    x: np.ndarray,
    W_gate: np.ndarray,
    W_up: np.ndarray,
    b_gate: np.ndarray | None = None,
    b_up: np.ndarray | None = None,
    return_cache: bool = False,
):
    """
    Fused GeGLU (GELU-Gated Linear Unit).

    Used in: T5, PaLM, some transformer variants.

    Formula: output = (x @ W_gate.T + b_gate) * GELU(x @ W_up.T + b_up)
    """

    def gelu(x):
        """Run gelu."""

        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        coeff = 0.044715
        return 0.5 * x * (1 + np.tanh(sqrt_2_over_pi * (x + coeff * x**3)))

    x = x.astype(np.float32)
    input_dim = x.shape[-1]
    hidden_dim = W_gate.shape[0]

    if x.ndim > 2:
        batch_seq = int(np.prod(x.shape[:-1]))
        x_2d = x.reshape(-1, input_dim)
    else:
        batch_seq = x.shape[0] if x.ndim == 2 else 1
        x_2d = x.reshape(batch_seq, -1)

    if "fused-geglu" not in self.shaders:
        gate = x_2d @ W_gate.T
        up = x_2d @ W_up.T
        if b_gate is not None:
            gate = gate + b_gate
        if b_up is not None:
            up = up + b_up
        output = gate * gelu(up)
        if return_cache:
            return output.astype(np.float32), gate.astype(np.float32), up.astype(np.float32)
        return output.astype(np.float32)

    x_flat = x_2d.flatten()
    w_gate_flat = W_gate.astype(np.float32).flatten()
    w_up_flat = W_up.astype(np.float32).flatten()
    output_size = batch_seq * hidden_dim * 4

    has_bias = 1 if (b_gate is not None and b_up is not None) else 0
    b_gate_flat = (
        b_gate.astype(np.float32).flatten() if has_bias else np.zeros(hidden_dim, dtype=np.float32)
    )
    b_up_flat = (
        b_up.astype(np.float32).flatten() if has_bias else np.zeros(hidden_dim, dtype=np.float32)
    )

    buf_input = self._acquire_buffer(x_flat.nbytes)
    buf_w_gate = self._acquire_buffer(w_gate_flat.nbytes)
    buf_w_up = self._acquire_buffer(w_up_flat.nbytes)
    buf_b_gate = self._acquire_buffer(b_gate_flat.nbytes)
    buf_b_up = self._acquire_buffer(b_up_flat.nbytes)
    buf_output = self._acquire_buffer(output_size)

    self._upload_buffer(buf_input, x_flat)
    self._upload_buffer(buf_w_gate, w_gate_flat)
    self._upload_buffer(buf_w_up, w_up_flat)
    self._upload_buffer(buf_b_gate, b_gate_flat)
    self._upload_buffer(buf_b_up, b_up_flat)

    pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
        "fused-geglu", 6, push_constant_size=16
    )

    descriptor_set = self.pipelines.get_cached_descriptor_set(
        "fused-geglu",
        [
            (self._get_buffer_handle(buf_input), x_flat.nbytes),
            (self._get_buffer_handle(buf_w_gate), w_gate_flat.nbytes),
            (self._get_buffer_handle(buf_w_up), w_up_flat.nbytes),
            (self._get_buffer_handle(buf_b_gate), b_gate_flat.nbytes),
            (self._get_buffer_handle(buf_b_up), b_up_flat.nbytes),
            (self._get_buffer_handle(buf_output), output_size),
        ],
    )

    push_constants = struct.pack("IIII", batch_seq, input_dim, hidden_dim, has_bias)
    workgroups_x = (hidden_dim + 15) // 16
    workgroups_y = (batch_seq + 15) // 16

    self.core._dispatch_compute(
        pipeline, pipeline_layout, descriptor_set, workgroups_x, push_constants, workgroups_y
    )

    result = self._download_buffer(buf_output, output_size, np.float32)
    self._release_buffers([buf_input, buf_w_gate, buf_w_up, buf_b_gate, buf_b_up, buf_output])

    output = result[: batch_seq * hidden_dim].reshape(batch_seq, hidden_dim)
    if return_cache:
        gate = x_2d @ W_gate.T + (b_gate if b_gate is not None else 0)
        up = x_2d @ W_up.T + (b_up if b_up is not None else 0)
        return output, gate.astype(np.float32), up.astype(np.float32)
    return output
