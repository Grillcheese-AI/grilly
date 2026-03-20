"""VSA Autograd Operations — differentiable bind/unbind/similarity.

Key insight: backward(unbind) = bind, backward(bind) = unbind.
FFT linearity means the gradient of circular correlation is circular
convolution, and vice versa. No new Vulkan shaders needed — we reuse
blockcode_bind.spv and blockcode_unbind.spv for both forward and backward.

Integrates with grilly's Variable autograd system and GradientTape.

Usage:
    from grilly.nn.vsa_autograd import vsa_unbind, vsa_bind, cosine_loss

    with GradientTape() as tape:
        unbound = vsa_unbind(prediction, role, bc)
        loss = cosine_loss(unbound, target)
    grads = tape.gradient(loss, [prediction])
"""

from __future__ import annotations

import numpy as np

from grilly.nn.autograd import Variable, _make_backward


def _ensure_variable(x):
    """Convert to Variable if not already one."""
    if hasattr(x, 'data') and hasattr(x, 'requires_grad') and hasattr(x, 'backward'):
        return x  # Already a Variable
    if isinstance(x, np.ndarray):
        return Variable(x.astype(np.float32), requires_grad=False)
    return Variable(np.array(x, dtype=np.float32), requires_grad=False)

# ── GPU bridge for block-code ops ───────────────────────────────────────────

_bridge = None
try:
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _bridge = _grilly_bridge
except Exception:
    pass


def _fft_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-block circular convolution (binding) via FFT."""
    fft_a = np.fft.fft(a.astype(np.float64), axis=-1)
    fft_b = np.fft.fft(b.astype(np.float64), axis=-1)
    return np.real(np.fft.ifft(fft_a * fft_b, axis=-1)).astype(np.float32)


def _fft_unbind(composite: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Per-block circular correlation (unbinding) via FFT."""
    fft_c = np.fft.fft(composite.astype(np.float64), axis=-1)
    fft_k = np.fft.fft(key.astype(np.float64), axis=-1)
    return np.real(np.fft.ifft(fft_c * np.conj(fft_k), axis=-1)).astype(np.float32)


def _gpu_bind(a: np.ndarray, b: np.ndarray, k: int, l: int) -> np.ndarray:
    """Bind via GPU bridge if available, else FFT fallback."""
    if _bridge is not None:
        try:
            a_flat = a.reshape(-1, k * l).astype(np.float32)
            b_flat = b.reshape(-1, k * l).astype(np.float32)
            result = _bridge.blockcode_bind(a_flat, b_flat, k, l)
            if result is not None:
                return np.asarray(result, dtype=np.float32).reshape(a.shape)
        except Exception:
            pass
    return _fft_bind(a, b)


def _gpu_unbind(composite: np.ndarray, key: np.ndarray, k: int, l: int) -> np.ndarray:
    """Unbind via GPU bridge if available, else FFT fallback."""
    if _bridge is not None:
        try:
            c_flat = composite.reshape(-1, k * l).astype(np.float32)
            k_flat = key.reshape(-1, k * l).astype(np.float32)
            result = _bridge.blockcode_unbind(c_flat, k_flat, k, l)
            if result is not None:
                return np.asarray(result, dtype=np.float32).reshape(composite.shape)
        except Exception:
            pass
    return _fft_unbind(composite, key)


# ── Differentiable VSA Operations ───────────────────────────────────────────


def vsa_bind(a: Variable, b: Variable, k: int = 8, l: int = 64) -> Variable:
    """Differentiable VSA binding (circular convolution).

    Forward:  c = IFFT(FFT(a) * FFT(b))
    Backward: grad_a = unbind(grad_c, b)  [circular correlation]
              grad_b = unbind(grad_c, a)

    Args:
        a: Variable with data shape (..., k, l) or (..., k*l).
        b: Variable with data shape (..., k, l) or (..., k*l).
        k: Number of blocks.
        l: Block length.

    Returns:
        Bound Variable.
    """
    a = _ensure_variable(a)
    b = _ensure_variable(b)

    a_data = a.data.reshape(-1, k, l) if a.data.shape[-1] == k * l else a.data
    b_data = b.data.reshape(-1, k, l) if b.data.shape[-1] == k * l else b.data

    result = _gpu_bind(a_data, b_data, k, l)

    def backward(grad_output):
        g = grad_output.reshape(a_data.shape)
        grad_a = _gpu_unbind(g, b_data, k, l).reshape(a.data.shape)
        grad_b = _gpu_unbind(g, a_data, k, l).reshape(b.data.shape)
        return grad_a, grad_b

    grad_fn = _make_backward("VSABind", [a, b], backward)
    req_grad = a.requires_grad or b.requires_grad
    return Variable(result.reshape(a.data.shape), requires_grad=req_grad, grad_fn=grad_fn)


def vsa_unbind(composite, key, k: int = 8, l: int = 64):
    """Differentiable VSA unbinding (circular correlation).

    Forward:  a = IFFT(FFT(composite) * conj(FFT(key)))
    Backward: grad_composite = bind(grad_a, key)  [THE MAGIC]

    The backward of unbind IS bind. No new shaders needed.
    """
    composite = _ensure_variable(composite)
    key = _ensure_variable(key)

    # Work on reshaped copies for FFT, keep originals for autograd
    c_np = composite.data.reshape(k, l) if composite.data.ndim <= 2 else composite.data
    k_np = key.data.reshape(k, l) if key.data.ndim <= 2 else key.data

    result = _gpu_unbind(c_np, k_np, k, l)

    def backward(grad_output):
        g = grad_output.reshape(k, l)
        # THE KEY INSIGHT: gradient of unbind = bind!
        grad_composite = _gpu_bind(g, k_np, k, l).reshape(composite.data.shape)
        grad_key = np.zeros_like(key.data)
        return grad_composite, grad_key

    grad_fn = _make_backward("VSAUnbind", [composite, key], backward)
    req_grad = composite.requires_grad or key.requires_grad
    if grad_fn is None and req_grad:
        # Fallback: create GradFn manually if _make_backward returned None
        from grilly.nn.autograd import GradFn
        grad_fn = GradFn("VSAUnbind", backward, [composite, key])
    return Variable(result.reshape(composite.data.shape), requires_grad=req_grad, grad_fn=grad_fn)


def cosine_similarity(a: Variable, b: Variable) -> Variable:
    """Differentiable cosine similarity.

    sim = (a . b) / (||a|| * ||b||)

    Args:
        a, b: Variables with matching shapes.

    Returns:
        Scalar Variable with similarity value.
    """
    a = _ensure_variable(a)
    b = _ensure_variable(b)

    a_flat = a.data.ravel().astype(np.float64)
    b_flat = b.data.ravel().astype(np.float64)

    a_norm = np.linalg.norm(a_flat) + 1e-8
    b_norm = np.linalg.norm(b_flat) + 1e-8

    sim = float(np.dot(a_flat, b_flat) / (a_norm * b_norm))

    def backward(grad_output):
        g = float(grad_output) if np.isscalar(grad_output) else float(grad_output.ravel()[0])
        grad_a = g * ((b_flat / b_norm - sim * a_flat / a_norm) / a_norm).astype(np.float32)
        grad_b = g * ((a_flat / a_norm - sim * b_flat / b_norm) / b_norm).astype(np.float32)
        return grad_a.reshape(a.data.shape), grad_b.reshape(b.data.shape)

    grad_fn = _make_backward("CosineSim", [a, b], backward)
    req_grad = a.requires_grad or b.requires_grad
    return Variable(np.array([sim], dtype=np.float32), requires_grad=req_grad, grad_fn=grad_fn)


def cosine_loss(prediction: Variable, target: Variable) -> Variable:
    """Cosine embedding loss: 1 - cosine_similarity.

    Args:
        prediction: CNN output Variable.
        target: Ground truth Variable (typically frozen codebook vector).

    Returns:
        Scalar loss Variable.
    """
    sim = cosine_similarity(prediction, target)
    return Variable(np.array([1.0], dtype=np.float32)) - sim


def additive_ce_loss_var(
    query: Variable,
    codebook_W: np.ndarray,
    target_indices: list[int],
    temperature: float = 5.0,
) -> Variable:
    """Additive Cross-Entropy as a differentiable Variable op.

    Args:
        query: CNN output Variable (d,).
        codebook_W: Frozen codebook (n, d) — plain numpy, not Variable.
        target_indices: Target codebook indices.
        temperature: Inverse temperature.

    Returns:
        Scalar loss Variable with gradient.
    """
    query = _ensure_variable(query)
    q = query.data.ravel().astype(np.float32)

    # Normalized
    q_norm = np.linalg.norm(q) + 1e-8
    w_norms = np.linalg.norm(codebook_W, axis=1, keepdims=True) + 1e-8
    q_hat = q / q_norm
    w_hat = (codebook_W / w_norms).astype(np.float32)

    # Cosine sims, temperature-scaled
    sims = np.clip(w_hat @ q_hat, -0.999, 0.999)
    logits = np.clip(temperature * sims, -50.0, 50.0)

    # Softmax
    logits_shifted = logits - logits.max()
    exp_l = np.exp(logits_shifted)
    probs = exp_l / (exp_l.sum() + 1e-8)

    # Loss
    target_probs = [max(probs[idx], 1e-8) for idx in target_indices]
    loss = -np.mean([np.log(p) for p in target_probs])

    # Gradient
    expected_w = probs @ w_hat
    target_w = np.mean([w_hat[idx] for idx in target_indices], axis=0)
    grad_q = temperature * (expected_w - target_w) / q_norm
    grad_q = np.clip(grad_q, -1.0, 1.0).astype(np.float32)

    def backward(grad_output):
        g = float(grad_output) if np.isscalar(grad_output) else float(grad_output.ravel()[0])
        return (g * grad_q.reshape(query.data.shape),)

    grad_fn = _make_backward("AdditiveCE", [query], backward)
    return Variable(np.array([loss], dtype=np.float32), requires_grad=True, grad_fn=grad_fn)
