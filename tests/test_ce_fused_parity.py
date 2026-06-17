"""
Parity tests for the FUSED cross-entropy loss+grad op (loss-ce-fused.glsl),
dispatched through the non-deprecated grilly_core C++ backend.

One GPU dispatch computes BOTH the per-row CE loss AND grad_logits, sharing a
single subgroup-reduced max + sum_exp pass per row (workgroup-per-row). This is
the T5 "Fused CE + softmax chain op": it replaces the separate
loss-cross-entropy (3-pass) + cross-entropy-backward (tree-reduction) dispatches
with one subgroup-reduced kernel.

Validation goes through grilly_core.cross_entropy_fused(device, logits, targets)
— the same C++ runtime/device the rest of the stack uses — NOT the deprecated
Python VulkanCore path. The bound entrypoint returns a (loss, grad) tuple of
grilly_core.Tensor; we convert via .numpy().

Oracle (must match functional/loss.py + cross-entropy-backward.glsl CPU paths):
    shifted   = clip(logits - max, -60, 60)
    sum_exp   = sum(exp(shifted))
    lse       = max + log(sum_exp)
    loss[row] = lse - logits[row, target]
    grad[row] = softmax(logits[row]) - one_hot(target)
"""

import os

import numpy as np
import pytest


def _ce_oracle(logits: np.ndarray, targets: np.ndarray):
    """Reference (loss_per_row, grad_logits) — mirrors the grilly CPU fallbacks."""
    logits = logits.astype(np.float32)
    row_max = np.max(logits, axis=1, keepdims=True)
    shifted = np.clip(logits - row_max, -60.0, 60.0)
    exp = np.exp(shifted)
    sum_exp = np.sum(exp, axis=1, keepdims=True)
    softmax = exp / np.maximum(sum_exp, 1e-12)

    lse = row_max.reshape(-1) + np.log(np.maximum(sum_exp.reshape(-1), 1e-12))
    tgt = targets.astype(np.int64)
    target_logit = logits[np.arange(logits.shape[0]), tgt]
    loss = (lse - target_logit).astype(np.float32)

    one_hot = np.zeros_like(logits)
    one_hot[np.arange(logits.shape[0]), tgt] = 1.0
    grad = (softmax - one_hot).astype(np.float32)
    return loss, grad


@pytest.fixture(scope="session")
def device():
    """A real grilly_core Vulkan device with shaders/spv loaded.

    Mirrors tests/test_losses_gpu.py: construct grilly_core.Device() and load
    the compiled SPIR-V from shaders/spv. This is the non-deprecated C++ path.
    """
    try:
        import grilly_core
    except ImportError as e:
        pytest.skip(f"grilly_core unavailable: {e}")

    if not hasattr(grilly_core, "cross_entropy_fused"):
        pytest.skip("grilly_core.cross_entropy_fused not present (rebuild grilly_core)")

    dev = grilly_core.Device()
    shader_dir = os.path.join(os.getcwd(), "shaders", "spv")
    try:
        dev.load_shaders(shader_dir)
    except Exception as e:
        pytest.skip(f"Could not init Vulkan device / load shaders: {e}")
    return dev


def _fused(device, logits, targets):
    """Call grilly_core.cross_entropy_fused and return numpy (loss, grad)."""
    import grilly_core

    logits = np.ascontiguousarray(logits, dtype=np.float32)
    targets = np.ascontiguousarray(targets, dtype=np.uint32)
    loss_t, grad_t = grilly_core.cross_entropy_fused(device, logits, targets)
    return np.asarray(loss_t.numpy()), np.asarray(grad_t.numpy())


@pytest.mark.parametrize("shape", [(32, 10), (8, 128), (1, 64)])
def test_ce_fused_parity(device, shape):
    rng = np.random.default_rng(42)
    batch, vocab = shape
    logits = rng.standard_normal((batch, vocab)).astype(np.float32)
    targets = rng.integers(0, vocab, size=(batch,)).astype(np.uint32)

    loss_ref, grad_ref = _ce_oracle(logits, targets)
    loss_gpu, grad_gpu = _fused(device, logits, targets)

    assert loss_gpu.shape == (batch,)
    assert grad_gpu.shape == (batch, vocab)
    np.testing.assert_allclose(loss_gpu, loss_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(grad_gpu, grad_ref, rtol=1e-4, atol=1e-4)


def test_ce_fused_grad_sums_to_zero(device):
    """Each grad row is softmax - one_hot, so it must sum to ~0."""
    rng = np.random.default_rng(7)
    logits = rng.standard_normal((16, 50)).astype(np.float32)
    targets = rng.integers(0, 50, size=(16,)).astype(np.uint32)

    _, grad_gpu = _fused(device, logits, targets)
    row_sums = np.sum(grad_gpu, axis=1)
    np.testing.assert_allclose(row_sums, np.zeros(16), atol=1e-4)


def test_ce_fused_loss_matches_separate_loss_op(device):
    """Fused per-row loss must match the standalone cross_entropy_loss C++ op.

    This pins the fused loss to the already-trusted loss-cross-entropy.spv path
    (which reads targets as uint32, same as the fused kernel) — a GPU-vs-GPU
    regression guard, not just numpy.

    NOTE: we deliberately do NOT cross-check grad against
    grilly_core.cross_entropy_backward here. That op's shader
    (cross-entropy-backward.glsl) declares its Targets buffer as `float` and
    does `uint(targets[idx])`, while the C++ uploads raw uint32 bytes — so it
    reads every target as ~0 (bit pattern of a small int reinterpreted as
    float). The fused kernel intentionally reads targets as uint32 (matching
    loss-cross-entropy.glsl), so its grad is correct and would not match the
    buggy backward op. Grad correctness is covered by test_ce_fused_parity
    against the numpy oracle.
    """
    import grilly_core

    rng = np.random.default_rng(123)
    logits = rng.standard_normal((24, 37)).astype(np.float32)
    targets = rng.integers(0, 37, size=(24,)).astype(np.uint32)

    loss_fused, _ = _fused(device, logits, targets)

    loss_sep = np.asarray(
        grilly_core.cross_entropy_loss(device, logits, targets).numpy()
    ).reshape(-1)

    np.testing.assert_allclose(loss_fused, loss_sep, rtol=1e-4, atol=1e-4)
