"""Parity + smoke test for the multiply-free ternary GEMM (ternary-gemm.glsl),
through the non-deprecated grilly_core C++ path.

  output = alpha * (activations @ trit^T),  trit in {-1,0,+1}
  weights 2-bit packed (16 trits/uint32), one per-tensor alpha = mean|W|.

Oracle: quantize W to {-alpha,0,+alpha} in numpy, do a plain fp32 matmul.
The GPU kernel must match EXACTLY (it computes the same sum, just multiply-
free) up to fp32 accumulation order — hence a modest rtol.
"""
import os

import numpy as np
import pytest


def _oracle(act, w):
    """alpha*(act @ trit^T) via numpy — the exact quantity the kernel computes."""
    alpha = float(np.abs(w).mean()) + 1e-5
    trit = np.clip(np.round(w / alpha), -1.0, 1.0).astype(np.float32)
    return (act @ trit.T) * alpha, alpha, trit


def _pack(w):
    from grilly.backend._bridge import pack_ternary
    return pack_ternary(w)


@pytest.fixture(scope="session")
def device():
    try:
        import grilly_core
    except ImportError as e:
        pytest.skip(f"grilly_core unavailable: {e}")
    if not hasattr(grilly_core, "ternary_gemm"):
        pytest.skip("grilly_core.ternary_gemm not present (rebuild grilly_core)")
    dev = grilly_core.Device()
    shader_dir = os.path.join(os.getcwd(), "shaders", "spv")
    try:
        dev.load_shaders(shader_dir)
    except Exception as e:
        pytest.skip(f"device/shader init failed: {e}")
    return dev


def _gpu(device, act, packed, K, alpha):
    import grilly_core
    return np.asarray(grilly_core.ternary_gemm(
        device, np.ascontiguousarray(act, np.float32),
        np.ascontiguousarray(packed, np.uint32), int(K), float(alpha)))


@pytest.mark.parametrize("shape", [(4, 32, 8), (16, 256, 64),
                                   (64, 512, 256), (8, 48, 32)])
def test_ternary_parity(device, shape):
    """(M, K, N): kernel == numpy quantized matmul. K=48 exercises the
    non-16-multiple tail (words_per_row guard)."""
    M, K, N = shape
    rng = np.random.default_rng(0)
    act = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    w = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)

    ref, alpha, _ = _oracle(act, w)
    packed, alpha_pk = _pack(w)
    assert abs(alpha - alpha_pk) < 1e-6            # packer alpha == oracle alpha
    out = _gpu(device, act, packed, K, alpha_pk)

    assert out.shape == (M, N)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_ternary_zero_weights(device):
    """All-zero weights -> all trits 0 -> output identically 0 (pure skip path)."""
    M, K, N = 8, 64, 16
    act = np.random.default_rng(1).standard_normal((M, K)).astype(np.float32)
    w = np.zeros((N, K), dtype=np.float32)
    packed, alpha = _pack(w)
    out = _gpu(device, act, packed, K, alpha)
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_ternary_sign_correctness(device):
    """A hand-built row: +1,-1,0,... must give alpha*(a0 - a1). Pins the
    conditional-negate sign convention (add pos, subtract neg)."""
    from grilly.backend._bridge import pack_ternary
    M, K, N = 1, 16, 1
    act = np.arange(1, K + 1, dtype=np.float32)[None, :]   # [1,2,...,16]
    w = np.zeros((N, K), dtype=np.float32)
    w[0, 0] = 5.0      # -> +1
    w[0, 1] = -5.0     # -> -1
    # rest 0
    packed, alpha = pack_ternary(w)
    out = _gpu(device, act, packed, K, alpha)
    # trit row = [+1,-1,0...]; sum = a0 - a1 = 1 - 2 = -1; * alpha
    expected = alpha * (act[0, 0] - act[0, 1])
    np.testing.assert_allclose(out[0, 0], expected, rtol=1e-4, atol=1e-4)
