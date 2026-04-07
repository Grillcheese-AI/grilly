"""grilly_core.vsa_lm_* — fused VSA-LM forward (GPU) + backward (CPU) tests."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

try:
    import grilly_core as gc
except ImportError:
    pytest.skip("grilly_core not available", allow_module_level=True)

try:
    from grilly.backend.base import VULKAN_AVAILABLE
except ImportError:
    VULKAN_AVAILABLE = False


def _shader_spv_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "shaders" / "spv"


def _addition_linear_numpy(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """y[s, o] = -sum_k |w[o, k] - x[s, k]| + b[o]"""
    out = np.empty((x.shape[0], w.shape[0]), dtype=np.float32)
    for s in range(x.shape[0]):
        for o in range(w.shape[0]):
            out[s, o] = -np.sum(np.abs(w[o] - x[s])) + b[o]
    return out


def _sign_activation(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1.0, -1.0).astype(np.float32)


def _layernorm_numpy(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                     eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps) * gamma + beta).astype(np.float32)


def vsa_lm_forward_numpy(
    embed_w, pos_w, ffn_up_ws, ffn_up_bs, ffn_down_ws, ffn_down_bs,
    ln_gammas, ln_betas, out_w, n_layers, input_ids,
):
    """NumPy reference forward."""
    s_len = input_ids.shape[0]
    d = embed_w.shape[1]

    x = embed_w[input_ids.astype(np.int64)] + pos_w[:s_len]

    for l in range(n_layers):
        h = _layernorm_numpy(x, ln_gammas[l], ln_betas[l])
        h_up = _addition_linear_numpy(h, ffn_up_ws[l], ffn_up_bs[l])
        h_sign = _sign_activation(h_up)
        h_ffn = _addition_linear_numpy(h_sign, ffn_down_ws[l], ffn_down_bs[l])
        x = x + h_ffn

    scale = 1.0 / np.sqrt(d).astype(np.float32)
    logits = (x @ out_w.T) * scale
    return logits.astype(np.float32)


def _make_weights(rng, vocab, d, d_ffn, max_seq, n_layers):
    embed_w = rng.standard_normal((vocab, d)).astype(np.float32) * 0.02
    pos_w = rng.standard_normal((max_seq, d)).astype(np.float32) * 0.02
    out_w = rng.standard_normal((vocab, d)).astype(np.float32) * 0.02

    ffn_up_ws = [rng.standard_normal((d_ffn, d)).astype(np.float32) * 0.02
                 for _ in range(n_layers)]
    ffn_up_bs = [np.zeros(d_ffn, dtype=np.float32) for _ in range(n_layers)]
    ffn_down_ws = [rng.standard_normal((d, d_ffn)).astype(np.float32) * 0.02
                   for _ in range(n_layers)]
    ffn_down_bs = [np.zeros(d, dtype=np.float32) for _ in range(n_layers)]
    ln_gammas = [np.ones(d, dtype=np.float32) for _ in range(n_layers)]
    ln_betas = [np.zeros(d, dtype=np.float32) for _ in range(n_layers)]

    return (embed_w, pos_w, out_w, ffn_up_ws, ffn_up_bs,
            ffn_down_ws, ffn_down_bs, ln_gammas, ln_betas)


@pytest.mark.gpu
@pytest.mark.cpp
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
def test_vsa_lm_forward_parity_small():
    """Forward parity: GPU vs NumPy at small scale."""
    if not _shader_spv_dir().exists():
        pytest.skip("shaders/spv not present")

    rng = np.random.default_rng(42)
    vocab, d, d_ffn, max_seq = 24, 8, 16, 12
    n_layers = 2
    s_len = 6
    input_ids = rng.integers(0, vocab, size=(s_len,), dtype=np.int32)

    (embed_w, pos_w, out_w, ffn_up_ws, ffn_up_bs,
     ffn_down_ws, ffn_down_bs, ln_gammas, ln_betas) = _make_weights(
        rng, vocab, d, d_ffn, max_seq, n_layers)

    ref = vsa_lm_forward_numpy(
        embed_w, pos_w, ffn_up_ws, ffn_up_bs, ffn_down_ws, ffn_down_bs,
        ln_gammas, ln_betas, out_w, n_layers, input_ids)

    dev = gc.Device()
    try:
        dev.load_shaders(str(_shader_spv_dir()))
    except Exception as e:
        pytest.skip(f"load_shaders failed: {e}")

    h = gc.vsa_lm_upload(
        dev, embed_w, pos_w,
        ffn_up_ws, ffn_up_bs, ffn_down_ws, ffn_down_bs,
        ln_gammas, ln_betas, out_w,
        n_layers, d, d_ffn)
    try:
        got = gc.vsa_lm_forward(dev, h, input_ids)
        assert got.shape == (s_len, vocab), f"shape: {got.shape}"
        np.testing.assert_allclose(got, ref, rtol=1e-2, atol=1e-2)
    finally:
        gc.vsa_lm_release(dev, h)


@pytest.mark.gpu
@pytest.mark.cpp
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
def test_vsa_lm_forward_parity_medium():
    """Forward parity at target-ish dimensions (d=64, 4 layers)."""
    if not _shader_spv_dir().exists():
        pytest.skip("shaders/spv not present")

    rng = np.random.default_rng(99)
    vocab, d, d_ffn, max_seq = 100, 64, 128, 32
    n_layers = 4
    s_len = 16
    input_ids = rng.integers(0, vocab, size=(s_len,), dtype=np.int32)

    (embed_w, pos_w, out_w, ffn_up_ws, ffn_up_bs,
     ffn_down_ws, ffn_down_bs, ln_gammas, ln_betas) = _make_weights(
        rng, vocab, d, d_ffn, max_seq, n_layers)

    ref = vsa_lm_forward_numpy(
        embed_w, pos_w, ffn_up_ws, ffn_up_bs, ffn_down_ws, ffn_down_bs,
        ln_gammas, ln_betas, out_w, n_layers, input_ids)

    dev = gc.Device()
    try:
        dev.load_shaders(str(_shader_spv_dir()))
    except Exception as e:
        pytest.skip(f"load_shaders failed: {e}")

    h = gc.vsa_lm_upload(
        dev, embed_w, pos_w,
        ffn_up_ws, ffn_up_bs, ffn_down_ws, ffn_down_bs,
        ln_gammas, ln_betas, out_w,
        n_layers, d, d_ffn)
    try:
        got = gc.vsa_lm_forward(dev, h, input_ids)
        assert got.shape == (s_len, vocab)
        np.testing.assert_allclose(got, ref, rtol=5e-2, atol=5e-2)
    finally:
        gc.vsa_lm_release(dev, h)


@pytest.mark.cpp
def test_vsa_lm_backward_cpu_shapes():
    """Backward is CPU-only — smoke-test shapes without Vulkan."""
    rng = np.random.default_rng(7)
    vocab, d, d_ffn, max_seq = 16, 4, 8, 8
    n_layers = 1
    s_len = 4
    input_ids = rng.integers(0, vocab, size=(s_len,), dtype=np.int32)
    grad_logits = rng.standard_normal((s_len, vocab)).astype(np.float32)

    (embed_w, pos_w, out_w, ffn_up_ws, ffn_up_bs,
     ffn_down_ws, ffn_down_bs, ln_gammas, ln_betas) = _make_weights(
        rng, vocab, d, d_ffn, max_seq, n_layers)

    dev = gc.Device()
    h = gc.vsa_lm_upload(
        dev, embed_w, pos_w,
        ffn_up_ws, ffn_up_bs, ffn_down_ws, ffn_down_bs,
        ln_gammas, ln_betas, out_w,
        n_layers, d, d_ffn)
    try:
        grads = gc.vsa_lm_backward(dev, h, input_ids, grad_logits)
        assert grads["grad_embed"].shape == (vocab, d)
        assert grads["grad_pos"].shape == (max_seq, d)
        assert grads["grad_out_w"].shape == (vocab, d)
        assert len(grads["grad_ffn_up_w"]) == n_layers
        assert grads["grad_ffn_up_w"][0].shape == (d_ffn, d)
        assert grads["grad_ffn_up_b"][0].shape == (d_ffn,)
        assert grads["grad_ffn_down_w"][0].shape == (d, d_ffn)
        assert grads["grad_ffn_down_b"][0].shape == (d,)
        assert grads["grad_ln_gamma"][0].shape == (d,)
        assert grads["grad_ln_beta"][0].shape == (d,)

        for key in ["grad_embed", "grad_pos", "grad_out_w"]:
            assert np.isfinite(grads[key]).all(), f"{key} has non-finite values"
    finally:
        gc.vsa_lm_release(dev, h)


@pytest.mark.cpp
def test_vsa_lm_update_weights():
    """Verify update_weights re-uploads without crash."""
    rng = np.random.default_rng(13)
    vocab, d, d_ffn, max_seq = 16, 4, 8, 8
    n_layers = 1

    (embed_w, pos_w, out_w, ffn_up_ws, ffn_up_bs,
     ffn_down_ws, ffn_down_bs, ln_gammas, ln_betas) = _make_weights(
        rng, vocab, d, d_ffn, max_seq, n_layers)

    dev = gc.Device()
    h = gc.vsa_lm_upload(
        dev, embed_w, pos_w,
        ffn_up_ws, ffn_up_bs, ffn_down_ws, ffn_down_bs,
        ln_gammas, ln_betas, out_w,
        n_layers, d, d_ffn)
    try:
        # Perturb all weights slightly and re-upload
        embed_w2 = embed_w + 0.001
        pos_w2 = pos_w + 0.001
        out_w2 = out_w + 0.001
        ffn_up_ws2 = [w + 0.001 for w in ffn_up_ws]
        ffn_up_bs2 = [b + 0.001 for b in ffn_up_bs]
        ffn_down_ws2 = [w + 0.001 for w in ffn_down_ws]
        ffn_down_bs2 = [b + 0.001 for b in ffn_down_bs]
        ln_gammas2 = [g + 0.001 for g in ln_gammas]
        ln_betas2 = [b + 0.001 for b in ln_betas]

        gc.vsa_lm_update_weights(
            dev, h, embed_w2, pos_w2,
            ffn_up_ws2, ffn_up_bs2, ffn_down_ws2, ffn_down_bs2,
            ln_gammas2, ln_betas2, out_w2)
    finally:
        gc.vsa_lm_release(dev, h)


@pytest.mark.gpu
@pytest.mark.cpp
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
def test_vsa_lm_large_scale_no_crash():
    """Ensure no segfault at production-ish shape (d=256, 6 layers)."""
    if not _shader_spv_dir().exists():
        pytest.skip("shaders/spv not present")

    rng = np.random.default_rng(55)
    vocab, d, d_ffn, max_seq = 1000, 256, 512, 512
    n_layers = 6
    s_len = 64
    input_ids = rng.integers(0, vocab, size=(s_len,), dtype=np.int32)

    (embed_w, pos_w, out_w, ffn_up_ws, ffn_up_bs,
     ffn_down_ws, ffn_down_bs, ln_gammas, ln_betas) = _make_weights(
        rng, vocab, d, d_ffn, max_seq, n_layers)

    dev = gc.Device()
    try:
        dev.load_shaders(str(_shader_spv_dir()))
    except Exception as e:
        pytest.skip(f"load_shaders failed: {e}")

    h = gc.vsa_lm_upload(
        dev, embed_w, pos_w,
        ffn_up_ws, ffn_up_bs, ffn_down_ws, ffn_down_bs,
        ln_gammas, ln_betas, out_w,
        n_layers, d, d_ffn)
    try:
        logits = gc.vsa_lm_forward(dev, h, input_ids)
        assert logits.shape == (s_len, vocab)
        assert np.isfinite(logits).all(), "logits contain non-finite values"

        grad_logits = rng.standard_normal((s_len, vocab)).astype(np.float32)
        grads = gc.vsa_lm_backward(dev, h, input_ids, grad_logits)
        assert grads["grad_embed"].shape == (vocab, d)
        assert np.isfinite(grads["grad_embed"]).all()
    finally:
        gc.vsa_lm_release(dev, h)
