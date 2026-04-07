"""grilly_core.moe_* — fused MoE forward (GPU) vs NumPy reference."""

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


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x, dtype=np.float32)
    return (e / e.sum()).astype(np.float32)


def moe_forward_numpy(
    embed_w: np.ndarray,
    pos_w: np.ndarray,
    expert_ws: list[np.ndarray],
    router_ws: list[np.ndarray],
    router_bs: list[np.ndarray],
    out_w: np.ndarray,
    n_layers: int,
    n_experts: int,
    input_ids: np.ndarray,
) -> np.ndarray:
    """Reference forward (CPU)."""
    s_len = int(input_ids.shape[0])
    d = embed_w.shape[1]
    x = embed_w[input_ids.astype(np.int64)] + pos_w[:s_len]
    for layer in range(n_layers):
        xm = x.mean(axis=0)
        logits = router_ws[layer] @ xm + router_bs[layer]
        p = _softmax(logits.astype(np.float32))
        blend = np.zeros_like(x, dtype=np.float32)
        for e in range(n_experts):
            we = expert_ws[layer * n_experts + e]
            y = x @ we.T
            blend += p[e] * y
        x = x + blend
    return x @ out_w.T


@pytest.mark.gpu
@pytest.mark.cpp
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
def test_moe_forward_parity_small():
    if not _shader_spv_dir().exists():
        pytest.skip("shaders/spv not present")

    rng = np.random.default_rng(7)
    vocab, d, max_seq = 24, 8, 12
    n_layers, n_experts = 2, 4
    s_len = 6

    embed_w = rng.standard_normal((vocab, d), dtype=np.float32)
    pos_w = rng.standard_normal((max_seq, d), dtype=np.float32)
    out_w = rng.standard_normal((vocab, d), dtype=np.float32)

    expert_ws = [rng.standard_normal((d, d), dtype=np.float32) for _ in range(n_layers * n_experts)]
    router_ws = [rng.standard_normal((n_experts, d), dtype=np.float32) for _ in range(n_layers)]
    router_bs = [rng.standard_normal((n_experts,), dtype=np.float32) for _ in range(n_layers)]

    input_ids = rng.integers(0, vocab, size=(s_len,), dtype=np.int32)

    ref = moe_forward_numpy(
        embed_w,
        pos_w,
        expert_ws,
        router_ws,
        router_bs,
        out_w,
        n_layers,
        n_experts,
        input_ids,
    )

    dev = gc.Device()
    try:
        dev.load_shaders(str(_shader_spv_dir()))
    except Exception as e:
        pytest.skip(f"load_shaders failed: {e}")

    h = gc.moe_upload(
        dev,
        embed_w,
        pos_w,
        expert_ws,
        router_ws,
        router_bs,
        out_w,
        n_layers,
        n_experts,
    )
    try:
        got = gc.moe_forward(dev, h, input_ids)
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)

        grad_logits = rng.standard_normal((s_len, vocab), dtype=np.float32)
        grads = gc.moe_backward(dev, h, input_ids, grad_logits)
        assert grads["grad_embed"].shape == (vocab, d)
        assert grads["grad_out_w"].shape == (vocab, d)
        assert len(grads["grad_experts"]) == n_layers * n_experts
        assert grads["grad_pos"].shape == (max_seq, d)
    finally:
        gc.moe_release(dev, h)


@pytest.mark.cpp
def test_moe_backward_cpu_shapes():
    """Backward is CPU-only; smoke-test shapes without Vulkan."""
    rng = np.random.default_rng(11)
    vocab, d, max_seq = 16, 4, 8
    n_layers, n_experts = 1, 2
    s_len = 4

    embed_w = rng.standard_normal((vocab, d), dtype=np.float32)
    pos_w = rng.standard_normal((max_seq, d), dtype=np.float32)
    out_w = rng.standard_normal((vocab, d), dtype=np.float32)
    expert_ws = [rng.standard_normal((d, d), dtype=np.float32) for _ in range(n_layers * n_experts)]
    router_ws = [rng.standard_normal((n_experts, d), dtype=np.float32) for _ in range(n_layers)]
    router_bs = [rng.standard_normal((n_experts,), dtype=np.float32) for _ in range(n_layers)]
    input_ids = rng.integers(0, vocab, size=(s_len,), dtype=np.int32)
    grad_logits = rng.standard_normal((s_len, vocab), dtype=np.float32)

    dev = gc.Device()
    h = gc.moe_upload(
        dev,
        embed_w,
        pos_w,
        expert_ws,
        router_ws,
        router_bs,
        out_w,
        n_layers,
        n_experts,
    )
    try:
        grads = gc.moe_backward(dev, h, input_ids, grad_logits)
        assert grads["grad_embed"].shape == (vocab, d)
        assert grads["grad_out_w"].shape == (vocab, d)
    finally:
        gc.moe_release(dev, h)
