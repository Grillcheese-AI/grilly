"""
Parity tests for the FUSED sampled-BCE head op (loss-sampled-bce-fused.glsl),
dispatched through the non-deprecated grilly_core C++ backend.

The softmax-free vocab head: each token scores ONLY its true target + K
sampled negatives with a sigmoid binary CE (NCE/SGNS). One submit computes
per-token loss, grad_hidden, and grad_table (atomic float scatter into the
tied (V, d) table). No softmax, no (N, V) logits, anywhere.

Validation goes through grilly_core.sampled_bce_fused(device, hidden, table,
ids) — the same C++ runtime/device the rest of the stack uses. Returns a
(losses, grad_hidden, grad_table) tuple of grilly_core.Tensor.

Oracle (must match the numpy fallback in cubby's step1_sampled_bce_head.py):
    s_pos      = dot(h, W[t])
    s_neg[j]   = dot(h, W[neg_j])
    loss       = softplus(-s_pos) + sum_j softplus(s_neg[j])   [collisions 0]
    dL/ds_pos  = -sigmoid(-s_pos) / N
    dL/ds_neg  =  sigmoid(s_neg) / N                            [collisions 0]
    dH[n]      = sum_j dL/ds[n,j] * W[ids[n,j]]
    dW[id]    += sum over (n,j) hitting id of dL/ds[n,j] * H[n]
"""

import os

import numpy as np
import pytest


def _softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _bce_oracle(H, W, ids):
    """Reference (loss_per_token, dH, dW) — mirrors the cubby numpy fallback."""
    H = H.astype(np.float64)
    W = W.astype(np.float64)
    N, d = H.shape
    C = ids.shape[1]
    t = ids[:, 0].astype(np.int64)
    keep = np.ones((N, C), dtype=bool)
    keep[:, 1:] = ids[:, 1:] != t[:, None]      # mask target collisions

    E = W[ids.astype(np.int64)]                 # (N, C, d)
    s = np.einsum("nd,ncd->nc", H, E)           # (N, C)

    loss = _softplus(-s[:, 0]) + (_softplus(s[:, 1:]) * keep[:, 1:]).sum(1)

    ds = np.empty_like(s)
    ds[:, 0] = -_sigmoid(-s[:, 0])
    ds[:, 1:] = _sigmoid(s[:, 1:]) * keep[:, 1:]
    ds /= N                                     # mean baked in (invN)

    dH = np.einsum("nc,ncd->nd", ds, E)
    dW = np.zeros_like(W)
    np.add.at(dW, ids.reshape(-1).astype(np.int64),
              (ds[:, :, None] * H[:, None, :]).reshape(-1, d))
    return (loss.astype(np.float32), dH.astype(np.float32),
            dW.astype(np.float32))


@pytest.fixture(scope="session")
def device():
    """A real grilly_core Vulkan device with shaders/spv loaded."""
    try:
        import grilly_core
    except ImportError as e:
        pytest.skip(f"grilly_core unavailable: {e}")

    if not hasattr(grilly_core, "sampled_bce_fused"):
        pytest.skip("grilly_core.sampled_bce_fused not present (rebuild grilly_core)")

    dev = grilly_core.Device()
    shader_dir = os.path.join(os.getcwd(), "shaders", "spv")
    try:
        dev.load_shaders(shader_dir)
    except Exception as e:
        pytest.skip(f"Could not init Vulkan device / load shaders: {e}")
    return dev


def _fused(device, H, W, ids):
    import grilly_core

    H = np.ascontiguousarray(H, dtype=np.float32)
    W = np.ascontiguousarray(W, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.uint32)
    l_t, gh_t, gw_t = grilly_core.sampled_bce_fused(device, H, W, ids)
    return (np.asarray(l_t.numpy()), np.asarray(gh_t.numpy()),
            np.asarray(gw_t.numpy()))


@pytest.mark.parametrize("shape", [(16, 8, 32, 4), (64, 32, 512, 16),
                                   (512, 256, 2048, 64)])
def test_sampled_bce_parity(device, shape):
    """(N, d, V, K): loss/dH/dW vs numpy oracle. dW uses a looser atol —
    float atomicAdd accumulation order is nondeterministic on GPU."""
    N, d, V, K = shape
    rng = np.random.default_rng(42)
    H = (rng.standard_normal((N, d)) * 0.5).astype(np.float32)
    W = (rng.standard_normal((V, d)) * 0.1).astype(np.float32)
    t = rng.integers(0, V, size=(N, 1))
    neg = rng.integers(0, V, size=(N, K))
    ids = np.concatenate([t, neg], axis=1).astype(np.uint32)

    loss_ref, dH_ref, dW_ref = _bce_oracle(H, W, ids)
    loss_gpu, dH_gpu, dW_gpu = _fused(device, H, W, ids)

    assert loss_gpu.shape == (N,)
    assert dH_gpu.shape == (N, d)
    assert dW_gpu.shape == (V, d)
    np.testing.assert_allclose(loss_gpu, loss_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dH_gpu, dH_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dW_gpu, dW_ref, rtol=1e-3, atol=1e-3)


def test_sampled_bce_collision_masking(device):
    """Negatives equal to their own target must contribute NOTHING: force
    every negative to collide -> loss is pure positive term, dW touches only
    target rows, and the negative slice of the gradient path is silent."""
    N, d, V = 8, 16, 64
    rng = np.random.default_rng(7)
    H = rng.standard_normal((N, d)).astype(np.float32)
    W = rng.standard_normal((V, d)).astype(np.float32)
    t = rng.integers(0, V, size=(N,))
    ids = np.repeat(t[:, None], 5, axis=1).astype(np.uint32)  # all cols = target

    loss_gpu, dH_gpu, dW_gpu = _fused(device, H, W, ids)

    s_pos = np.einsum("nd,nd->n", H, W[t])
    np.testing.assert_allclose(loss_gpu, _softplus(-s_pos), rtol=1e-4, atol=1e-4)
    # dH must equal the positive-only term: -sigmoid(-s_pos)/N * W[t]
    dH_ref = (-_sigmoid(-s_pos) / N)[:, None] * W[t]
    np.testing.assert_allclose(dH_gpu, dH_ref, rtol=1e-4, atol=1e-4)
    untouched = np.setdiff1d(np.arange(V), t)
    np.testing.assert_allclose(dW_gpu[untouched], 0.0, atol=1e-6)


def test_sampled_bce_dw_row_conservation(device):
    """Sum over dW rows equals sum over (ds * H) contributions — a global
    conservation check that survives atomic reordering."""
    N, d, V, K = 128, 32, 256, 8
    rng = np.random.default_rng(3)
    H = rng.standard_normal((N, d)).astype(np.float32)
    W = (rng.standard_normal((V, d)) * 0.2).astype(np.float32)
    ids = np.concatenate([rng.integers(0, V, (N, 1)),
                          rng.integers(0, V, (N, K))], axis=1).astype(np.uint32)

    _, _, dW_gpu = _fused(device, H, W, ids)
    _, _, dW_ref = _bce_oracle(H, W, ids)
    np.testing.assert_allclose(dW_gpu.sum(axis=0), dW_ref.sum(axis=0),
                               rtol=1e-3, atol=1e-3)
