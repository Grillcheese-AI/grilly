"""
Numerical parity: `grilly.functional` vs numpy and (optional) PyTorch references.
"""

import numpy as np
import pytest


def _ref_linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y.astype(np.float32, copy=False)


@pytest.mark.parity
def test_linear_matches_numpy_reference():
    from grilly.functional import linear

    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    w = rng.standard_normal((16, 32)).astype(np.float32)
    b = rng.standard_normal((16,)).astype(np.float32)
    expected = _ref_linear(x, w, b)
    got = linear(x, w, b)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parity
def test_linear_matches_numpy_no_bias():
    from grilly.functional import linear

    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 64)).astype(np.float32)
    w = rng.standard_normal((24, 64)).astype(np.float32)
    expected = _ref_linear(x, w, None)
    got = linear(x, w, None)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parity
def test_relu_matches_numpy_reference():
    from grilly.functional import relu

    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, 17)).astype(np.float32)
    expected = np.maximum(0.0, x).astype(np.float32)
    got = relu(x)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parity
def test_linear_relu_chain_matches_numpy_reference():
    from grilly.functional import linear, relu

    rng = np.random.default_rng(3)
    x = rng.standard_normal((6, 20)).astype(np.float32)
    w = rng.standard_normal((12, 20)).astype(np.float32)
    b = rng.standard_normal((12,)).astype(np.float32)
    expected = np.maximum(0.0, _ref_linear(x, w, b)).astype(np.float32)
    got = relu(linear(x, w, b))
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parity
def test_linear_matches_torch_functional():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from grilly.functional import linear

    rng = np.random.default_rng(4)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    w = rng.standard_normal((16, 32)).astype(np.float32)
    b = rng.standard_normal((16,)).astype(np.float32)

    xt = torch.from_numpy(x)
    wt = torch.from_numpy(w)
    bt = torch.from_numpy(b)
    expected = F.linear(xt, wt, bt).detach().numpy().astype(np.float32)
    got = linear(x, w, b)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parity
def test_relu_matches_torch():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from grilly.functional import relu

    rng = np.random.default_rng(5)
    x = rng.standard_normal((5, 17)).astype(np.float32)
    xt = torch.from_numpy(x)
    expected = F.relu(xt).detach().numpy().astype(np.float32)
    got = relu(x)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)
