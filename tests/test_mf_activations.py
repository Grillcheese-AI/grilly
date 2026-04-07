"""Multiplication-free activations: numpy + autograd."""

import numpy as np
import pytest
from grilly.functional.mf_activations import (
    mf_sigmoid,
    mf_sigmoid_01,
    mf_softmax,
    mf_softplus,
)
from grilly.nn import autograd as ag


def test_mf_softmax_rows_sum_to_one():
    x = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    y = mf_softmax(x, dim=-1)
    assert y.shape == x.shape
    np.testing.assert_allclose(y.sum(axis=-1), np.ones(2), rtol=1e-5)


def test_mf_softplus_matches_algebra():
    x = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
    b = 1.0
    c = 4.0 / (b * b)
    want = 0.5 * (x + np.sqrt(x * x + c))
    got = mf_softplus(x, beta=b)
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_mf_sigmoid_bounds():
    x = np.linspace(-5, 5, 11, dtype=np.float32)
    y = mf_sigmoid(x)
    assert float(y.min()) >= -1.0 and float(y.max()) <= 1.0
    z = mf_sigmoid_01(x)
    assert float(z.min()) >= 0.0 and float(z.max()) <= 1.0


def test_mf_softmax_autograd():
    v = ag.Variable(np.array([[1.0, 2.0]], dtype=np.float32), requires_grad=True)
    y = ag.mf_softmax(v, dim=-1)
    assert y.data.sum() == pytest.approx(1.0)
    y.backward()
    assert v.grad is not None


def test_mf_softplus_autograd():
    v = ag.Variable(np.array([0.5], dtype=np.float32), requires_grad=True)
    y = ag.mf_softplus(v, beta=1.0)
    y.backward()
    assert v.grad is not None
