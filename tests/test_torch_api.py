"""Smoke tests for ``grilly.torch_api`` (torch-style facade, no PyTorch)."""

import grilly.torch_api as torch
import numpy as np
import pytest
from grilly.nn import autograd as ag


def test_device_and_vulkan():
    d = torch.device("vulkan")
    assert "vulkan" in str(d)
    assert isinstance(torch.vulkan.is_available(), bool)


def test_tensor_long_and_ops():
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    assert x.shape == (2, 2)
    z = torch.zeros(3)
    z.uniform_(-1, 1)
    assert z.shape == (3,)
    p = torch.randperm(10)
    assert p.shape == (10,)
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    d = torch.cdist(a.unsqueeze(0), b.unsqueeze(0), p=1)
    assert d.shape == (1, 2, 2)


def test_functional_cross_entropy_sum():
    logits = ag.randn(5, 3, requires_grad=True)
    target = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    loss = torch.nn.functional.cross_entropy(logits, target, reduction="sum")
    loss.backward()
    assert loss.data.size == 1


def test_amp_namespace():
    assert hasattr(torch.amp, "autocast")
    assert hasattr(torch.amp, "GradScaler")
    s = torch.amp.GradScaler("vulkan", enabled=False)
    assert s.get_scale() == 1.0


def test_grl_save_load_roundtrip(tmp_path):
    path = tmp_path / "t.grl"
    state = {"model": {"w": np.ones((2, 2), dtype=np.float32)}, "step": 3, "best_ppl": 1.25}
    torch.save(state, path)
    out = torch.load(path, map_location=torch.device("cpu"))
    assert "model" in out or "metadata" in out
