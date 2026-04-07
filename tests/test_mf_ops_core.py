"""grilly_core mf_softmax / mf_softplus / mf_sigmoid — GPU vs NumPy reference."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

try:
    import grilly_core as gc
except ImportError:
    pytest.skip("grilly_core not available", allow_module_level=True)

from grilly.functional.mf_activations import mf_sigmoid, mf_softmax, mf_softplus


def _shader_spv_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "shaders" / "spv"


def _require_mf_symbols() -> None:
    for name in ("mf_softmax", "mf_softplus", "mf_sigmoid"):
        if not hasattr(gc, name):
            pytest.skip(f"grilly_core.{name} not in this build — rebuild extension")


@pytest.mark.gpu
@pytest.mark.cpp
def test_mf_softmax_gpu_matches_numpy() -> None:
    _require_mf_symbols()
    if not _shader_spv_dir().exists():
        pytest.skip("shaders/spv not present")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 7), dtype=np.float32)

    dev = gc.Device()
    try:
        dev.load_shaders(str(_shader_spv_dir()))
    except Exception as e:
        pytest.skip(f"load_shaders failed: {e}")

    got = gc.mf_softmax(dev, x, -1)
    ref = mf_softmax(x, dim=-1)
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.cpp
def test_mf_softplus_and_sigmoid_gpu_match_numpy() -> None:
    _require_mf_symbols()
    if not _shader_spv_dir().exists():
        pytest.skip("shaders/spv not present")

    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 5), dtype=np.float32)

    dev = gc.Device()
    try:
        dev.load_shaders(str(_shader_spv_dir()))
    except Exception as e:
        pytest.skip(f"load_shaders failed: {e}")

    g_sig = gc.mf_sigmoid(dev, x)
    np.testing.assert_allclose(g_sig.numpy(), mf_sigmoid(x), rtol=1e-5, atol=1e-6)

    g_sp = gc.mf_softplus(dev, x, 1.25)
    np.testing.assert_allclose(g_sp.numpy(), mf_softplus(x, 1.25), rtol=1e-4, atol=1e-5)
