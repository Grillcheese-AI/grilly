"""
Reference check: conv2d backward weight (GEMM path) vs PyTorch (Workstream C1).

When `convd_im2col` + `gemm_mnk` + `tensor-transpose` are loaded, weight gradients
should match PyTorch within tolerance.
"""

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

try:
    from grilly.backend.compute import VulkanCompute
    from grilly.backend.base import VULKAN_PYTHON_LEGACY_BACKEND_AVAILABLE
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_PYTHON_LEGACY_BACKEND_AVAILABLE, reason="Vulkan not available")
def test_conv2d_backward_weight_matches_torch_gemm_path():
    """Small conv: dW from Grilly GEMM path vs torch autograd."""
    np.random.seed(123)
    torch.manual_seed(123)

    batch_size, in_ch, out_ch = 2, 4, 6
    h, w = 8, 8
    kh, kw = 3, 3
    stride = (1, 1)
    padding = (1, 1)

    x_np = np.random.randn(batch_size, in_ch, h, w).astype(np.float32)
    w_np = np.random.randn(out_ch, in_ch, kh, kw).astype(np.float32)
    b_np = np.random.randn(out_ch).astype(np.float32)

    xt = torch.from_numpy(x_np).requires_grad_(True)
    wt = torch.from_numpy(w_np).requires_grad_(True)
    bt = torch.from_numpy(b_np).requires_grad_(True)

    y_t = F.conv2d(xt, wt, bt, stride=stride, padding=padding)
    go = np.random.randn(*y_t.shape).astype(np.float32)
    y_t.backward(torch.from_numpy(go))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        backend = VulkanCompute()
    try:
        if "convd_im2col" not in backend.conv.shaders:
            pytest.skip("convd_im2col not available")
        gw, gb = backend.conv.conv2d_backward_weight(
            go,
            x_np,
            (kh, kw),
            stride=stride,
            padding=padding,
            dilation=(1, 1),
            groups=1,
            has_bias=True,
        )
        np.testing.assert_allclose(gw, wt.grad.numpy(), rtol=5e-3, atol=5e-4)
        np.testing.assert_allclose(gb, bt.grad.numpy(), rtol=5e-3, atol=5e-4)
    finally:
        backend.cleanup()
