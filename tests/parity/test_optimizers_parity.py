"""
Optimizer stepping parity vs PyTorch (optional `torch`).
"""

import numpy as np
import pytest


@pytest.mark.parity
def test_sgd_no_momentum_matches_torch():
    torch = pytest.importorskip("torch")
    import torch.optim as optim
    from grilly.nn import Parameter
    from grilly.optim import SGD

    np.random.seed(7)
    w = np.random.randn(5, 4).astype(np.float32)
    g = np.random.randn(5, 4).astype(np.float32)
    lr = 0.01

    wt = torch.from_numpy(w.copy())
    wt.requires_grad_(True)
    wt.grad = torch.from_numpy(g.copy())
    topt = optim.SGD([wt], lr=lr)
    topt.step()
    torch_result = wt.detach().numpy()

    p = Parameter(w.copy(), requires_grad=True)
    p.grad = g.copy()
    gopt = SGD([p], lr=lr, momentum=0.0, use_gpu=False)
    gopt.step()

    np.testing.assert_allclose(np.asarray(p, dtype=np.float32), torch_result, rtol=1e-5, atol=1e-6)


@pytest.mark.parity
def test_adam_cpu_matches_torch_single_tensor():
    torch = pytest.importorskip("torch")
    import torch.optim as optim
    from grilly.nn import Parameter
    from grilly.optim import Adam

    np.random.seed(8)
    w = np.random.randn(3, 5).astype(np.float32)
    g = np.random.randn(3, 5).astype(np.float32)
    lr = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-8

    wt = torch.from_numpy(w.copy())
    wt.requires_grad_(True)
    wt.grad = torch.from_numpy(g.copy())
    topt = optim.Adam([wt], lr=lr, betas=betas, eps=eps)
    topt.step()
    torch_result = wt.detach().numpy()

    p = Parameter(w.copy(), requires_grad=True)
    p.grad = g.copy()
    gopt = Adam([p], lr=lr, betas=betas, eps=eps, use_gpu=False)
    gopt.step()

    np.testing.assert_allclose(np.asarray(p, dtype=np.float32), torch_result, rtol=1e-4, atol=1e-5)


@pytest.mark.parity
def test_adamw_cpu_matches_torch_single_tensor():
    torch = pytest.importorskip("torch")
    import torch.optim as optim
    from grilly.nn import Parameter
    from grilly.optim import AdamW

    np.random.seed(9)
    w = np.random.randn(4, 6).astype(np.float32)
    g = np.random.randn(4, 6).astype(np.float32)
    lr = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0.01

    wt = torch.from_numpy(w.copy())
    wt.requires_grad_(True)
    wt.grad = torch.from_numpy(g.copy())
    topt = optim.AdamW([wt], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    topt.step()
    torch_result = wt.detach().numpy()

    p = Parameter(w.copy(), requires_grad=True)
    p.grad = g.copy()
    gopt = AdamW(
        [p], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, use_gpu=False
    )
    gopt.step()

    np.testing.assert_allclose(np.asarray(p, dtype=np.float32), torch_result, rtol=1e-4, atol=1e-5)
