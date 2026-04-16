import numpy as np
import pytest

from cubemind.training.losses import (
    mse_loss,
    cross_entropy_loss,
    cosine_similarity_loss,
    CIWLoss,
    DROPSLoss,
)


@pytest.fixture(scope="session")
def device():
    import grilly_core
    import os
    dev = grilly_core.Device()
    shader_dir = os.path.join(os.getcwd(), "shaders", "spv")
    try:
        dev.load_shaders(shader_dir)
        return dev
    except Exception as e:
        pytest.skip(f"Could not init Vulkan device: {e}")


def test_mse_loss_parity(device):
    np.random.seed(42)
    preds = np.random.randn(10, 5).astype(np.float32)
    targs = np.random.randn(10, 5).astype(np.float32)

    loss_cpu = mse_loss(preds, targs)
    loss_gpu = mse_loss(preds, targs, device=device)

    np.testing.assert_allclose(loss_cpu, loss_gpu, rtol=1e-5, atol=1e-5)


def test_cross_entropy_parity(device):
    np.random.seed(42)
    logits = np.random.randn(32, 10).astype(np.float32)
    labels = np.random.randint(0, 10, size=(32,)).astype(np.uint32)

    loss_cpu = cross_entropy_loss(logits, labels, from_logits=True)
    loss_gpu = cross_entropy_loss(logits, labels, from_logits=True, device=device)

    np.testing.assert_allclose(loss_cpu, loss_gpu, rtol=1e-5, atol=1e-5)


def test_cosine_similarity_parity(device):
    np.random.seed(42)
    preds = np.random.randn(16, 64).astype(np.float32)
    targs = np.random.randn(16, 64).astype(np.float32)

    loss_cpu = cosine_similarity_loss(preds, targs)
    loss_gpu = cosine_similarity_loss(preds, targs, device=device)

    np.testing.assert_allclose(loss_cpu, loss_gpu, rtol=1e-5, atol=1e-5)


def test_ciw_loss_parity(device):
    np.random.seed(42)
    logits = np.random.randn(16, 10).astype(np.float32)
    labels = np.random.randint(0, 10, size=(16,)).astype(np.uint32)

    ciw = CIWLoss()
    ciw._iteration = 5  # past burn-in
    
    # We must instantiate two CIWLoss objects so internal EMA/iteration isn't shared/advanced twice
    ciw_cpu = CIWLoss()
    ciw_cpu._iteration = 5
    ciw_gpu = CIWLoss()
    ciw_gpu._iteration = 5

    loss_cpu = ciw_cpu(logits, labels)
    loss_gpu = ciw_gpu(logits, labels, device=device)

    np.testing.assert_allclose(loss_cpu, loss_gpu, rtol=1e-5, atol=1e-5)


def test_drops_loss_parity(device):
    np.random.seed(42)
    logits = np.random.randn(16, 10).astype(np.float32)
    labels = np.random.randint(0, 10, size=(16,)).astype(np.uint32)
    
    drops_cpu = DROPSLoss()
    drops_gpu = DROPSLoss()

    # Need multiple steps to test EMA behavior
    for i in range(3):
        # same random data for both paths for a fair test
        log_i = np.random.randn(16, 10).astype(np.float32)
        lab_i = np.random.randint(0, 10, size=(16,)).astype(np.uint32)
        
        loss_cpu = drops_cpu(log_i, lab_i)
        loss_gpu = drops_gpu(log_i, lab_i, device=device)
        
        np.testing.assert_allclose(loss_cpu, loss_gpu, rtol=1e-5, atol=1e-5)
