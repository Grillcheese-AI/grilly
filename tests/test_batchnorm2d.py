"""
Tests for BatchNorm2d layer.
Validates correctness against PyTorch and benchmarks performance.
"""

import numpy as np
import pytest

# Try to import PyTorch for reference comparisons
try:
    import torch
    import torch.nn as torch_nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.fixture
def backend():
    """Initialize Vulkan backend"""
    from grilly import Compute

    compute = Compute()
    yield compute
    compute.cleanup()


@pytest.mark.gpu
class TestBatchNorm2dBasic:
    """Basic BatchNorm2d functionality tests"""

    def test_batchnorm2d_import(self):
        """Test BatchNorm2d can be imported"""
        from grilly.nn import BatchNorm2d

        assert BatchNorm2d is not None

    def test_batchnorm2d_init(self):
        """Test BatchNorm2d initialization"""
        from grilly.nn import BatchNorm2d

        bn = BatchNorm2d(64)
        assert bn.num_features == 64
        assert bn.eps == 1e-5
        assert bn.momentum == 0.1
        assert bn.affine is True
        assert bn.track_running_stats is True
        assert bn.weight.shape == (64,)
        assert bn.bias.shape == (64,)
        assert bn.running_mean.shape == (64,)
        assert bn.running_var.shape == (64,)

    def test_batchnorm2d_forward_shape(self, backend):
        """Test BatchNorm2d forward pass output shape"""
        from grilly.nn import BatchNorm2d

        bn = BatchNorm2d(64)
        x = np.random.randn(32, 64, 56, 56).astype(np.float32)

        output = bn(x)

        assert output.shape == (32, 64, 56, 56)
        assert output.dtype == np.float32

    def test_batchnorm2d_train_mode(self, backend):
        """Test BatchNorm2d in training mode"""
        from grilly.nn import BatchNorm2d

        bn = BatchNorm2d(16)
        bn.train()
        assert bn.training is True

        x = np.random.randn(8, 16, 32, 32).astype(np.float32)
        bn(x)

        # Running stats should be updated
        assert not np.allclose(bn.running_mean, 0.0)
        assert not np.allclose(bn.running_var, 1.0)

    def test_batchnorm2d_eval_mode(self, backend):
        """Test BatchNorm2d in eval mode"""
        from grilly.nn import BatchNorm2d

        bn = BatchNorm2d(16)

        # Train first to populate running stats
        bn.train()
        x_train = np.random.randn(8, 16, 32, 32).astype(np.float32)
        _ = bn(x_train)

        # Switch to eval
        bn.eval()
        assert bn.training is False

        x_eval = np.random.randn(8, 16, 32, 32).astype(np.float32)
        output = bn(x_eval)

        # Output should use running stats
        assert output.shape == (8, 16, 32, 32)

    def test_batchnorm2d_no_affine(self, backend):
        """Test BatchNorm2d without learnable parameters"""
        from grilly.nn import BatchNorm2d

        bn = BatchNorm2d(16, affine=False)
        assert bn.weight is None
        assert bn.bias is None

        x = np.random.randn(8, 16, 32, 32).astype(np.float32)
        output = bn(x)
        assert output.shape == (8, 16, 32, 32)

    def test_batchnorm2d_no_tracking(self, backend):
        """Test BatchNorm2d without running stats tracking"""
        from grilly.nn import BatchNorm2d

        bn = BatchNorm2d(16, track_running_stats=False)
        assert bn.running_mean is None
        assert bn.running_var is None

        x = np.random.randn(8, 16, 32, 32).astype(np.float32)
        output = bn(x)
        assert output.shape == (8, 16, 32, 32)


@pytest.mark.gpu
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestBatchNorm2dVsPyTorch:
    """Tests comparing Grilly BatchNorm2d with PyTorch"""

    def test_batchnorm2d_correctness_train(self, backend):
        """Compare Grilly BatchNorm2d output with PyTorch (training mode)"""
        from grilly.nn import BatchNorm2d as GrillyBN

        # Set seeds
        np.random.seed(42)
        torch.manual_seed(42)

        # Create layers
        grilly_bn = GrillyBN(16, eps=1e-5, momentum=0.1)
        torch_bn = torch_nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)

        # Initialize with same weights
        weight_np = np.asarray(grilly_bn.weight, dtype=np.float32)
        bias_np = np.asarray(grilly_bn.bias, dtype=np.float32)
        torch_bn.weight.data = torch.from_numpy(weight_np)
        torch_bn.bias.data = torch.from_numpy(bias_np)

        # Same running stats
        torch_bn.running_mean.data = torch.from_numpy(grilly_bn.running_mean.copy())
        torch_bn.running_var.data = torch.from_numpy(grilly_bn.running_var.copy())

        # Test input
        x_np = np.random.randn(4, 16, 8, 8).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        # Forward pass in training mode
        grilly_bn.train()
        torch_bn.train()

        grilly_out = grilly_bn(x_np)
        torch_out = torch_bn(x_torch).detach().numpy()

        # Compare outputs (allow small numerical difference)
        np.testing.assert_allclose(grilly_out, torch_out, rtol=1e-4, atol=1e-5)

        # Compare running stats (slightly relaxed tolerance for EMA accumulation)
        np.testing.assert_allclose(
            grilly_bn.running_mean, torch_bn.running_mean.numpy(), rtol=1e-3, atol=1e-4
        )
        np.testing.assert_allclose(
            grilly_bn.running_var, torch_bn.running_var.numpy(), rtol=1e-3, atol=1e-4
        )

    def test_batchnorm2d_correctness_eval(self, backend):
        """Compare with PyTorch (eval mode)"""
        from grilly.nn import BatchNorm2d as GrillyBN

        np.random.seed(42)
        torch.manual_seed(42)

        grilly_bn = GrillyBN(16)
        torch_bn = torch_nn.BatchNorm2d(16)

        # Train first
        grilly_bn.train()
        torch_bn.train()

        x_train = np.random.randn(4, 16, 8, 8).astype(np.float32)
        _ = grilly_bn(x_train)
        _ = torch_bn(torch.from_numpy(x_train))

        # Sync running stats
        torch_bn.running_mean.data = torch.from_numpy(grilly_bn.running_mean.copy())
        torch_bn.running_var.data = torch.from_numpy(grilly_bn.running_var.copy())

        # Eval mode
        grilly_bn.eval()
        torch_bn.eval()

        x_eval = np.random.randn(4, 16, 8, 8).astype(np.float32)
        grilly_out = grilly_bn(x_eval)
        torch_out = torch_bn(torch.from_numpy(x_eval)).detach().numpy()

        np.testing.assert_allclose(grilly_out, torch_out, rtol=1e-4, atol=1e-5)

    def test_batchnorm2d_backward_correctness(self, backend):
        """Compare gradients with PyTorch"""
        from grilly.nn import BatchNorm2d as GrillyBN

        np.random.seed(42)
        torch.manual_seed(42)

        # Create layers
        grilly_bn = GrillyBN(8, eps=1e-5)
        torch_bn = torch_nn.BatchNorm2d(8, eps=1e-5)

        # Sync weights
        weight_np = np.asarray(grilly_bn.weight, dtype=np.float32)
        bias_np = np.asarray(grilly_bn.bias, dtype=np.float32)
        torch_bn.weight.data = torch.from_numpy(weight_np)
        torch_bn.bias.data = torch.from_numpy(bias_np)

        # Input
        x_np = np.random.randn(2, 8, 4, 4).astype(np.float32)
        x_torch = torch.from_numpy(x_np).requires_grad_(True)

        # Forward
        grilly_bn.train()
        torch_bn.train()

        grilly_out = grilly_bn(x_np)
        torch_out = torch_bn(x_torch)

        # Backward
        grad_output = np.random.randn(*grilly_out.shape).astype(np.float32)
        torch_out.backward(torch.from_numpy(grad_output))
        grilly_grad_input = grilly_bn.backward(grad_output)

        # Compare input gradients
        np.testing.assert_allclose(grilly_grad_input, x_torch.grad.numpy(), rtol=1e-3, atol=1e-4)

        # Compare weight gradients
        np.testing.assert_allclose(
            grilly_bn.weight.grad, torch_bn.weight.grad.numpy(), rtol=1e-3, atol=1e-4
        )

        # Compare bias gradients
        np.testing.assert_allclose(
            grilly_bn.bias.grad, torch_bn.bias.grad.numpy(), rtol=1e-3, atol=1e-4
        )


@pytest.mark.gpu
class TestBatchNorm2dEdgeCases:
    """Edge case tests"""

    def test_batchnorm2d_single_channel(self, backend):
        """Test with single channel"""
        from grilly.nn import BatchNorm2d

        bn = BatchNorm2d(1)
        x = np.random.randn(4, 1, 8, 8).astype(np.float32)
        output = bn(x)
        assert output.shape == (4, 1, 8, 8)

    def test_batchnorm2d_large_batch(self, backend):
        """Test with large batch size"""
        from grilly.nn import BatchNorm2d

        bn = BatchNorm2d(16)
        x = np.random.randn(128, 16, 8, 8).astype(np.float32)
        output = bn(x)
        assert output.shape == (128, 16, 8, 8)

    def test_batchnorm2d_small_spatial(self, backend):
        """Test with small spatial dimensions"""
        from grilly.nn import BatchNorm2d

        bn = BatchNorm2d(32)
        x = np.random.randn(16, 32, 1, 1).astype(np.float32)
        output = bn(x)
        assert output.shape == (16, 32, 1, 1)

    def test_batchnorm2d_batch_size_1(self, backend):
        """Test with batch size = 1"""
        from grilly.nn import BatchNorm2d

        bn = BatchNorm2d(16)
        x = np.random.randn(1, 16, 8, 8).astype(np.float32)
        output = bn(x)
        assert output.shape == (1, 16, 8, 8)


@pytest.mark.gpu
class TestBatchNorm1d:
    """Tests for BatchNorm1d wrapper"""

    def test_batchnorm1d_import(self):
        """Test BatchNorm1d can be imported"""
        from grilly.nn import BatchNorm1d

        assert BatchNorm1d is not None

    def test_batchnorm1d_2d_input(self, backend):
        """Test BatchNorm1d with 2D input (N, C)"""
        from grilly.nn import BatchNorm1d

        bn = BatchNorm1d(64)
        x = np.random.randn(32, 64).astype(np.float32)
        output = bn(x)
        assert output.shape == (32, 64)

    def test_batchnorm1d_3d_input(self, backend):
        """Test BatchNorm1d with 3D input (N, C, L)"""
        from grilly.nn import BatchNorm1d

        bn = BatchNorm1d(64)
        x = np.random.randn(32, 64, 100).astype(np.float32)
        output = bn(x)
        assert output.shape == (32, 64, 100)


@pytest.mark.gpu
@pytest.mark.benchmark
class TestBatchNorm2dPerformance:
    """Performance benchmarking tests"""

    def test_batchnorm2d_benchmark_small(self, backend, benchmark):
        """Benchmark small batchnorm (typical ResNet block)"""
        from grilly.nn import BatchNorm2d

        bn = BatchNorm2d(64)
        x = np.random.randn(32, 64, 56, 56).astype(np.float32)

        def run_bn():
            return bn(x)

        result = benchmark(run_bn)
        assert result.shape == (32, 64, 56, 56)

    def test_batchnorm2d_benchmark_large(self, backend, benchmark):
        """Benchmark large batchnorm"""
        from grilly.nn import BatchNorm2d

        bn = BatchNorm2d(256)
        x = np.random.randn(16, 256, 28, 28).astype(np.float32)

        def run_bn():
            return bn(x)

        result = benchmark(run_bn)
        assert result.shape == (16, 256, 28, 28)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_batchnorm2d_speedup_vs_cpu(self, backend):
        """Measure speedup vs PyTorch CPU"""
        import time

        from grilly.nn import BatchNorm2d as GrillyBN

        # Create layers
        grilly_bn = GrillyBN(128)
        torch_bn = torch_nn.BatchNorm2d(128)

        # Test input
        x_np = np.random.randn(32, 128, 28, 28).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        # Warm-up
        _ = grilly_bn(x_np)
        _ = torch_bn(x_torch)

        # Benchmark Grilly (GPU)
        n_iters = 10
        start = time.time()
        for _ in range(n_iters):
            _ = grilly_bn(x_np)
        grilly_time = (time.time() - start) / n_iters

        # Benchmark PyTorch (CPU)
        start = time.time()
        for _ in range(n_iters):
            _ = torch_bn(x_torch)
        torch_time = (time.time() - start) / n_iters

        speedup = torch_time / grilly_time
        print(f"\nSpeedup vs PyTorch CPU: {speedup:.2f}x")
        print(f"Grilly: {grilly_time * 1000:.2f}ms, PyTorch CPU: {torch_time * 1000:.2f}ms")

        assert speedup > 0  # Just verify it runs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
