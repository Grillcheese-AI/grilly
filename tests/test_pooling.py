"""
Tests for pooling layers (MaxPool2d, AvgPool2d).
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
class TestMaxPool2dBasic:
    """Basic MaxPool2d functionality tests"""

    def test_maxpool2d_import(self):
        """Test MaxPool2d can be imported"""
        from grilly.nn import MaxPool2d

        assert MaxPool2d is not None

    def test_maxpool2d_init(self):
        """Test MaxPool2d initialization"""
        from grilly.nn import MaxPool2d

        pool = MaxPool2d(kernel_size=2, stride=2)
        assert pool.kernel_size == 2
        assert pool.stride == 2
        assert pool.padding == 0
        assert pool.dilation == 1
        assert not pool.return_indices

    def test_maxpool2d_forward_shape(self, backend):
        """Test MaxPool2d forward pass output shape"""
        from grilly.nn import MaxPool2d

        pool = MaxPool2d(kernel_size=2, stride=2)
        x = np.random.randn(2, 3, 8, 8).astype(np.float32)
        output = pool(x)

        assert output.shape == (2, 3, 4, 4)
        assert output.dtype == np.float32

    def test_maxpool2d_with_indices(self, backend):
        """Test MaxPool2d returns indices when requested"""
        from grilly.nn import MaxPool2d

        pool = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        x = np.random.randn(2, 3, 8, 8).astype(np.float32)
        output, indices = pool(x)

        assert output.shape == (2, 3, 4, 4)
        assert indices.shape == (2, 3, 4, 4)
        assert indices.dtype == np.uint32

    def test_maxpool2d_stride_1(self, backend):
        """Test MaxPool2d with stride=1"""
        from grilly.nn import MaxPool2d

        pool = MaxPool2d(kernel_size=3, stride=1, padding=1)
        x = np.random.randn(1, 4, 8, 8).astype(np.float32)
        output = pool(x)

        assert output.shape == (1, 4, 8, 8)

    def test_maxpool2d_non_square_kernel(self, backend):
        """Test MaxPool2d with non-square kernel"""
        from grilly.nn import MaxPool2d

        pool = MaxPool2d(kernel_size=(2, 3), stride=(2, 3))
        x = np.random.randn(1, 4, 8, 9).astype(np.float32)
        output = pool(x)

        assert output.shape == (1, 4, 4, 3)


@pytest.mark.gpu
class TestAvgPool2dBasic:
    """Basic AvgPool2d functionality tests"""

    def test_avgpool2d_import(self):
        """Test AvgPool2d can be imported"""
        from grilly.nn import AvgPool2d

        assert AvgPool2d is not None

    def test_avgpool2d_init(self):
        """Test AvgPool2d initialization"""
        from grilly.nn import AvgPool2d

        pool = AvgPool2d(kernel_size=2, stride=2)
        assert pool.kernel_size == 2
        assert pool.stride == 2
        assert pool.padding == 0
        assert pool.count_include_pad

    def test_avgpool2d_forward_shape(self, backend):
        """Test AvgPool2d forward pass output shape"""
        from grilly.nn import AvgPool2d

        pool = AvgPool2d(kernel_size=2, stride=2)
        x = np.random.randn(2, 3, 8, 8).astype(np.float32)
        output = pool(x)

        assert output.shape == (2, 3, 4, 4)
        assert output.dtype == np.float32

    def test_avgpool2d_count_include_pad_false(self, backend):
        """Test AvgPool2d with count_include_pad=False"""
        from grilly.nn import AvgPool2d

        pool = AvgPool2d(kernel_size=2, stride=2, padding=1, count_include_pad=False)
        x = np.random.randn(1, 4, 8, 8).astype(np.float32)
        output = pool(x)

        assert output.shape == (1, 4, 5, 5)

    def test_avgpool2d_non_square_kernel(self, backend):
        """Test AvgPool2d with non-square kernel"""
        from grilly.nn import AvgPool2d

        pool = AvgPool2d(kernel_size=(2, 3), stride=(2, 3))
        x = np.random.randn(1, 4, 8, 9).astype(np.float32)
        output = pool(x)

        assert output.shape == (1, 4, 4, 3)


@pytest.mark.gpu
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestMaxPool2dVsPyTorch:
    """Tests comparing Grilly MaxPool2d with PyTorch"""

    def test_maxpool2d_correctness_basic(self, backend):
        """Compare Grilly MaxPool2d output with PyTorch (basic case)"""
        from grilly.nn import MaxPool2d as GrillyMaxPool

        np.random.seed(42)
        torch.manual_seed(42)

        # Create layers
        grilly_pool = GrillyMaxPool(kernel_size=2, stride=2)
        torch_pool = torch_nn.MaxPool2d(kernel_size=2, stride=2)

        # Test input
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        # Forward pass
        grilly_out = grilly_pool(x_np)
        torch_out = torch_pool(x_torch).numpy()

        # Compare outputs
        np.testing.assert_allclose(grilly_out, torch_out, rtol=1e-5, atol=1e-6)

    def test_maxpool2d_correctness_with_padding(self, backend):
        """Compare with PyTorch (with padding)"""
        from grilly.nn import MaxPool2d as GrillyMaxPool

        np.random.seed(42)
        torch.manual_seed(42)

        grilly_pool = GrillyMaxPool(kernel_size=3, stride=2, padding=1)
        torch_pool = torch_nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        x_np = np.random.randn(2, 4, 7, 7).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        grilly_out = grilly_pool(x_np)
        torch_out = torch_pool(x_torch).numpy()

        np.testing.assert_allclose(grilly_out, torch_out, rtol=1e-5, atol=1e-6)

    def test_maxpool2d_correctness_with_dilation(self, backend):
        """Compare with PyTorch (with dilation)"""
        from grilly.nn import MaxPool2d as GrillyMaxPool

        np.random.seed(42)
        torch.manual_seed(42)

        grilly_pool = GrillyMaxPool(kernel_size=2, stride=1, dilation=2)
        torch_pool = torch_nn.MaxPool2d(kernel_size=2, stride=1, dilation=2)

        x_np = np.random.randn(1, 3, 10, 10).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        grilly_out = grilly_pool(x_np)
        torch_out = torch_pool(x_torch).numpy()

        np.testing.assert_allclose(grilly_out, torch_out, rtol=1e-5, atol=1e-6)

    def test_maxpool2d_backward_correctness(self, backend):
        """Compare gradients with PyTorch"""
        from grilly.nn import MaxPool2d as GrillyMaxPool

        np.random.seed(42)
        torch.manual_seed(42)

        # Create layers
        grilly_pool = GrillyMaxPool(kernel_size=2, stride=2)
        torch_pool = torch_nn.MaxPool2d(kernel_size=2, stride=2)

        # Input
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x_torch = torch.from_numpy(x_np).requires_grad_(True)

        # Forward
        grilly_out = grilly_pool(x_np)
        torch_out = torch_pool(x_torch)

        # Backward
        grad_output = np.random.randn(*grilly_out.shape).astype(np.float32)
        torch_out.backward(torch.from_numpy(grad_output))
        grilly_grad_input = grilly_pool.backward(grad_output)

        # Compare input gradients
        np.testing.assert_allclose(grilly_grad_input, x_torch.grad.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestAvgPool2dVsPyTorch:
    """Tests comparing Grilly AvgPool2d with PyTorch"""

    def test_avgpool2d_correctness_basic(self, backend):
        """Compare Grilly AvgPool2d output with PyTorch (basic case)"""
        from grilly.nn import AvgPool2d as GrillyAvgPool

        np.random.seed(42)
        torch.manual_seed(42)

        # Create layers
        grilly_pool = GrillyAvgPool(kernel_size=2, stride=2)
        torch_pool = torch_nn.AvgPool2d(kernel_size=2, stride=2)

        # Test input
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        # Forward pass
        grilly_out = grilly_pool(x_np)
        torch_out = torch_pool(x_torch).numpy()

        # Compare outputs
        np.testing.assert_allclose(grilly_out, torch_out, rtol=1e-5, atol=1e-6)

    def test_avgpool2d_correctness_with_padding(self, backend):
        """Compare with PyTorch (with padding, count_include_pad=True)"""
        from grilly.nn import AvgPool2d as GrillyAvgPool

        np.random.seed(42)
        torch.manual_seed(42)

        grilly_pool = GrillyAvgPool(kernel_size=3, stride=2, padding=1, count_include_pad=True)
        torch_pool = torch_nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=True)

        x_np = np.random.randn(2, 4, 7, 7).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        grilly_out = grilly_pool(x_np)
        torch_out = torch_pool(x_torch).numpy()

        np.testing.assert_allclose(grilly_out, torch_out, rtol=1e-5, atol=1e-6)

    def test_avgpool2d_correctness_exclude_padding(self, backend):
        """Compare with PyTorch (count_include_pad=False)"""
        from grilly.nn import AvgPool2d as GrillyAvgPool

        np.random.seed(42)
        torch.manual_seed(42)

        grilly_pool = GrillyAvgPool(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        torch_pool = torch_nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        x_np = np.random.randn(2, 4, 7, 7).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        grilly_out = grilly_pool(x_np)
        torch_out = torch_pool(x_torch).numpy()

        np.testing.assert_allclose(grilly_out, torch_out, rtol=1e-5, atol=1e-6)

    def test_avgpool2d_backward_correctness(self, backend):
        """Compare gradients with PyTorch"""
        from grilly.nn import AvgPool2d as GrillyAvgPool

        np.random.seed(42)
        torch.manual_seed(42)

        # Create layers
        grilly_pool = GrillyAvgPool(kernel_size=2, stride=2)
        torch_pool = torch_nn.AvgPool2d(kernel_size=2, stride=2)

        # Input
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x_torch = torch.from_numpy(x_np).requires_grad_(True)

        # Forward
        grilly_out = grilly_pool(x_np)
        torch_out = torch_pool(x_torch)

        # Backward
        grad_output = np.random.randn(*grilly_out.shape).astype(np.float32)
        torch_out.backward(torch.from_numpy(grad_output))
        grilly_grad_input = grilly_pool.backward(grad_output)

        # Compare input gradients
        np.testing.assert_allclose(grilly_grad_input, x_torch.grad.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
class TestAdaptivePooling:
    """Tests for adaptive pooling layers"""

    def test_adaptive_maxpool2d_import(self):
        """Test AdaptiveMaxPool2d can be imported"""
        from grilly.nn import AdaptiveMaxPool2d

        assert AdaptiveMaxPool2d is not None

    def test_adaptive_avgpool2d_import(self):
        """Test AdaptiveAvgPool2d can be imported"""
        from grilly.nn import AdaptiveAvgPool2d

        assert AdaptiveAvgPool2d is not None

    def test_adaptive_maxpool2d_forward(self, backend):
        """Test AdaptiveMaxPool2d forward pass"""
        from grilly.nn import AdaptiveMaxPool2d

        pool = AdaptiveMaxPool2d(output_size=(7, 7))
        x = np.random.randn(1, 512, 14, 14).astype(np.float32)
        output = pool(x)

        assert output.shape == (1, 512, 7, 7)

    def test_adaptive_avgpool2d_forward(self, backend):
        """Test AdaptiveAvgPool2d forward pass (global pooling)"""
        from grilly.nn import AdaptiveAvgPool2d

        pool = AdaptiveAvgPool2d(output_size=(1, 1))
        x = np.random.randn(1, 512, 7, 7).astype(np.float32)
        output = pool(x)

        assert output.shape == (1, 512, 1, 1)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_adaptive_avgpool2d_vs_pytorch(self, backend):
        """Compare AdaptiveAvgPool2d with PyTorch"""
        from grilly.nn import AdaptiveAvgPool2d as GrillyAdaptiveAvgPool

        np.random.seed(42)
        torch.manual_seed(42)

        grilly_pool = GrillyAdaptiveAvgPool(output_size=(1, 1))
        torch_pool = torch_nn.AdaptiveAvgPool2d(output_size=(1, 1))

        x_np = np.random.randn(2, 64, 7, 7).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        grilly_out = grilly_pool(x_np)
        torch_out = torch_pool(x_torch).numpy()

        np.testing.assert_allclose(grilly_out, torch_out, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
class TestPoolingEdgeCases:
    """Edge case tests for pooling layers"""

    def test_maxpool2d_single_channel(self, backend):
        """Test MaxPool2d with single channel"""
        from grilly.nn import MaxPool2d

        pool = MaxPool2d(kernel_size=2, stride=2)
        x = np.random.randn(4, 1, 8, 8).astype(np.float32)
        output = pool(x)
        assert output.shape == (4, 1, 4, 4)

    def test_avgpool2d_single_channel(self, backend):
        """Test AvgPool2d with single channel"""
        from grilly.nn import AvgPool2d

        pool = AvgPool2d(kernel_size=2, stride=2)
        x = np.random.randn(4, 1, 8, 8).astype(np.float32)
        output = pool(x)
        assert output.shape == (4, 1, 4, 4)

    def test_maxpool2d_large_batch(self, backend):
        """Test MaxPool2d with large batch size"""
        from grilly.nn import MaxPool2d

        pool = MaxPool2d(kernel_size=2, stride=2)
        x = np.random.randn(128, 16, 8, 8).astype(np.float32)
        output = pool(x)
        assert output.shape == (128, 16, 4, 4)

    def test_avgpool2d_batch_size_1(self, backend):
        """Test AvgPool2d with batch size = 1"""
        from grilly.nn import AvgPool2d

        pool = AvgPool2d(kernel_size=2, stride=2)
        x = np.random.randn(1, 16, 8, 8).astype(np.float32)
        output = pool(x)
        assert output.shape == (1, 16, 4, 4)


@pytest.mark.gpu
@pytest.mark.benchmark
class TestPoolingPerformance:
    """Performance benchmarking tests"""

    def test_maxpool2d_benchmark_small(self, backend, benchmark):
        """Benchmark small maxpool (typical ResNet block)"""
        from grilly.nn import MaxPool2d

        pool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        x = np.random.randn(32, 64, 56, 56).astype(np.float32)

        def run_pool():
            return pool(x)

        result = benchmark(run_pool)
        assert result.shape == (32, 64, 28, 28)

    def test_avgpool2d_benchmark_small(self, backend, benchmark):
        """Benchmark small avgpool"""
        from grilly.nn import AvgPool2d

        pool = AvgPool2d(kernel_size=2, stride=2)
        x = np.random.randn(32, 64, 56, 56).astype(np.float32)

        def run_pool():
            return pool(x)

        result = benchmark(run_pool)
        assert result.shape == (32, 64, 28, 28)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_maxpool2d_speedup_vs_cpu(self, backend):
        """Measure speedup vs PyTorch CPU"""
        import time

        from grilly.nn import MaxPool2d as GrillyMaxPool

        # Create layers
        grilly_pool = GrillyMaxPool(kernel_size=2, stride=2)
        torch_pool = torch_nn.MaxPool2d(kernel_size=2, stride=2)

        # Test input
        x_np = np.random.randn(32, 128, 28, 28).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        # Warm-up
        _ = grilly_pool(x_np)
        _ = torch_pool(x_torch)

        # Benchmark Grilly (GPU)
        n_iters = 10
        start = time.time()
        for _ in range(n_iters):
            _ = grilly_pool(x_np)
        grilly_time = (time.time() - start) / n_iters

        # Benchmark PyTorch (CPU)
        start = time.time()
        for _ in range(n_iters):
            _ = torch_pool(x_torch)
        torch_time = (time.time() - start) / n_iters

        speedup = torch_time / grilly_time
        print(f"\nSpeedup vs PyTorch CPU: {speedup:.2f}x")
        print(f"Grilly: {grilly_time * 1000:.2f}ms, PyTorch CPU: {torch_time * 1000:.2f}ms")

        assert speedup > 0  # Just verify it runs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
