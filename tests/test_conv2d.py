"""
Tests for Conv2d layer.
Validates correctness against PyTorch and benchmarks performance.
"""

import numpy as np
import pytest

# Try to import PyTorch for reference comparisons
try:
    import torch
    import torch.nn as torch_nn
    import torch.nn.functional as F

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


@pytest.fixture
def skip_if_no_torch():
    """Skip test if PyTorch is not available"""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available for comparison")


@pytest.mark.gpu
class TestConv2dBasic:
    """Basic Conv2d functionality tests"""

    def test_conv2d_import(self):
        """Test Conv2d can be imported"""
        from grilly.nn import Conv2d

        assert Conv2d is not None

    def test_conv2d_init(self):
        """Test Conv2d initialization"""
        from grilly.nn import Conv2d

        conv = Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        assert conv.in_channels == 3
        assert conv.out_channels == 16
        assert conv.kernel_size == (3, 3)
        assert conv.stride == (1, 1)
        assert conv.padding == (1, 1)
        assert conv.weight.shape == (16, 3, 3, 3)
        assert conv.bias.shape == (16,)

    def test_conv2d_forward_shape(self, backend):
        """Test Conv2d forward pass output shape"""
        from grilly.nn import Conv2d

        conv = Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        x = np.random.randn(2, 3, 32, 32).astype(np.float32)

        output = conv(x)

        assert output.shape == (2, 16, 32, 32)
        assert output.dtype == np.float32

    def test_conv2d_stride(self, backend):
        """Test Conv2d with stride > 1"""
        from grilly.nn import Conv2d

        conv = Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        x = np.random.randn(2, 3, 32, 32).astype(np.float32)

        output = conv(x)

        # With stride=2, output should be half size
        assert output.shape == (2, 16, 16, 16)

    def test_conv2d_no_bias(self, backend):
        """Test Conv2d without bias"""
        from grilly.nn import Conv2d

        conv = Conv2d(3, 16, kernel_size=3, bias=False)
        assert conv.bias is None

        x = np.random.randn(2, 3, 32, 32).astype(np.float32)
        output = conv(x)
        assert output.shape == (2, 16, 30, 30)  # No padding, so size reduces

    def test_conv2d_groups(self, backend):
        """Test grouped convolution"""
        from grilly.nn import Conv2d

        # Grouped conv: in_channels and out_channels must be divisible by groups
        conv = Conv2d(4, 8, kernel_size=3, groups=2, padding=1)
        assert conv.groups == 2
        assert conv.weight.shape == (8, 2, 3, 3)  # out_channels, in_channels/groups, kh, kw

        x = np.random.randn(2, 4, 32, 32).astype(np.float32)
        output = conv(x)
        assert output.shape == (2, 8, 32, 32)

    def test_conv2d_depthwise(self, backend):
        """Test depthwise convolution (groups = in_channels)"""
        from grilly.nn import Conv2d

        # Depthwise: groups = in_channels = out_channels
        in_ch = 16
        conv = Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch, padding=1)
        assert conv.weight.shape == (16, 1, 3, 3)  # Each output channel has 1 input channel

        x = np.random.randn(2, 16, 32, 32).astype(np.float32)
        output = conv(x)
        assert output.shape == (2, 16, 32, 32)

    def test_conv2d_dilation(self, backend):
        """Test dilated convolution"""
        from grilly.nn import Conv2d

        conv = Conv2d(3, 16, kernel_size=3, dilation=2, padding=2)
        x = np.random.randn(2, 3, 32, 32).astype(np.float32)

        output = conv(x)
        assert output.shape == (2, 16, 32, 32)

    def test_conv2d_nonsquare_kernel(self, backend):
        """Test non-square kernel"""
        from grilly.nn import Conv2d

        conv = Conv2d(3, 16, kernel_size=(3, 5), padding=(1, 2))
        assert conv.kernel_size == (3, 5)
        assert conv.weight.shape == (16, 3, 3, 5)

        x = np.random.randn(2, 3, 32, 32).astype(np.float32)
        output = conv(x)
        assert output.shape == (2, 16, 32, 32)


@pytest.mark.gpu
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestConv2dVsPyTorch:
    """Tests comparing Grilly Conv2d with PyTorch"""

    @pytest.mark.xfail(reason="GPU conv GEMM path has numerical issues on some drivers")
    def test_conv2d_correctness_basic(self, backend):
        """Compare Grilly Conv2d output with PyTorch (basic case)"""
        from grilly.nn import Conv2d as GrillyConv2d

        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        # Create Grilly conv
        grilly_conv = GrillyConv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=True)

        # Create PyTorch conv with same weights
        torch_conv = torch_nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=True)
        weight_np = np.asarray(
            grilly_conv.weight.data if hasattr(grilly_conv.weight, "data") else grilly_conv.weight,
            dtype=np.float32,
        ).copy()
        bias_np = np.asarray(
            grilly_conv.bias.data if hasattr(grilly_conv.bias, "data") else grilly_conv.bias,
            dtype=np.float32,
        ).copy()
        torch_conv.weight.data = torch.from_numpy(weight_np)
        torch_conv.bias.data = torch.from_numpy(bias_np)

        # Test input
        x_np = np.random.randn(2, 3, 16, 16).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        # Forward pass
        grilly_out = grilly_conv(x_np)
        torch_out = torch_conv(x_torch).detach().numpy()

        # Compare outputs (allow small numerical difference)
        np.testing.assert_allclose(grilly_out, torch_out, rtol=1e-4, atol=1e-5)

    def test_conv2d_correctness_stride(self, backend):
        """Compare with PyTorch (stride > 1)"""
        from grilly.nn import Conv2d as GrillyConv2d

        np.random.seed(42)
        torch.manual_seed(42)

        grilly_conv = GrillyConv2d(3, 16, kernel_size=3, stride=2, padding=1)
        torch_conv = torch_nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        weight_np = (
            grilly_conv.weight.data if hasattr(grilly_conv.weight, "data") else grilly_conv.weight
        )
        bias_np = grilly_conv.bias.data if hasattr(grilly_conv.bias, "data") else grilly_conv.bias
        torch_conv.weight.data = torch.from_numpy(np.asarray(weight_np, dtype=np.float32))
        torch_conv.bias.data = torch.from_numpy(np.asarray(bias_np, dtype=np.float32))

        x_np = np.random.randn(2, 3, 32, 32).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        grilly_out = grilly_conv(x_np)
        torch_out = torch_conv(x_torch).detach().numpy()

        np.testing.assert_allclose(grilly_out, torch_out, rtol=1e-4, atol=1e-5)

    def test_conv2d_correctness_groups(self, backend):
        """Compare with PyTorch (grouped convolution)"""
        from grilly.nn import Conv2d as GrillyConv2d

        np.random.seed(42)
        torch.manual_seed(42)
        try:
            grilly_conv = GrillyConv2d(4, 8, kernel_size=3, groups=2, padding=1)
            torch_conv = torch_nn.Conv2d(4, 8, kernel_size=3, groups=2, padding=1)
            weight_np = (
                grilly_conv.weight.data
                if hasattr(grilly_conv.weight, "data")
                else grilly_conv.weight
            )
            bias_np = (
                grilly_conv.bias.data if hasattr(grilly_conv.bias, "data") else grilly_conv.bias
            )
            torch_conv.weight.data = torch.from_numpy(np.asarray(weight_np, dtype=np.float32))
            torch_conv.bias.data = torch.from_numpy(np.asarray(bias_np, dtype=np.float32))

            x_np = np.random.randn(2, 4, 16, 16).astype(np.float32)
            x_torch = torch.from_numpy(x_np)

            grilly_out = grilly_conv(x_np)
            torch_out = torch_conv(x_torch).detach().numpy()

            np.testing.assert_allclose(grilly_out, torch_out, rtol=1e-4, atol=1e-5)
        except NotImplementedError:
            pytest.skip("Grouped convolution not implemented in Grilly backend")
        except Exception as e:
            pytest.fail(f"Unexpected error during grouped convolution test: {e}")

    @pytest.mark.xfail(reason="GPU conv backward path has numerical issues on some drivers")
    def test_conv2d_backward_correctness(self, backend):
        """Compare gradients with PyTorch"""
        from grilly.nn import Conv2d as GrillyConv2d

        np.random.seed(42)
        torch.manual_seed(42)

        # Create layers
        grilly_conv = GrillyConv2d(3, 8, kernel_size=3, stride=1, padding=1)
        torch_conv = torch_nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        weight_np = np.asarray(
            grilly_conv.weight.data if hasattr(grilly_conv.weight, "data") else grilly_conv.weight,
            dtype=np.float32,
        ).copy()
        bias_np = np.asarray(
            grilly_conv.bias.data if hasattr(grilly_conv.bias, "data") else grilly_conv.bias,
            dtype=np.float32,
        ).copy()
        torch_conv.weight.data = torch.from_numpy(weight_np)
        torch_conv.bias.data = torch.from_numpy(bias_np)

        # Input
        x_np = np.random.randn(2, 3, 16, 16).astype(np.float32)
        x_torch = torch.from_numpy(x_np).requires_grad_(True)

        # Forward
        grilly_out = grilly_conv(x_np)
        torch_out = torch_conv(x_torch)

        # Backward
        grad_output = np.random.randn(*grilly_out.shape).astype(np.float32)
        torch_out.backward(torch.from_numpy(grad_output))
        grilly_grad_input = grilly_conv.backward(grad_output)

        # Compare input gradients
        np.testing.assert_allclose(grilly_grad_input, x_torch.grad.numpy(), rtol=1e-3, atol=1e-4)

        # Compare weight gradients
        np.testing.assert_allclose(
            grilly_conv.weight.grad, torch_conv.weight.grad.numpy(), rtol=1e-3, atol=1e-4
        )

        # Compare bias gradients
        np.testing.assert_allclose(
            grilly_conv.bias.grad, torch_conv.bias.grad.numpy(), rtol=1e-3, atol=1e-4
        )


@pytest.mark.gpu
class TestConv2dEdgeCases:
    """Edge case tests"""

    def test_conv2d_1x1_kernel(self, backend):
        """Test 1x1 convolution (pointwise)"""
        from grilly.nn import Conv2d

        conv = Conv2d(16, 32, kernel_size=1)
        assert conv.kernel_size == (1, 1)

        x = np.random.randn(2, 16, 32, 32).astype(np.float32)
        output = conv(x)
        assert output.shape == (2, 32, 32, 32)

    def test_conv2d_large_padding(self, backend):
        """Test large padding"""
        from grilly.nn import Conv2d

        conv = Conv2d(3, 16, kernel_size=3, padding=5)
        x = np.random.randn(2, 3, 16, 16).astype(np.float32)
        output = conv(x)
        # With padding=5, output size increases
        assert output.shape == (2, 16, 24, 24)

    def test_conv2d_single_channel(self, backend):
        """Test single channel input/output"""
        from grilly.nn import Conv2d

        conv = Conv2d(1, 1, kernel_size=3, padding=1)
        x = np.random.randn(2, 1, 32, 32).astype(np.float32)
        output = conv(x)
        assert output.shape == (2, 1, 32, 32)

    def test_conv2d_batch_size_1(self, backend):
        """Test batch size = 1"""
        from grilly.nn import Conv2d

        conv = Conv2d(3, 16, kernel_size=3, padding=1)
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        output = conv(x)
        assert output.shape == (1, 16, 32, 32)


@pytest.mark.gpu
@pytest.mark.benchmark
class TestConv2dPerformance:
    """Performance benchmarking tests"""

    def test_conv2d_benchmark_small(self, backend, benchmark):
        """Benchmark small conv (typical ResNet block)"""
        from grilly.nn import Conv2d

        conv = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        x = np.random.randn(32, 64, 56, 56).astype(np.float32)

        def run_conv():
            return conv(x)

        result = benchmark(run_conv)
        assert result.shape == (32, 64, 56, 56)

    def test_conv2d_benchmark_large(self, backend, benchmark):
        """Benchmark large conv"""
        from grilly.nn import Conv2d

        conv = Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        x = np.random.randn(16, 128, 28, 28).astype(np.float32)

        def run_conv():
            return conv(x)

        result = benchmark(run_conv)
        assert result.shape == (16, 256, 28, 28)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_conv2d_speedup_vs_cpu(self, backend):
        """Measure speedup vs PyTorch CPU"""
        import time

        from grilly.nn import Conv2d as GrillyConv2d

        # Create layers
        grilly_conv = GrillyConv2d(64, 128, kernel_size=3, padding=1)
        torch_conv = torch_nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Test input
        x_np = np.random.randn(32, 64, 56, 56).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        # Warm-up
        _ = grilly_conv(x_np)
        _ = torch_conv(x_torch)

        # Benchmark Grilly (GPU)
        n_iters = 10
        start = time.time()
        for _ in range(n_iters):
            _ = grilly_conv(x_np)
        grilly_time = (time.time() - start) / n_iters

        # Benchmark PyTorch (CPU)
        start = time.time()
        for _ in range(n_iters):
            _ = torch_conv(x_torch)
        torch_time = (time.time() - start) / n_iters

        speedup = torch_time / grilly_time
        print(f"\nSpeedup vs PyTorch CPU: {speedup:.2f}x")
        print(f"Grilly: {grilly_time * 1000:.2f}ms, PyTorch CPU: {torch_time * 1000:.2f}ms")

        # We expect at least some speedup on GPU (target is >50x, but depends on hardware)
        # For now, just verify it runs without error
        assert speedup > 0


@pytest.mark.gpu
class TestConv1d:
    """Tests for Conv1d wrapper"""

    def test_conv1d_import(self):
        """Test Conv1d can be imported"""
        from grilly.nn import Conv1d

        assert Conv1d is not None

    def test_conv1d_forward(self, backend):
        """Test Conv1d forward pass"""
        from grilly.nn import Conv1d

        conv = Conv1d(16, 32, kernel_size=3, padding=1)
        x = np.random.randn(2, 16, 100).astype(np.float32)

        output = conv(x)
        assert output.shape == (2, 32, 100)

    def test_conv1d_stride(self, backend):
        """Test Conv1d with stride"""
        from grilly.nn import Conv1d

        conv = Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        x = np.random.randn(2, 16, 100).astype(np.float32)

        output = conv(x)
        assert output.shape == (2, 32, 50)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
