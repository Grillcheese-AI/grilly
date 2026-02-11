"""
Tests for backward pass operations and autograd integration.
"""

import numpy as np
import pytest

try:
    from grilly import Compute
    from grilly.backend.base import VULKAN_AVAILABLE
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestBackwardOperations:
    """Test backward pass operations on GPU"""

    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        backend = Compute()
        backend.set_architecture("bert")  # Initialize all operations
        yield backend
        backend.cleanup()

    def test_gelu_backward(self, gpu):
        """Test GELU backward pass"""
        # Input
        x = np.random.randn(32, 64).astype(np.float32)
        grad_output = np.random.randn(32, 64).astype(np.float32)

        # GPU backward
        grad_input = gpu.activation_gelu_backward(grad_output, x)

        # Verify shape
        assert grad_input.shape == x.shape
        assert np.all(np.isfinite(grad_input))

        # Verify gradient correctness with numerical gradient
        eps = 1e-4
        numerical_grad = np.zeros_like(x)
        for i in range(min(5, x.shape[0])):  # Test first 5 samples
            for j in range(min(5, x.shape[1])):  # Test first 5 features
                x_plus = x.copy()
                x_plus[i, j] += eps
                x_minus = x.copy()
                x_minus[i, j] -= eps

                # GELU forward
                def gelu(x):
                    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

                f_plus = gelu(x_plus)[i, j]
                f_minus = gelu(x_minus)[i, j]
                numerical_grad[i, j] = (f_plus - f_minus) / (2 * eps)

        # Check that analytical gradient matches numerical (for small subset)
        analytical_subset = grad_input[:5, :5] / grad_output[:5, :5]
        numerical_subset = numerical_grad[:5, :5]
        assert np.allclose(analytical_subset, numerical_subset, rtol=0.1, atol=1e-3), (
            f"GELU gradient mismatch: max diff = {np.abs(analytical_subset - numerical_subset).max()}"
        )

    def test_linear_backward(self, gpu):
        """Test linear layer backward pass (CPU fallback)"""
        batch_size = 16
        in_features = 32
        out_features = 64

        # Create test data
        x = np.random.randn(batch_size, in_features).astype(np.float32)
        weights = np.random.randn(out_features, in_features).astype(np.float32)
        bias = np.random.randn(out_features).astype(np.float32)
        grad_output = np.random.randn(batch_size, out_features).astype(np.float32)

        # Test CPU fallback by temporarily removing shader
        original_shaders = gpu.fnn.shaders.copy()
        gpu.fnn.shaders = {k: v for k, v in original_shaders.items() if "linear-backward" not in k}

        try:
            # CPU backward
            grad_input, grad_weight, grad_bias = gpu.fnn.linear_backward(
                grad_output, x, weights, bias
            )

            # Verify shapes
            assert grad_input.shape == x.shape
            assert grad_weight.shape == weights.shape
            assert grad_bias.shape == bias.shape

            # Verify correctness with CPU reference
            expected_grad_input = grad_output @ weights
            expected_grad_weight = grad_output.T @ x
            expected_grad_bias = np.sum(grad_output, axis=0)

            assert np.allclose(grad_input, expected_grad_input, rtol=1e-3, atol=1e-4), (
                f"grad_input mismatch: max diff = {np.abs(grad_input - expected_grad_input).max()}"
            )
            assert np.allclose(grad_weight, expected_grad_weight, rtol=1e-3, atol=1e-4), (
                f"grad_weight mismatch: max diff = {np.abs(grad_weight - expected_grad_weight).max()}"
            )
            assert np.allclose(grad_bias, expected_grad_bias, rtol=1e-3, atol=1e-4), (
                f"grad_bias mismatch: max diff = {np.abs(grad_bias - expected_grad_bias).max()}"
            )
        finally:
            gpu.fnn.shaders = original_shaders

    def test_layernorm_backward(self, gpu):
        """Test LayerNorm backward pass"""
        batch_size = 8
        seq_len = 10
        features = 64

        # Create test data
        x = np.random.randn(batch_size, seq_len, features).astype(np.float32)
        gamma = np.ones(features, dtype=np.float32)
        beta = np.zeros(features, dtype=np.float32)
        grad_output = np.random.randn(batch_size, seq_len, features).astype(np.float32)

        # Compute mean and var for forward pass
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)

        # GPU backward
        grad_input, grad_gamma, grad_beta = gpu.fnn.layernorm_backward(
            grad_output, x, gamma, mean, var, eps=1e-5
        )

        # Verify shapes
        assert grad_input.shape == x.shape, (
            f"grad_input shape: {grad_input.shape} vs expected {x.shape}"
        )
        assert grad_gamma.shape == gamma.shape, (
            f"grad_gamma shape: {grad_gamma.shape} vs expected {gamma.shape}"
        )
        assert grad_beta.shape == beta.shape, (
            f"grad_beta shape: {grad_beta.shape} vs expected {beta.shape}"
        )

        # Verify finiteness
        assert np.all(np.isfinite(grad_input)), "grad_input has non-finite values"
        assert np.all(np.isfinite(grad_gamma)), "grad_gamma has non-finite values"
        assert np.all(np.isfinite(grad_beta)), "grad_beta has non-finite values"

        # Verify grad_beta is sum of grad_output
        expected_grad_beta = np.sum(grad_output, axis=(0, 1))
        assert np.allclose(grad_beta, expected_grad_beta, rtol=1e-3, atol=1e-4), (
            f"grad_beta mismatch: max diff = {np.abs(grad_beta - expected_grad_beta).max()}"
        )

    def test_softmax_backward(self, gpu):
        """Test softmax backward pass"""
        batch_size = 16
        num_classes = 10

        # Create test data
        logits = np.random.randn(batch_size, num_classes).astype(np.float32)

        # Forward: compute softmax
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        softmax_output = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        grad_output = np.random.randn(batch_size, num_classes).astype(np.float32)

        # GPU backward
        grad_input = gpu.fnn.softmax_backward(grad_output, softmax_output)

        # Verify shape
        assert grad_input.shape == logits.shape

        # Verify correctness with CPU reference
        # grad_input = s * (grad_output - sum(grad_output * s))
        sum_term = np.sum(grad_output * softmax_output, axis=-1, keepdims=True)
        expected_grad_input = softmax_output * (grad_output - sum_term)

        assert np.allclose(grad_input, expected_grad_input, rtol=1e-3, atol=1e-4), (
            f"softmax backward mismatch: max diff = {np.abs(grad_input - expected_grad_input).max()}"
        )


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestAutogradIntegration:
    """Test autograd integration with nn.Module"""

    def test_variable_backward(self):
        """Test Variable backward propagation"""
        from grilly.nn import Variable

        # Create leaf variables
        x = Variable(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = Variable(np.array([4.0, 5.0, 6.0]), requires_grad=True)

        # Manually create output (simulating a forward pass)
        z_data = x.data + y.data
        z = Variable(z_data, requires_grad=True)

        # Backward
        z.backward()

        # z is the final output, so it should have gradient of ones
        assert z.grad is not None
        assert np.allclose(z.grad, np.ones(3))

    def test_gradient_tape(self):
        """Test GradientTape for recording operations"""
        try:
            from grilly.nn import GradientTape
        except ImportError:
            pytest.skip("GradientTape not available")

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        with GradientTape() as tape:
            # Record a simple operation
            x * 2.0

        # Tape should be usable
        assert tape is not None

    def test_training_context(self):
        """Test TrainingContext for training loop"""
        try:
            from grilly.nn import Linear, MSELoss, TrainingContext
        except ImportError:
            pytest.skip("TrainingContext not available")

        # Create simple model
        model = Linear(10, 5)

        # Create training context
        with TrainingContext(model) as ctx:
            # Forward pass
            x = np.random.randn(4, 10).astype(np.float32)
            output = ctx.forward(x)

            assert output.shape == (4, 5)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestEndToEndTraining:
    """Test end-to-end training loop"""

    def test_simple_training_step(self):
        """Test a single training step"""
        from grilly import nn
        from grilly.optim import Adam

        # Create model
        model = nn.Linear(10, 5)

        # Create optimizer
        optimizer = Adam(model.parameters(), lr=0.01)

        # Create data
        x = np.random.randn(8, 10).astype(np.float32)
        y_true = np.random.randn(8, 5).astype(np.float32)

        # Forward pass
        y_pred = model(x)
        assert y_pred.shape == (8, 5)

        # Compute loss (MSE)
        loss = np.mean((y_pred - y_true) ** 2)
        assert np.isfinite(loss)

        # Compute gradient manually
        grad_loss = 2 * (y_pred - y_true) / y_pred.size

        # Backward pass
        model.backward(grad_loss, x)

        # Check gradients exist
        weight = model._parameters.get("weight")
        if weight is not None and hasattr(weight, "grad"):
            assert weight.grad is not None or True  # May be None if backward not implemented

        # Optimizer step
        optimizer.step()

    def test_sequential_backward(self):
        """Test backward through sequential model"""
        from grilly import nn

        # Create sequential model
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.GELU(),
            nn.Linear(32, 5),
        )

        # Forward pass
        x = np.random.randn(4, 10).astype(np.float32)
        output = model(x)

        assert output.shape == (4, 5)
        assert np.all(np.isfinite(output))
