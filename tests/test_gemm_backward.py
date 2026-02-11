"""
Tests for GEMM-based backward passes.
Validates correctness and performance of GEMM optimizations.
"""

import numpy as np
import pytest


@pytest.mark.gpu
class TestLinearGEMMBackward:
    """Test Linear layer backward pass with GEMM optimization"""

    def test_linear_backward_gemm_small(self):
        """Test GEMM backward with small problem (may use fallback shader)"""
        from grilly.nn import Linear

        batch = 4
        in_features = 8
        out_features = 16

        # Create linear layer
        linear = Linear(in_features, out_features, bias=True)

        # Forward pass
        x = np.random.randn(batch, in_features).astype(np.float32)
        linear.forward(x)

        # Backward pass
        grad_output = np.random.randn(batch, out_features).astype(np.float32)
        grad_input = linear.backward(grad_output, x)

        # Verify shapes
        assert grad_input.shape == (batch, in_features)
        assert linear.weight.grad.shape == (out_features, in_features)
        assert linear.bias.grad.shape == (out_features,)

        # Verify gradients are not NaN or Inf
        assert not np.any(np.isnan(grad_input))
        assert not np.any(np.isinf(grad_input))
        assert not np.any(np.isnan(linear.weight.grad))
        assert not np.any(np.isnan(linear.bias.grad))

    def test_linear_backward_gemm_large(self):
        """Test GEMM backward with large problem (should use GEMM path)"""
        from grilly.nn import Linear

        # Large problem size to trigger GEMM path (batch * in_features >= 4096)
        batch = 64
        in_features = 128
        out_features = 256

        # Create linear layer
        linear = Linear(in_features, out_features, bias=True)

        # Forward pass
        x = np.random.randn(batch, in_features).astype(np.float32)
        linear.forward(x)

        # Backward pass
        grad_output = np.random.randn(batch, out_features).astype(np.float32)
        grad_input = linear.backward(grad_output, x)

        # Verify shapes
        assert grad_input.shape == (batch, in_features)
        assert linear.weight.grad.shape == (out_features, in_features)
        assert linear.bias.grad.shape == (out_features,)

        # Compute reference gradients (CPU)
        weight = linear.weight.data if hasattr(linear.weight, "data") else np.asarray(linear.weight)
        grad_input_ref = grad_output @ weight
        grad_weight_ref = grad_output.T @ x
        grad_bias_ref = np.sum(grad_output, axis=0)

        # Verify correctness
        np.testing.assert_allclose(grad_input, grad_input_ref, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(linear.weight.grad, grad_weight_ref, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(linear.bias.grad, grad_bias_ref, rtol=1e-4, atol=1e-5)

    def test_linear_backward_gemm_no_bias(self):
        """Test GEMM backward without bias"""
        from grilly.nn import Linear

        batch = 64
        in_features = 128
        out_features = 256

        # Create linear layer without bias
        linear = Linear(in_features, out_features, bias=False)

        # Forward pass
        x = np.random.randn(batch, in_features).astype(np.float32)
        linear.forward(x)

        # Backward pass
        grad_output = np.random.randn(batch, out_features).astype(np.float32)
        grad_input = linear.backward(grad_output, x)

        # Verify shapes
        assert grad_input.shape == (batch, in_features)
        assert linear.weight.grad.shape == (out_features, in_features)
        assert linear.bias is None

        # Compute reference gradients
        weight = linear.weight.data if hasattr(linear.weight, "data") else np.asarray(linear.weight)
        grad_input_ref = grad_output @ weight
        grad_weight_ref = grad_output.T @ x

        # Verify correctness
        np.testing.assert_allclose(grad_input, grad_input_ref, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(linear.weight.grad, grad_weight_ref, rtol=1e-4, atol=1e-5)

    def test_linear_backward_gemm_3d_input(self):
        """Test GEMM backward with 3D input (batch, seq, features)"""
        from grilly.nn import Linear

        # Use larger sizes to ensure GEMM path is taken (batch*seq*in_features >= 4096)
        batch = 64
        seq_len = 32
        in_features = 128
        out_features = 256

        # Create linear layer
        linear = Linear(in_features, out_features, bias=True)

        # Forward pass with 3D input
        x = np.random.randn(batch, seq_len, in_features).astype(np.float32)
        linear.forward(x)

        # Zero gradients before backward
        linear.weight.grad = None
        linear.bias.grad = None

        # Backward pass
        grad_output = np.random.randn(batch, seq_len, out_features).astype(np.float32)
        grad_input = linear.backward(grad_output, x)

        # Verify shapes
        assert grad_input.shape == (batch, seq_len, in_features)
        assert linear.weight.grad.shape == (out_features, in_features)
        assert linear.bias.grad.shape == (out_features,)

        # Flatten for reference computation
        x_flat = x.reshape(-1, in_features)
        grad_output_flat = grad_output.reshape(-1, out_features)

        # Compute reference gradients
        weight = linear.weight.data if hasattr(linear.weight, "data") else np.asarray(linear.weight)
        grad_input_ref = (grad_output_flat @ weight).reshape(batch, seq_len, in_features)
        grad_weight_ref = grad_output_flat.T @ x_flat
        grad_bias_ref = np.sum(grad_output_flat, axis=0)

        # Verify correctness (relaxed tolerance for GEMM fp32 precision)
        np.testing.assert_allclose(grad_input, grad_input_ref, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(linear.weight.grad, grad_weight_ref, rtol=1e-3, atol=2e-5)
        np.testing.assert_allclose(linear.bias.grad, grad_bias_ref, rtol=1e-4, atol=1e-5)

    def test_linear_backward_gradient_accumulation(self):
        """Test gradient accumulation across multiple backward passes"""
        from grilly.nn import Linear

        batch = 64
        in_features = 128
        out_features = 256

        # Create linear layer
        linear = Linear(in_features, out_features, bias=True)

        # First forward and backward
        x1 = np.random.randn(batch, in_features).astype(np.float32)
        linear.forward(x1)
        grad_output1 = np.random.randn(batch, out_features).astype(np.float32)
        linear.backward(grad_output1, x1)

        # Save first gradients
        grad_weight1 = linear.weight.grad.copy()
        grad_bias1 = linear.bias.grad.copy()

        # Second forward and backward (should accumulate)
        x2 = np.random.randn(batch, in_features).astype(np.float32)
        linear.forward(x2)
        grad_output2 = np.random.randn(batch, out_features).astype(np.float32)
        linear.backward(grad_output2, x2)

        # Verify gradients accumulated
        linear.weight.data if hasattr(linear.weight, "data") else np.asarray(linear.weight)
        grad_weight_ref = grad_weight1 + grad_output2.T @ x2
        grad_bias_ref = grad_bias1 + np.sum(grad_output2, axis=0)

        np.testing.assert_allclose(linear.weight.grad, grad_weight_ref, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(linear.bias.grad, grad_bias_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
class TestMultiheadAttentionGEMMBackward:
    """Test MultiheadAttention backward pass with GEMM-backed Linear layers"""

    def test_attention_backward_uses_gemm(self):
        """Test that attention Q/K/V projections use GEMM in backward"""
        from grilly.nn import MultiheadAttention

        batch = 32
        seq_len = 16
        embed_dim = 256
        num_heads = 8

        # Create attention module
        attn = MultiheadAttention(embed_dim, num_heads)

        # Forward pass
        query = np.random.randn(batch, seq_len, embed_dim).astype(np.float32)
        key = np.random.randn(batch, seq_len, embed_dim).astype(np.float32)
        value = np.random.randn(batch, seq_len, embed_dim).astype(np.float32)
        output, attn_weights = attn.forward(query, key, value)

        # Backward pass
        grad_output = np.random.randn(batch, seq_len, embed_dim).astype(np.float32)
        grad_query, grad_key, grad_value = attn.backward(grad_output, query, key, value)

        # Verify shapes
        assert grad_query.shape == query.shape
        assert grad_key.shape == key.shape
        assert grad_value.shape == value.shape

        # Verify projections have gradients
        assert attn.q_proj.weight.grad is not None
        assert attn.k_proj.weight.grad is not None
        assert attn.v_proj.weight.grad is not None
        assert attn.out_proj.weight.grad is not None

        # Verify gradients are reasonable
        assert not np.any(np.isnan(grad_query))
        assert not np.any(np.isnan(grad_key))
        assert not np.any(np.isnan(grad_value))


@pytest.mark.gpu
class TestGEMMPerformance:
    """Performance tests to verify GEMM is faster than fallback"""

    @pytest.mark.benchmark
    def test_gemm_vs_cpu_performance(self):
        """Benchmark GEMM vs CPU fallback (requires manual verification)"""
        import time

        from grilly.nn import Linear

        # Large problem size
        batch = 256
        in_features = 512
        out_features = 1024

        # Create linear layer
        linear = Linear(in_features, out_features, bias=True)

        # Warm-up
        x = np.random.randn(batch, in_features).astype(np.float32)
        linear.forward(x)
        grad_output = np.random.randn(batch, out_features).astype(np.float32)
        linear.backward(grad_output, x)

        # Benchmark GPU backward
        num_iterations = 10
        start = time.perf_counter()
        for _ in range(num_iterations):
            x = np.random.randn(batch, in_features).astype(np.float32)
            linear.forward(x)
            grad_output = np.random.randn(batch, out_features).astype(np.float32)
            linear.backward(grad_output, x)
        gpu_time = (time.perf_counter() - start) / num_iterations

        # Benchmark CPU reference (ensure weight is a numpy ndarray for .T)
        start = time.perf_counter()
        for _ in range(num_iterations):
            x = np.random.randn(batch, in_features).astype(np.float32)
            w = linear.weight.data if hasattr(linear.weight, "data") else linear.weight
            weight = np.asarray(w, dtype=np.float32).copy()
            x @ weight.T
            grad_output = np.random.randn(batch, out_features).astype(np.float32)
            grad_output @ weight
            grad_output.T @ x
        cpu_time = (time.perf_counter() - start) / num_iterations

        print(f"\nGEMM backward (GPU): {gpu_time * 1000:.2f} ms")
        print(f"CPU backward:        {cpu_time * 1000:.2f} ms")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")

        # GEMM should be faster for large problems (but test may vary by hardware)
        # We don't assert here since it's hardware-dependent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
