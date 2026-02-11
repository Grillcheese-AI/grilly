"""
Tests for AdamW optimizer.
Validates correctness against PyTorch and verifies decoupled weight decay.
"""

import numpy as np
import pytest

# Try to import PyTorch for reference comparisons
try:
    import torch
    import torch.optim as torch_optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestAdamWBasic:
    """Basic AdamW functionality tests"""

    def test_adamw_import(self):
        """Test AdamW can be imported"""
        from grilly.optim import AdamW

        assert AdamW is not None

    def test_adamw_init(self):
        """Test AdamW initialization"""
        from grilly.nn import Parameter
        from grilly.optim import AdamW

        param = Parameter(np.random.randn(10, 10).astype(np.float32))
        optimizer = AdamW([param], lr=0.001, weight_decay=0.01)

        assert optimizer.defaults["lr"] == 0.001
        assert optimizer.defaults["betas"] == (0.9, 0.999)
        assert optimizer.defaults["eps"] == 1e-8
        assert optimizer.defaults["weight_decay"] == 0.01
        assert not optimizer.defaults["amsgrad"]

    def test_adamw_step(self):
        """Test AdamW performs update step"""
        from grilly.nn import Parameter
        from grilly.optim import AdamW

        param = Parameter(np.ones((5, 5), dtype=np.float32))
        param.grad = np.ones_like(param.data) * 0.1  # Small gradient

        optimizer = AdamW([param], lr=0.01, weight_decay=0.0)  # No weight decay for simple test

        param_before = np.asarray(param.data, dtype=np.float32).copy()
        optimizer.step()
        param_after = np.asarray(param.data, dtype=np.float32)

        # Parameters should have changed
        assert not np.allclose(param_before, param_after)
        # With positive gradient, parameters should decrease
        assert np.all(param_after < param_before)

    def test_adamw_state_initialization(self):
        """Test AdamW initializes optimizer state"""
        from grilly.nn import Parameter
        from grilly.optim import AdamW

        param = Parameter(np.random.randn(5, 5).astype(np.float32))
        param.grad = np.random.randn(5, 5).astype(np.float32)

        optimizer = AdamW([param], lr=0.001)
        optimizer.step()

        param_id = id(param)
        state = optimizer.state[param_id]

        assert "step" in state
        assert "exp_avg" in state
        assert "exp_avg_sq" in state
        assert state["step"] == 1
        assert state["exp_avg"].shape == param.shape
        assert state["exp_avg_sq"].shape == param.shape

    def test_adamw_amsgrad(self):
        """Test AdamW with AMSGrad variant"""
        from grilly.nn import Parameter
        from grilly.optim import AdamW

        param = Parameter(np.random.randn(5, 5).astype(np.float32))
        param.grad = np.random.randn(5, 5).astype(np.float32)

        optimizer = AdamW([param], lr=0.001, amsgrad=True)
        optimizer.step()

        param_id = id(param)
        state = optimizer.state[param_id]

        assert "max_exp_avg_sq" in state


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestAdamWVsPyTorch:
    """Tests comparing Grilly AdamW with PyTorch"""

    def test_adamw_decoupled_weight_decay(self):
        """Verify AdamW uses decoupled weight decay (vs Adam's coupled)"""
        from grilly.nn import Parameter as GrillyParameter
        from grilly.optim import AdamW as GrillyAdamW

        np.random.seed(42)
        torch.manual_seed(42)

        # Create identical parameters
        init_param = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        grilly_param = GrillyParameter(init_param.copy())
        torch_param = torch.nn.Parameter(torch.from_numpy(init_param.copy()))

        # Create optimizers with weight decay
        grilly_opt = GrillyAdamW(
            [grilly_param], lr=0.1, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8
        )
        torch_opt = torch_optim.AdamW(
            [torch_param], lr=0.1, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8
        )

        # Same gradient
        grad = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        grilly_param.grad = grad.copy()
        torch_param.grad = torch.from_numpy(grad.copy())

        # Step
        grilly_opt.step()
        torch_opt.step()

        # Compare results
        np.testing.assert_allclose(
            grilly_param.data,
            torch_param.data.numpy(),
            rtol=1e-4,
            atol=1e-5,
            err_msg="AdamW parameters should match PyTorch after one step",
        )

    def test_adamw_multiple_steps(self):
        """Test AdamW matches PyTorch over multiple optimization steps"""
        from grilly.nn import Parameter as GrillyParameter
        from grilly.optim import AdamW as GrillyAdamW

        np.random.seed(42)
        torch.manual_seed(42)

        # Larger parameter for more robust test
        init_param = np.random.randn(10, 10).astype(np.float32)

        grilly_param = GrillyParameter(init_param.copy())
        torch_param = torch.nn.Parameter(torch.from_numpy(init_param.copy()))

        # Create optimizers
        grilly_opt = GrillyAdamW([grilly_param], lr=0.01, weight_decay=0.01)
        torch_opt = torch_optim.AdamW([torch_param], lr=0.01, weight_decay=0.01)

        # Run 10 steps
        for i in range(10):
            # Random gradient each step
            grad = np.random.randn(10, 10).astype(np.float32) * 0.1
            grilly_param.grad = grad.copy()
            torch_param.grad = torch.from_numpy(grad.copy())

            grilly_opt.step()
            torch_opt.step()

            # Verify match at each step
            np.testing.assert_allclose(
                grilly_param.data,
                torch_param.data.numpy(),
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"AdamW mismatch at step {i + 1}",
            )

    def test_adamw_vs_adam_weight_decay(self):
        """Verify AdamW behaves differently from Adam with weight decay"""
        from grilly.nn import Parameter as GrillyParameter
        from grilly.optim import Adam as GrillyAdam
        from grilly.optim import AdamW as GrillyAdamW

        np.random.seed(42)

        # Same initial parameters
        init_param = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        adamw_param = GrillyParameter(init_param.copy())
        adam_param = GrillyParameter(init_param.copy())

        # Same hyperparameters
        lr, wd = 0.1, 0.1
        adamw_opt = GrillyAdamW([adamw_param], lr=lr, weight_decay=wd, betas=(0.9, 0.999))
        adam_opt = GrillyAdam([adam_param], lr=lr, weight_decay=wd, betas=(0.9, 0.999))

        # Same gradient
        grad = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        adamw_param.grad = grad.copy()
        adam_param.grad = grad.copy()

        # Step
        adamw_opt.step()
        adam_opt.step()

        # Results should be DIFFERENT (AdamW has decoupled weight decay)
        assert not np.allclose(adamw_param.data, adam_param.data, rtol=1e-5)

    def test_adamw_no_weight_decay(self):
        """Test AdamW with weight_decay=0 matches Adam"""
        from grilly.nn import Parameter as GrillyParameter
        from grilly.optim import Adam as GrillyAdam
        from grilly.optim import AdamW as GrillyAdamW

        np.random.seed(42)

        init_param = np.random.randn(5, 5).astype(np.float32)

        adamw_param = GrillyParameter(init_param.copy())
        adam_param = GrillyParameter(init_param.copy())

        adamw_opt = GrillyAdamW([adamw_param], lr=0.01, weight_decay=0.0)
        adam_opt = GrillyAdam([adam_param], lr=0.01, weight_decay=0.0)

        # Multiple steps
        for _ in range(5):
            grad = np.random.randn(5, 5).astype(np.float32) * 0.1
            adamw_param.grad = grad.copy()
            adam_param.grad = grad.copy()

            adamw_opt.step()
            adam_opt.step()

        # Should match when weight_decay=0
        np.testing.assert_allclose(
            adamw_param.data,
            adam_param.data,
            rtol=1e-4,
            atol=1e-5,
            err_msg="AdamW with weight_decay=0 should match Adam",
        )

    def test_adamw_bias_correction(self):
        """Test AdamW bias correction matches PyTorch"""
        from grilly.nn import Parameter as GrillyParameter
        from grilly.optim import AdamW as GrillyAdamW

        np.random.seed(42)
        torch.manual_seed(42)

        init_param = np.ones((3, 3), dtype=np.float32)

        grilly_param = GrillyParameter(init_param.copy())
        torch_param = torch.nn.Parameter(torch.from_numpy(init_param.copy()))

        # Use different betas to test bias correction
        grilly_opt = GrillyAdamW([grilly_param], lr=0.01, betas=(0.8, 0.9), weight_decay=0.01)
        torch_opt = torch_optim.AdamW([torch_param], lr=0.01, betas=(0.8, 0.9), weight_decay=0.01)

        # First step (bias correction most significant)
        grad = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
        grilly_param.grad = grad.copy()
        torch_param.grad = torch.from_numpy(grad.copy())

        grilly_opt.step()
        torch_opt.step()

        np.testing.assert_allclose(
            grilly_param.data, torch_param.data.numpy(), rtol=1e-4, atol=1e-5
        )


class TestAdamWEdgeCases:
    """Edge case tests for AdamW"""

    def test_adamw_zero_gradient(self):
        """Test AdamW with zero gradients (no NaN/Inf)"""
        from grilly.nn import Parameter
        from grilly.optim import AdamW

        param = Parameter(np.ones((5, 5), dtype=np.float32))
        param.grad = np.zeros_like(param.data)

        optimizer = AdamW([param], lr=0.01, weight_decay=0.01)

        param_before = np.asarray(param.data, dtype=np.float32).copy()
        optimizer.step()
        param_after = np.asarray(param.data, dtype=np.float32)

        # With weight decay, parameters should still change even with zero gradient
        assert not np.allclose(param_before, param_after)
        assert not np.any(np.isnan(param_after))
        assert not np.any(np.isinf(param_after))

    def test_adamw_large_gradient(self):
        """Test AdamW handles large gradients without overflow"""
        from grilly.nn import Parameter
        from grilly.optim import AdamW

        param = Parameter(np.ones((5, 5), dtype=np.float32))
        param.grad = np.ones_like(param.data) * 1000.0  # Large gradient

        optimizer = AdamW([param], lr=0.001, weight_decay=0.01)
        optimizer.step()

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(param.data))
        assert not np.any(np.isinf(param.data))

    def test_adamw_multiple_param_groups(self):
        """Test AdamW with multiple parameter groups"""
        from grilly.nn import Parameter
        from grilly.optim import AdamW

        param1 = Parameter(np.random.randn(5, 5).astype(np.float32))
        param2 = Parameter(np.random.randn(3, 3).astype(np.float32))

        param1.grad = np.random.randn(5, 5).astype(np.float32) * 0.1
        param2.grad = np.random.randn(3, 3).astype(np.float32) * 0.1

        optimizer = AdamW([param1, param2], lr=0.01, weight_decay=0.01)

        param1_before = np.asarray(param1.data, dtype=np.float32).copy()
        param2_before = np.asarray(param2.data, dtype=np.float32).copy()

        optimizer.step()

        # Both parameters should update
        assert not np.allclose(param1.data, param1_before)
        assert not np.allclose(param2.data, param2_before)

    def test_adamw_gradient_clearing(self):
        """Test AdamW clears gradients after step"""
        from grilly.nn import Parameter
        from grilly.optim import AdamW

        param = Parameter(np.random.randn(5, 5).astype(np.float32))
        param.grad = np.random.randn(5, 5).astype(np.float32)

        optimizer = AdamW([param], lr=0.01)
        optimizer.step()

        # Gradient should be cleared after step
        assert param.grad is None or np.all(param.grad == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
