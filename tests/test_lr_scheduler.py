"""
Tests for Learning Rate Schedulers.
Validates correctness against PyTorch schedulers.
"""

import numpy as np
import pytest

# Try to import PyTorch for reference comparisons
try:
    import torch
    import torch.optim as torch_optim
    from torch.optim.lr_scheduler import CosineAnnealingLR as TorchCosineAnnealingLR
    from torch.optim.lr_scheduler import OneCycleLR as TorchOneCycleLR
    from torch.optim.lr_scheduler import ReduceLROnPlateau as TorchReduceLROnPlateau
    from torch.optim.lr_scheduler import StepLR as TorchStepLR

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestSchedulersBasic:
    """Basic scheduler functionality tests"""

    def test_steplr_import(self):
        """Test StepLR can be imported"""
        from grilly.optim import StepLR

        assert StepLR is not None

    def test_cosine_import(self):
        """Test CosineAnnealingLR can be imported"""
        from grilly.optim import CosineAnnealingLR

        assert CosineAnnealingLR is not None

    def test_plateau_import(self):
        """Test ReduceLROnPlateau can be imported"""
        from grilly.optim import ReduceLROnPlateau

        assert ReduceLROnPlateau is not None

    def test_onecycle_import(self):
        """Test OneCycleLR can be imported"""
        from grilly.optim import OneCycleLR

        assert OneCycleLR is not None

    def test_steplr_init(self):
        """Test StepLR initialization"""
        from grilly.nn import Parameter
        from grilly.optim import SGD, StepLR

        param = Parameter(np.random.randn(10, 10).astype(np.float32))
        optimizer = SGD([param], lr=0.1)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        assert scheduler.step_size == 5
        assert scheduler.gamma == 0.1
        assert len(scheduler.base_lrs) == 1
        assert scheduler.base_lrs[0] == 0.1

    def test_steplr_decay(self):
        """Test StepLR decays learning rate"""
        from grilly.nn import Parameter
        from grilly.optim import SGD, StepLR

        param = Parameter(np.random.randn(10, 10).astype(np.float32))
        optimizer = SGD([param], lr=0.1)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

        # Initial LR should be 0.1
        assert optimizer.param_groups[0]["lr"] == 0.1

        # After 3 steps, should decay
        scheduler.step()
        scheduler.step()
        scheduler.step()
        assert abs(optimizer.param_groups[0]["lr"] - 0.01) < 1e-6

        # Another 3 steps, decay again
        scheduler.step()
        scheduler.step()
        scheduler.step()
        assert abs(optimizer.param_groups[0]["lr"] - 0.001) < 1e-6


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestStepLRVsPyTorch:
    """Test StepLR against PyTorch"""

    def test_steplr_matches_pytorch(self):
        """Verify StepLR matches PyTorch exactly"""
        from grilly.nn import Parameter as GrillyParameter
        from grilly.optim import SGD as GrillySGD
        from grilly.optim import StepLR as GrillyStepLR

        # Create identical optimizers
        param_init = np.random.randn(5, 5).astype(np.float32)
        grilly_param = GrillyParameter(param_init.copy())
        torch_param = torch.nn.Parameter(torch.from_numpy(param_init.copy()))

        grilly_opt = GrillySGD([grilly_param], lr=0.1)
        torch_opt = torch_optim.SGD([torch_param], lr=0.1)

        grilly_scheduler = GrillyStepLR(grilly_opt, step_size=5, gamma=0.5)
        torch_scheduler = TorchStepLR(torch_opt, step_size=5, gamma=0.5)

        # Test over 20 steps
        for i in range(20):
            grilly_scheduler.step()
            torch_scheduler.step()

            grilly_lr = grilly_opt.param_groups[0]["lr"]
            torch_lr = torch_opt.param_groups[0]["lr"]

            assert abs(grilly_lr - torch_lr) < 1e-6, (
                f"LR mismatch at step {i}: {grilly_lr} vs {torch_lr}"
            )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestCosineAnnealingLRVsPyTorch:
    """Test CosineAnnealingLR against PyTorch"""

    def test_cosine_matches_pytorch(self):
        """Verify CosineAnnealingLR matches PyTorch"""
        from grilly.nn import Parameter as GrillyParameter
        from grilly.optim import SGD as GrillySGD
        from grilly.optim import CosineAnnealingLR as GrillyCosineAnnealingLR

        param_init = np.random.randn(5, 5).astype(np.float32)
        grilly_param = GrillyParameter(param_init.copy())
        torch_param = torch.nn.Parameter(torch.from_numpy(param_init.copy()))

        grilly_opt = GrillySGD([grilly_param], lr=0.1)
        torch_opt = torch_optim.SGD([torch_param], lr=0.1)

        grilly_scheduler = GrillyCosineAnnealingLR(grilly_opt, T_max=10, eta_min=0.001)
        torch_scheduler = TorchCosineAnnealingLR(torch_opt, T_max=10, eta_min=0.001)

        # Test over full cycle
        for i in range(20):
            grilly_scheduler.step()
            torch_scheduler.step()

            grilly_lr = grilly_opt.param_groups[0]["lr"]
            torch_lr = torch_opt.param_groups[0]["lr"]

            assert abs(grilly_lr - torch_lr) < 1e-6, (
                f"LR mismatch at step {i}: {grilly_lr} vs {torch_lr}"
            )

    def test_cosine_min_max(self):
        """Test CosineAnnealingLR reaches min and max values"""
        from grilly.nn import Parameter
        from grilly.optim import SGD, CosineAnnealingLR

        param = Parameter(np.random.randn(5, 5).astype(np.float32))
        optimizer = SGD([param], lr=0.1)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)

        lrs = []
        for i in range(20):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Should reach maximum at start
        assert abs(max(lrs) - 0.1) < 1e-6

        # Should reach near minimum at T_max/2
        assert min(lrs[:10]) < 0.01


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestReduceLROnPlateauVsPyTorch:
    """Test ReduceLROnPlateau against PyTorch"""

    def test_plateau_matches_pytorch(self):
        """Verify ReduceLROnPlateau matches PyTorch"""
        from grilly.nn import Parameter as GrillyParameter
        from grilly.optim import SGD as GrillySGD
        from grilly.optim import ReduceLROnPlateau as GrillyReduceLROnPlateau

        param_init = np.random.randn(5, 5).astype(np.float32)
        grilly_param = GrillyParameter(param_init.copy())
        torch_param = torch.nn.Parameter(torch.from_numpy(param_init.copy()))

        grilly_opt = GrillySGD([grilly_param], lr=0.1)
        torch_opt = torch_optim.SGD([torch_param], lr=0.1)

        grilly_scheduler = GrillyReduceLROnPlateau(grilly_opt, mode="min", factor=0.1, patience=5)
        torch_scheduler = TorchReduceLROnPlateau(torch_opt, mode="min", factor=0.1, patience=5)

        # Simulate training with plateauing metric
        metrics = [1.0, 0.9, 0.8, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]

        for metric in metrics:
            grilly_scheduler.step(metric)
            torch_scheduler.step(metric)

            grilly_lr = grilly_opt.param_groups[0]["lr"]
            torch_lr = torch_opt.param_groups[0]["lr"]

            assert abs(grilly_lr - torch_lr) < 1e-6, f"LR mismatch: {grilly_lr} vs {torch_lr}"

    def test_plateau_reduces_on_plateau(self):
        """Test ReduceLROnPlateau actually reduces LR on plateau"""
        from grilly.nn import Parameter
        from grilly.optim import SGD, ReduceLROnPlateau

        param = Parameter(np.random.randn(5, 5).astype(np.float32))
        optimizer = SGD([param], lr=0.1)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        initial_lr = optimizer.param_groups[0]["lr"]

        # Simulate plateau (same loss for patience+1 epochs)
        for _ in range(5):
            scheduler.step(0.5)

        # LR should have been reduced
        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestOneCycleLRVsPyTorch:
    """Test OneCycleLR against PyTorch"""

    def test_onecycle_matches_pytorch(self):
        """Verify OneCycleLR matches PyTorch"""
        from grilly.nn import Parameter as GrillyParameter
        from grilly.optim import SGD as GrillySGD
        from grilly.optim import OneCycleLR as GrillyOneCycleLR

        param_init = np.random.randn(5, 5).astype(np.float32)
        grilly_param = GrillyParameter(param_init.copy())
        torch_param = torch.nn.Parameter(torch.from_numpy(param_init.copy()))

        grilly_opt = GrillySGD([grilly_param], lr=0.01, momentum=0.9)
        torch_opt = torch_optim.SGD([torch_param], lr=0.01, momentum=0.9)

        total_steps = 100
        grilly_scheduler = GrillyOneCycleLR(grilly_opt, max_lr=0.1, total_steps=total_steps)
        torch_scheduler = TorchOneCycleLR(torch_opt, max_lr=0.1, total_steps=total_steps)

        # Test over full cycle
        for i in range(total_steps):
            grilly_scheduler.step()
            torch_scheduler.step()

            grilly_lr = grilly_opt.param_groups[0]["lr"]
            torch_lr = torch_opt.param_groups[0]["lr"]

            # Allow small tolerance for floating-point differences
            assert abs(grilly_lr - torch_lr) < 1e-5, (
                f"LR mismatch at step {i}: {grilly_lr} vs {torch_lr}"
            )

    def test_onecycle_reaches_max(self):
        """Test OneCycleLR reaches max_lr"""
        from grilly.nn import Parameter
        from grilly.optim import SGD, OneCycleLR

        param = Parameter(np.random.randn(5, 5).astype(np.float32))
        optimizer = SGD([param], lr=0.01, momentum=0.9)
        scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=100, pct_start=0.3)

        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Should reach max_lr around pct_start of cycle
        assert max(lrs) >= 0.099  # Allow small tolerance

    def test_onecycle_with_adam(self):
        """Test OneCycleLR with Adam optimizer (uses betas instead of momentum)"""
        from grilly.nn import Parameter
        from grilly.optim import Adam, OneCycleLR

        param = Parameter(np.random.randn(5, 5).astype(np.float32))
        optimizer = Adam([param], lr=0.001)
        scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=50)

        # Should not raise error
        for _ in range(50):
            scheduler.step()

        # Beta1 should have been cycled
        assert "betas" in optimizer.param_groups[0]


class TestSchedulersEdgeCases:
    """Edge case tests for schedulers"""

    def test_steplr_single_step(self):
        """Test StepLR with step_size=1"""
        from grilly.nn import Parameter
        from grilly.optim import SGD, StepLR

        param = Parameter(np.random.randn(5, 5).astype(np.float32))
        optimizer = SGD([param], lr=0.1)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

        for i in range(10):
            expected_lr = 0.1 * (0.9**i)
            current_lr = optimizer.param_groups[0]["lr"]
            assert abs(current_lr - expected_lr) < 1e-6
            scheduler.step()

    def test_cosine_single_cycle(self):
        """Test CosineAnnealingLR completes one full cycle"""
        from grilly.nn import Parameter
        from grilly.optim import SGD, CosineAnnealingLR

        param = Parameter(np.random.randn(5, 5).astype(np.float32))
        optimizer = SGD([param], lr=1.0)
        T_max = 10
        eta_min = 0.0
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        lrs = []
        for _ in range(T_max):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # At T_max, should return to near max
        # Due to cosine, it won't be exactly 1.0, but close to eta_min
        assert lrs[-1] < 0.1  # Should be near minimum

    def test_plateau_mode_max(self):
        """Test ReduceLROnPlateau with mode='max'"""
        from grilly.nn import Parameter
        from grilly.optim import SGD, ReduceLROnPlateau

        param = Parameter(np.random.randn(5, 5).astype(np.float32))
        optimizer = SGD([param], lr=0.1)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        initial_lr = optimizer.param_groups[0]["lr"]

        # Improving metric (increasing)
        for i in range(3):
            scheduler.step(0.9 + i * 0.01)

        # LR should not have changed
        assert optimizer.param_groups[0]["lr"] == initial_lr

        # Plateau (same value)
        for _ in range(5):
            scheduler.step(0.92)

        # LR should have been reduced
        assert optimizer.param_groups[0]["lr"] < initial_lr

    def test_onecycle_error_on_no_total_steps(self):
        """Test OneCycleLR raises error without total_steps"""
        from grilly.nn import Parameter
        from grilly.optim import SGD, OneCycleLR

        param = Parameter(np.random.randn(5, 5).astype(np.float32))
        optimizer = SGD([param], lr=0.1)

        with pytest.raises(ValueError):
            OneCycleLR(optimizer, max_lr=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
