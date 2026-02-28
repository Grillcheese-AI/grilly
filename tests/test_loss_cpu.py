"""
CPU-only tests for nn/loss.py — MSELoss, CrossEntropyLoss, BCELoss.

All tests use pure numpy, no GPU required.
"""

import numpy as np
import pytest

try:
    from grilly.nn.loss import BCELoss, CrossEntropyLoss, MSELoss
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


@pytest.fixture(autouse=True)
def seed_rng():
    """Fix random seed for reproducibility."""
    np.random.seed(42)


# ---------------------------------------------------------------------------
# MSELoss
# ---------------------------------------------------------------------------
class TestMSELoss:
    """Tests for Mean Squared Error loss."""

    def test_forward_mean(self):
        """Default reduction='mean' matches manual np.mean((pred - target)**2)."""
        pred = np.random.randn(8, 4).astype(np.float32)
        target = np.random.randn(8, 4).astype(np.float32)

        loss_fn = MSELoss(reduction="mean")
        loss = loss_fn.forward(pred, target)

        expected = np.mean((pred - target) ** 2)
        np.testing.assert_allclose(loss, expected, rtol=1e-6)

    def test_forward_sum(self):
        """reduction='sum' returns total squared error."""
        pred = np.random.randn(8, 4).astype(np.float32)
        target = np.random.randn(8, 4).astype(np.float32)

        loss_fn = MSELoss(reduction="sum")
        loss = loss_fn.forward(pred, target)

        expected = np.sum((pred - target) ** 2)
        np.testing.assert_allclose(loss, expected, rtol=1e-6)

    def test_forward_none(self):
        """reduction='none' returns per-element squared error."""
        pred = np.random.randn(8, 4).astype(np.float32)
        target = np.random.randn(8, 4).astype(np.float32)

        loss_fn = MSELoss(reduction="none")
        loss = loss_fn.forward(pred, target)

        expected = (pred - target) ** 2
        np.testing.assert_allclose(loss, expected, rtol=1e-6)

    def test_output_shape_mean(self):
        """Mean reduction produces a scalar output."""
        pred = np.random.randn(16, 10).astype(np.float32)
        target = np.random.randn(16, 10).astype(np.float32)

        loss = MSELoss(reduction="mean").forward(pred, target)
        assert np.ndim(loss) == 0, "Mean-reduced MSE should be a scalar"

    def test_output_shape_none(self):
        """No reduction preserves the input shape."""
        pred = np.random.randn(16, 10).astype(np.float32)
        target = np.random.randn(16, 10).astype(np.float32)

        loss = MSELoss(reduction="none").forward(pred, target)
        assert loss.shape == pred.shape

    def test_backward_mean(self):
        """Backward for mean reduction: grad = 2*(pred-target)/N."""
        pred = np.random.randn(8, 4).astype(np.float32)
        target = np.random.randn(8, 4).astype(np.float32)

        loss_fn = MSELoss(reduction="mean")
        loss = loss_fn.forward(pred, target)
        grad_output = np.ones_like(loss)
        grad = loss_fn.backward(grad_output, pred, target)

        assert grad.shape == pred.shape
        N = pred.size
        expected_grad = 2.0 * (pred - target) / N
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-6)

    def test_backward_sum(self):
        """Backward for sum reduction: grad = 2*(pred-target)."""
        pred = np.random.randn(8, 4).astype(np.float32)
        target = np.random.randn(8, 4).astype(np.float32)

        loss_fn = MSELoss(reduction="sum")
        loss = loss_fn.forward(pred, target)
        grad_output = np.ones_like(loss)
        grad = loss_fn.backward(grad_output, pred, target)

        assert grad.shape == pred.shape
        expected_grad = 2.0 * (pred - target)
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-6)

    def test_backward_none(self):
        """Backward for none reduction: grad = 2*(pred-target) per element."""
        pred = np.random.randn(8, 4).astype(np.float32)
        target = np.random.randn(8, 4).astype(np.float32)

        loss_fn = MSELoss(reduction="none")
        loss = loss_fn.forward(pred, target)
        grad_output = np.ones_like(loss)
        grad = loss_fn.backward(grad_output, pred, target)

        assert grad.shape == pred.shape
        expected_grad = 2.0 * (pred - target)
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-6)

    def test_backward_requires_forward(self):
        """Backward without input/target raises ValueError."""
        loss_fn = MSELoss()
        with pytest.raises(ValueError):
            loss_fn.backward(np.array(1.0))


# ---------------------------------------------------------------------------
# CrossEntropyLoss
# ---------------------------------------------------------------------------
class TestCrossEntropyLoss:
    """Tests for Cross Entropy loss with softmax."""

    # -- forward ----------------------------------------------------------

    def test_forward_index_targets_mean(self):
        """Index targets with mean reduction match manual cross-entropy."""
        batch, num_classes = 8, 5
        logits = np.random.randn(batch, num_classes).astype(np.float32)
        targets = np.random.randint(0, num_classes, size=(batch,))

        loss_fn = CrossEntropyLoss(reduction="mean")
        loss = loss_fn.forward(logits, targets)

        # Manual: softmax -> log -> pick correct class -> negate -> mean
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(shifted)
        softmax = exp_l / exp_l.sum(axis=-1, keepdims=True)
        log_probs = np.log(softmax + 1e-8)
        per_sample = -log_probs[np.arange(batch), targets]
        expected = np.mean(per_sample)

        np.testing.assert_allclose(loss, expected, rtol=1e-5)

    def test_forward_index_targets_sum(self):
        """Index targets with sum reduction."""
        batch, num_classes = 8, 5
        logits = np.random.randn(batch, num_classes).astype(np.float32)
        targets = np.random.randint(0, num_classes, size=(batch,))

        loss_fn = CrossEntropyLoss(reduction="sum")
        loss = loss_fn.forward(logits, targets)

        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(shifted)
        softmax = exp_l / exp_l.sum(axis=-1, keepdims=True)
        log_probs = np.log(softmax + 1e-8)
        per_sample = -log_probs[np.arange(batch), targets]
        expected = np.sum(per_sample)

        np.testing.assert_allclose(loss, expected, rtol=1e-5)

    def test_forward_index_targets_none(self):
        """Index targets with no reduction returns per-sample losses."""
        batch, num_classes = 8, 5
        logits = np.random.randn(batch, num_classes).astype(np.float32)
        targets = np.random.randint(0, num_classes, size=(batch,))

        loss_fn = CrossEntropyLoss(reduction="none")
        loss = loss_fn.forward(logits, targets)

        assert loss.shape == (batch,)

        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(shifted)
        softmax = exp_l / exp_l.sum(axis=-1, keepdims=True)
        log_probs = np.log(softmax + 1e-8)
        per_sample = -log_probs[np.arange(batch), targets]

        np.testing.assert_allclose(loss, per_sample, rtol=1e-5)

    def test_forward_onehot_targets(self):
        """One-hot float targets produce the same loss as index targets."""
        batch, num_classes = 8, 5
        logits = np.random.randn(batch, num_classes).astype(np.float32)
        indices = np.random.randint(0, num_classes, size=(batch,))
        one_hot = np.zeros((batch, num_classes), dtype=np.float32)
        one_hot[np.arange(batch), indices] = 1.0

        loss_idx = CrossEntropyLoss(reduction="mean").forward(logits, indices)
        loss_oh = CrossEntropyLoss(reduction="mean").forward(logits, one_hot)

        np.testing.assert_allclose(loss_idx, loss_oh, rtol=1e-5)

    def test_forward_3d_input(self):
        """3D input (batch, seq_len, classes) with 2D index targets."""
        batch, seq_len, num_classes = 4, 6, 10
        logits = np.random.randn(batch, seq_len, num_classes).astype(np.float32)
        targets = np.random.randint(0, num_classes, size=(batch, seq_len))

        loss_fn = CrossEntropyLoss(reduction="mean")
        loss = loss_fn.forward(logits, targets)

        assert np.isscalar(loss) or np.ndim(loss) == 0
        assert np.isfinite(loss)
        assert loss > 0

    def test_forward_ignore_index(self):
        """Ignored indices do not contribute to the mean loss."""
        batch, num_classes = 8, 5
        logits = np.random.randn(batch, num_classes).astype(np.float32)
        targets = np.random.randint(0, num_classes, size=(batch,))

        # Mark half the samples as ignored
        targets_masked = targets.copy()
        targets_masked[::2] = -100

        loss_fn = CrossEntropyLoss(reduction="mean", ignore_index=-100)
        loss_masked = loss_fn.forward(logits, targets_masked)

        # Manually compute loss only for non-masked samples
        valid_indices = np.where(targets_masked != -100)[0]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(shifted)
        softmax = exp_l / exp_l.sum(axis=-1, keepdims=True)
        log_probs = np.log(softmax + 1e-8)
        valid_losses = -log_probs[valid_indices, targets_masked[valid_indices]]
        expected = np.mean(valid_losses)

        np.testing.assert_allclose(loss_masked, expected, rtol=1e-5)

    # -- backward ---------------------------------------------------------

    def test_backward_index_mean(self):
        """Backward with index targets, mean reduction: grad = (softmax - one_hot) / N."""
        batch, num_classes = 8, 5
        logits = np.random.randn(batch, num_classes).astype(np.float32)
        targets = np.random.randint(0, num_classes, size=(batch,))

        loss_fn = CrossEntropyLoss(reduction="mean")
        loss_fn.forward(logits, targets)
        grad = loss_fn.backward(np.float32(1.0), logits, targets)

        assert grad.shape == logits.shape

        # Expected: (softmax - one_hot) / batch_size
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(shifted)
        softmax = exp_l / exp_l.sum(axis=-1, keepdims=True)
        one_hot = np.zeros_like(logits)
        one_hot[np.arange(batch), targets] = 1.0
        expected_grad = (softmax - one_hot) / batch

        np.testing.assert_allclose(grad, expected_grad, rtol=1e-5)

    def test_backward_index_sum(self):
        """Backward with index targets, sum reduction: grad = softmax - one_hot."""
        batch, num_classes = 8, 5
        logits = np.random.randn(batch, num_classes).astype(np.float32)
        targets = np.random.randint(0, num_classes, size=(batch,))

        loss_fn = CrossEntropyLoss(reduction="sum")
        loss_fn.forward(logits, targets)
        grad = loss_fn.backward(np.float32(1.0), logits, targets)

        assert grad.shape == logits.shape

        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(shifted)
        softmax = exp_l / exp_l.sum(axis=-1, keepdims=True)
        one_hot = np.zeros_like(logits)
        one_hot[np.arange(batch), targets] = 1.0
        expected_grad = softmax - one_hot

        np.testing.assert_allclose(grad, expected_grad, rtol=1e-5)

    def test_backward_onehot(self):
        """Backward with one-hot targets matches index-target backward."""
        batch, num_classes = 8, 5
        logits = np.random.randn(batch, num_classes).astype(np.float32)
        indices = np.random.randint(0, num_classes, size=(batch,))
        one_hot = np.zeros((batch, num_classes), dtype=np.float32)
        one_hot[np.arange(batch), indices] = 1.0

        loss_fn_idx = CrossEntropyLoss(reduction="mean")
        loss_fn_oh = CrossEntropyLoss(reduction="mean")

        loss_fn_idx.forward(logits, indices)
        loss_fn_oh.forward(logits, one_hot)

        grad_idx = loss_fn_idx.backward(np.float32(1.0), logits, indices)
        grad_oh = loss_fn_oh.backward(np.float32(1.0), logits, one_hot)

        np.testing.assert_allclose(grad_idx, grad_oh, rtol=1e-5)

    def test_backward_3d(self):
        """Backward with 3D input produces matching-shape gradients."""
        batch, seq_len, num_classes = 4, 6, 10
        logits = np.random.randn(batch, seq_len, num_classes).astype(np.float32)
        targets = np.random.randint(0, num_classes, size=(batch, seq_len))

        loss_fn = CrossEntropyLoss(reduction="mean")
        loss_fn.forward(logits, targets)
        grad = loss_fn.backward(np.float32(1.0), logits, targets)

        assert grad.shape == logits.shape
        assert np.all(np.isfinite(grad))

    def test_backward_ignore_index(self):
        """Masked positions have zero gradient."""
        batch, num_classes = 8, 5
        logits = np.random.randn(batch, num_classes).astype(np.float32)
        targets = np.random.randint(0, num_classes, size=(batch,))
        targets_masked = targets.copy()
        targets_masked[::2] = -100

        loss_fn = CrossEntropyLoss(reduction="mean", ignore_index=-100)
        loss_fn.forward(logits, targets_masked)
        grad = loss_fn.backward(np.float32(1.0), logits, targets_masked)

        assert grad.shape == logits.shape

        # Masked positions should have zero gradient across all classes
        masked_positions = np.where(targets_masked == -100)[0]
        for pos in masked_positions:
            np.testing.assert_allclose(
                grad[pos], np.zeros(num_classes, dtype=np.float32), atol=1e-8
            )

        # Non-masked positions should have non-zero gradient
        valid_positions = np.where(targets_masked != -100)[0]
        for pos in valid_positions:
            assert np.any(np.abs(grad[pos]) > 1e-8)

    def test_backward_requires_forward(self):
        """Backward without input/target raises ValueError."""
        loss_fn = CrossEntropyLoss()
        with pytest.raises(ValueError):
            loss_fn.backward(np.array(1.0))


# ---------------------------------------------------------------------------
# BCELoss
# ---------------------------------------------------------------------------
class TestBCELoss:
    """Tests for Binary Cross Entropy loss (with sigmoid)."""

    def test_forward_mean(self):
        """Mean reduction matches manual -[t*log(sig) + (1-t)*log(1-sig)]."""
        pred = np.random.randn(8, 4).astype(np.float32)
        target = np.random.randint(0, 2, size=(8, 4)).astype(np.float32)

        loss_fn = BCELoss(reduction="mean")
        loss = loss_fn.forward(pred, target)

        sigmoid = 1.0 / (1.0 + np.exp(-pred))
        sigmoid = np.clip(sigmoid, 1e-8, 1.0 - 1e-8)
        expected = np.mean(-(target * np.log(sigmoid) + (1 - target) * np.log(1 - sigmoid)))

        np.testing.assert_allclose(loss, expected, rtol=1e-6)

    def test_forward_sum(self):
        """Sum reduction returns total BCE loss."""
        pred = np.random.randn(8, 4).astype(np.float32)
        target = np.random.randint(0, 2, size=(8, 4)).astype(np.float32)

        loss_fn = BCELoss(reduction="sum")
        loss = loss_fn.forward(pred, target)

        sigmoid = 1.0 / (1.0 + np.exp(-pred))
        sigmoid = np.clip(sigmoid, 1e-8, 1.0 - 1e-8)
        expected = np.sum(-(target * np.log(sigmoid) + (1 - target) * np.log(1 - sigmoid)))

        np.testing.assert_allclose(loss, expected, rtol=1e-6)

    def test_forward_none(self):
        """No reduction returns per-element loss with matching shape."""
        pred = np.random.randn(8, 4).astype(np.float32)
        target = np.random.randint(0, 2, size=(8, 4)).astype(np.float32)

        loss_fn = BCELoss(reduction="none")
        loss = loss_fn.forward(pred, target)

        assert loss.shape == pred.shape

        sigmoid = 1.0 / (1.0 + np.exp(-pred))
        sigmoid = np.clip(sigmoid, 1e-8, 1.0 - 1e-8)
        expected = -(target * np.log(sigmoid) + (1 - target) * np.log(1 - sigmoid))

        np.testing.assert_allclose(loss, expected, rtol=1e-6)

    def test_backward_mean(self):
        """Backward for mean reduction: grad = (sigmoid - target) / N."""
        pred = np.random.randn(8, 4).astype(np.float32)
        target = np.random.randint(0, 2, size=(8, 4)).astype(np.float32)

        loss_fn = BCELoss(reduction="mean")
        loss = loss_fn.forward(pred, target)
        grad_output = np.ones_like(loss)
        grad = loss_fn.backward(grad_output, pred, target)

        assert grad.shape == pred.shape

        sigmoid = 1.0 / (1.0 + np.exp(-pred))
        N = pred.size
        expected_grad = (sigmoid - target) / N

        np.testing.assert_allclose(grad, expected_grad, rtol=1e-5)

        # Numerical gradient check: perturb each element and compare
        eps = 1e-4
        numerical_grad = np.zeros_like(pred)
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                pred_plus = pred.copy()
                pred_minus = pred.copy()
                pred_plus[i, j] += eps
                pred_minus[i, j] -= eps
                loss_plus = BCELoss(reduction="mean").forward(pred_plus, target)
                loss_minus = BCELoss(reduction="mean").forward(pred_minus, target)
                numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(grad, numerical_grad, atol=1e-3, rtol=1e-3)

    def test_backward_sum(self):
        """Backward for sum reduction: grad = sigmoid - target."""
        pred = np.random.randn(8, 4).astype(np.float32)
        target = np.random.randint(0, 2, size=(8, 4)).astype(np.float32)

        loss_fn = BCELoss(reduction="sum")
        loss = loss_fn.forward(pred, target)
        grad_output = np.ones_like(loss)
        grad = loss_fn.backward(grad_output, pred, target)

        assert grad.shape == pred.shape

        sigmoid = 1.0 / (1.0 + np.exp(-pred))
        expected_grad = sigmoid - target

        np.testing.assert_allclose(grad, expected_grad, rtol=1e-5)

    def test_backward_requires_forward(self):
        """Backward without input/target raises ValueError."""
        loss_fn = BCELoss()
        with pytest.raises(ValueError):
            loss_fn.backward(np.array(1.0))
