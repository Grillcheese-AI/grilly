"""
Loss Functions (PyTorch-like)

Loss functions with backward pass support for training.
"""

import numpy as np

from .module import Module


class MSELoss(Module):
    """
    Mean Squared Error Loss

    Loss = mean((input - target)^2)
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: 'mean' (default) or 'sum' or 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute MSE loss.

        Args:
            input: Predictions (any shape)
            target: Targets (same shape as input)

        Returns:
            Loss value (scalar if reduction='mean' or 'sum', array if reduction='none')
        """
        diff = input - target
        squared = diff**2

        if self.reduction == "mean":
            return np.mean(squared)
        elif self.reduction == "sum":
            return np.sum(squared)
        else:  # 'none'
            return squared

    def backward(
        self, grad_output: np.ndarray = None, input: np.ndarray = None, target: np.ndarray = None
    ) -> np.ndarray:
        """
        Backward pass for MSE loss.

        Gradient: d/dx MSE = 2 * (input - target) / N (for mean reduction)

        Args:
            grad_output: Gradient w.r.t. loss (usually 1.0 for loss functions)
            input: Input from forward pass
            target: Target from forward pass

        Returns:
            grad_input: Gradient w.r.t. input
        """
        if input is None or target is None:
            raise ValueError("input and target must be provided for backward pass")

        if grad_output is None:
            grad_output = 1.0

        # Gradient: 2 * (input - target) / N for mean, 2 * (input - target) for sum
        diff = input - target

        if self.reduction == "mean":
            N = input.size
            grad_input = 2.0 * diff / N * grad_output
        elif self.reduction == "sum":
            grad_input = 2.0 * diff * grad_output
        else:  # 'none'
            grad_input = 2.0 * diff * grad_output

        return grad_input


class CrossEntropyLoss(Module):
    """
    Cross Entropy Loss (with softmax)

    Loss = -sum(target * log(softmax(input))) / N
    """

    def __init__(self, reduction: str = "mean", ignore_index: int = -100):
        """
        Args:
            reduction: 'mean' (default) or 'sum' or 'none'
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self._log_probs = None  # Store log probabilities for backward

    def forward(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute cross entropy loss.

        Args:
            input: Logits (batch, num_classes) or (batch, seq_len, num_classes)
            target: Class indices (batch,) or (batch, seq_len)

        Returns:
            Loss value (scalar if reduction='mean' or 'sum', array if reduction='none')
        """
        # Compute softmax
        # Shift input for numerical stability
        input_max = np.max(input, axis=-1, keepdims=True)
        exp_input = np.exp(input - input_max)
        softmax = exp_input / np.sum(exp_input, axis=-1, keepdims=True)

        # Compute log probabilities
        log_probs = np.log(softmax + 1e-8)  # Add small epsilon for numerical stability
        self._log_probs = log_probs  # Store for backward pass

        # Get target as one-hot or indices
        if target.ndim == input.ndim - 1:
            # Target is indices
            batch_size = target.shape[0]
            if input.ndim == 3:
                # (batch, seq_len, num_classes) -> (batch, seq_len)
                seq_len = target.shape[1] if len(target.shape) > 1 else 1
                num_classes = input.shape[-1]
                target_one_hot = np.zeros((batch_size, seq_len, num_classes), dtype=np.float32)
                if len(target.shape) == 1:
                    target_one_hot[np.arange(batch_size), target] = 1.0
                else:
                    for b in range(batch_size):
                        for s in range(seq_len):
                            if target[b, s] != self.ignore_index:
                                target_one_hot[b, s, target[b, s]] = 1.0
            else:
                # (batch, num_classes) -> (batch,)
                num_classes = input.shape[-1]
                target_one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
                target_one_hot[np.arange(batch_size), target] = 1.0
        else:
            # Target is already one-hot
            target_one_hot = target

        # Compute loss: -sum(target * log_probs)
        loss = -np.sum(target_one_hot * log_probs, axis=-1)

        # Apply ignore_index mask
        if target.ndim == input.ndim - 1:
            mask = (target != self.ignore_index).astype(np.float32)
            loss = loss * mask

        if self.reduction == "mean":
            if target.ndim == input.ndim - 1:
                valid = np.sum(mask)
                return np.sum(loss) / max(valid, 1.0)
            return np.mean(loss)
        elif self.reduction == "sum":
            return np.sum(loss)
        else:  # 'none'
            return loss

    def backward(
        self, grad_output: np.ndarray = None, input: np.ndarray = None, target: np.ndarray = None
    ) -> np.ndarray:
        """
        Backward pass for cross entropy loss.

        Gradient: d/dx CE = softmax(input) - target

        Args:
            grad_output: Gradient w.r.t. loss (usually 1.0 for loss functions)
            input: Input from forward pass
            target: Target from forward pass

        Returns:
            grad_input: Gradient w.r.t. input
        """
        if input is None or target is None:
            raise ValueError("input and target must be provided for backward pass")

        if grad_output is None:
            grad_output = 1.0

        # Recompute softmax (or use stored log_probs)
        input_max = np.max(input, axis=-1, keepdims=True)
        exp_input = np.exp(input - input_max)
        softmax = exp_input / np.sum(exp_input, axis=-1, keepdims=True)

        # Get target as one-hot
        if target.ndim == input.ndim - 1:
            # Target is indices
            batch_size = target.shape[0]
            if input.ndim == 3:
                # (batch, seq_len, num_classes) -> (batch, seq_len)
                seq_len = target.shape[1] if len(target.shape) > 1 else 1
                num_classes = input.shape[-1]
                target_one_hot = np.zeros((batch_size, seq_len, num_classes), dtype=np.float32)
                if len(target.shape) == 1:
                    target_one_hot[np.arange(batch_size), target] = 1.0
                else:
                    for b in range(batch_size):
                        for s in range(seq_len):
                            if target[b, s] != self.ignore_index:
                                target_one_hot[b, s, target[b, s]] = 1.0
            else:
                # (batch, num_classes) -> (batch,)
                num_classes = input.shape[-1]
                target_one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
                target_one_hot[np.arange(batch_size), target] = 1.0
        else:
            # Target is already one-hot
            target_one_hot = target

        # Gradient: softmax - target
        grad_input = softmax - target_one_hot

        # Apply ignore_index mask
        if target.ndim == input.ndim - 1:
            mask = (target != self.ignore_index).astype(np.float32)
            if input.ndim >= 2:
                mask = mask[..., np.newaxis]
            grad_input = grad_input * mask

        # Apply reduction scaling
        if self.reduction == "mean":
            if target.ndim == input.ndim - 1:
                valid = np.sum(mask[..., 0])
                grad_input = grad_input / max(valid, 1.0)
            else:
                N = input.size / input.shape[-1]  # Number of samples
                grad_input = grad_input / N
        # For 'sum' and 'none', no additional scaling needed

        return grad_input * grad_output


class BCELoss(Module):
    """
    Binary Cross Entropy Loss

    Loss = -sum(target * log(sigmoid(input)) + (1 - target) * log(1 - sigmoid(input))) / N
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: 'mean' (default) or 'sum' or 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute BCE loss.

        Args:
            input: Logits (any shape)
            target: Targets in [0, 1] (same shape as input)

        Returns:
            Loss value (scalar if reduction='mean' or 'sum', array if reduction='none')
        """
        # Compute sigmoid
        sigmoid = 1.0 / (1.0 + np.exp(-input))

        # Clamp to avoid log(0)
        sigmoid = np.clip(sigmoid, 1e-8, 1.0 - 1e-8)

        # Compute loss
        loss = -(target * np.log(sigmoid) + (1.0 - target) * np.log(1.0 - sigmoid))

        if self.reduction == "mean":
            return np.mean(loss)
        elif self.reduction == "sum":
            return np.sum(loss)
        else:  # 'none'
            return loss

    def backward(
        self, grad_output: np.ndarray = None, input: np.ndarray = None, target: np.ndarray = None
    ) -> np.ndarray:
        """
        Backward pass for BCE loss.

        Gradient: d/dx BCE = sigmoid(input) - target

        Args:
            grad_output: Gradient w.r.t. loss (usually 1.0 for loss functions)
            input: Input from forward pass
            target: Target from forward pass

        Returns:
            grad_input: Gradient w.r.t. input
        """
        if input is None or target is None:
            raise ValueError("input and target must be provided for backward pass")

        if grad_output is None:
            grad_output = 1.0

        # Compute sigmoid
        sigmoid = 1.0 / (1.0 + np.exp(-input))

        # Gradient: sigmoid - target
        grad_input = sigmoid - target

        # Apply reduction scaling
        if self.reduction == "mean":
            N = input.size
            grad_input = grad_input / N
        # For 'sum' and 'none', no additional scaling needed

        return grad_input * grad_output
