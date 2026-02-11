"""
SGD Optimizer

Stochastic Gradient Descent optimizer.
"""

from collections.abc import Iterator

import numpy as np

from .base import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Implements: param = param - lr * grad

    Note: SGD is simple enough that CPU implementation is efficient.
    For GPU acceleration, we could use a generic update shader in the future.
    """

    def __init__(
        self,
        params: Iterator[np.ndarray],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        use_gpu: bool = False,
    ):
        """
        Initialize SGD optimizer.

        Args:
            params: Iterator of parameter arrays to optimize
            lr: Learning rate (default: 1e-3)
            momentum: Momentum factor (default: 0.0)
            weight_decay: Weight decay (L2 penalty) (default: 0.0)
            dampening: Dampening for momentum (default: 0.0)
            nesterov: Enable Nesterov momentum (default: False)
            use_gpu: Whether to attempt GPU acceleration (default: False, CPU is efficient)
        """
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "dampening": dampening,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)
        self.use_gpu = use_gpu
        self._backend = None

    def _get_backend(self):
        """Get or create backend instance"""
        if self._backend is None:
            try:
                from grilly import Compute

                self._backend = Compute()
            except Exception:
                self._backend = None
        return self._backend

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure that reevaluates the model and returns loss
        """
        loss = None
        if closure is not None:
            loss = closure()

        self._get_backend()

        lr = self.defaults["lr"]
        momentum = self.defaults["momentum"]
        weight_decay = self.defaults["weight_decay"]
        dampening = self.defaults["dampening"]
        nesterov = self.defaults["nesterov"]

        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue

                param_id = id(p)
                state = self.state[param_id]

                # Initialize momentum buffer if needed
                if momentum != 0 and "momentum_buffer" not in state:
                    state["momentum_buffer"] = np.zeros_like(p, dtype=np.float32)

                # Get gradients (from backward pass)
                grad = getattr(p, "grad", None)
                if grad is None:
                    continue

                # Extract data if parameter is wrapped
                p_data = p.data if hasattr(p, "data") and not isinstance(p, np.ndarray) else p
                # Ensure numpy array
                if not isinstance(p_data, np.ndarray):
                    p_data = np.array(p_data, dtype=np.float32)

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad + weight_decay * p_data

                # Apply momentum
                if momentum != 0:
                    buf = state["momentum_buffer"]
                    buf = momentum * buf + (1 - dampening) * grad
                    state["momentum_buffer"] = buf

                    if nesterov:
                        grad = grad + momentum * buf
                    else:
                        grad = buf

                # Update parameters (in-place)
                p_data -= lr * grad

                # Update parameter (handle wrapper or direct numpy array)
                if hasattr(p, "data") and not isinstance(p, np.ndarray):
                    # Parameter wrapper or custom class
                    p.data = p_data
                else:
                    # Direct numpy array - update in-place
                    p[:] = p_data

                # Clear gradient after update
                if hasattr(p, "grad") and p.grad is not None:
                    if hasattr(p, "zero_grad"):
                        p.zero_grad()
                    else:
                        p.grad = None

        return loss
