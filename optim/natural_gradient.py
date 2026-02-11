"""
Natural Gradient Optimizer

Uses: fisher-natural-gradient.glsl

Implements natural gradient descent using Fisher information matrix.
"""

from collections.abc import Iterator

import numpy as np

from .base import Optimizer


class NaturalGradient(Optimizer):
    """
    Natural Gradient optimizer using Fisher information matrix.

    Uses: fisher-natural-gradient.glsl

    Implements natural gradient descent:
    - F = Fisher information matrix
    - param = param - lr * F^(-1) * grad

    Reference: grilly/backend/learning.py natural_gradient
    """

    def __init__(
        self,
        params: Iterator[np.ndarray],
        lr: float = 1e-3,
        fisher_momentum: float = 0.9,
        use_gpu: bool = True,
    ):
        """
        Initialize Natural Gradient optimizer.

        Args:
            params: Iterator of parameter arrays to optimize
            lr: Learning rate (default: 1e-3)
            fisher_momentum: Momentum for Fisher information estimate (default: 0.9)
            use_gpu: Whether to use GPU acceleration (default: True)
        """
        defaults = {
            "lr": lr,
            "fisher_momentum": fisher_momentum,
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

        backend = self._get_backend()
        use_gpu = self.use_gpu and backend is not None
        lr = self.defaults["lr"]
        fisher_momentum = self.defaults["fisher_momentum"]

        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue

                param_id = id(p)
                state = self.state[param_id]

                # Initialize Fisher information if needed
                if "fisher" not in state:
                    state["fisher"] = np.ones_like(p, dtype=np.float32) * 1e-6

                fisher = state["fisher"]

                # Get gradients
                grad = getattr(p, "grad", None)
                if grad is None:
                    continue

                # Get parameter data
                p_data = p.data if hasattr(p, "data") and not isinstance(p, np.ndarray) else p
                # Ensure numpy array
                if not isinstance(p_data, np.ndarray):
                    p_data = np.array(p_data, dtype=np.float32)

                # Try GPU update if available
                if use_gpu and backend is not None and hasattr(backend, "learning"):
                    try:
                        # Update Fisher information
                        if hasattr(backend.learning, "fisher_info_update"):
                            fisher = backend.learning.fisher_info_update(
                                grad.flatten(), fisher.flatten(), momentum=fisher_momentum
                            )
                            fisher = fisher.reshape(p.shape)
                            state["fisher"] = fisher

                        # Apply natural gradient
                        if hasattr(backend.learning, "natural_gradient"):
                            natural_grad = backend.learning.natural_gradient(
                                grad.flatten(), fisher.flatten(), learning_rate=lr, epsilon=1e-8
                            )
                            natural_grad = natural_grad.reshape(p.shape)
                            p_data -= natural_grad  # natural_gradient already includes lr

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
                            continue
                    except Exception as e:
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.debug(
                            f"GPU Natural Gradient update failed: {e}, falling back to CPU"
                        )
                        pass  # Fall back to CPU

                # CPU fallback
                # Update Fisher information (simplified - use gradient squared)
                fisher = fisher_momentum * fisher + (1 - fisher_momentum) * (grad**2)
                state["fisher"] = fisher

                # Apply natural gradient: F^(-1) * grad ≈ grad / (fisher + eps)
                eps = 1e-8
                natural_grad = grad / (fisher + eps)
                p_data -= lr * natural_grad

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
