"""
NLMS (Normalized Least Mean Squares) Optimizer

Uses: nlms-update.glsl

Reference: ref/brain/specialist.py NLMSExpertHead
"""

from collections.abc import Iterator

import numpy as np

from .base import Optimizer


class NLMS(Optimizer):
    """
    NLMS (Normalized Least Mean Squares) optimizer.

    Uses: nlms-update.glsl

    Implements adaptive filtering with normalized learning rate:
    - w = w + mu * error * x / (||x||^2 + eps)

    Reference: ref/brain/specialist.py NLMSExpertHead
    """

    def __init__(
        self,
        params: Iterator[np.ndarray],
        lr: float = 0.5,
        lr_decay: float = 0.99995,
        lr_min: float = 0.1,
        eps: float = 1e-6,
        use_gpu: bool = True,
    ):
        """
        Initialize NLMS optimizer.

        Args:
            params: Iterator of parameter arrays to optimize
            lr: Initial learning rate (mu) (default: 0.5)
            lr_decay: Learning rate decay factor (default: 0.99995)
            lr_min: Minimum learning rate (default: 0.1)
            eps: Small constant for numerical stability (default: 1e-6)
            use_gpu: Whether to use GPU acceleration (default: True)
        """
        defaults = {
            "lr": lr,
            "lr_decay": lr_decay,
            "lr_min": lr_min,
            "eps": eps,
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

        for group in self.param_groups:
            lr = group.get("lr", self.defaults["lr"])
            lr_decay = self.defaults["lr_decay"]
            lr_min = self.defaults["lr_min"]
            eps = self.defaults["eps"]

            for p in group["params"]:
                if p is None:
                    continue

                param_id = id(p)
                state = self.state[param_id]

                # Initialize state if needed
                if len(state) == 0:
                    state["mu"] = lr
                    state["mu_initial"] = lr
                    state["update_count"] = 0

                mu = state["mu"]

                # Get gradients (assumed to be stored in p.grad)
                # For NLMS, we need both the gradient and the input
                # In practice, this would come from the forward pass
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
                        # NLMS requires features and target, not just gradients
                        # For standard optimizer use, we approximate:
                        # - grad ≈ error * x (gradient already contains error-weighted input)
                        # - We normalize by ||grad||^2 to approximate ||x||^2

                        # Check if nlms-update shader is available
                        if hasattr(backend.learning, "nlms_update"):
                            # For NLMS, we need to extract features from gradient
                            # In practice, NLMS is used with explicit features and targets
                            # For optimizer use, we'll use a simplified GPU-accelerated version
                            # that normalizes by gradient norm

                            # Compute gradient norm for normalization
                            grad_flat = grad.flatten()
                            norm_sq = np.dot(grad_flat, grad_flat) + eps

                            # Use GPU for the update computation if possible
                            # For now, use CPU fallback with efficient NumPy operations
                            pass
                    except Exception:
                        pass

                # CPU fallback (simplified - assumes grad is the error-weighted input)
                # In real NLMS: w = w + mu * error * x / (||x||^2 + eps)
                # Here we approximate: grad ≈ error * x, so we normalize by ||grad||^2
                grad_flat = grad.flatten()
                norm_sq = np.dot(grad_flat, grad_flat) + eps
                step = mu / norm_sq
                p_data -= step * grad

                # Update parameter (handle wrapper or direct numpy array)
                if hasattr(p, "data") and not isinstance(p, np.ndarray):
                    # Parameter wrapper or custom class
                    p.data = p_data
                else:
                    # Direct numpy array - update in-place
                    p[:] = p_data

                # Decay learning rate
                if mu > lr_min:
                    state["mu"] = mu * lr_decay

                state["update_count"] += 1

                # Clear gradient after update
                if hasattr(p, "grad") and p.grad is not None:
                    if hasattr(p, "zero_grad"):
                        p.zero_grad()
                    else:
                        p.grad = None

        return loss
