"""
Adam Optimizer

Uses: adam-update.glsl, affect-adam.glsl
"""

from collections.abc import Iterator

import numpy as np

from .base import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer using GPU-accelerated shaders.

    Uses: adam-update.glsl

    Implements the Adam algorithm:
    - m = beta1 * m + (1 - beta1) * grad
    - v = beta2 * v + (1 - beta2) * grad^2
    - m_hat = m / (1 - beta1^t)
    - v_hat = v / (1 - beta2^t)
    - param = param - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: Iterator[np.ndarray],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        use_gpu: bool = True,
    ):
        """
        Initialize Adam optimizer.

        Args:
            params: Iterator of parameter arrays to optimize
            lr: Learning rate (default: 1e-3)
            betas: Coefficients for computing running averages (default: (0.9, 0.999))
            eps: Term added to denominator for numerical stability (default: 1e-8)
            weight_decay: Weight decay (L2 penalty) (default: 0.0)
            use_gpu: Whether to use GPU acceleration (default: True)
        """
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
        self.use_gpu = use_gpu
        self._backend = None
        self._step_count = 0

    def _get_backend(self):
        """Get or create backend instance"""
        if self._backend is None:
            try:
                from grilly import Compute

                self._backend = Compute()
            except Exception:
                self._backend = None
        return self._backend

    def step(self, closure=None, gradients=None):
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure that reevaluates the model and returns loss
            gradients: Optional dict mapping parameter IDs to gradients.
                      If None, tries to get gradients from param.grad attribute.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Store gradients for use in parameter updates
        self._gradients = gradients

        self._step_count += 1
        beta1, beta2 = self.defaults["betas"]
        lr = self.defaults["lr"]
        eps = self.defaults["eps"]
        weight_decay = self.defaults["weight_decay"]

        # Bias correction terms
        beta1_t = beta1**self._step_count
        beta2_t = beta2**self._step_count

        backend = self._get_backend()
        use_gpu = self.use_gpu and backend is not None

        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue

                param_id = id(p)
                state = self.state[param_id]

                # Initialize state if needed
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = np.zeros_like(p, dtype=np.float32)
                    state["exp_avg_sq"] = np.zeros_like(p, dtype=np.float32)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Get gradients from backward pass
                # Gradients are stored in param.grad after calling backward()
                grad = None

                # First, try to get from gradients dict (if manually provided for compatibility)
                if hasattr(self, "_gradients") and self._gradients is not None:
                    grad = self._gradients.get(param_id, None)

                # Then, try to get from param.grad (from backward pass)
                if grad is None:
                    grad = getattr(p, "grad", None)

                # If still no gradient, skip this parameter
                if grad is None:
                    continue

                # Ensure gradient is numpy array (extract from wrapper if needed)
                if hasattr(grad, "data"):
                    grad = grad.data
                if not isinstance(grad, np.ndarray):
                    grad = np.array(grad, dtype=np.float32)

                # Extract parameter data if it's wrapped
                if hasattr(p, "data"):
                    p_data = p.data
                else:
                    p_data = p

                # Ensure p_data is numpy array (handle memoryview)
                if not isinstance(p_data, np.ndarray):
                    p_data = np.asarray(p_data, dtype=np.float32)

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad + weight_decay * p_data

                # Try GPU update if available
                if use_gpu and backend is not None:
                    try:
                        # Check if adam-update or affect-adam shader is available
                        shaders_available = hasattr(backend, "core") and hasattr(
                            backend.core, "shaders"
                        )
                        shader_name = None
                        if shaders_available:
                            if "adam-update" in backend.core.shaders:
                                shader_name = "adam-update"
                            elif "affect-adam" in backend.core.shaders:
                                shader_name = "affect-adam"

                        if shader_name is not None:
                            p_data, exp_avg, exp_avg_sq = self._adam_update_gpu(
                                backend,
                                p_data,
                                grad,
                                exp_avg,
                                exp_avg_sq,
                                lr,
                                beta1,
                                beta2,
                                eps,
                                beta1_t,
                                beta2_t,
                                shader_name=shader_name,
                            )
                            # Update parameter (handle wrapper)
                            if hasattr(p, "data"):
                                p.data = p_data
                            else:
                                p[:] = p_data
                            state["exp_avg"] = exp_avg
                            state["exp_avg_sq"] = exp_avg_sq
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
                        logger.debug(f"GPU Adam update failed: {e}, falling back to CPU")
                        pass  # Fall back to CPU

                # CPU fallback
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad

                # Bias correction
                m_hat = exp_avg / (1 - beta1_t) if beta1_t < 1.0 else exp_avg
                v_hat = exp_avg_sq / (1 - beta2_t) if beta2_t < 1.0 else exp_avg_sq

                # Update parameters (in-place)
                p_data -= lr * m_hat / (np.sqrt(v_hat) + eps)

                # Update parameter (handle wrapper or direct numpy array)
                if hasattr(p, "data") and not isinstance(p, np.ndarray):
                    # Parameter wrapper or custom class
                    p.data = p_data
                else:
                    # Direct numpy array - update in-place
                    p[:] = p_data

                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

                # Clear gradient after update (from backward pass)
                if hasattr(p, "grad") and p.grad is not None:
                    if hasattr(p, "zero_grad"):
                        p.zero_grad()
                    else:
                        p.grad = None

        return loss

    def _adam_update_gpu(
        self, backend, param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, beta1_t, beta2_t
    ):
        """
        GPU-accelerated Adam update using adam-update.glsl shader.
        """
        try:
            # Use backend's learning module
            if hasattr(backend, "learning") and hasattr(backend.learning, "adam_update"):
                param, exp_avg, exp_avg_sq = backend.learning.adam_update(
                    weights=param,
                    gradients=grad,
                    moment1=exp_avg,
                    moment2=exp_avg_sq,
                    learning_rate=lr,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=eps,
                    beta1_t=beta1_t,
                    beta2_t=beta2_t,
                    clear_grad=False,
                )
                return param, exp_avg, exp_avg_sq
        except Exception:
            # Fall back to CPU if GPU fails
            pass

        # CPU fallback
        exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
        m_hat = exp_avg / (1 - beta1_t) if beta1_t < 1.0 else exp_avg
        v_hat = exp_avg_sq / (1 - beta2_t) if beta2_t < 1.0 else exp_avg_sq
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return param, exp_avg, exp_avg_sq


class AffectAdam(Adam):
    """
    Affect-aware Adam optimizer.

    Uses: affect-adam.glsl

    Similar to Adam but optimized for affect/emotion processing.
    """

    def __init__(
        self,
        params: Iterator[np.ndarray],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        use_gpu: bool = True,
    ):
        """
        Initialize AffectAdam optimizer.

        Args are the same as Adam.
        """
        super().__init__(params, lr, betas, eps, weight_decay, use_gpu)

    def _adam_update_gpu(
        self, backend, param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, beta1_t, beta2_t
    ):
        """
        GPU-accelerated AffectAdam update using affect-adam.glsl shader.
        """
        # TODO: Implement GPU shader dispatch when backend method is available
        # For now, use CPU fallback (same as Adam)
        return super()._adam_update_gpu(
            backend, param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, beta1_t, beta2_t
        )
