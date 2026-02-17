"""
AdamW Optimizer

AdamW (Adam with decoupled Weight decay) - more effective regularization than Adam.

Key difference from Adam:
- Adam: weight decay is added to gradient before moment updates (coupled)
- AdamW: weight decay is applied directly to parameters after Adam step (decoupled)

Reference: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)

Uses: adamw-update.glsl
"""

from collections.abc import Iterator

import numpy as np

from .base import Optimizer


class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay.

    Implements the AdamW algorithm:
    - m = beta1 * m + (1 - beta1) * grad
    - v = beta2 * v + (1 - beta2) * grad^2
    - m_hat = m / (1 - beta1^t)
    - v_hat = v / (1 - beta2^t)
    - param = param - lr * m_hat / (sqrt(v_hat) + eps)  # Adam step
    - param = param - lr * weight_decay * param  # Decoupled weight decay

    This decoupling improves generalization compared to Adam's coupled weight decay.
    """

    def __init__(
        self,
        params: Iterator[np.ndarray],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,  # Default 0.01 (higher than Adam's 0.0)
        amsgrad: bool = False,
        use_gpu: bool = True,
    ):
        """
        Initialize AdamW optimizer.

        Args:
            params: Iterator of parameter arrays to optimize
            lr: Learning rate (default: 1e-3)
            betas: Coefficients for computing running averages (default: (0.9, 0.999))
            eps: Term added to denominator for numerical stability (default: 1e-8)
            weight_decay: Decoupled weight decay coefficient (default: 0.01)
            amsgrad: Whether to use AMSGrad variant (default: False)
            use_gpu: Whether to use GPU acceleration (default: True)
        """
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
        }
        super().__init__(params, defaults)
        self.use_gpu = use_gpu
        self._backend = None
        self._step_count = 0
        self.last_backend = "uninitialized"
        self.last_error = None

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
        amsgrad = self.defaults["amsgrad"]

        # Bias correction terms
        beta1_t = beta1**self._step_count
        beta2_t = beta2**self._step_count

        backend = self._get_backend()
        use_gpu = self.use_gpu and backend is not None
        used_gpu_shader = False
        used_cpu_fallback = False

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
                    if amsgrad:
                        state["max_exp_avg_sq"] = np.zeros_like(p, dtype=np.float32)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Get gradients
                grad = None
                if hasattr(self, "_gradients") and self._gradients is not None:
                    grad = self._gradients.get(param_id, None)
                if grad is None:
                    grad = getattr(p, "grad", None)
                if grad is None:
                    continue

                # Ensure gradient is numpy array
                if hasattr(grad, "data"):
                    grad = grad.data
                if not isinstance(grad, np.ndarray):
                    grad = np.array(grad, dtype=np.float32)

                # Extract parameter data
                if hasattr(p, "data"):
                    p_data = p.data
                else:
                    p_data = p

                # Ensure p_data is numpy array (handle memoryview)
                if not isinstance(p_data, np.ndarray):
                    p_data = np.asarray(p_data, dtype=np.float32)

                # Try GPU update if available
                if use_gpu and backend is not None:
                    try:
                        shaders_available = hasattr(backend, "core") and hasattr(
                            backend.core, "shaders"
                        )
                        if shaders_available and (
                            "adamw-update" in backend.core.shaders
                            or "adam-update" in backend.core.shaders
                        ):
                            p_data, exp_avg, exp_avg_sq = self._adamw_update_gpu(
                                backend,
                                p_data,
                                grad,
                                exp_avg,
                                exp_avg_sq,
                                lr,
                                beta1,
                                beta2,
                                eps,
                                weight_decay,
                                beta1_t,
                                beta2_t,
                                amsgrad,
                            )
                            if hasattr(p, "data") and not isinstance(p, np.ndarray):
                                p.data = p_data
                            else:
                                p[:] = p_data
                            state["exp_avg"] = exp_avg
                            state["exp_avg_sq"] = exp_avg_sq
                            used_gpu_shader = True
                            if hasattr(p, "grad") and p.grad is not None:
                                if hasattr(p, "zero_grad"):
                                    p.zero_grad()
                                else:
                                    p.grad = None
                            continue
                    except Exception as e:
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.debug(f"GPU AdamW update failed: {e}, falling back to CPU")
                        self.last_error = str(e)
                        pass

                # CPU fallback
                # Update biased first and second moment estimates
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad

                if amsgrad:
                    # Maintain max of all second moment running averages
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    max_exp_avg_sq = np.maximum(max_exp_avg_sq, exp_avg_sq)
                    state["max_exp_avg_sq"] = max_exp_avg_sq
                    denom = np.sqrt(max_exp_avg_sq / (1 - beta2_t)) + eps
                else:
                    denom = np.sqrt(exp_avg_sq / (1 - beta2_t)) + eps

                # Bias-corrected first moment estimate
                step_size = lr / (1 - beta1_t)

                # Decoupled weight decay (applied to original parameter)
                if weight_decay != 0:
                    p_data = p_data * (1 - lr * weight_decay)

                # Adam update (applied to decayed parameter)
                p_data -= step_size * exp_avg / denom

                # Update parameter
                if hasattr(p, "data") and not isinstance(p, np.ndarray):
                    p.data = p_data
                else:
                    p[:] = p_data

                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
                used_cpu_fallback = True

                # Clear gradient
                if hasattr(p, "grad") and p.grad is not None:
                    if hasattr(p, "zero_grad"):
                        p.zero_grad()
                    else:
                        p.grad = None

        if used_gpu_shader and not used_cpu_fallback:
            self.last_backend = "gpu_shader"
        elif used_gpu_shader and used_cpu_fallback:
            self.last_backend = "mixed_gpu_cpu"
        else:
            self.last_backend = "cpu_fallback"

        return loss

    def _adamw_update_gpu(
        self,
        backend,
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        beta1_t,
        beta2_t,
        amsgrad,
    ):
        """
        GPU-accelerated AdamW update using adamw-update.glsl shader.
        """
        try:
            if hasattr(backend, "learning") and hasattr(backend.learning, "adamw_update"):
                param, exp_avg, exp_avg_sq = backend.learning.adamw_update(
                    weights=param,
                    gradients=grad,
                    moment1=exp_avg,
                    moment2=exp_avg_sq,
                    learning_rate=lr,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=eps,
                    weight_decay=weight_decay,
                    beta1_t=beta1_t,
                    beta2_t=beta2_t,
                    clear_grad=False,
                )
                return param, exp_avg, exp_avg_sq
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
                if weight_decay != 0:
                    param = param * (1 - lr * weight_decay)
                return param, exp_avg, exp_avg_sq
        except Exception:
            pass

        # CPU fallback
        exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad

        if amsgrad:
            # Not implemented for CPU fallback - just use regular
            denom = np.sqrt(exp_avg_sq / (1 - beta2_t)) + eps
        else:
            denom = np.sqrt(exp_avg_sq / (1 - beta2_t)) + eps

        step_size = lr / (1 - beta1_t)

        # Decoupled weight decay (applied to original parameter)
        if weight_decay != 0:
            param = param * (1 - lr * weight_decay)

        # Adam update (applied to decayed parameter)
        param -= step_size * exp_avg / denom

        return param, exp_avg, exp_avg_sq
