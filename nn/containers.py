"""
Container modules — Sequential and Residual.
"""

import numpy as np

from ._helpers import _get_param_array
from .module import Module

# Fused activation map for Sequential auto-fusion
_FUSED_ACTIVATION_MAP = {
    "ReLU": "fused_linear_relu",
    "GELU": "fused_linear_gelu",
    "SiLU": "fused_linear_silu",
}


class Sequential(Module):
    """
    Sequential container for modules

    Caches intermediate activations during forward pass for efficient backward pass.
    Automatically fuses Linear+Activation pairs when fused GPU shaders are available.
    When a fused shader is missing, `fused_linear_relu` may use a two-kernel **single-submit**
    path (`VulkanFNN._linear_relu_recorded_chain`) before falling back to separate calls.
    """

    def __init__(self, *modules):
        """Initialize the instance."""

        super().__init__()
        for i, module in enumerate(modules):
            self._modules[str(i)] = module
        self._cached_activations = []  # Cache intermediate activations
        self._fusion_plan = None  # Lazily computed

    def _compute_fusion_plan(self):
        """Scan module list for fusible Linear → Activation pairs.

        Returns a list of tuples:
          ('fuse', linear_idx, act_idx, fused_method_name)  — fused pair
          ('run', idx)                                       — run module normally
        """
        # Import Linear here to avoid circular imports
        from .linear import Linear

        modules_list = list(self._modules.values())
        n = len(modules_list)
        plan = []
        i = 0
        while i < n:
            if (
                i + 1 < n
                and isinstance(modules_list[i], Linear)
                and type(modules_list[i + 1]).__name__ in _FUSED_ACTIVATION_MAP
            ):
                method_name = _FUSED_ACTIVATION_MAP[type(modules_list[i + 1]).__name__]
                plan.append(("fuse", i, i + 1, method_name))
                i += 2
            else:
                plan.append(("run", i))
                i += 1
        return plan

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with automatic Linear+Activation fusion."""
        # Import Linear here to avoid circular imports

        # Clear cached activations
        self._cached_activations = [x]  # Store initial input

        modules_list = list(self._modules.values())

        # Lazily compute fusion plan (invalidated if modules change)
        if self._fusion_plan is None or len(modules_list) != sum(
            2 if s[0] == "fuse" else 1 for s in self._fusion_plan
        ):
            self._fusion_plan = self._compute_fusion_plan()

        current = x
        for step in self._fusion_plan:
            if step[0] == "fuse":
                _, lin_idx, act_idx, method_name = step
                linear_mod = modules_list[lin_idx]
                weight = _get_param_array(linear_mod.weight)
                bias = _get_param_array(linear_mod.bias) if linear_mod.bias is not None else None
                backend = linear_mod._get_backend()
                fused_fn = getattr(backend.fnn, method_name, None) if backend is not None else None
                if fused_fn is not None:
                    try:
                        current = fused_fn(
                            current,
                            weight,
                            bias,
                            return_gpu_tensor=linear_mod._return_gpu_tensor,
                        )
                        # Push two entries for backward indexing consistency
                        self._cached_activations.append(current)  # output of Linear
                        self._cached_activations.append(current)  # output of Activation
                        continue
                    except Exception:
                        pass  # Fall back to sequential execution
                # Fallback: run both modules individually
                current = modules_list[lin_idx](current)
                self._cached_activations.append(current)
                current = modules_list[act_idx](current)
                self._cached_activations.append(current)
            else:
                _, idx = step
                current = modules_list[idx](current)
                self._cached_activations.append(current)

        return current

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass through all modules in reverse order.

        Uses cached activations from forward pass.

        Args:
            grad_output: Gradient w.r.t. output
            x: Original input (optional, uses cached if available)

        Returns:
            grad_input: Gradient w.r.t. input
        """
        # Use cached activations if available
        if len(self._cached_activations) == 0:
            # If no cached activations, we can't do proper backward
            # This shouldn't happen if forward was called first
            raise RuntimeError("No cached activations found. Call forward() before backward().")

        grad = grad_output
        modules_list = list(self._modules.values())

        # Backward through modules in reverse order
        # cached_activations[0] is input, cached_activations[i+1] is output of module i
        for i in range(len(modules_list) - 1, -1, -1):
            module = modules_list[i]
            module_input = self._cached_activations[i]  # Input to module i
            self._cached_activations[i + 1]  # Output of module i

            if hasattr(module, "backward"):
                try:
                    # Pass both input and output for backward (some modules need both)
                    grad = module.backward(grad, module_input)
                except TypeError:
                    # Some backward methods only take grad_output
                    grad = module.backward(grad)
                except NotImplementedError:
                    # If backward not implemented, just pass through
                    pass

        return grad

    def __repr__(self):
        """Return a debug representation."""

        modules_str = ",\n  ".join([str(m) for m in self._modules.values()])
        return f"Sequential(\n  {modules_str}\n)"


class Residual(Module):
    """
    Residual connection: output = input + module(input)
    Uses: fnn-residual.glsl
    """

    def __init__(self, module: Module):
        """Initialize the instance."""

        super().__init__()
        self.module = module
        self._modules["module"] = module
        self._cached_input = None
        self._cached_module_output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: x + module(x)"""
        backend = self._get_backend()
        module_out = self.module(x)

        # Cache input for backward pass
        self._cached_input = x
        self._cached_module_output = module_out

        # Try GPU shader if available
        if hasattr(backend, "fnn") and hasattr(backend.fnn, "residual"):
            try:
                return backend.fnn.residual(x, module_out)
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        return x + module_out

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for Residual.

        Residual: output = input + module(input)
        Gradient: grad_input = grad_output + grad_module(input)

        Args:
            grad_output: Gradient w.r.t. output
            x: Input from forward pass (optional, uses cached if available)

        Returns:
            grad_input: Gradient w.r.t. input
        """
        # Use cached input if available
        if x is None:
            x = self._cached_input
            if x is None:
                raise ValueError("Input x is required for Residual backward pass")

        # Residual backward: grad_input = grad_output + grad_module
        # The gradient flows through both the residual connection and the module
        if hasattr(self.module, "backward"):
            try:
                grad_module = self.module.backward(grad_output, x)
                return grad_output + grad_module
            except (TypeError, NotImplementedError):
                # If backward not properly implemented, just pass through residual gradient
                return grad_output
        return grad_output

    def __repr__(self):
        """Return a debug representation."""

        return f"Residual({self.module})"
