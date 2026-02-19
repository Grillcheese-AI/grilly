"""
Base Module class (PyTorch-like)
"""

from typing import Any

import numpy as np

# Try to import tensor conversion utilities
try:
    from ..utils.tensor_conversion import ensure_vulkan_compatible, to_vulkan

    TENSOR_CONVERSION_AVAILABLE = True
except ImportError:
    TENSOR_CONVERSION_AVAILABLE = False

# Try to import Parameter class
try:
    from .parameter import Parameter

    PARAMETER_AVAILABLE = True
except ImportError:
    PARAMETER_AVAILABLE = False

    # Fallback: create a simple Parameter-like class
    class Parameter(np.ndarray):
        """Fallback trainable array with gradient storage."""

        def __new__(cls, data, requires_grad=True):
            """Create a parameter view from raw data."""
            obj = np.asarray(data, dtype=np.float32).view(cls)
            obj.requires_grad = requires_grad
            obj.grad = None
            return obj

        def __array_finalize__(self, obj):
            """Propagate metadata when numpy creates array views."""
            if obj is None:
                return
            self.requires_grad = getattr(obj, "requires_grad", True)
            self.grad = getattr(obj, "grad", None)

        def zero_grad(self):
            """Reset gradients to zeros."""
            if self.grad is not None:
                self.grad.fill(0.0)
            else:
                self.grad = np.zeros_like(self, dtype=np.float32)


class Module:
    """Base class for Grilly neural network modules."""

    def __init__(self):
        """Initialize the instance."""

        self.training = True
        self._parameters = {}
        self._buffers = {}
        self._modules = {}
        self._backend = None
        self._device = "vulkan"
        self._grad_enabled = True  # Enable gradients by default
        self._return_gpu_tensor = False  # GPU-resident output mode
        self._use_device_local = False  # DEVICE_LOCAL VRAM buffers

    def _get_backend(self):
        """Execute get backend."""

        if self._backend is None:
            from grilly import Compute

            self._backend = Compute()
        return self._backend

    def _convert_input(self, x: np.ndarray | Any):
        """
        Convert input to Vulkan-compatible format.

        Automatically handles PyTorch tensors, converting them to numpy.
        When GPU-resident mode is enabled, VulkanTensor inputs are passed
        through without downloading to CPU, avoiding unnecessary round-trips.
        Preserves integer dtypes for index arrays (e.g., token IDs).

        Args:
            x: Input (PyTorch tensor, numpy array, VulkanTensor, or other)

        Returns:
            numpy array or VulkanTensor (when GPU-resident mode is active)
        """
        # Handle VulkanTensor (GPU-resident)
        if TENSOR_CONVERSION_AVAILABLE:
            from ..utils.tensor_conversion import VulkanTensor

            if isinstance(x, VulkanTensor):
                # In GPU mode, pass VulkanTensor through to avoid CPU round-trip
                if self._return_gpu_tensor:
                    return x
                return x.numpy()
            return ensure_vulkan_compatible(x)
        else:
            # Fallback conversion - preserve integer types for indexing
            if isinstance(x, np.ndarray):
                # Preserve integer dtypes (needed for embedding lookups, etc.)
                if np.issubdtype(x.dtype, np.integer):
                    return x
                return x.astype(np.float32) if x.dtype != np.float32 else x
            elif hasattr(x, "cpu"):  # PyTorch tensor
                arr = x.detach().cpu().numpy()
                if np.issubdtype(arr.dtype, np.integer):
                    return arr
                return arr.astype(np.float32)
            elif hasattr(x, "numpy"):  # TensorFlow tensor or VulkanTensor
                result = x.numpy()
                if np.issubdtype(result.dtype, np.integer):
                    return result
                return result.astype(np.float32) if result.dtype != np.float32 else result
            else:
                return np.array(x, dtype=np.float32)

    def forward(self, *args, **kwargs):
        """Execute forward."""

        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        # Automatically convert PyTorch tensor inputs to numpy
        """Invoke the callable instance."""

        converted_args = tuple(self._convert_input(arg) for arg in args)
        # Convert keyword arguments that might be tensors
        converted_kwargs = {
            k: self._convert_input(v) if self._is_tensor_like(v) else v for k, v in kwargs.items()
        }
        return self.forward(*converted_args, **converted_kwargs)

    def _is_tensor_like(self, obj: Any) -> bool:
        """Check if object is tensor-like and needs conversion"""
        if isinstance(obj, np.ndarray):
            return False  # Already numpy, no conversion needed
        if hasattr(obj, "cpu") and hasattr(obj, "numpy"):  # PyTorch tensor
            return True
        if hasattr(obj, "numpy") and not isinstance(obj, np.ndarray):  # TensorFlow
            return True
        return False

    def train(self, mode: bool = True):
        """Execute train."""

        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        """Execute eval."""

        return self.train(False)

    def parameters(self):
        """Return iterator over all parameters"""
        for name, param in self._parameters.items():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self):
        """Return iterator over all named parameters"""
        for name, param in self._parameters.items():
            yield name, param
        for prefix, module in self._modules.items():
            for name, param in module.named_parameters():
                yield f"{prefix}.{name}", param

    def zero_grad(self):
        """Clear gradients for all parameters"""
        for param in self.parameters():
            if hasattr(param, "grad") and param.grad is not None:
                param.zero_grad()
            elif hasattr(param, "grad"):
                # Initialize grad if it doesn't exist
                param.grad = np.zeros_like(param, dtype=np.float32)

    def backward(self, loss: np.ndarray):
        """
        Backward pass - compute gradients for all parameters.

        This is a placeholder that should be implemented by subclasses
        or through automatic differentiation.

        Args:
            loss: Loss value (scalar or tensor)
        """
        # For now, this is a placeholder
        # In a full implementation, this would:
        # 1. Compute gradients through the computation graph
        # 2. Store gradients in param.grad for each parameter
        # 3. Use backward shaders from the backend

        # Basic implementation: if loss is a scalar, we need to start backprop
        # For now, we'll just ensure gradients are initialized
        if not self._grad_enabled:
            return

        # Initialize gradients for all parameters
        for param in self.parameters():
            if hasattr(param, "requires_grad") and param.requires_grad:
                if not hasattr(param, "grad") or param.grad is None:
                    param.grad = np.zeros_like(param, dtype=np.float32)

        # Note: Actual gradient computation should be implemented by specific modules
        # or through an autograd system. This is a framework for it.

    def register_parameter(self, name: str, param: np.ndarray | None):
        """
        Register a parameter with the module.

        Args:
            name: Parameter name
            param: Parameter array (will be converted to Parameter if needed)
        """
        if param is None:
            self._parameters.pop(name, None)
            return

        # Convert to Parameter if not already
        if not isinstance(param, Parameter):
            param = Parameter(param, requires_grad=True)

        self._parameters[name] = param

    def state_dict(self) -> dict[str, Any]:
        """Execute state dict."""

        state = {}
        for name, param in self._parameters.items():
            # Extract underlying array for state dict
            if isinstance(param, Parameter):
                state[name] = np.array(param, copy=True)
            else:
                state[name] = param.copy() if isinstance(param, np.ndarray) else param
        for name, module in self._modules.items():
            state[name] = module.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Execute load state dict."""

        for name, param in self._parameters.items():
            if name in state_dict:
                if isinstance(param, np.ndarray):
                    self._parameters[name] = state_dict[name].copy()
                else:
                    self._parameters[name] = state_dict[name]
        for name, module in self._modules.items():
            if name in state_dict:
                module.load_state_dict(state_dict[name])

    def gpu_mode(self, enable=True, device_local=True):
        """Enable GPU-resident output (returns VulkanTensor instead of numpy).

        When enabled, operations will accept VulkanTensor inputs without
        downloading to CPU, and subclasses that support it will return
        VulkanTensor outputs.  This eliminates CPU round-trips for chained
        operations (e.g. linear -> relu -> linear).

        Args:
            enable: True to enable, False to disable
            device_local: When True (default), intermediate buffers use
                DEVICE_LOCAL VRAM instead of HOST_VISIBLE memory. This
                keeps activations in fast GDDR6/HBM (384 GB/s) rather than
                crossing the PCIe bus (14 GB/s). Only effective when
                *enable* is also True.

        Returns:
            self (for chaining)
        """
        self._return_gpu_tensor = enable
        self._use_device_local = enable and device_local
        for module in self._modules.values():
            module.gpu_mode(enable, device_local)
        return self

    def to(self, device=None):
        """Execute to."""

        if device is None:
            return self
        device = str(device).lower()
        if device in ("cuda", "vulkan", "llama-cpp", "cpu"):
            self._device = device
        return self

    def cpu(self):
        """Execute cpu."""

        return self.to("cpu")

    def cuda(self):
        """Execute cuda."""

        return self.to("cuda")

    def vulkan(self):
        """Execute vulkan."""

        return self.to("vulkan")

    def llama_cpp(self):
        """Execute llama cpp."""

        return self.to("llama-cpp")

    def __repr__(self):
        """Return a debug representation."""

        return f"{self.__class__.__name__}()"
