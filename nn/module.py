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
        """Get the legacy VulkanCompute backend if available, else None.

        The legacy backend (``backend.compute.VulkanCompute``) requires the
        PyPI ``vulkan`` Python package (ctypes bindings) and is only used
        today by a few slow-path fallbacks (e.g. ``fnn.xavier_init`` at
        layer construction time). The real forward/backward path goes
        through the C++ ``_bridge`` which only needs ``grilly_core`` and has
        no Python ``vulkan`` dependency.

        Returning ``None`` when the legacy backend can't init lets callers
        that already guard with ``hasattr(backend, "fnn")`` gracefully fall
        through to their CPU reference path (e.g. numpy xavier init), which
        is harmless at construction time since the fast path takes over at
        forward time via the bridge.
        """
        if self._backend is None:
            try:
                from ..utils.device_manager import get_device_manager

                self._backend = get_device_manager().vulkan
            except Exception:
                self._backend = None
        return self._backend

    def _convert_input(self, x: np.ndarray | Any):
        """
        Convert input to Vulkan-compatible format (GPU-first).

        Pass-through priority:
        1. ``torch_api.Tensor`` / ``LongTensor`` — kept as-is so user
           ``forward()`` code keeps torch-style methods (``.unsqueeze``,
           ``.reshape``, ``.mean(dim=...)``). Works alongside grilly's
           existing numpy-native layers because ``Variable.__array__``
           lets ``np.matmul``/``np.dot`` operate on Tensor inputs directly.
        2. ``VulkanTensor`` — GPU-resident, always pass through.
        3. Everything else (numpy, PyTorch, …) — convert to Vulkan-compatible.

        Preserves integer dtypes for index arrays (e.g., token IDs).
        """
        # 1. torch_api Tensor wrappers + autograd Variable: pass through
        # unchanged. Variable matters because Tensor inherits from Variable
        # and ops like ``.reshape`` / ``.mean(dim=...)`` / arithmetic return
        # raw Variable instances rather than the Tensor subclass.
        try:
            from grilly.nn.autograd import Variable as _Variable
            from grilly.torch_api.tensor import LongTensor as _TorchLong
            from grilly.torch_api.tensor import Tensor as _TorchTensor

            if isinstance(x, (_TorchTensor, _TorchLong, _Variable)):
                return x
        except ImportError:
            pass

        # 2. GPU-first: always pass VulkanTensor through without CPU round-trip
        if TENSOR_CONVERSION_AVAILABLE:
            from ..utils.tensor_conversion import VulkanTensor

            if isinstance(x, VulkanTensor):
                return x  # GPU-first: always pass through
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
        """Invoke the callable instance.

        If any positional arg was torch-style (``torch_api.Tensor``,
        ``LongTensor``, or autograd ``Variable``) and the forward returned a
        raw ndarray, wrap the output in ``Tensor`` so chained calls
        (``x = embed(ids); x = layer(x); x = next(x)``) preserve torch-style
        type through user-defined Module subclasses.

        Three classes count as a "torch input":
        - ``LongTensor`` — model entry points use it for token ids; the next
          layer (Embedding) returns floats and the chain has to start in
          torch-style mode from the first step.
        - ``Tensor`` — the public torch_api wrapper users construct directly.
        - ``Variable`` — what ``Tensor.reshape``/``.mean(dim=...)``/arithmetic
          ops actually return today (they delegate to module-level functions
          that build raw Variable, not the Tensor subclass). Without this
          branch the type is silently lost after the first reshape and the
          chain falls back to numpy.

        ``Parameter`` (ndarray subclass) is left untouched on the way out so
        re-wrapping a stored weight doesn't break gradient bookkeeping.
        """
        try:
            from grilly.nn.autograd import Variable as _Variable
            from grilly.torch_api.tensor import LongTensor as _TorchLong
            from grilly.torch_api.tensor import Tensor as _TorchTensor

            saw_torch_input = any(
                isinstance(a, (_TorchTensor, _TorchLong, _Variable)) for a in args
            )
        except ImportError:
            _TorchTensor = None  # type: ignore[assignment]
            _TorchLong = None  # type: ignore[assignment]
            _Variable = None  # type: ignore[assignment]
            saw_torch_input = False

        converted_args = tuple(self._convert_input(arg) for arg in args)
        converted_kwargs = {
            k: self._convert_input(v) if self._is_tensor_like(v) else v
            for k, v in kwargs.items()
        }
        out = self.forward(*converted_args, **converted_kwargs)

        # Wrap raw ndarray outputs back to Tensor when a torch input was seen.
        # Skip ndarray subclasses (Parameter, LongTensor) and Variable outputs
        # — those already carry a torch-compatible API.
        if saw_torch_input and _TorchTensor is not None:
            if isinstance(out, np.ndarray) and type(out) is np.ndarray:
                return _TorchTensor(out, requires_grad=False)
        return out

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

    def register_buffer(self, name: str, tensor: np.ndarray | Any | None, persistent: bool = True):
        """
        Register a non-trainable buffer (e.g. running state not in autograd).

        Args:
            name: Buffer attribute name
            tensor: Numpy array or None to remove
            persistent: If False, omitted from state_dict (PyTorch parity; optional)
        """
        if tensor is None:
            self._buffers.pop(name, None)
            if hasattr(self, name):
                delattr(self, name)
            return
        try:
            from grilly.torch_api.tensor import Tensor as _TorchTensor
        except ImportError:
            _TorchTensor = None  # type: ignore[assignment]
        if _TorchTensor is not None and isinstance(tensor, _TorchTensor):
            self._buffers[name] = tensor
            setattr(self, name, tensor)
        elif hasattr(tensor, "data") and not isinstance(tensor, np.ndarray):
            arr = np.asarray(tensor.data)
            self._buffers[name] = arr
            setattr(self, name, arr)
        else:
            arr = np.asarray(tensor)
            self._buffers[name] = arr
            setattr(self, name, arr)
        if not persistent:
            # Mark non-persistent with a private set on the buffer entry
            setattr(self, f"_np_buffer_{name}", True)

    def state_dict(self) -> dict[str, Any]:
        """Execute state dict."""

        state = {}
        for name, param in self._parameters.items():
            # Extract underlying array for state dict
            # Handles VulkanTensor (.numpy()), Parameter (np subclass), ParamWrapper (.data)
            if hasattr(param, "numpy") and not isinstance(param, np.ndarray):
                state[name] = param.numpy().copy()  # VulkanTensor
            elif isinstance(param, Parameter):
                state[name] = np.array(param, copy=True)
            elif isinstance(param, np.ndarray):
                state[name] = param.copy()
            elif hasattr(param, "data") and isinstance(param.data, np.ndarray):
                state[name] = param.data.copy()  # ParamWrapper
            else:
                state[name] = param
        try:
            from grilly.torch_api.tensor import Tensor as _TorchTensor
        except ImportError:
            _TorchTensor = None  # type: ignore[assignment]
        for name, buf in self._buffers.items():
            if getattr(self, f"_np_buffer_{name}", False):
                continue
            if _TorchTensor is not None and isinstance(buf, _TorchTensor):
                state[name] = buf.data.copy()
            elif isinstance(buf, np.ndarray):
                state[name] = buf.copy()
            else:
                state[name] = np.asarray(buf)
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
        try:
            from grilly.torch_api.tensor import Tensor as _TorchTensor
        except ImportError:
            _TorchTensor = None  # type: ignore[assignment]
        for name in list(self._buffers.keys()):
            if name in state_dict:
                prev = self._buffers.get(name)
                raw = state_dict[name]
                if _TorchTensor is not None and isinstance(prev, _TorchTensor):
                    self._buffers[name] = _TorchTensor(
                        np.asarray(raw, dtype=np.float32).copy(), requires_grad=False
                    )
                else:
                    val = np.asarray(raw, dtype=np.float32)
                    if val.dtype != np.float32 and np.issubdtype(val.dtype, np.floating):
                        val = val.astype(np.float32)
                    self._buffers[name] = val
                setattr(self, name, self._buffers[name])
        for name, module in self._modules.items():
            if name in state_dict:
                module.load_state_dict(state_dict[name])

    def named_buffers(self):
        """Yield (name, buffer) for all buffers including child modules."""
        yield from self._buffers.items()
        for prefix, module in self._modules.items():
            for n, b in module.named_buffers():
                yield f"{prefix}.{n}", b

    def buffers(self):
        """Iterator over buffers."""
        for _, b in self.named_buffers():
            yield b

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

    def __setattr__(self, name: str, value: Any) -> None:
        """Auto-register ``Parameter`` / child ``Module`` attributes (PyTorch parity).

        Without this hook, ``self.weight = nn.Parameter(...)`` and
        ``self.lin = nn.Linear(...)`` only land in ``self.__dict__``, never in
        ``self._parameters`` / ``self._modules``. ``parameters()`` then walks
        empty dicts and the optimizer sees no params to update — silently
        breaking every Module subclass that uses the standard PyTorch
        attribute-assignment idiom.
        """
        if isinstance(value, Parameter):
            # Ensure base __init__ has run; if not, fall back to plain assign.
            params = self.__dict__.get("_parameters")
            if params is not None:
                # Drop any prior child module / buffer with the same name first.
                self.__dict__.get("_modules", {}).pop(name, None)
                self.__dict__.get("_buffers", {}).pop(name, None)
                params[name] = value
                object.__setattr__(self, name, value)
                return
        elif isinstance(value, Module):
            modules = self.__dict__.get("_modules")
            if modules is not None:
                self.__dict__.get("_parameters", {}).pop(name, None)
                self.__dict__.get("_buffers", {}).pop(name, None)
                modules[name] = value
                object.__setattr__(self, name, value)
                return
        object.__setattr__(self, name, value)

    def __repr__(self):
        """Return a debug representation."""

        return f"{self.__class__.__name__}()"
