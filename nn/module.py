"""
Base Module class (PyTorch-like)
"""
import numpy as np
from typing import Dict, Any, Optional, Union

# Try to import tensor conversion utilities
try:
    from ..utils.tensor_conversion import to_vulkan, ensure_vulkan_compatible
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
        def __new__(cls, data, requires_grad=True):
            obj = np.asarray(data, dtype=np.float32).view(cls)
            obj.requires_grad = requires_grad
            obj.grad = None
            return obj
        def __array_finalize__(self, obj):
            if obj is None:
                return
            self.requires_grad = getattr(obj, 'requires_grad', True)
            self.grad = getattr(obj, 'grad', None)
        def zero_grad(self):
            if self.grad is not None:
                self.grad.fill(0.0)
            else:
                self.grad = np.zeros_like(self, dtype=np.float32)


class Module:
    def __init__(self):
        self.training = True
        self._parameters = {}
        self._buffers = {}
        self._modules = {}
        self._backend = None
        self._device = 'vulkan'
        self._grad_enabled = True  # Enable gradients by default
    
    def _get_backend(self):
        if self._backend is None:
            from grilly import Compute
            self._backend = Compute()
        return self._backend
    
    def _convert_input(self, x: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Convert input to Vulkan-compatible numpy array.
        
        Automatically handles PyTorch tensors, converting them to numpy.
        For VulkanTensor, extracts numpy array (handles GPU download if needed).
        
        Args:
            x: Input (PyTorch tensor, numpy array, VulkanTensor, or other)
        
        Returns:
            numpy array (float32) ready for Vulkan operations
        """
        # Handle VulkanTensor (GPU-resident)
        if TENSOR_CONVERSION_AVAILABLE:
            from ..utils.tensor_conversion import VulkanTensor
            if isinstance(x, VulkanTensor):
                # Extract numpy from GPU tensor
                return x.numpy()
            return ensure_vulkan_compatible(x)
        else:
            # Fallback conversion
            if isinstance(x, np.ndarray):
                return x.astype(np.float32) if x.dtype != np.float32 else x
            elif hasattr(x, 'cpu'):  # PyTorch tensor
                return x.detach().cpu().numpy().astype(np.float32)
            elif hasattr(x, 'numpy'):  # TensorFlow tensor or VulkanTensor
                result = x.numpy()
                return result.astype(np.float32) if result.dtype != np.float32 else result
            else:
                return np.array(x, dtype=np.float32)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        # Automatically convert PyTorch tensor inputs to numpy
        converted_args = tuple(self._convert_input(arg) for arg in args)
        # Convert keyword arguments that might be tensors
        converted_kwargs = {k: self._convert_input(v) if self._is_tensor_like(v) else v 
                           for k, v in kwargs.items()}
        return self.forward(*converted_args, **converted_kwargs)
    
    def _is_tensor_like(self, obj: Any) -> bool:
        """Check if object is tensor-like and needs conversion"""
        if isinstance(obj, np.ndarray):
            return False  # Already numpy, no conversion needed
        if hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):  # PyTorch tensor
            return True
        if hasattr(obj, 'numpy') and not isinstance(obj, np.ndarray):  # TensorFlow
            return True
        return False
    
    def train(self, mode: bool = True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self):
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
            if hasattr(param, 'grad') and param.grad is not None:
                param.zero_grad()
            elif hasattr(param, 'grad'):
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
            if hasattr(param, 'requires_grad') and param.requires_grad:
                if not hasattr(param, 'grad') or param.grad is None:
                    param.grad = np.zeros_like(param, dtype=np.float32)
        
        # Note: Actual gradient computation should be implemented by specific modules
        # or through an autograd system. This is a framework for it.
    
    def register_parameter(self, name: str, param: Optional[np.ndarray]):
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
    
    def state_dict(self) -> Dict[str, Any]:
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
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        for name, param in self._parameters.items():
            if name in state_dict:
                if isinstance(param, np.ndarray):
                    self._parameters[name] = state_dict[name].copy()
                else:
                    self._parameters[name] = state_dict[name]
        for name, module in self._modules.items():
            if name in state_dict:
                module.load_state_dict(state_dict[name])
    
    def to(self, device=None):
        if device is None:
            return self
        device = str(device).lower()
        if device in ('cuda', 'vulkan', 'llama-cpp', 'cpu'):
            self._device = device
        return self
    
    def cpu(self):
        return self.to('cpu')
    
    def cuda(self):
        return self.to('cuda')
    
    def vulkan(self):
        return self.to('vulkan')
    
    def llama_cpp(self):
        return self.to('llama-cpp')
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
