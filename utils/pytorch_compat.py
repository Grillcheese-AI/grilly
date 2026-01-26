"""
PyTorch Compatibility Layer

Provides PyTorch-like tensor operations and utilities for seamless
integration with PyTorch workflows while using Vulkan backend.
"""
from typing import Optional, Union, Tuple, List, Any
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .device_manager import get_device_manager


class Tensor:
    """
    PyTorch-compatible tensor wrapper that uses Vulkan backend.
    
    Provides PyTorch-like API while using Vulkan for computation.
    """
    
    def __init__(self, data: Union[np.ndarray, 'Tensor', Any], device: Optional[str] = None):
        """
        Initialize tensor.
        
        Args:
            data: numpy array, PyTorch tensor, or other array-like
            device: Device ('vulkan', 'cuda', 'cpu')
        """
        self.device_manager = get_device_manager()
        
        # Convert to numpy
        if isinstance(data, Tensor):
            self._data = data._data.copy()
        elif isinstance(data, np.ndarray):
            self._data = data.astype(np.float32)
        elif TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            self._data = data.detach().cpu().numpy().astype(np.float32)
        else:
            self._data = np.array(data, dtype=np.float32)
        
        self.device = device or self.device_manager.get_device()
        self.shape = self._data.shape
        self.dtype = self._data.dtype
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        return self._data
    
    def cpu(self) -> 'Tensor':
        """Move to CPU"""
        return Tensor(self._data, device='cpu')
    
    def cuda(self, device: Optional[int] = None) -> torch.Tensor:
        """Move to CUDA (returns PyTorch tensor)"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for CUDA operations")
        return self.device_manager.to_cuda(self._data)
    
    def to(self, device: Union[str, torch.device]) -> Union['Tensor', torch.Tensor]:
        """Move to device"""
        if isinstance(device, str):
            if device == 'cpu':
                return self.cpu()
            elif device.startswith('cuda'):
                return self.cuda()
            else:
                return Tensor(self._data, device=device)
        else:
            # PyTorch device
            return self.cuda()
    
    def __add__(self, other):
        """Element-wise addition"""
        if isinstance(other, Tensor):
            return Tensor(self._data + other._data, device=self.device)
        return Tensor(self._data + other, device=self.device)
    
    def __sub__(self, other):
        """Element-wise subtraction"""
        if isinstance(other, Tensor):
            return Tensor(self._data - other._data, device=self.device)
        return Tensor(self._data - other, device=self.device)
    
    def __mul__(self, other):
        """Element-wise multiplication"""
        if isinstance(other, Tensor):
            return Tensor(self._data * other._data, device=self.device)
        return Tensor(self._data * other, device=self.device)
    
    def __truediv__(self, other):
        """Element-wise division"""
        if isinstance(other, Tensor):
            return Tensor(self._data / other._data, device=self.device)
        return Tensor(self._data / other, device=self.device)
    
    def __matmul__(self, other):
        """Matrix multiplication"""
        if isinstance(other, Tensor):
            return Tensor(self._data @ other._data, device=self.device)
        return Tensor(self._data @ other, device=self.device)
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"
    
    def detach(self) -> 'Tensor':
        """Detach (no-op for numpy backend)"""
        return Tensor(self._data.copy(), device=self.device)
    
    def requires_grad_(self, requires_grad: bool = True) -> 'Tensor':
        """Set requires_grad (no-op for numpy backend)"""
        return self
    
    @property
    def grad(self):
        """Gradient (not supported)"""
        return None
    
    def backward(self, gradient: Optional[np.ndarray] = None):
        """Backward pass (not supported)"""
        raise NotImplementedError("Backward pass not supported for numpy backend")


# PyTorch-like functions
def tensor(data: Union[np.ndarray, List, Any], device: Optional[str] = None) -> Tensor:
    """Create a tensor (PyTorch-like API)"""
    return Tensor(data, device=device)


def zeros(shape: Tuple[int, ...], device: Optional[str] = None) -> Tensor:
    """Create tensor of zeros"""
    return Tensor(np.zeros(shape, dtype=np.float32), device=device)


def ones(shape: Tuple[int, ...], device: Optional[str] = None) -> Tensor:
    """Create tensor of ones"""
    return Tensor(np.ones(shape, dtype=np.float32), device=device)


def randn(*shape: int, device: Optional[str] = None) -> Tensor:
    """Create tensor with random normal values"""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(np.random.randn(*shape).astype(np.float32), device=device)


def arange(start: int, end: Optional[int] = None, step: int = 1, device: Optional[str] = None) -> Tensor:
    """Create tensor with range"""
    if end is None:
        return Tensor(np.arange(start, dtype=np.float32), device=device)
    return Tensor(np.arange(start, end, step, dtype=np.float32), device=device)


def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """Concatenate tensors"""
    arrays = [t._data for t in tensors]
    return Tensor(np.concatenate(arrays, axis=dim), device=tensors[0].device if tensors else None)


def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """Stack tensors"""
    arrays = [t._data for t in tensors]
    return Tensor(np.stack(arrays, axis=dim), device=tensors[0].device if tensors else None)


def from_numpy(array: np.ndarray) -> Tensor:
    """Create tensor from numpy array"""
    return Tensor(array)


def to_numpy(tensor: Union[Tensor, torch.Tensor]) -> np.ndarray:
    """Convert tensor to numpy"""
    if isinstance(tensor, Tensor):
        return tensor.numpy()
    elif TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return np.array(tensor)
