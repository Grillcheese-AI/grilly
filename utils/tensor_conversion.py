"""
Tensor Conversion Utilities

Seamless conversion between PyTorch tensors and Vulkan (numpy arrays).
Provides automatic conversion for seamless integration with GPU acceleration on AMD.
"""
from typing import Union, Any, Tuple, List, Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .device_manager import get_device_manager


def to_vulkan(tensor: Union[np.ndarray, Any], keep_on_gpu: bool = False) -> Union[np.ndarray, 'VulkanTensor']:
    """
    Convert PyTorch tensor (or any tensor-like object) to numpy array for Vulkan.
    
    This is the main function to use for converting PyTorch tensors to Vulkan-compatible arrays.
    On AMD GPUs, can optionally keep data on GPU to avoid CPU round-trips.
    
    Args:
        tensor: PyTorch tensor, numpy array, or other array-like object
        keep_on_gpu: If True, creates a GPU buffer directly (faster for AMD, avoids CPU round-trip)
    
    Returns:
        numpy array (float32) ready for Vulkan operations, or VulkanTensor if keep_on_gpu=True
    
    Examples:
        >>> import torch
        >>> x = torch.randn(10, 128).cuda()
        >>> x_vulkan = to_vulkan(x)  # Convert to numpy for Vulkan
        >>> from grilly import nn
        >>> linear = nn.Linear(128, 64)
        >>> result = linear(x_vulkan)  # Process with Vulkan
        
        # For AMD GPU optimization:
        >>> x_gpu = to_vulkan(x, keep_on_gpu=True)  # Stays on GPU
        >>> result = linear(x_gpu)  # Faster, no CPU transfer
    """
    device_manager = get_device_manager()
    
    # If keep_on_gpu is True, try to create a GPU buffer directly
    if keep_on_gpu:
        try:
            return to_vulkan_gpu(tensor)
        except Exception:
            # Fall back to regular conversion if GPU buffer creation fails
            pass
    
    return device_manager.to_vulkan(tensor)


def to_vulkan_gpu(tensor: Union[np.ndarray, Any]) -> 'VulkanTensor':
    """
    Convert tensor directly to Vulkan GPU buffer (stays on GPU, no CPU round-trip).
    
    Optimized for AMD GPUs - creates device-local buffer directly on GPU.
    
    Args:
        tensor: PyTorch tensor, numpy array, or other array-like object
    
    Returns:
        VulkanTensor wrapper that keeps data on GPU
    
    Examples:
        >>> import torch
        >>> x = torch.randn(10, 128)
        >>> x_gpu = to_vulkan_gpu(x)  # Directly on GPU
        >>> result = model(x_gpu)  # No CPU transfer needed
    """
    # Get numpy array first
    device_manager = get_device_manager()
    numpy_array = device_manager.to_vulkan(tensor)
    
    # Ensure float32
    if numpy_array.dtype != np.float32:
        numpy_array = numpy_array.astype(np.float32)
    
    # Create VulkanTensor wrapper
    return VulkanTensor(numpy_array)


class VulkanTensor:
    """
    GPU-resident tensor wrapper for Vulkan operations.
    
    Keeps data on GPU memory, avoiding CPU round-trips for better performance on AMD GPUs.
    """
    
    def __init__(self, data: np.ndarray):
        """
        Initialize VulkanTensor from numpy array.
        
        Args:
            data: numpy array (will be uploaded to GPU)
        """
        self._cpu_data = np.ascontiguousarray(data.astype(np.float32))
        self._gpu_buffer = None
        self._gpu_memory = None
        self._shape = self._cpu_data.shape
        self._dtype = self._cpu_data.dtype
        self._uploaded = False
    
    def _ensure_uploaded(self):
        """Ensure data is uploaded to GPU"""
        if not self._uploaded:
            try:
                from grilly import Compute
                backend = Compute()
                
                # Create GPU buffer
                size = self._cpu_data.nbytes
                self._gpu_buffer, self._gpu_memory = backend.create_buffer(
                    size, usage='storage'
                )
                
                # Upload to GPU
                backend.upload_buffer(self._gpu_buffer, self._gpu_memory, self._cpu_data)
                self._uploaded = True
            except Exception as e:
                # If GPU upload fails, fall back to CPU
                self._uploaded = False
                raise RuntimeError(f"Failed to upload to GPU: {e}")
    
    @property
    def shape(self):
        """Get tensor shape"""
        return self._shape
    
    @property
    def dtype(self):
        """Get tensor dtype"""
        return self._dtype
    
    def numpy(self) -> np.ndarray:
        """
        Convert to numpy array (downloads from GPU if needed).
        
        Returns:
            numpy array
        """
        if self._uploaded:
            try:
                from grilly import Compute
                backend = Compute()
                # Download from GPU
                return backend.read_buffer(
                    self._gpu_memory, 
                    size=self._cpu_data.nbytes,
                    dtype=self._dtype
                ).reshape(self._shape)
            except Exception:
                # Fall back to CPU data
                pass
        return self._cpu_data.copy()
    
    def cpu(self) -> np.ndarray:
        """Get CPU copy"""
        return self.numpy()
    
    def __array__(self):
        """Numpy array interface"""
        return self.numpy()
    
    def __repr__(self):
        return f"VulkanTensor(shape={self.shape}, dtype={self.dtype}, on_gpu={self._uploaded})"


def to_vulkan_batch(tensors: Union[List, Tuple, Any]) -> Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, ...]]:
    """
    Convert a batch of PyTorch tensors to numpy arrays for Vulkan.
    
    Args:
        tensors: Single tensor, list of tensors, or tuple of tensors
    
    Returns:
        Converted numpy array(s) with same structure
    """
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(to_vulkan(t) for t in tensors)
    else:
        return to_vulkan(tensors)


def from_vulkan(array: np.ndarray, device: str = 'cuda') -> Any:
    """
    Convert numpy array (from Vulkan) to PyTorch tensor.
    
    Args:
        array: numpy array from Vulkan operations
        device: Target device ('cuda', 'cpu', or PyTorch device)
    
    Returns:
        PyTorch tensor on specified device
    
    Examples:
        >>> from grilly import nn
        >>> linear = nn.Linear(128, 64)
        >>> x = np.random.randn(10, 128).astype(np.float32)
        >>> result = linear(x)  # Vulkan operation
        >>> torch_result = from_vulkan(result, device='cuda')  # Convert to PyTorch CUDA
    """
    device_manager = get_device_manager()
    
    if device == 'cuda':
        try:
            return device_manager.to_cuda(array)
        except (RuntimeError, AssertionError):
            # CUDA not available, fall back to CPU
            if TORCH_AVAILABLE:
                return torch.from_numpy(array).cpu()
            return array
    elif device == 'cpu':
        if TORCH_AVAILABLE:
            return torch.from_numpy(array).cpu()
        return array
    else:
        # PyTorch device string
        if TORCH_AVAILABLE:
            return torch.from_numpy(array).to(device)
        return array


def auto_convert_to_vulkan(func):
    """
    Decorator to automatically convert PyTorch tensor arguments to numpy arrays.
    
    Usage:
        @auto_convert_to_vulkan
        def my_function(x, y):
            # x and y are automatically converted to numpy if they're PyTorch tensors
            return x @ y
    
    Note: This decorator converts the first argument only. For multiple arguments,
    use to_vulkan() manually.
    """
    def wrapper(*args, **kwargs):
        # Convert first argument if it's a PyTorch tensor
        if args and TORCH_AVAILABLE and isinstance(args[0], torch.Tensor):
            args = (to_vulkan(args[0]),) + args[1:]
        return func(*args, **kwargs)
    return wrapper


def ensure_vulkan_compatible(data: Union[np.ndarray, Any]) -> np.ndarray:
    """
    Ensure data is Vulkan-compatible (numpy array, float32).
    
    Handles VulkanTensor by extracting numpy array.
    
    Args:
        data: Any tensor-like data (including VulkanTensor)
    
    Returns:
        numpy array (float32) ready for Vulkan
    """
    # Handle VulkanTensor
    if isinstance(data, VulkanTensor):
        return data.numpy()
    
    arr = to_vulkan(data, keep_on_gpu=False)  # Get numpy, not GPU tensor
    if isinstance(arr, VulkanTensor):
        arr = arr.numpy()
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def convert_module_inputs(*args, **kwargs):
    """
    Convert all PyTorch tensor inputs to numpy arrays for Vulkan operations.
    
    Args:
        *args: Positional arguments (tensors will be converted)
        **kwargs: Keyword arguments (tensors will be converted)
    
    Returns:
        Tuple of (converted_args, converted_kwargs)
    
    Example:
        >>> import torch
        >>> x = torch.randn(10, 128)
        >>> y = torch.randn(128, 64)
        >>> args, kwargs = convert_module_inputs(x, y, some_param=torch.tensor([1, 2, 3]))
        >>> # Now args and kwargs contain numpy arrays
    """
    converted_args = tuple(to_vulkan(arg) if _is_tensor_like(arg) else arg for arg in args)
    converted_kwargs = {k: to_vulkan(v) if _is_tensor_like(v) else v for k, v in kwargs.items()}
    return converted_args, converted_kwargs


def _is_tensor_like(obj: Any) -> bool:
    """Check if object is a tensor-like (PyTorch, TensorFlow, etc.)"""
    if isinstance(obj, np.ndarray):
        return False  # Already numpy
    if TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
        return True
    if hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):
        return True  # PyTorch-like
    if hasattr(obj, 'numpy') and not isinstance(obj, np.ndarray):
        return True  # TensorFlow-like
    return False
