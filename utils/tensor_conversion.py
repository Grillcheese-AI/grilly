"""
Tensor Conversion Utilities

Seamless conversion between PyTorch tensors and Vulkan (numpy arrays).
Provides automatic conversion for seamless integration with GPU acceleration on AMD.
"""

from typing import Any, Union

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .device_manager import get_device_manager


def to_vulkan(
    tensor: np.ndarray | Any, keep_on_gpu: bool = False
) -> Union[np.ndarray, "VulkanTensor"]:
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


def to_vulkan_gpu(tensor: np.ndarray | Any) -> "VulkanTensor":
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

    Features:
    - Lazy transfer: Only uploads to GPU when actually needed
    - Dirty tracking: Knows when CPU/GPU copies are out of sync
    - Buffer pooling: Reuses GPU buffers for efficiency
    - PyTorch bridge: Seamless conversion to/from PyTorch tensors

    Example:
        >>> x = VulkanTensor(np.random.randn(10, 128).astype(np.float32))
        >>> # Data stays on CPU until needed
        >>> result = model(x)  # Triggers upload to GPU
        >>> result_np = result.numpy()  # Downloads from GPU (lazy)
    """

    def __init__(self, data: np.ndarray, lazy: bool = True):
        """
        Initialize VulkanTensor from numpy array.

        Args:
            data: numpy array (will be uploaded to GPU lazily)
            lazy: If True (default), defer GPU upload until needed
        """
        # Handle integer types - preserve them
        if np.issubdtype(data.dtype, np.integer):
            self._cpu_data = np.ascontiguousarray(data)
        else:
            self._cpu_data = np.ascontiguousarray(data.astype(np.float32))

        self._gpu_buffer = None
        self._gpu_memory = None
        self._pooled_buffer = None  # For buffer pool integration
        self._core = None  # VulkanCore reference for fast download (avoids re-init)
        self._is_device_local = False  # True when buffer is DEVICE_LOCAL VRAM
        self._shape = self._cpu_data.shape
        self._dtype = self._cpu_data.dtype

        # State tracking
        self._gpu_valid = False  # GPU has valid copy
        self._cpu_valid = True  # CPU has valid copy
        self._uploaded = False  # Backwards compatibility

        # Lazy upload
        if not lazy:
            self._ensure_uploaded()

    @classmethod
    def from_torch(cls, tensor, lazy: bool = True) -> "VulkanTensor":
        """
        Create VulkanTensor from PyTorch tensor.

        Optimized path that avoids unnecessary copies when possible.

        Args:
            tensor: PyTorch tensor (CPU or CUDA)
            lazy: If True, defer GPU upload

        Returns:
            VulkanTensor
        """
        if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            # Get numpy array efficiently
            if tensor.is_cuda:
                # CUDA tensor - must go through CPU
                arr = tensor.detach().cpu().numpy()
            else:
                # CPU tensor - try to avoid copy
                arr = tensor.detach().numpy()
                if not arr.flags["C_CONTIGUOUS"]:
                    arr = np.ascontiguousarray(arr)
        else:
            arr = np.asarray(tensor)

        return cls(arr, lazy=lazy)

    @classmethod
    def empty(cls, shape: tuple, dtype=np.float32) -> "VulkanTensor":
        """
        Create uninitialized VulkanTensor.

        Useful for output buffers where data will be written by GPU.

        Args:
            shape: Tensor shape
            dtype: Data type

        Returns:
            VulkanTensor with uninitialized data
        """
        data = np.empty(shape, dtype=dtype)
        tensor = cls(data, lazy=True)
        tensor._cpu_valid = False  # CPU data is garbage
        return tensor

    @classmethod
    def zeros(cls, shape: tuple, dtype=np.float32) -> "VulkanTensor":
        """Create zero-initialized VulkanTensor"""
        return cls(np.zeros(shape, dtype=dtype), lazy=True)

    @classmethod
    def ones(cls, shape: tuple, dtype=np.float32) -> "VulkanTensor":
        """Create ones-initialized VulkanTensor"""
        return cls(np.ones(shape, dtype=dtype), lazy=True)

    def _ensure_uploaded(self):
        """Ensure data is uploaded to GPU (lazy upload)"""
        if self._gpu_valid:
            return  # Already valid on GPU

        if not self._cpu_valid:
            raise RuntimeError("Cannot upload: no valid CPU data")

        try:
            from grilly import Compute

            backend = Compute()

            size = self._cpu_data.nbytes

            # Try to use buffer pool if available
            try:
                from grilly.backend.buffer_pool import acquire_buffer

                self._pooled_buffer = acquire_buffer(size, core=backend.core)
                self._gpu_buffer = self._pooled_buffer.handle
                self._gpu_memory = self._pooled_buffer.memory
            except (ImportError, Exception):
                # Fallback to direct allocation
                self._gpu_buffer, self._gpu_memory = backend.create_buffer(size, usage="storage")

            # Upload to GPU — use VMA path if pooled buffer
            if self._pooled_buffer is not None and hasattr(self._pooled_buffer, "pool") and self._pooled_buffer.pool is not None:
                self._pooled_buffer.pool.upload_data(self._pooled_buffer, self._cpu_data)
            else:
                backend.upload_buffer(self._gpu_buffer, self._gpu_memory, self._cpu_data)
            self._gpu_valid = True
            self._uploaded = True  # Backwards compatibility

        except Exception as e:
            self._gpu_valid = False
            raise RuntimeError(f"Failed to upload to GPU: {e}")

    def _ensure_downloaded(self):
        """Ensure CPU data is current (lazy download)"""
        if self._cpu_valid:
            return  # Already valid on CPU

        if not self._gpu_valid:
            raise RuntimeError("Cannot download: no valid GPU data")

        # DEVICE_LOCAL buffers cannot be mapped — must stage through readback
        if getattr(self, "_is_device_local", False):
            self._download_via_staging()
            return

        try:
            size = self._cpu_data.nbytes

            # VMA path: use pool's vmaMapMemory (not vkMapMemory)
            pooled = getattr(self, "_pooled_buffer", None)
            if (
                pooled is not None
                and hasattr(pooled, "pool")
                and pooled.pool is not None
                and hasattr(pooled.pool, "download_data")
            ):
                self._cpu_data = pooled.pool.download_data(
                    pooled, size, dtype=self._dtype
                ).reshape(self._shape)
                self._cpu_valid = True
                return

            # Legacy path: use core._download_buffer (vkMapMemory)
            core = self._core
            if core is None:
                from grilly import Compute

                backend = Compute()
                core = backend.core
                self._core = core

            self._cpu_data = core._download_buffer(
                self._gpu_memory, size, dtype=self._dtype
            ).reshape(self._shape)
            self._cpu_valid = True

        except Exception as e:
            raise RuntimeError(f"Failed to download from GPU: {e}")

    def _download_via_staging(self):
        """Download DEVICE_LOCAL buffer via staging readback buffer.

        Uses CommandRecorder + transfer barrier + vkCmdCopyBuffer to copy
        DEVICE_LOCAL VRAM → HOST_VISIBLE readback → numpy.
        """
        core = self._core
        if core is None:
            from grilly import Compute
            backend = Compute()
            core = backend.core
            self._core = core

        pooled = getattr(self, "_pooled_buffer", None)
        pool = pooled.pool if pooled is not None and hasattr(pooled, "pool") else None

        if pool is None or not hasattr(pool, "acquire_staging"):
            raise RuntimeError("Cannot download DEVICE_LOCAL: no pool with staging support")

        size = self._cpu_data.nbytes

        readback = pool.acquire_staging(size, for_upload=False)

        # Get handles
        dl_handle = self._gpu_buffer
        rb_handle = readback.get_vulkan_handle() if hasattr(readback, "get_vulkan_handle") else readback.handle

        with core.record_commands() as rec:
            rec.transfer_barrier()
            rec.copy_buffer(dl_handle, rb_handle, size)

        self._cpu_data = pool.download_data(readback, size, dtype=self._dtype).reshape(self._shape)
        self._cpu_valid = True
        readback.release()

    def mark_gpu_modified(self):
        """Mark that GPU data has been modified (CPU copy is now stale)"""
        self._gpu_valid = True
        self._cpu_valid = False

    def mark_cpu_modified(self):
        """Mark that CPU data has been modified (GPU copy is now stale)"""
        self._cpu_valid = True
        self._gpu_valid = False

    @property
    def shape(self):
        """Get tensor shape"""
        return self._shape

    @property
    def dtype(self):
        """Get tensor dtype"""
        return self._dtype

    @property
    def ndim(self):
        """Get number of dimensions"""
        return len(self._shape)

    @property
    def nbytes(self):
        """Get size in bytes"""
        return self._cpu_data.nbytes

    @property
    def on_gpu(self) -> bool:
        """Check if tensor has valid GPU copy"""
        return self._gpu_valid

    @property
    def gpu_buffer(self):
        """Get GPU buffer handle (ensures upload)"""
        self._ensure_uploaded()
        return self._gpu_buffer

    @property
    def gpu_memory(self):
        """Get GPU memory handle (ensures upload)"""
        self._ensure_uploaded()
        return self._gpu_memory

    def numpy(self) -> np.ndarray:
        """
        Convert to numpy array (downloads from GPU if needed).

        Uses lazy download - only transfers if GPU has newer data.

        Returns:
            numpy array
        """
        self._ensure_downloaded()
        return self._cpu_data.copy()

    def cpu(self) -> np.ndarray:
        """Get CPU copy (alias for numpy())"""
        return self.numpy()

    def to_torch(self, device: str = "cpu"):
        """
        Convert to PyTorch tensor.

        Args:
            device: Target device ('cpu', 'cuda', etc.)

        Returns:
            PyTorch tensor
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        self._ensure_downloaded()
        tensor = torch.from_numpy(self._cpu_data)

        if device != "cpu":
            tensor = tensor.to(device)

        return tensor

    def upload(self):
        """Force upload to GPU"""
        self._ensure_uploaded()
        return self

    def download(self):
        """Force download from GPU"""
        self._ensure_downloaded()
        return self

    def release_gpu(self):
        """Release GPU buffer (return to pool if using pooling)"""
        if self._pooled_buffer is not None:
            self._pooled_buffer.release()
            self._pooled_buffer = None
        self._gpu_buffer = None
        self._gpu_memory = None
        self._gpu_valid = False
        self._uploaded = False

    def __array__(self, dtype=None):
        """NumPy array interface (supports optional dtype coercion)."""
        arr = self.numpy()
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr

    def __len__(self):
        """Length (first dimension)"""
        return self._shape[0] if self._shape else 0

    def __getitem__(self, key):
        """Indexing (operates on CPU data)"""
        self._ensure_downloaded()
        return self._cpu_data[key]

    def __setitem__(self, key, value):
        """Assignment (operates on CPU data, marks GPU stale)"""
        self._ensure_downloaded()
        self._cpu_data[key] = value
        self.mark_cpu_modified()

    def reshape(self, *shape) -> "VulkanTensor":
        """Reshape tensor (returns new VulkanTensor)"""
        self._ensure_downloaded()
        new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return VulkanTensor(self._cpu_data.reshape(new_shape), lazy=True)

    def __repr__(self):
        """Return a debug representation."""

        status = []
        if self._gpu_valid:
            status.append("gpu")
        if self._cpu_valid:
            status.append("cpu")
        return f"VulkanTensor(shape={self.shape}, dtype={self.dtype}, valid=[{','.join(status)}])"

    def __del__(self):
        """Cleanup - release GPU buffer on destruction"""
        try:
            self.release_gpu()
        except Exception:
            pass  # Ignore cleanup errors


def to_vulkan_batch(
    tensors: list | tuple | Any,
) -> np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]:
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


def from_vulkan(array: np.ndarray, device: str = "cuda") -> Any:
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

    if device == "cuda":
        try:
            return device_manager.to_cuda(array)
        except (RuntimeError, AssertionError):
            # CUDA not available, fall back to CPU
            if TORCH_AVAILABLE:
                return torch.from_numpy(array).cpu()
            return array
    elif device == "cpu":
        if TORCH_AVAILABLE:
            return torch.from_numpy(array).cpu()
        return array
    else:
        # PyTorch device string
        if TORCH_AVAILABLE:
            return torch.from_numpy(array).to(device)
        return array


def auto_convert_to_vulkan(func):
    """Decorate a function to auto-convert the first tensor argument."""

    def wrapper(*args, **kwargs):
        # Convert first argument if it is a PyTorch tensor.
        if args and TORCH_AVAILABLE and isinstance(args[0], torch.Tensor):
            args = (to_vulkan(args[0]),) + args[1:]
        return func(*args, **kwargs)

    return wrapper


def ensure_vulkan_compatible(data: np.ndarray | Any) -> np.ndarray:
    """
    Ensure data is Vulkan-compatible numpy array.

    Handles VulkanTensor by extracting numpy array.
    Preserves integer dtypes for index arrays (e.g., token IDs).

    Args:
        data: Any tensor-like data (including VulkanTensor)

    Returns:
        numpy array ready for Vulkan (float32 for floats, preserved for integers)
    """
    # Handle VulkanTensor
    if isinstance(data, VulkanTensor):
        return data.numpy()

    arr = to_vulkan(data, keep_on_gpu=False)  # Get numpy, not GPU tensor
    if isinstance(arr, VulkanTensor):
        arr = arr.numpy()
    # Preserve integer dtypes (needed for embedding lookups, indices, etc.)
    if np.issubdtype(arr.dtype, np.integer):
        return arr
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
    if hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return True  # PyTorch-like
    if hasattr(obj, "numpy") and not isinstance(obj, np.ndarray):
        return True  # TensorFlow-like
    return False
