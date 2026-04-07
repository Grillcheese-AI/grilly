"""
Tensor Conversion Utilities

Seamless conversion between PyTorch tensors and Vulkan (numpy arrays).
Provides automatic conversion for seamless integration with GPU acceleration on AMD.

When the C++ grilly_core module is available, VulkanTensor wraps grilly_core.Tensor
for GPU-first operation with dual CPU/GPU validity tracking. Otherwise, falls back
to a pure-numpy implementation with lazy Vulkan buffer management.
"""

from typing import Any, Union

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import grilly_core as _gc

    _CPP_AVAILABLE = True
except ImportError:
    _gc = None
    _CPP_AVAILABLE = False

from .device_manager import get_device_manager

_vulkan_backend_cache = None


def _get_vulkan_backend():
    """Get cached Vulkan backend without using deprecated Compute()."""
    global _vulkan_backend_cache
    if _vulkan_backend_cache is not None:
        return _vulkan_backend_cache
    try:
        _vulkan_backend_cache = get_device_manager().vulkan
        return _vulkan_backend_cache
    except Exception:
        return None


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


# ═══════════════════════════════════════════════════════════════════════════════
# VulkanTensor — C++ backed (preferred) or pure-numpy fallback
# ═══════════════════════════════════════════════════════════════════════════════

if _CPP_AVAILABLE:

    class VulkanTensor:
        """GPU-resident tensor backed by C++ grilly_core.Tensor.
        Legacy attributes (_gpu_buffer, _pooled_buffer, etc.) are kept so
        backend/base.py can inject pooled-buffer handles for zero-copy dispatch."""

        __slots__ = (
            "_t", "_pooled_buffer", "_gpu_buffer", "_gpu_memory",
            "_core", "_is_device_local", "_gpu_valid", "_cpu_valid", "_uploaded",
            "grad", "grad_fn", "_is_leaf", "_retain_grad",
        )

        def __init__(self, data=None, lazy: bool = True, *, _cpp_tensor=None, **kwargs):
            self._pooled_buffer = None
            self._gpu_buffer = None
            self._gpu_memory = None
            self._core = None
            self._is_device_local = False
            self._gpu_valid = False
            self._cpu_valid = True
            self._uploaded = False
            self.grad = None
            self.grad_fn = None
            self._is_leaf = True
            self._retain_grad = False

            if _cpp_tensor is not None:
                self._t = _cpp_tensor
                self._gpu_valid = _cpp_tensor.on_gpu
                self._cpu_valid = _cpp_tensor.on_cpu
                self._uploaded = _cpp_tensor.on_gpu
            elif data is not None:
                if isinstance(data, np.ndarray):
                    # Preserve integer types (needed for embedding lookups)
                    if np.issubdtype(data.dtype, np.integer):
                        arr = np.ascontiguousarray(data)
                    else:
                        arr = np.ascontiguousarray(data.astype(np.float32))
                    self._t = _gc.Tensor.from_numpy(arr)
                elif TORCH_AVAILABLE and isinstance(data, torch.Tensor):
                    if data.is_cuda:
                        arr = data.detach().cpu().numpy()
                    else:
                        arr = data.detach().numpy()
                    arr = np.ascontiguousarray(arr.astype(np.float32))
                    self._t = _gc.Tensor.from_numpy(arr)
                elif hasattr(data, "numpy"):
                    arr = np.ascontiguousarray(np.asarray(data).astype(np.float32))
                    self._t = _gc.Tensor.from_numpy(arr)
                else:
                    arr = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
                    self._t = _gc.Tensor.from_numpy(arr)
                self._cpu_valid = True
                self._gpu_valid = False
            else:
                raise ValueError("Must provide data or _cpp_tensor")

        @classmethod
        def _new_shell(cls):
            """Create an uninitialized VulkanTensor with all legacy slots zeroed."""
            t = cls.__new__(cls)
            t._pooled_buffer = t._gpu_buffer = t._gpu_memory = t._core = None
            t._is_device_local = False
            t._gpu_valid = t._cpu_valid = False
            t._uploaded = False
            t.grad = None
            t.grad_fn = None
            t._is_leaf = True
            t._retain_grad = False
            return t

        @classmethod
        def from_cpp(cls, cpp_tensor):
            """Wrap a C++ Tensor directly (zero-copy)."""
            t = cls._new_shell()
            t._t = cpp_tensor
            t._gpu_valid = cpp_tensor.on_gpu
            t._cpu_valid = cpp_tensor.on_cpu
            t._uploaded = cpp_tensor.on_gpu
            return t

        @classmethod
        def from_torch(cls, tensor, lazy: bool = True) -> "VulkanTensor":
            """Create VulkanTensor from PyTorch tensor."""
            if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
                if tensor.is_cuda:
                    arr = tensor.detach().cpu().numpy()
                else:
                    arr = tensor.detach().numpy()
                    if not arr.flags["C_CONTIGUOUS"]:
                        arr = np.ascontiguousarray(arr)
            else:
                arr = np.asarray(tensor)
            return cls(arr, lazy=lazy)

        @classmethod
        def empty(cls, shape: tuple, dtype=np.float32) -> "VulkanTensor":
            """Create uninitialized VulkanTensor."""
            t = cls._new_shell()
            t._t = _gc.Tensor.empty(list(shape))
            return t

        @classmethod
        def zeros(cls, shape: tuple, dtype=np.float32) -> "VulkanTensor":
            """Create zero-initialized VulkanTensor."""
            t = cls._new_shell()
            t._t = _gc.Tensor.zeros(list(shape))
            t._cpu_valid = True
            return t

        @classmethod
        def ones(cls, shape: tuple, dtype=np.float32) -> "VulkanTensor":
            """Create ones-initialized VulkanTensor."""
            return cls(np.ones(shape, dtype=dtype), lazy=True)

        @property
        def shape(self):
            return tuple(self._t.shape)

        @property
        def dtype(self):
            return np.float32

        @property
        def ndim(self):
            return self._t.ndim

        @property
        def nbytes(self):
            return self._t.nbytes

        @property
        def on_gpu(self) -> bool:
            return self._gpu_valid or self._t.on_gpu

        @property
        def gpu_buffer(self):
            self._ensure_uploaded()
            return self._gpu_buffer if self._gpu_buffer is not None else self._t.gpu_handle()

        @property
        def gpu_memory(self):
            self._ensure_uploaded()
            return self._gpu_memory

        @property
        def requires_grad(self):
            return self._t.requires_grad

        @requires_grad.setter
        def requires_grad(self, val):
            self._t.requires_grad = val

        @property
        def is_leaf(self) -> bool:
            """True if this tensor was created directly (not by an operation)."""
            return self._is_leaf

        def detach(self) -> "VulkanTensor":
            """Return a new VulkanTensor detached from the computation graph."""
            t = VulkanTensor(self.numpy())
            t.grad_fn = None
            t._is_leaf = True
            return t

        def retain_grad(self):
            """Request gradient retention for non-leaf tensors."""
            self._retain_grad = True

        def backward(self, grad_output=None, retain_graph: bool = False):
            """Run backward pass through the computation graph."""
            if self.grad_fn is None:
                return
            if grad_output is None:
                if self.numpy().size == 1:
                    grad_output = np.ones_like(self.numpy())
                else:
                    raise RuntimeError(
                        "backward() requires grad_output for non-scalar tensors"
                    )
            self.grad_fn.backward(grad_output, retain_graph=retain_graph)

        def numpy(self) -> np.ndarray:
            """Convert to numpy array (downloads from GPU if needed)."""
            if self._gpu_valid and not self._cpu_valid and self._pooled_buffer is not None:
                self._ensure_downloaded()
                return self._t.numpy().copy()
            return self._t.numpy().copy()

        def cpu(self) -> np.ndarray:
            return self.numpy()

        def item(self) -> float:
            return float(self.numpy().ravel()[0])

        def to_torch(self, device: str = "cpu"):
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch not available")
            t = torch.from_numpy(self.numpy().copy())
            return t if device == "cpu" else t.to(device)

        def upload(self):
            self._ensure_uploaded()
            return self

        def download(self):
            self._ensure_downloaded()
            return self

        def release_gpu(self):
            if self._pooled_buffer is not None:
                self._pooled_buffer.release()
                self._pooled_buffer = None
            self._gpu_buffer = None
            self._gpu_memory = None
            self._gpu_valid = False
            self._uploaded = False
            try:
                self._t.release_gpu()
            except Exception:
                pass

        def mark_gpu_modified(self):
            self._gpu_valid = True
            self._cpu_valid = False
            try:
                self._t.mark_gpu_modified()
            except Exception:
                pass

        def mark_cpu_modified(self):
            self._cpu_valid = True
            self._gpu_valid = False
            try:
                self._t.mark_cpu_modified()
            except Exception:
                pass

        def _try_bind_cpp_gpu_buffer(self) -> bool:
            """If C++ Tensor already holds a GPU buffer, mirror its VkBuffer handle for Python dispatch.

            Avoids allocating a second pooled buffer and re-uploading when the tensor is
            GPU-resident only (e.g. output of a prior op) but Python slots were not set.
            """
            if self._pooled_buffer is not None or self._gpu_buffer is not None:
                return True
            try:
                h = int(self._t.gpu_handle_if_valid())
            except Exception:
                return False
            if h == 0:
                return False
            try:
                import vulkan as vk

                self._gpu_buffer = vk.ffi.cast("VkBuffer", h)
                self._gpu_memory = None
                if self._core is None:
                    backend = _get_vulkan_backend()
                    if backend is not None:
                        self._core = backend.core
                self._gpu_valid = True
                self._uploaded = True
                self._cpu_valid = bool(self._t.on_cpu)
                return True
            except Exception:
                return False

        def prepare_for_dispatch(self) -> None:
            """Ensure this tensor can supply a Vulkan buffer for kernels (minimal CPU↔GPU traffic)."""
            if self._try_bind_cpp_gpu_buffer():
                return
            if self._t.on_gpu:
                try:
                    self._t.ensure_gpu()
                except Exception:
                    pass
                if self._try_bind_cpp_gpu_buffer():
                    return
            self._ensure_uploaded()

        def _ensure_uploaded(self):
            if self._try_bind_cpp_gpu_buffer():
                return

            if not self._cpu_valid and not self._t.on_cpu:
                if self._t.on_gpu:
                    try:
                        self._t.ensure_gpu()
                    except Exception:
                        pass
                    if self._try_bind_cpp_gpu_buffer():
                        return
                raise RuntimeError("Cannot upload: no valid CPU data")

            try:
                self._t.ensure_gpu()
                self._gpu_valid = True
                self._uploaded = True
                if self._try_bind_cpp_gpu_buffer():
                    return
            except Exception:
                # C++ Tensor may not have Vulkan backend — fall back to Python path
                try:
                    backend = _get_vulkan_backend()
                    if backend is None:
                        raise RuntimeError("Vulkan backend is unavailable")
                    cpu_data = self._t.numpy()
                    size = cpu_data.nbytes

                    try:
                        from grilly.backend.buffer_pool import acquire_buffer

                        self._pooled_buffer = acquire_buffer(size, core=backend.core)
                        self._gpu_buffer = self._pooled_buffer.handle
                        self._gpu_memory = self._pooled_buffer.memory
                    except (ImportError, Exception):
                        self._gpu_buffer, self._gpu_memory = backend.create_buffer(
                            size, usage="storage"
                        )

                    if (
                        self._pooled_buffer is not None
                        and hasattr(self._pooled_buffer, "pool")
                        and self._pooled_buffer.pool is not None
                    ):
                        self._pooled_buffer.pool.upload_data(self._pooled_buffer, cpu_data)
                    else:
                        backend.upload_buffer(self._gpu_buffer, self._gpu_memory, cpu_data)
                    self._gpu_valid = True
                    self._uploaded = True
                except Exception as e:
                    self._gpu_valid = False
                    raise RuntimeError(f"Failed to upload to GPU: {e}")

        def _ensure_downloaded(self):
            if self._cpu_valid:
                return

            if not self._gpu_valid and not self._t.on_gpu:
                raise RuntimeError("Cannot download: no valid GPU data")

            # DEVICE_LOCAL buffers require staging readback
            if self._is_device_local:
                self._download_via_staging()
                return

            # Try VMA pool download first
            pooled = self._pooled_buffer
            if (
                pooled is not None
                and hasattr(pooled, "pool")
                and pooled.pool is not None
                and hasattr(pooled.pool, "download_data")
            ):
                size = self._t.nbytes
                dtype = np.float32
                cpu_data = pooled.pool.download_data(pooled, size, dtype=dtype).reshape(
                    self.shape
                )
                # Re-create C++ tensor from downloaded data
                self._t = _gc.Tensor.from_numpy(np.ascontiguousarray(cpu_data))
                self._cpu_valid = True
                return

            # Legacy path with core._download_buffer
            core = self._core
            if core is None:
                try:
                    backend = _get_vulkan_backend()
                    if backend is None:
                        raise RuntimeError("Vulkan backend is unavailable")
                    core = backend.core
                    self._core = core
                except Exception:
                    pass

            if core is not None and self._gpu_memory is not None:
                size = self._t.nbytes
                dtype = np.float32
                cpu_data = core._download_buffer(
                    self._gpu_memory, size, dtype=dtype
                ).reshape(self.shape)
                self._t = _gc.Tensor.from_numpy(np.ascontiguousarray(cpu_data))
                self._cpu_valid = True
                return

            # Fallback: C++ tensor may know how to download itself
            try:
                _ = self._t.numpy()
                self._cpu_valid = True
            except Exception as e:
                raise RuntimeError(f"Failed to download from GPU: {e}")

        def _download_via_staging(self):
            core = self._core
            if core is None:
                backend = _get_vulkan_backend()
                if backend is None:
                    raise RuntimeError("Vulkan backend is unavailable")
                core = backend.core
                self._core = core

            pooled = self._pooled_buffer
            pool = pooled.pool if pooled is not None and hasattr(pooled, "pool") else None

            if pool is None or not hasattr(pool, "acquire_staging"):
                raise RuntimeError("Cannot download DEVICE_LOCAL: no pool with staging support")

            size = self._t.nbytes
            readback = pool.acquire_staging(size, for_upload=False)

            dl_handle = self._gpu_buffer
            rb_handle = (
                readback.get_vulkan_handle()
                if hasattr(readback, "get_vulkan_handle")
                else readback.handle
            )

            with core.record_commands() as rec:
                rec.transfer_barrier()
                rec.copy_buffer(dl_handle, rb_handle, size)

            cpu_data = pool.download_data(readback, size, dtype=np.float32).reshape(self.shape)
            self._t = _gc.Tensor.from_numpy(np.ascontiguousarray(cpu_data))
            self._cpu_valid = True
            readback.release()

        def __array__(self, dtype=None):
            arr = self.numpy()
            if dtype is not None:
                return arr.astype(dtype, copy=False)
            return arr

        def __len__(self):
            return self.shape[0] if self.shape else 0

        def __getitem__(self, key):
            return self.numpy()[key]

        def __setitem__(self, key, value):
            arr = self.numpy()
            arr[key] = value if isinstance(value, np.ndarray) else np.asarray(value)
            self._t = _gc.Tensor.from_numpy(np.ascontiguousarray(arr))
            self.mark_cpu_modified()

        def reshape(self, *shape) -> "VulkanTensor":
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                new_shape = list(shape[0])
            else:
                new_shape = list(shape)
            return VulkanTensor.from_cpp(self._t.reshape(new_shape))

        # ── Repr / cleanup ─────────────────────────────────────────────────

        def __repr__(self):
            """Return a debug representation."""
            status = []
            if self._gpu_valid or self._t.on_gpu:
                status.append("gpu")
            if self._cpu_valid or self._t.on_cpu:
                status.append("cpu")
            return (
                f"VulkanTensor(shape={self.shape}, dtype={self.dtype}, "
                f"valid=[{','.join(status)}])"
            )

        def __del__(self):
            """Cleanup - release GPU buffer on destruction."""
            try:
                if self._pooled_buffer is not None:
                    self._pooled_buffer.release()
                    self._pooled_buffer = None
            except Exception:
                pass  # Ignore cleanup errors

else:
    # ═══════════════════════════════════════════════════════════════════════
    # Fallback: pure-numpy VulkanTensor (no C++ backend)
    # ═══════════════════════════════════════════════════════════════════════

    class VulkanTensor:  # type: ignore[no-redef]
        """
        GPU-resident tensor wrapper for Vulkan operations (fallback, no C++ backend).

        Features:
        - Lazy transfer: Only uploads to GPU when actually needed
        - Dirty tracking: Knows when CPU/GPU copies are out of sync
        - Buffer pooling: Reuses GPU buffers for efficiency
        - PyTorch bridge: Seamless conversion to/from PyTorch tensors
        """

        def __init__(self, data: np.ndarray, lazy: bool = True, **kwargs):
            # Handle integer types - preserve them
            if np.issubdtype(data.dtype, np.integer):
                self._cpu_data = np.ascontiguousarray(data)
            else:
                self._cpu_data = np.ascontiguousarray(data.astype(np.float32))

            self._gpu_buffer = None
            self._gpu_memory = None
            self._pooled_buffer = None
            self._core = None
            self._is_device_local = False
            self._shape = self._cpu_data.shape
            self._dtype = self._cpu_data.dtype

            self._gpu_valid = False
            self._cpu_valid = True
            self._uploaded = False

            # Autograd fields
            self.requires_grad = kwargs.get('requires_grad', False)
            self.grad = None
            self.grad_fn = None
            self._is_leaf = True
            self._retain_grad = False

            if not lazy:
                self._ensure_uploaded()

        @classmethod
        def from_torch(cls, tensor, lazy: bool = True) -> "VulkanTensor":
            if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
                if tensor.is_cuda:
                    arr = tensor.detach().cpu().numpy()
                else:
                    arr = tensor.detach().numpy()
                    if not arr.flags["C_CONTIGUOUS"]:
                        arr = np.ascontiguousarray(arr)
            else:
                arr = np.asarray(tensor)
            return cls(arr, lazy=lazy)

        @classmethod
        def empty(cls, shape: tuple, dtype=np.float32) -> "VulkanTensor":
            data = np.empty(shape, dtype=dtype)
            tensor = cls(data, lazy=True)
            tensor._cpu_valid = False
            return tensor

        @classmethod
        def zeros(cls, shape: tuple, dtype=np.float32) -> "VulkanTensor":
            return cls(np.zeros(shape, dtype=dtype), lazy=True)

        @classmethod
        def ones(cls, shape: tuple, dtype=np.float32) -> "VulkanTensor":
            return cls(np.ones(shape, dtype=dtype), lazy=True)

        def _ensure_uploaded(self):
            if self._gpu_valid:
                return
            if not self._cpu_valid:
                raise RuntimeError("Cannot upload: no valid CPU data")
            try:
                backend = _get_vulkan_backend()
                if backend is None:
                    raise RuntimeError("Vulkan backend is unavailable")
                size = self._cpu_data.nbytes
                try:
                    from grilly.backend.buffer_pool import acquire_buffer

                    self._pooled_buffer = acquire_buffer(size, core=backend.core)
                    self._gpu_buffer = self._pooled_buffer.handle
                    self._gpu_memory = self._pooled_buffer.memory
                except (ImportError, Exception):
                    self._gpu_buffer, self._gpu_memory = backend.create_buffer(
                        size, usage="storage"
                    )
                if (
                    self._pooled_buffer is not None
                    and hasattr(self._pooled_buffer, "pool")
                    and self._pooled_buffer.pool is not None
                ):
                    self._pooled_buffer.pool.upload_data(self._pooled_buffer, self._cpu_data)
                else:
                    backend.upload_buffer(self._gpu_buffer, self._gpu_memory, self._cpu_data)
                self._gpu_valid = True
                self._uploaded = True
            except Exception as e:
                self._gpu_valid = False
                raise RuntimeError(f"Failed to upload to GPU: {e}")

        def _ensure_downloaded(self):
            if self._cpu_valid:
                return
            if not self._gpu_valid:
                raise RuntimeError("Cannot download: no valid GPU data")
            if getattr(self, "_is_device_local", False):
                self._download_via_staging()
                return
            try:
                size = self._cpu_data.nbytes
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
                core = self._core
                if core is None:
                    backend = _get_vulkan_backend()
                    if backend is None:
                        raise RuntimeError("Vulkan backend is unavailable")
                    core = backend.core
                    self._core = core
                self._cpu_data = core._download_buffer(
                    self._gpu_memory, size, dtype=self._dtype
                ).reshape(self._shape)
                self._cpu_valid = True
            except Exception as e:
                raise RuntimeError(f"Failed to download from GPU: {e}")

        def _download_via_staging(self):
            core = self._core
            if core is None:
                backend = _get_vulkan_backend()
                if backend is None:
                    raise RuntimeError("Vulkan backend is unavailable")
                core = backend.core
                self._core = core
            pooled = getattr(self, "_pooled_buffer", None)
            pool = pooled.pool if pooled is not None and hasattr(pooled, "pool") else None
            if pool is None or not hasattr(pool, "acquire_staging"):
                raise RuntimeError("Cannot download DEVICE_LOCAL: no pool with staging support")
            size = self._cpu_data.nbytes
            readback = pool.acquire_staging(size, for_upload=False)
            dl_handle = self._gpu_buffer
            rb_handle = (
                readback.get_vulkan_handle()
                if hasattr(readback, "get_vulkan_handle")
                else readback.handle
            )
            with core.record_commands() as rec:
                rec.transfer_barrier()
                rec.copy_buffer(dl_handle, rb_handle, size)
            self._cpu_data = pool.download_data(readback, size, dtype=self._dtype).reshape(
                self._shape
            )
            self._cpu_valid = True
            readback.release()

        def mark_gpu_modified(self):
            self._gpu_valid = True
            self._cpu_valid = False

        def mark_cpu_modified(self):
            self._cpu_valid = True
            self._gpu_valid = False

        def prepare_for_dispatch(self) -> None:
            self._ensure_uploaded()

        @property
        def is_leaf(self) -> bool:
            """True if this tensor was created directly (not by an operation)."""
            return self._is_leaf

        def detach(self) -> "VulkanTensor":
            """Return a new VulkanTensor detached from the computation graph."""
            t = VulkanTensor(self.numpy())
            t.requires_grad = False
            t.grad_fn = None
            t._is_leaf = True
            return t

        def retain_grad(self):
            """Request gradient retention for non-leaf tensors."""
            self._retain_grad = True

        def backward(self, grad_output=None, retain_graph: bool = False):
            """Run backward pass through the computation graph."""
            if self.grad_fn is None:
                return
            if grad_output is None:
                if self._cpu_data.size == 1:
                    grad_output = np.ones_like(self._cpu_data)
                else:
                    raise RuntimeError(
                        "backward() requires grad_output for non-scalar tensors"
                    )
            self.grad_fn.backward(grad_output, retain_graph=retain_graph)

        @property
        def shape(self):
            return self._shape

        @property
        def dtype(self):
            return self._dtype

        @property
        def ndim(self):
            return len(self._shape)

        @property
        def nbytes(self):
            return self._cpu_data.nbytes

        @property
        def on_gpu(self) -> bool:
            return self._gpu_valid

        @property
        def gpu_buffer(self):
            self._ensure_uploaded()
            return self._gpu_buffer

        @property
        def gpu_memory(self):
            self._ensure_uploaded()
            return self._gpu_memory

        def numpy(self) -> np.ndarray:
            self._ensure_downloaded()
            return self._cpu_data.copy()

        def cpu(self) -> np.ndarray:
            return self.numpy()

        def item(self) -> float:
            return float(self.numpy().ravel()[0])

        def to_torch(self, device: str = "cpu"):
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch not available")
            self._ensure_downloaded()
            tensor = torch.from_numpy(self._cpu_data)
            if device != "cpu":
                tensor = tensor.to(device)
            return tensor

        def upload(self):
            self._ensure_uploaded()
            return self

        def download(self):
            self._ensure_downloaded()
            return self

        def release_gpu(self):
            if self._pooled_buffer is not None:
                self._pooled_buffer.release()
                self._pooled_buffer = None
            self._gpu_buffer = None
            self._gpu_memory = None
            self._gpu_valid = False
            self._uploaded = False

        def __array__(self, dtype=None):
            arr = self.numpy()
            if dtype is not None:
                return arr.astype(dtype, copy=False)
            return arr

        def __len__(self):
            return self._shape[0] if self._shape else 0

        def __getitem__(self, key):
            self._ensure_downloaded()
            return self._cpu_data[key]

        def __setitem__(self, key, value):
            self._ensure_downloaded()
            self._cpu_data[key] = value
            self.mark_cpu_modified()

        def reshape(self, *shape) -> "VulkanTensor":
            self._ensure_downloaded()
            new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
            return VulkanTensor(self._cpu_data.reshape(new_shape), lazy=True)

        def __repr__(self):
            status = []
            if self._gpu_valid:
                status.append("gpu")
            if self._cpu_valid:
                status.append("cpu")
            return (
                f"VulkanTensor(shape={self.shape}, dtype={self.dtype}, "
                f"valid=[{','.join(status)}])"
            )

        def __del__(self):
            try:
                self.release_gpu()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# Utility functions (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════


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
