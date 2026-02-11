"""
Multi-Backend Device Manager

Manages multiple backends (Vulkan, CUDA, CPU) and allows seamless switching
between them. Supports HuggingFace models on CUDA while using Vulkan for
custom operations.
"""

from typing import Any

import numpy as np

# Backend instances (lazy initialization)
_vulkan_backend: Any | None = None
_cuda_backend: Any | None = None
_cpu_backend: Any | None = None

# Current device settings
_current_device: str = "vulkan"
_default_cuda_device: str | None = None


class DeviceManager:
    """
    Multi-backend device manager for Vulkan, CUDA, and CPU.

    Allows running HuggingFace models on CUDA while using Vulkan for
    custom GPU-accelerated operations.
    """

    def __init__(self):
        """Initialize the instance."""

        self._vulkan = None
        self._cuda = None
        self._torch = None
        self._current = "vulkan"
        self._cuda_device = None

    @property
    def vulkan(self):
        """Get Vulkan backend (lazy initialization)"""
        if self._vulkan is None:
            try:
                from ..backend.compute import VulkanCompute

                self._vulkan = VulkanCompute()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Vulkan backend: {e}")
        return self._vulkan

    @property
    def cuda(self):
        """Get CUDA backend (PyTorch) (lazy initialization)"""
        if self._cuda is None:
            try:
                import torch

                self._torch = torch
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available")
                self._cuda_device = torch.device(
                    "cuda:0" if self._cuda_device is None else self._cuda_device
                )
            except ImportError:
                raise RuntimeError("PyTorch is not installed. Install with: pip install torch")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize CUDA backend: {e}")
        return self._cuda

    @property
    def torch(self):
        """Get PyTorch module"""
        if self._torch is None:
            try:
                import torch

                self._torch = torch
            except ImportError:
                raise RuntimeError("PyTorch is not installed. Install with: pip install torch")
        return self._torch

    def set_device(self, device: str, cuda_index: int | None = None):
        """
        Set current device.

        Args:
            device: Device name ('vulkan', 'cuda', 'cpu')
            cuda_index: CUDA device index (for CUDA device)
        """
        device = device.lower()
        if device not in ("vulkan", "cuda", "cpu"):
            raise ValueError(f"Unknown device: {device}. Must be 'vulkan', 'cuda', or 'cpu'")

        self._current = device

        if device == "cuda" and cuda_index is not None:
            if self._torch is None:
                import torch

                self._torch = torch
            self._cuda_device = self._torch.device(f"cuda:{cuda_index}")
        elif device == "cuda":
            if self._torch is None:
                import torch

                self._torch = torch
            self._cuda_device = self._torch.device("cuda:0")

    def get_device(self) -> str:
        """Get current device name"""
        return self._current

    def get_cuda_device(self):
        """Get PyTorch CUDA device"""
        if self._torch is None:
            import torch

            self._torch = torch
        if self._cuda_device is None:
            self._cuda_device = self._torch.device("cuda:0")
        return self._cuda_device

    def to_vulkan(self, tensor: np.ndarray | Any) -> np.ndarray:
        """
        Convert tensor to numpy array for Vulkan operations.

        Args:
            tensor: PyTorch tensor or numpy array

        Returns:
            numpy array
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        elif hasattr(tensor, "cpu"):  # PyTorch tensor
            return tensor.detach().cpu().numpy()
        elif hasattr(tensor, "numpy"):  # TensorFlow tensor
            return tensor.numpy()
        else:
            return np.array(tensor)

    def to_cuda(self, array: np.ndarray, dtype: Any | None = None):
        """
        Convert numpy array to CUDA tensor.

        Args:
            array: numpy array
            dtype: Optional torch dtype

        Returns:
            PyTorch CUDA tensor
        """
        if self._torch is None:
            import torch

            self._torch = torch

        tensor = self._torch.from_numpy(array)
        if dtype is not None:
            tensor = tensor.to(dtype)

        return tensor.to(self.get_cuda_device())

    def device_count(self, backend: str | None = None) -> int:
        """
        Get number of available devices for a backend.

        Args:
            backend: Backend name ('vulkan', 'cuda', 'cpu') or None for current

        Returns:
            Number of devices
        """
        backend = backend or self._current

        if backend == "cuda":
            if self._torch is None:
                try:
                    import torch

                    return torch.cuda.device_count() if torch.cuda.is_available() else 0
                except ImportError:
                    return 0
            return self._torch.cuda.device_count() if self._torch.cuda.is_available() else 0
        elif backend == "vulkan":
            return 1  # Vulkan device count would require enumeration
        else:
            return 1  # CPU

    def synchronize(self, backend: str | None = None):
        """
        Synchronize operations on a backend.

        Args:
            backend: Backend name or None for current
        """
        backend = backend or self._current

        if backend == "cuda":
            if self._torch is None:
                import torch

                self._torch = torch
            self._torch.cuda.synchronize()
        elif backend == "vulkan":
            # Vulkan synchronization would be done via queue submission
            pass


# Global device manager instance
_device_manager: DeviceManager | None = None


def get_device_manager() -> DeviceManager:
    """Get global device manager instance"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager


def set_device(device: str, cuda_index: int | None = None):
    """Set current device (convenience function)"""
    get_device_manager().set_device(device, cuda_index)


def get_device() -> str:
    """Get current device (convenience function)"""
    return get_device_manager().get_device()


def get_vulkan_backend():
    """Get Vulkan backend instance"""
    return get_device_manager().vulkan


def get_cuda_backend():
    """Get CUDA backend (PyTorch)"""
    return get_device_manager().cuda


def get_torch():
    """Get PyTorch module"""
    return get_device_manager().torch
