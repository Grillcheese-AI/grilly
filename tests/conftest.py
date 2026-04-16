"""
Pytest configuration and fixtures for Grilly tests
"""

import numpy as np
import pytest


def pytest_configure(config):
    """Register custom markers and configure environment."""
    # Set non-interactive backend for everything to avoid Tkinter issues in tests
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        pass

    config.addinivalue_line(
        "markers", "gpu: marks tests that require Vulkan/GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks GPU benchmark tests (deselect with '-m \"not benchmark\"')"
    )
    config.addinivalue_line(
        "markers", "cpp: marks tests that require C++ backend (deselect with '-m \"not cpp\"')"
    )
    config.addinivalue_line(
        "markers", "parity: marks numerical parity tests (numpy / optional PyTorch reference)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests that are slow or heavy (deselect with '-m \"not slow\"')"
    )


try:
    import grilly
    from grilly.backend.base import (
        VULKAN_AVAILABLE,
        VULKAN_PYTHON_LEGACY_BACKEND_AVAILABLE,
    )

    GRILLY_AVAILABLE = True
except (ImportError, AttributeError, Exception):
    GRILLY_AVAILABLE = False
    VULKAN_AVAILABLE = False
    VULKAN_PYTHON_LEGACY_BACKEND_AVAILABLE = False

# C++ backend (grilly_core with NN framework classes)
try:
    from grilly_core import Tensor as _CppTensor

    CPP_AVAILABLE = True
except (ImportError, AttributeError):
    CPP_AVAILABLE = False


@pytest.fixture
def gpu_backend():
    """Fixture for GPU backend (skips if not available)"""
    if not VULKAN_PYTHON_LEGACY_BACKEND_AVAILABLE:
        pytest.skip(
            "Vulkan Compute() not available (needs C++ grilly_core GPU + pip install vulkan)"
        )
    try:
        from grilly import Compute

        backend = Compute()
        yield backend
        # Cleanup
        if hasattr(backend, "cleanup"):
            backend.cleanup()
    except Exception as e:
        pytest.skip(f"GPU backend not available: {e}")


@pytest.fixture
def cpu_backend():
    """Fixture for CPU fallback testing"""
    # For tests that should work without GPU
    return None


@pytest.fixture
def test_data():
    """Fixture providing test data"""
    np.random.seed(42)
    return {
        "small_vector": np.random.randn(100).astype(np.float32),
        "medium_vector": np.random.randn(1000).astype(np.float32),
        "large_vector": np.random.randn(10000).astype(np.float32),
        "embedding": np.random.randn(384).astype(np.float32),
        "matrix_100x128": np.random.randn(100, 128).astype(np.float32),
        "matrix_10x384": np.random.randn(10, 384).astype(np.float32),
    }
