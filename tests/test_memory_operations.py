"""
Tests for memory operations
"""

import numpy as np
import pytest

try:
    from grilly import Compute
    from grilly.backend.base import VULKAN_AVAILABLE
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestMemoryOperations:
    """Test memory read/write operations"""

    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_memory_write(self, gpu):
        """Test memory write operation"""
        n_memories = 10
        embedding_dim = 384

        keys = np.zeros((n_memories, embedding_dim), dtype=np.float32)
        values = np.zeros((n_memories, embedding_dim), dtype=np.float32)

        new_key = np.random.randn(embedding_dim).astype(np.float32)
        new_value = np.random.randn(embedding_dim).astype(np.float32)
        write_index = 5

        updated_keys, updated_values = gpu.memory_write(
            new_key, new_value, keys, values, write_index, write_mode=0
        )

        # Check the memory was written
        np.testing.assert_allclose(updated_keys[write_index], new_key, rtol=1e-4)
        np.testing.assert_allclose(updated_values[write_index], new_value, rtol=1e-4)

    def test_memory_read(self, gpu):
        """Test memory read operation"""
        n_memories = 10
        embedding_dim = 384

        keys = np.random.randn(n_memories, embedding_dim).astype(np.float32)
        values = np.random.randn(n_memories, embedding_dim).astype(np.float32)

        queries = np.random.randn(1, embedding_dim).astype(np.float32)

        result = gpu.memory_read(queries, keys, values)

        assert result.shape[0] == 1
        assert result.shape[1] == embedding_dim
