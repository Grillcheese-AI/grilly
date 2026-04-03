"""
GPU Flash Attention 2 at medium/long sequence lengths (Workstream C3).

Smoke tests: output shape, finiteness. Tight parity vs a reference attention
implementation is tracked separately (FA2 uses online softmax; paths may diverge).
"""

import numpy as np
import pytest

try:
    from grilly import Compute
    from grilly.backend.base import VULKAN_AVAILABLE
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


@pytest.mark.gpu
@pytest.mark.parametrize("seq_len", [128, 256, 512])
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
def test_flash_attention2_long_sequence_finite(seq_len):
    backend = Compute()
    try:
        batch_size = 1
        num_heads = 2
        head_dim = 32
        rng = np.random.default_rng(42 + seq_len)
        q = rng.standard_normal((batch_size, seq_len, num_heads * head_dim), dtype=np.float32)
        k_arr = rng.standard_normal((batch_size, seq_len, num_heads * head_dim), dtype=np.float32)
        v_arr = rng.standard_normal((batch_size, seq_len, num_heads * head_dim), dtype=np.float32)

        out = backend.flash_attention2(
            q,
            k_arr,
            v_arr,
            num_heads,
            head_dim,
            tile_size_q=32,
            tile_size_k=32,
        )
        assert out.shape == (batch_size, seq_len, num_heads, head_dim)
        assert np.all(np.isfinite(out))
    finally:
        backend.cleanup()


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
def test_flash_attention2_very_long_sequence_finite(seq_len):
    backend = Compute()
    try:
        batch_size = 1
        num_heads = 2
        head_dim = 32
        rng = np.random.default_rng(7)
        q = rng.standard_normal((batch_size, seq_len, num_heads * head_dim), dtype=np.float32)
        k_arr = rng.standard_normal((batch_size, seq_len, num_heads * head_dim), dtype=np.float32)
        v_arr = rng.standard_normal((batch_size, seq_len, num_heads * head_dim), dtype=np.float32)

        out = backend.flash_attention2(
            q,
            k_arr,
            v_arr,
            num_heads,
            head_dim,
            tile_size_q=64,
            tile_size_k=64,
        )
        assert out.shape == (batch_size, seq_len, num_heads, head_dim)
        assert np.all(np.isfinite(out))
    finally:
        backend.cleanup()
