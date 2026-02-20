"""Tests for Phase 7: SNN Attention"""

import numpy as np
import pytest


class TestTemporalWiseAttention:
    """Test TWA squeeze-excite on T dimension."""

    def test_shape(self):
        """TWA should preserve [T, N, C, H, W] shape."""
        from grilly.nn.snn_attention import TemporalWiseAttention

        twa = TemporalWiseAttention(T=4, channels=16, reduction=4)
        x = np.random.randn(4, 2, 16, 8, 8).astype(np.float32)
        out = twa(x)
        assert out.shape == (4, 2, 16, 8, 8)

    def test_output_range(self):
        """TWA sigmoid attention should scale but not wildly change values."""
        from grilly.nn.snn_attention import TemporalWiseAttention

        twa = TemporalWiseAttention(T=4, channels=8, reduction=2)
        x = np.ones((4, 2, 8, 4, 4), dtype=np.float32)
        out = twa(x)
        # Should be positive (sigmoid * positive input)
        assert np.all(out >= 0)

    def test_has_parameters(self):
        """TWA should have learnable FC weights."""
        from grilly.nn.snn_attention import TemporalWiseAttention

        twa = TemporalWiseAttention(T=4, channels=16, reduction=4)
        params = list(twa.parameters())
        assert len(params) == 4  # fc1_weight, fc1_bias, fc2_weight, fc2_bias


class TestMultiDimensionalAttention:
    """Test MA-SNN multi-dimensional attention."""

    def test_shape(self):
        """MDA should preserve shape."""
        from grilly.nn.snn_attention import MultiDimensionalAttention

        mda = MultiDimensionalAttention(T=4, channels=8, reduction=2)
        x = np.random.randn(4, 2, 8, 4, 4).astype(np.float32)
        out = mda(x)
        assert out.shape == (4, 2, 8, 4, 4)


class TestSpikingSelfAttention:
    """Test Spikformer-style spiking self-attention."""

    def test_shape(self):
        """SSA should process [N, L, D] input."""
        from grilly.nn.snn_attention import SpikingSelfAttention

        ssa = SpikingSelfAttention(embed_dim=32, num_heads=4)
        x = np.random.randn(2, 8, 32).astype(np.float32)
        out = ssa(x)
        assert out.shape == (2, 8, 32)

    def test_invalid_heads(self):
        """embed_dim not divisible by num_heads should error."""
        from grilly.nn.snn_attention import SpikingSelfAttention

        with pytest.raises(ValueError, match="not divisible"):
            SpikingSelfAttention(embed_dim=33, num_heads=4)


class TestQKAttention:
    """Test QK attention variants."""

    def test_qk_shape(self):
        """QKAttention should return attention scores."""
        from grilly.nn.snn_attention import QKAttention

        qk = QKAttention(embed_dim=16, num_heads=2)
        x = np.random.randn(2, 4, 16).astype(np.float32)
        out = qk(x)
        assert out.shape == (2, 4, 4)  # (N, L, L) attention matrix

    def test_token_qk_shape(self):
        """TokenQKAttention should return attended tokens."""
        from grilly.nn.snn_attention import TokenQKAttention

        tqk = TokenQKAttention(embed_dim=16, num_heads=2)
        x = np.random.randn(2, 4, 16).astype(np.float32)
        out = tqk(x)
        assert out.shape == (2, 4, 16)

    def test_channel_qk_shape(self):
        """ChannelQKAttention should return attended channels."""
        from grilly.nn.snn_attention import ChannelQKAttention

        cqk = ChannelQKAttention(embed_dim=16, num_heads=2)
        x = np.random.randn(2, 4, 16).astype(np.float32)
        out = cqk(x)
        assert out.shape == (2, 4, 16)
