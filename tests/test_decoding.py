"""Tests for nn.decoding module."""

import numpy as np
import pytest

try:
    from grilly.nn.decoding import GreedyDecoder, SampleDecoder
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


class TestGreedyDecoder:
    """Tests for GreedyDecoder (CPU fallback path)."""

    def test_greedy_decoder_2d(self):
        """GreedyDecoder forward with (batch, vocab_size) logits."""
        decoder = GreedyDecoder(vocab_size=10)
        logits = np.random.randn(4, 10).astype(np.float32)
        out = decoder.forward(logits)
        assert out.shape in ((4,), (4, 1))
        assert out.dtype in (np.int32, np.int64, np.uint32)
        assert np.all((out >= 0) & (out < 10))

    def test_greedy_decoder_3d(self):
        """GreedyDecoder forward with (batch, seq_len, vocab_size) logits."""
        decoder = GreedyDecoder(vocab_size=20)
        logits = np.random.randn(2, 5, 20).astype(np.float32)
        out = decoder.forward(logits)
        assert out.shape == (2, 5)
        assert np.all((out >= 0) & (out < 20))

    def test_greedy_decoder_argmax(self):
        """GreedyDecoder returns argmax indices."""
        decoder = GreedyDecoder(vocab_size=5)
        # logits where index 3 is max
        logits = np.array([[-1, -1, -1, 10, -1]], dtype=np.float32)
        out = decoder.forward(logits)
        assert out[0] == 3

    def test_greedy_decoder_repr(self):
        """GreedyDecoder has repr."""
        decoder = GreedyDecoder(vocab_size=100)
        assert "GreedyDecoder" in repr(decoder)


class TestSampleDecoder:
    """Tests for SampleDecoder (CPU fallback path)."""

    def test_sample_decoder_2d(self):
        """SampleDecoder forward with (batch, vocab_size) logits."""
        decoder = SampleDecoder(vocab_size=10, temperature=1.0)
        np.random.seed(42)
        logits = np.random.randn(4, 10).astype(np.float32)
        out = decoder.forward(logits)
        assert out.shape in ((4,), (4, 1))
        assert np.all((out >= 0) & (out < 10))

    def test_sample_decoder_3d(self):
        """SampleDecoder forward with (batch, seq_len, vocab_size) logits."""
        decoder = SampleDecoder(vocab_size=20, temperature=0.5)
        np.random.seed(123)
        logits = np.random.randn(2, 3, 20).astype(np.float32)
        out = decoder.forward(logits)
        assert out.shape == (2, 3)

    def test_sample_decoder_high_temperature(self):
        """SampleDecoder with temperature > 1."""
        decoder = SampleDecoder(vocab_size=5, temperature=2.0)
        logits = np.random.randn(2, 5).astype(np.float32)
        out = decoder.forward(logits)
        assert out.shape in ((2,), (2, 1))
        assert np.all((out >= 0) & (out < 5))
