"""Tests for utils/batch_embedder.py.

All tests mock GrillyOnnxModel.forward() so no real ONNX file is required.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("onnx")
from grilly.utils.batch_embedder import BatchEmbedder, WhitespaceTokenizer, _pad_batch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBED_DIM = 32


def _make_mock_model(batch_size: int, seq_len: int, embed_dim: int = EMBED_DIM):
    """Return a mock GrillyOnnxModel whose forward() returns (B, L, D)."""
    model = MagicMock()
    model.forward.return_value = np.random.randn(batch_size, seq_len, embed_dim).astype(
        np.float32
    )
    return model


def _make_embedder(model) -> BatchEmbedder:
    """Build a BatchEmbedder from a pre-built mock model (no file I/O)."""
    return BatchEmbedder(model=model)


# ---------------------------------------------------------------------------
# WhitespaceTokenizer
# ---------------------------------------------------------------------------


class TestWhitespaceTokenizer:
    def test_basic_encode(self):
        tok = WhitespaceTokenizer()
        ids = tok.encode("hello world")
        assert isinstance(ids, list)
        assert len(ids) == 2
        assert all(isinstance(i, int) for i in ids)

    def test_empty_string_returns_unk(self):
        tok = WhitespaceTokenizer()
        ids = tok.encode("")
        assert ids == [WhitespaceTokenizer.UNK_ID]

    def test_max_length_truncation(self):
        tok = WhitespaceTokenizer(max_length=3)
        ids = tok.encode("a b c d e f")
        assert len(ids) == 3

    def test_pad_token_id_is_zero(self):
        assert WhitespaceTokenizer().pad_token_id == 0

    def test_vocabulary_grows_incrementally(self):
        tok = WhitespaceTokenizer()
        ids1 = tok.encode("foo")
        ids2 = tok.encode("bar")
        ids3 = tok.encode("foo")
        # Same word → same id
        assert ids1 == ids3
        # Different words → different ids
        assert ids1 != ids2


# ---------------------------------------------------------------------------
# _pad_batch
# ---------------------------------------------------------------------------


class TestPadBatch:
    def test_pads_to_max_length(self):
        seqs = [[1, 2, 3], [4, 5]]
        result = _pad_batch(seqs, pad_id=0)
        assert result.shape == (2, 3)
        assert result[1, 2] == 0  # padded position

    def test_dtype_is_int64(self):
        result = _pad_batch([[1, 2]], pad_id=0)
        assert result.dtype == np.int64

    def test_single_sequence(self):
        result = _pad_batch([[7, 8, 9]], pad_id=0)
        assert result.shape == (1, 3)
        np.testing.assert_array_equal(result[0], [7, 8, 9])


# ---------------------------------------------------------------------------
# BatchEmbedder initialisation
# ---------------------------------------------------------------------------


class TestBatchEmbedderInit:
    def test_requires_model_path_or_model(self):
        with pytest.raises(ValueError, match="model_path or model"):
            BatchEmbedder()

    def test_init_with_preloaded_model(self):
        mock_model = _make_mock_model(1, 5)
        embedder = _make_embedder(mock_model)
        assert embedder._model is mock_model

    def test_uses_whitespace_tokenizer_by_default(self):
        mock_model = _make_mock_model(1, 5)
        embedder = _make_embedder(mock_model)
        assert isinstance(embedder._tokenizer, WhitespaceTokenizer)

    def test_custom_tokenizer_accepted(self):
        mock_model = _make_mock_model(1, 5)
        custom_tok = MagicMock()
        custom_tok.encode.return_value = [1, 2, 3]
        custom_tok.pad_token_id = 0
        embedder = BatchEmbedder(model=mock_model, tokenizer=custom_tok)
        assert embedder._tokenizer is custom_tok

    @patch("grilly.utils.batch_embedder.OnnxModelLoader.load")
    def test_loads_model_from_path(self, mock_load):
        mock_model = _make_mock_model(1, 5)
        mock_load.return_value = mock_model
        embedder = BatchEmbedder("dummy.onnx")
        mock_load.assert_called_once_with("dummy.onnx")
        assert embedder._model is mock_model


# ---------------------------------------------------------------------------
# BatchEmbedder.encode — shape correctness
# ---------------------------------------------------------------------------


class TestEncode:
    def test_encode_returns_correct_shape(self):
        n_texts = 5
        model = _make_mock_model(n_texts, 10, EMBED_DIM)
        embedder = _make_embedder(model)
        texts = [f"sentence {i}" for i in range(n_texts)]
        result = embedder.encode(texts)
        assert result.shape == (n_texts, EMBED_DIM)

    def test_encode_returns_float32(self):
        model = _make_mock_model(3, 8, EMBED_DIM)
        embedder = _make_embedder(model)
        result = embedder.encode(["a", "b", "c"])
        assert result.dtype == np.float32

    def test_encode_empty_raises(self):
        model = _make_mock_model(1, 5)
        embedder = _make_embedder(model)
        with pytest.raises(ValueError):
            embedder.encode([])

    def test_batch_size_splits_correctly(self):
        """Verify forward() is called the expected number of times."""
        texts = [f"text {i}" for i in range(10)]
        call_count = 0

        def side_effect(input_ids):
            nonlocal call_count
            b = input_ids.shape[0]
            call_count += 1
            return np.random.randn(b, 5, EMBED_DIM).astype(np.float32)

        model = MagicMock()
        model.forward.side_effect = side_effect
        embedder = _make_embedder(model)

        embedder.encode(texts, batch_size=3)
        # ceil(10 / 3) = 4 batches
        assert call_count == 4

    def test_encode_concatenates_all_batches(self):
        texts = [f"t{i}" for i in range(7)]

        def side_effect(input_ids):
            b = input_ids.shape[0]
            return np.ones((b, 4, EMBED_DIM), dtype=np.float32)

        model = MagicMock()
        model.forward.side_effect = side_effect
        embedder = _make_embedder(model)

        result = embedder.encode(texts, batch_size=4)
        assert result.shape == (7, EMBED_DIM)

    def test_mean_pool_applied_to_3d_output(self):
        """Forward returns (B, L, D); result should be (B, D) after pooling."""
        B, L, D = 2, 6, 16
        fake_output = np.zeros((B, L, D), dtype=np.float32)
        fake_output[:, :, 0] = 1.0  # first dim always 1 after mean pool

        model = MagicMock()
        model.forward.return_value = fake_output
        embedder = _make_embedder(model)

        result = embedder.encode(["hello", "world"])
        assert result.shape == (B, D)
        np.testing.assert_allclose(result[:, 0], 1.0)

    def test_already_pooled_2d_output(self):
        """If model returns (B, D), encode should pass it through unchanged."""
        B, D = 3, 64
        fake_output = np.random.randn(B, D).astype(np.float32)

        model = MagicMock()
        model.forward.return_value = fake_output
        embedder = _make_embedder(model)

        result = embedder.encode(["a", "b", "c"])
        assert result.shape == (B, D)
        np.testing.assert_array_almost_equal(result, fake_output)


# ---------------------------------------------------------------------------
# BatchEmbedder.encode_one
# ---------------------------------------------------------------------------


class TestEncodeOne:
    def test_returns_1d_vector(self):
        model = MagicMock()
        model.forward.return_value = np.random.randn(1, 5, EMBED_DIM).astype(np.float32)
        embedder = _make_embedder(model)
        result = embedder.encode_one("hello world")
        assert result.ndim == 1
        assert result.shape == (EMBED_DIM,)

    def test_consistent_with_encode(self):
        """encode_one("x") should match encode(["x"])[0]."""
        model = MagicMock()
        fixed = np.random.randn(1, 4, EMBED_DIM).astype(np.float32)
        model.forward.return_value = fixed
        embedder = _make_embedder(model)

        one = embedder.encode_one("test sentence")
        model.forward.return_value = fixed  # reset for second call
        batch = embedder.encode(["test sentence"])[0]

        np.testing.assert_array_almost_equal(one, batch)
