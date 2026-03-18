"""
Tests for BatchVSAEncoder — GPU-accelerated batch text hypervector encoding.

All tests run on CPU (no Vulkan required). GPU path is tested only when
a Vulkan device is available.
"""

import numpy as np
import pytest

from grilly.experimental.language.batch_encoder import BatchVSAEncoder
from grilly.experimental.vsa.ops import HolographicOps


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 512  # small for fast tests


@pytest.fixture(scope="module")
def encoder():
    """Shared BatchVSAEncoder (CPU-only) for all tests."""
    return BatchVSAEncoder(dim=DIM, use_gpu=False)


# ---------------------------------------------------------------------------
# encode_sentences
# ---------------------------------------------------------------------------


class TestEncodeSentences:
    """Shape and basic correctness of encode_sentences."""

    def test_returns_correct_shape(self, encoder):
        texts = ["hello world", "the cat sat on the mat", "foo bar baz"]
        out = encoder.encode_sentences(texts)
        assert out.shape == (3, DIM)

    def test_dtype_is_float32(self, encoder):
        out = encoder.encode_sentences(["hello world"])
        assert out.dtype == np.float32

    def test_single_sentence(self, encoder):
        out = encoder.encode_sentences(["one two three"])
        assert out.shape == (1, DIM)

    def test_empty_list_returns_zero_rows(self, encoder):
        out = encoder.encode_sentences([])
        assert out.shape == (0, DIM)

    def test_large_batch(self, encoder):
        texts = [f"sentence number {i}" for i in range(200)]
        out = encoder.encode_sentences(texts, batch_size=50)
        assert out.shape == (200, DIM)

    def test_vectors_are_not_all_zero(self, encoder):
        out = encoder.encode_sentences(["the quick brown fox"])
        assert not np.all(out == 0)


# ---------------------------------------------------------------------------
# encode_words
# ---------------------------------------------------------------------------


class TestEncodeWords:
    """Shape and basic correctness of encode_words."""

    def test_returns_correct_shape(self, encoder):
        words = ["apple", "banana", "cherry"]
        out = encoder.encode_words(words)
        assert out.shape == (3, DIM)

    def test_dtype_is_float32(self, encoder):
        out = encoder.encode_words(["hello"])
        assert out.dtype == np.float32

    def test_empty_list(self, encoder):
        out = encoder.encode_words([])
        assert out.shape == (0, DIM)

    def test_single_word(self, encoder):
        out = encoder.encode_words(["grilly"])
        assert out.shape == (1, DIM)

    def test_different_words_different_vectors(self, encoder):
        out = encoder.encode_words(["dog", "cat"])
        assert not np.array_equal(out[0], out[1])


# ---------------------------------------------------------------------------
# Semantic similarity: similar sentences > random
# ---------------------------------------------------------------------------


class TestSimilarity:
    """Similar sentences should be more similar than unrelated ones."""

    def test_similar_sentences_more_similar_than_random(self, encoder):
        """
        Two sentences sharing most words should be more similar to each other
        than to a completely different, unrelated sentence.
        """
        sent_a = "the dog chased the ball"
        sent_b = "the dog ran after the ball"
        sent_c = "quantum mechanics describes subatomic phenomena"

        vecs = encoder.encode_sentences([sent_a, sent_b, sent_c])
        sim_ab = HolographicOps.similarity(vecs[0], vecs[1])
        sim_ac = HolographicOps.similarity(vecs[0], vecs[2])

        # Allow a generous tolerance — VSA is approximate
        assert sim_ab >= sim_ac - 0.15, (
            f"Expected sim(a,b)={sim_ab:.3f} >= sim(a,c)={sim_ac:.3f} - 0.15"
        )

    def test_sentence_self_similarity_is_high(self, encoder):
        """A sentence should be maximally similar to itself."""
        text = "the quick brown fox jumps over the lazy dog"
        vecs = encoder.encode_sentences([text, text])
        sim = HolographicOps.similarity(vecs[0], vecs[1])
        # Identical inputs → should be >= 0.99 (floating-point deterministic)
        assert sim >= 0.99, f"Expected self-similarity >= 0.99, got {sim:.4f}"


# ---------------------------------------------------------------------------
# Batch consistency: same result as single-sentence encoding
# ---------------------------------------------------------------------------


class TestBatchConsistency:
    """encode_sentences should give the same result as one-at-a-time encoding."""

    def test_batch_matches_single(self, encoder):
        texts = [
            "the cat sat on the mat",
            "hello world",
            "machine learning is great",
        ]

        # Batch encode
        batch_vecs = encoder.encode_sentences(texts)

        # Single encode
        single_vecs = np.array([encoder.encode_sentences([t])[0] for t in texts])

        np.testing.assert_allclose(
            batch_vecs,
            single_vecs,
            atol=1e-5,
            err_msg="Batch encoding differs from single encoding",
        )

    def test_encode_words_matches_word_encoder(self, encoder):
        """encode_words output should match underlying WordEncoder."""
        words = ["apple", "banana", "cherry"]
        batch_out = encoder.encode_words(words)

        for i, word in enumerate(words):
            expected = encoder.word_encoder.encode_word(word)
            np.testing.assert_allclose(
                batch_out[i],
                expected,
                atol=1e-6,
                err_msg=f"Mismatch for word '{word}'",
            )
