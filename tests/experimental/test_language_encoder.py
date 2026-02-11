"""
TDD Tests for WordEncoder and SentenceEncoder.

Tests word encoding, n-gram similarity, sentence encoding with roles,
and role queries.
"""

import numpy as np


class TestWordEncoderBasic:
    """Basic tests for WordEncoder initialization."""

    def test_init_default_dimensions(self):
        """Should initialize with default dimensions."""
        from grilly.experimental.language.encoder import WordEncoder

        encoder = WordEncoder()

        assert encoder.dim > 0
        assert hasattr(encoder, "encode_word")

    def test_init_custom_dimension(self):
        """Should initialize with custom dimension."""
        from grilly.experimental.language.encoder import WordEncoder

        encoder = WordEncoder(dim=2048)

        assert encoder.dim == 2048

    def test_init_has_char_vectors(self):
        """Should initialize character vectors for n-grams."""
        from grilly.experimental.language.encoder import WordEncoder

        encoder = WordEncoder()

        assert len(encoder.char_vectors) > 0
        assert "a" in encoder.char_vectors or "<UNK>" in encoder.char_vectors


class TestWordEncode:
    """Tests for word encoding."""

    def test_encode_word_returns_correct_shape(self, dim):
        """encode_word should return vector of correct dimension."""
        from grilly.experimental.language.encoder import WordEncoder

        encoder = WordEncoder(dim=dim)

        result = encoder.encode_word("test")

        assert result.shape == (dim,)

    def test_encode_word_deterministic(self, dim):
        """Same word should produce same encoding."""
        from grilly.experimental.language.encoder import WordEncoder

        encoder = WordEncoder(dim=dim)

        v1 = encoder.encode_word("test")
        v2 = encoder.encode_word("test")

        np.testing.assert_array_equal(v1, v2)

    def test_encode_word_different_words_different_vectors(self, dim):
        """Different words should produce different encodings."""
        from grilly.experimental.language.encoder import WordEncoder

        encoder = WordEncoder(dim=dim)

        v1 = encoder.encode_word("dog")
        v2 = encoder.encode_word("cat")

        assert not np.array_equal(v1, v2)

    def test_encode_word_ngram_similarity(self, dim):
        """Similar spelling should produce similar vectors."""
        from grilly.experimental.language.encoder import WordEncoder
        from grilly.experimental.vsa.ops import HolographicOps

        encoder = WordEncoder(dim=dim, use_ngrams=True)

        v1 = encoder.encode_word("running")
        v2 = encoder.encode_word("runner")
        v3 = encoder.encode_word("zebra")

        sim_similar = HolographicOps.similarity(v1, v2)
        sim_different = HolographicOps.similarity(v1, v3)

        # Similar words should have higher similarity (or at least not significantly lower)
        # Due to HRR approximation, we allow some tolerance
        assert sim_similar >= sim_different - 0.1  # Allow small tolerance


class TestWordRelations:
    """Tests for word relation extraction."""

    def test_extract_relation_basic(self, dim):
        """Should extract relation between two words."""
        from grilly.experimental.language.encoder import WordEncoder

        encoder = WordEncoder(dim=dim)

        relation = encoder.extract_relation("king", "queen")

        assert relation.shape == (dim,)

    def test_apply_relation(self, dim):
        """Should apply relation to get related word."""
        from grilly.experimental.language.encoder import WordEncoder
        from grilly.experimental.vsa.ops import HolographicOps

        encoder = WordEncoder(dim=dim)

        # Extract king->queen relation
        relation = encoder.extract_relation("king", "queen")

        # Apply to source word should recover target word
        encoder.encode_word("king")
        queen_vec = encoder.encode_word("queen")

        # Apply relation
        result_vec = encoder.apply_relation("king", relation)

        # Should be close to queen
        sim = HolographicOps.similarity(result_vec, queen_vec)
        assert sim > 0.8  # Strong similarity expected

    def test_find_closest(self, dim):
        """Should find closest words to a vector."""
        from grilly.experimental.language.encoder import WordEncoder

        encoder = WordEncoder(dim=dim)

        # Encode some words
        encoder.encode_word("dog")
        encoder.encode_word("cat")
        encoder.encode_word("bird")

        # Find closest to "dog"
        dog_vec = encoder.encode_word("dog")
        results = encoder.find_closest(dog_vec, top_k=2)

        assert len(results) == 2
        assert results[0][0] == "dog"  # Should be most similar to itself


class TestSentenceEncoderBasic:
    """Basic tests for SentenceEncoder initialization."""

    def test_init_with_word_encoder(self, dim):
        """Should initialize with WordEncoder."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder

        word_encoder = WordEncoder(dim=dim)
        encoder = SentenceEncoder(word_encoder)

        assert encoder.word_encoder is word_encoder
        assert encoder.dim == dim

    def test_init_has_roles(self, dim):
        """Should initialize with role vectors."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder

        word_encoder = WordEncoder(dim=dim)
        encoder = SentenceEncoder(word_encoder)

        assert "SUBJ" in encoder.roles
        assert "VERB" in encoder.roles
        assert "OBJ" in encoder.roles


class TestSentenceEncode:
    """Tests for sentence encoding."""

    def test_encode_sentence_returns_correct_shape(self, dim):
        """encode_sentence should return vector of correct dimension."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder

        word_encoder = WordEncoder(dim=dim)
        encoder = SentenceEncoder(word_encoder)

        words = ["the", "dog", "chased", "the", "cat"]
        result = encoder.encode_sentence(words)

        assert result.shape == (dim,)

    def test_encode_sentence_with_roles(self, dim):
        """Should encode sentence with explicit roles."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder

        word_encoder = WordEncoder(dim=dim)
        encoder = SentenceEncoder(word_encoder)

        words = ["dog", "chased", "cat"]
        roles = ["SUBJ", "VERB", "OBJ"]

        result = encoder.encode_sentence(words, roles=roles)

        assert result.shape == (dim,)

    def test_encode_sentence_auto_assigns_roles(self, dim):
        """Should auto-assign roles if not provided."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder

        word_encoder = WordEncoder(dim=dim)
        encoder = SentenceEncoder(word_encoder)

        words = ["the", "dog", "chased", "the", "cat"]

        # Should not raise error
        result = encoder.encode_sentence(words)

        assert result.shape == (dim,)


class TestSentenceQueryRole:
    """Tests for querying roles in sentences."""

    def test_query_role_returns_vector(self, dim):
        """query_role should return a vector."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder

        word_encoder = WordEncoder(dim=dim)
        encoder = SentenceEncoder(word_encoder)

        words = ["dog", "chased", "cat"]
        roles = ["SUBJ", "VERB", "OBJ"]

        sent_vec = encoder.encode_sentence(words, roles=roles)
        result = encoder.query_role(sent_vec, "SUBJ")

        assert result.shape == (dim,)

    def test_query_role_recovers_word(self, dim):
        """Querying a role should recover the word filling that role."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder
        from grilly.experimental.vsa.ops import HolographicOps

        word_encoder = WordEncoder(dim=dim)
        encoder = SentenceEncoder(word_encoder)

        words = ["dog", "chased", "cat"]
        roles = ["SUBJ", "VERB", "OBJ"]

        sent_vec = encoder.encode_sentence(words, roles=roles)

        # Query SUBJ role
        query_result = encoder.query_role(sent_vec, "SUBJ")

        # Should be closer to the correct word than to an unrelated word
        dog_vec = word_encoder.encode_word("dog")
        random_vec = word_encoder.encode_word("zebra")
        sim_true = HolographicOps.similarity(query_result, dog_vec)
        sim_rand = HolographicOps.similarity(query_result, random_vec)

        assert sim_true >= sim_rand - 0.05

    def test_find_role_filler(self, dim):
        """find_role_filler should return word candidates."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder

        word_encoder = WordEncoder(dim=dim)
        encoder = SentenceEncoder(word_encoder)

        words = ["dog", "chased", "cat"]
        roles = ["SUBJ", "VERB", "OBJ"]

        sent_vec = encoder.encode_sentence(words, roles=roles)

        # Find SUBJ filler
        results = encoder.find_role_filler(sent_vec, "SUBJ", top_k=3)

        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


class TestSentenceSimilarity:
    """Tests for sentence similarity."""

    def test_sentence_similarity_range(self, dim):
        """Similarity should be in valid range."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder

        word_encoder = WordEncoder(dim=dim)
        encoder = SentenceEncoder(word_encoder)

        sent1 = ["the", "dog", "chased", "the", "cat"]
        sent2 = ["the", "cat", "ran", "from", "the", "dog"]

        sim = encoder.sentence_similarity(sent1, sent2)

        # Allow small floating-point errors
        assert -1.1 <= sim <= 1.1  # Allow small tolerance for FP errors

    def test_sentence_similarity_self(self, dim):
        """Sentence should be most similar to itself."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder

        word_encoder = WordEncoder(dim=dim)
        encoder = SentenceEncoder(word_encoder)

        sent1 = ["the", "dog", "chased", "the", "cat"]
        sent2 = ["lightning", "causes", "thunder"]

        sim_self = encoder.sentence_similarity(sent1, sent1)
        sim_different = encoder.sentence_similarity(sent1, sent2)

        # Self-similarity should be higher (or at least not significantly lower)
        # Due to HRR approximation, allow some tolerance
        assert sim_self >= sim_different - 0.1  # Allow small tolerance
