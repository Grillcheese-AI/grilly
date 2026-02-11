"""
TDD Tests for ResonatorParser.

Tests sentence parsing via resonator factorization.
"""

import numpy as np


class TestResonatorParserBasic:
    """Basic tests for ResonatorParser initialization."""

    def test_init_with_sentence_encoder(self, dim):
        """Should initialize with SentenceEncoder."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder
        from grilly.experimental.language.parser import ResonatorParser

        word_encoder = WordEncoder(dim=dim)
        sentence_encoder = SentenceEncoder(word_encoder)
        parser = ResonatorParser(sentence_encoder)

        assert parser.encoder is sentence_encoder
        assert parser.dim == dim


class TestParse:
    """Tests for sentence parsing."""

    def test_parse_returns_word_role_pairs(self, dim):
        """parse should return list of (word, role, confidence) tuples."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder
        from grilly.experimental.language.parser import ResonatorParser

        word_encoder = WordEncoder(dim=dim)
        sentence_encoder = SentenceEncoder(word_encoder)
        parser = ResonatorParser(sentence_encoder)

        # Encode a sentence
        words = ["dog", "chased", "cat"]
        roles = ["SUBJ", "VERB", "OBJ"]
        sent_vec = sentence_encoder.encode_sentence(words, roles=roles)

        results = parser.parse(sent_vec, num_slots=3)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 3 for r in results)

    def test_parse_recovers_words(self, dim):
        """parse should recover words from sentence vector."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder
        from grilly.experimental.language.parser import ResonatorParser

        word_encoder = WordEncoder(dim=dim)
        sentence_encoder = SentenceEncoder(word_encoder)
        parser = ResonatorParser(sentence_encoder)

        # Encode words first
        words = ["dog", "chased", "cat"]
        for word in words:
            word_encoder.encode_word(word)

        roles = ["SUBJ", "VERB", "OBJ"]
        sent_vec = sentence_encoder.encode_sentence(words, roles=roles)

        results = parser.parse(sent_vec, num_slots=3)

        # Should recover at least some words
        recovered_words = [r[0] for r in results]
        assert len(recovered_words) > 0


class TestParallelParse:
    """Tests for parallel parsing with known vocabulary."""

    def test_parallel_parse_returns_dict(self, dim):
        """parallel_parse should return word->(role, confidence) mapping."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder
        from grilly.experimental.language.parser import ResonatorParser

        word_encoder = WordEncoder(dim=dim)
        sentence_encoder = SentenceEncoder(word_encoder)
        parser = ResonatorParser(sentence_encoder)

        # Encode words
        words = ["dog", "chased", "cat"]
        for word in words:
            word_encoder.encode_word(word)

        roles = ["SUBJ", "VERB", "OBJ"]
        sent_vec = sentence_encoder.encode_sentence(words, roles=roles)

        results = parser.parallel_parse(sent_vec, known_words=words)

        assert isinstance(results, dict)
        assert len(results) > 0

    def test_parallel_parse_assigns_roles(self, dim):
        """parallel_parse should assign roles to known words."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder
        from grilly.experimental.language.parser import ResonatorParser

        word_encoder = WordEncoder(dim=dim)
        sentence_encoder = SentenceEncoder(word_encoder)
        parser = ResonatorParser(sentence_encoder)

        words = ["dog", "chased", "cat"]
        for word in words:
            word_encoder.encode_word(word)

        roles = ["SUBJ", "VERB", "OBJ"]
        sent_vec = sentence_encoder.encode_sentence(words, roles=roles)

        results = parser.parallel_parse(sent_vec, known_words=words)

        # Should assign roles
        for word, (role, conf) in results.items():
            assert isinstance(role, str)
            assert isinstance(conf, (float, np.floating))
            assert conf >= 0
