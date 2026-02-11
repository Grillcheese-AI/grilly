"""
TDD Tests for SentenceGenerator.

Tests template-based sentence generation and relation-based generation.
"""


class TestSentenceGeneratorBasic:
    """Basic tests for SentenceGenerator initialization."""

    def test_init_with_sentence_encoder(self, dim):
        """Should initialize with SentenceEncoder."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder
        from grilly.experimental.language.generator import SentenceGenerator

        word_encoder = WordEncoder(dim=dim)
        sentence_encoder = SentenceEncoder(word_encoder)
        generator = SentenceGenerator(sentence_encoder)

        assert generator.encoder is sentence_encoder
        assert generator.word_encoder is word_encoder


class TestGenerateFromRoles:
    """Tests for template-based generation."""

    def test_generate_from_roles_returns_words(self, dim):
        """generate_from_roles should return list of words."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder
        from grilly.experimental.language.generator import SentenceGenerator

        word_encoder = WordEncoder(dim=dim)
        sentence_encoder = SentenceEncoder(word_encoder)
        generator = SentenceGenerator(sentence_encoder)

        role_fillers = {"SUBJ": "The dog", "VERB": "chased", "OBJ": "the cat"}

        words = generator.generate_from_roles(role_fillers, "simple_transitive")

        assert isinstance(words, list)
        assert len(words) > 0

    def test_generate_from_roles_fills_template(self, dim):
        """Should fill template slots with role fillers."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder
        from grilly.experimental.language.generator import SentenceGenerator

        word_encoder = WordEncoder(dim=dim)
        sentence_encoder = SentenceEncoder(word_encoder)
        generator = SentenceGenerator(sentence_encoder)

        role_fillers = {"SUBJ": "The robot", "VERB": "learned", "OBJ": "language"}

        words = generator.generate_from_roles(role_fillers, "simple_transitive")

        # Should contain the fillers
        words_str = " ".join(words).lower()
        assert "robot" in words_str or "learned" in words_str or "language" in words_str


class TestGenerateFromRelation:
    """Tests for relation-based generation."""

    def test_generate_from_relation_returns_words(self, dim):
        """generate_from_relation should return list of words."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder
        from grilly.experimental.language.generator import SentenceGenerator

        word_encoder = WordEncoder(dim=dim)
        sentence_encoder = SentenceEncoder(word_encoder)
        generator = SentenceGenerator(sentence_encoder)

        words = generator.generate_from_relation("lightning", "causes", "thunder")

        assert isinstance(words, list)
        assert len(words) > 0

    def test_generate_from_relation_expresses_relation(self, dim):
        """Should generate sentence expressing the relation."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder
        from grilly.experimental.language.generator import SentenceGenerator

        word_encoder = WordEncoder(dim=dim)
        sentence_encoder = SentenceEncoder(word_encoder)
        generator = SentenceGenerator(sentence_encoder)

        words = generator.generate_from_relation("lightning", "causes", "thunder")

        words_str = " ".join(words).lower()
        # Should contain subject and object
        assert "lightning" in words_str or "thunder" in words_str


class TestCompleteSentence:
    """Tests for sentence completion."""

    def test_complete_sentence_returns_candidates(self, dim):
        """complete_sentence should return word candidates."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder
        from grilly.experimental.language.generator import SentenceGenerator

        word_encoder = WordEncoder(dim=dim)
        sentence_encoder = SentenceEncoder(word_encoder)
        generator = SentenceGenerator(sentence_encoder)

        partial = ["the", "dog", "chased"]
        partial_roles = ["DET", "SUBJ", "VERB"]

        results = generator.complete_sentence(partial, partial_roles, "OBJ")

        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
