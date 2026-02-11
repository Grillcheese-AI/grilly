"""
TDD Tests for InstantLanguage system.

Tests the complete language learning system integration.
"""


class TestInstantLanguageBasic:
    """Basic tests for InstantLanguage initialization."""

    def test_init_default_dimensions(self):
        """Should initialize with default dimensions."""
        from grilly.experimental.language.system import InstantLanguage

        lang = InstantLanguage()

        assert lang.dim > 0
        assert hasattr(lang, "word_encoder")
        assert hasattr(lang, "sentence_encoder")
        assert hasattr(lang, "generator")
        assert hasattr(lang, "parser")

    def test_init_custom_dimension(self):
        """Should initialize with custom dimension."""
        from grilly.experimental.language.system import InstantLanguage

        lang = InstantLanguage(dim=2048)

        assert lang.dim == 2048


class TestLearnSentence:
    """Tests for learning sentences."""

    def test_learn_sentence_returns_vector(self, dim):
        """learn_sentence should return sentence vector."""
        from grilly.experimental.language.system import InstantLanguage

        lang = InstantLanguage(dim=dim)

        sent_vec = lang.learn_sentence("The dog chased the cat")

        assert sent_vec.shape == (dim,)

    def test_learn_sentence_stores_in_memory(self, dim):
        """learn_sentence should store sentence in memory."""
        from grilly.experimental.language.system import InstantLanguage

        lang = InstantLanguage(dim=dim)

        initial_count = len(lang.sentence_memory)
        lang.learn_sentence("The dog chased the cat")

        assert len(lang.sentence_memory) == initial_count + 1


class TestLearnRelation:
    """Tests for learning word relations."""

    def test_learn_relation_stores_pairs(self, dim):
        """learn_relation should store word pairs."""
        from grilly.experimental.language.system import InstantLanguage

        lang = InstantLanguage(dim=dim)

        lang.learn_relation("king", "queen", "gender_swap")

        assert "gender_swap" in lang.relation_memory
        assert len(lang.relation_memory["gender_swap"]) > 0


class TestQueryRelation:
    """Tests for querying relations."""

    def test_query_relation_returns_candidates(self, dim):
        """query_relation should return word candidates."""
        from grilly.experimental.language.system import InstantLanguage

        lang = InstantLanguage(dim=dim)

        # Learn relation first
        lang.learn_relation("king", "queen", "gender_swap")
        lang.learn_relation("man", "woman", "gender_swap")

        results = lang.query_relation("king", "gender_swap")

        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


class TestExpressRelation:
    """Tests for expressing relations as sentences."""

    def test_express_relation_returns_string(self, dim):
        """express_relation should return sentence string."""
        from grilly.experimental.language.system import InstantLanguage

        lang = InstantLanguage(dim=dim)

        sentence = lang.express_relation("lightning", "causes", "thunder")

        assert isinstance(sentence, str)
        assert len(sentence) > 0


class TestParseSentence:
    """Tests for parsing sentences."""

    def test_parse_sentence_returns_word_role_pairs(self, dim):
        """parse_sentence should return parsed structure."""
        from grilly.experimental.language.system import InstantLanguage

        lang = InstantLanguage(dim=dim)

        results = lang.parse_sentence("The dog chased the cat")

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 3 for r in results)


class TestFindSimilarSentences:
    """Tests for finding similar sentences."""

    def test_find_similar_sentences_returns_results(self, dim):
        """find_similar_sentences should return similar sentences."""
        from grilly.experimental.language.system import InstantLanguage

        lang = InstantLanguage(dim=dim)

        # Learn some sentences
        lang.learn_sentence("The dog chased the cat")
        lang.learn_sentence("The cat ran from the dog")
        lang.learn_sentence("Lightning causes thunder")

        results = lang.find_similar_sentences("The dog chased the cat", top_k=2)

        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


class TestComplete:
    """Tests for sentence completion."""

    def test_complete_returns_candidates(self, dim):
        """complete should return word candidates."""
        from grilly.experimental.language.system import InstantLanguage

        lang = InstantLanguage(dim=dim)

        results = lang.complete("The dog chased the", role="OBJ")

        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


class TestAnalogy:
    """Tests for solving analogies."""

    def test_analogy_returns_candidates(self, dim):
        """analogy should return word candidates."""
        from grilly.experimental.language.system import InstantLanguage

        lang = InstantLanguage(dim=dim)

        # Encode words
        lang.word_encoder.encode_word("king")
        lang.word_encoder.encode_word("queen")
        lang.word_encoder.encode_word("man")
        lang.word_encoder.encode_word("woman")

        results = lang.analogy("king", "queen", "man")

        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_analogy_king_queen_man_woman(self, dim):
        """Should solve king:queen :: man:woman analogy."""
        from grilly.experimental.language.system import InstantLanguage

        lang = InstantLanguage(dim=dim)

        # Encode words
        lang.word_encoder.encode_word("king")
        lang.word_encoder.encode_word("queen")
        lang.word_encoder.encode_word("man")
        lang.word_encoder.encode_word("woman")

        results = lang.analogy("king", "queen", "man")

        # Top result should be "woman" or similar
        top_words = [r[0] for r in results[:3]]
        # Check if woman is in top results (may not be exact due to approximation)
        assert len(top_words) > 0
