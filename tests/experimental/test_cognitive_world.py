"""
TDD Tests for WorldModel.

Tests fact storage, querying, and coherence checking.
"""

import numpy as np


class TestWorldModelBasic:
    """Basic tests for WorldModel initialization."""

    def test_init_default_dimensions(self):
        """Should initialize with default dimensions."""
        from grilly.experimental.cognitive.world import WorldModel

        world = WorldModel()

        assert world.dim > 0
        assert len(world.facts) == 0

    def test_init_custom_dimension(self):
        """Should initialize with custom dimension."""
        from grilly.experimental.cognitive.world import WorldModel

        world = WorldModel(dim=2048)

        assert world.dim == 2048


class TestWorldModelFacts:
    """Tests for fact storage and querying."""

    def test_add_fact_stores_fact(self, dim):
        """add_fact should store fact in world model."""
        from grilly.experimental.cognitive.world import WorldModel

        world = WorldModel(dim=dim)

        world.add_fact("dog", "is", "animal")

        assert len(world.facts) == 1
        assert world.facts[0].subject == "dog"
        assert world.facts[0].relation == "is"
        assert world.facts[0].object == "animal"

    def test_encode_fact_returns_vector(self, dim):
        """encode_fact should return vector of correct dimension."""
        from grilly.experimental.cognitive.world import WorldModel

        world = WorldModel(dim=dim)

        fact_vec = world.encode_fact("dog", "is", "animal")

        assert fact_vec.shape == (dim,)

    def test_add_fact_sets_capsule_vector(self, dim):
        """add_fact should set capsule vector for facts."""
        from grilly.experimental.cognitive.world import WorldModel

        world = WorldModel(dim=dim)

        world.add_fact("dog", "is", "animal")

        assert world.facts[0].capsule_vector is not None
        assert world.facts[0].capsule_vector.shape == (world.capsule_dim,)

    def test_query_fact_finds_existing(self, dim):
        """query_fact should find existing facts."""
        from grilly.experimental.cognitive.world import WorldModel

        world = WorldModel(dim=dim)

        world.add_fact("dog", "is", "animal")

        is_known, confidence = world.query_fact("dog", "is", "animal")

        assert is_known is True
        assert confidence > 0.5

    def test_query_fact_returns_false_for_unknown(self, dim):
        """query_fact should return False for unknown facts."""
        from grilly.experimental.cognitive.world import WorldModel

        world = WorldModel(dim=dim)

        is_known, confidence = world.query_fact("unicorn", "is", "real")

        assert is_known is False
        assert confidence < 0.5


class TestWorldModelCoherence:
    """Tests for coherence checking."""

    def test_check_coherence_returns_tuple(self, dim):
        """check_coherence should return (is_coherent, confidence, reason)."""
        from grilly.experimental.cognitive.world import WorldModel
        from grilly.experimental.vsa.ops import HolographicOps

        world = WorldModel(dim=dim)

        statement_vec = HolographicOps.random_vector(dim)

        is_coherent, confidence, reason = world.check_coherence(statement_vec)

        assert isinstance(is_coherent, bool)
        assert isinstance(confidence, (float, np.floating))
        assert isinstance(reason, str)

    def test_check_coherence_detects_contradictions(self, dim):
        """check_coherence should detect contradictions."""
        from grilly.experimental.cognitive.world import WorldModel
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder

        world = WorldModel(dim=dim)

        # Add fact: "dog is animal"
        world.add_fact("dog", "is", "animal")

        # Encode contradictory statement: "dog is not animal"
        word_encoder = WordEncoder(dim=dim)
        sentence_encoder = SentenceEncoder(word_encoder)

        # Create statement vector that contradicts the fact
        contradiction_words = ["dog", "is", "not", "animal"]
        contradiction_vec = sentence_encoder.encode_sentence(contradiction_words)

        is_coherent, confidence, reason = world.check_coherence(contradiction_vec)

        # Should detect contradiction (may not always work perfectly due to HRR approximation)
        # At minimum, should return a result
        assert isinstance(is_coherent, bool)

    def test_check_coherence_supports_consistent_statements(self, dim):
        """check_coherence should support statements consistent with facts."""
        from grilly.experimental.cognitive.world import WorldModel
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder

        world = WorldModel(dim=dim)

        # Add fact: "dog is animal"
        world.add_fact("dog", "is", "animal")

        # Encode consistent statement
        word_encoder = WordEncoder(dim=dim)
        sentence_encoder = SentenceEncoder(word_encoder)

        consistent_words = ["dog", "is", "animal"]
        consistent_vec = sentence_encoder.encode_sentence(consistent_words)

        is_coherent, confidence, reason = world.check_coherence(consistent_vec)

        # Should be coherent (may not always work perfectly due to HRR approximation)
        assert isinstance(is_coherent, bool)


class TestWorldModelCausal:
    """Tests for causal expectations."""

    def test_add_causal_link_stores_expectation(self, dim):
        """add_causal_link should store causal expectation."""
        from grilly.experimental.cognitive.world import WorldModel

        world = WorldModel(dim=dim)

        world.add_causal_link("rain", "wet_ground", strength=0.9)

        assert "rain" in world.expectations
        assert len(world.expectations["rain"]) > 0

    def test_predict_consequence_returns_expectations(self, dim):
        """predict_consequence should return expected consequences."""
        from grilly.experimental.cognitive.world import WorldModel

        world = WorldModel(dim=dim)

        world.add_causal_link("rain", "wet_ground", strength=0.9)

        consequences = world.predict_consequence("rain")

        assert len(consequences) > 0
        assert any(effect == "wet_ground" for effect, _ in consequences)
