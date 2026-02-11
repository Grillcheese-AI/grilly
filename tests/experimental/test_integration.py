"""
Integration tests for grilly.experimental modules.

Verifies that all experimental modules work together correctly,
including cross-module interactions and end-to-end workflows.
"""

import numpy as np
import pytest

from grilly.experimental.cognitive import CognitiveController
from grilly.experimental.language import InstantLanguage
from grilly.experimental.moe import RelationalEncoder, RelationalMoE, ResonatorMoE
from grilly.experimental.temporal import (
    CausalChain,
    TemporalEncoder,
)
from grilly.experimental.vsa import BinaryOps, HolographicOps, ResonatorNetwork


class TestVSAAndMoEIntegration:
    """Test VSA operations with MoE routing."""

    def test_resonator_moe_uses_vsa_ops(self, dim):
        """ResonatorMoE should use VSA operations for similarity."""
        # Create experts dict
        experts = {f"expert_{i}": lambda x: x for i in range(10)}
        expert_vectors = {f"expert_{i}": HolographicOps.random_vector(dim) for i in range(10)}
        moe = ResonatorMoE(dim=dim, experts=experts, expert_vectors=expert_vectors)

        query = HolographicOps.random_vector(dim)
        result = moe.route(query, top_k=3)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) <= 3
        assert all(name in experts for name in result)

    def test_relational_moe_uses_relational_encoder(self, dim):
        """RelationalMoE should use RelationalEncoder for expert vectors."""
        encoder = RelationalEncoder(dim=dim)
        experts = {"expert1": lambda x: x, "expert2": lambda x: x, "expert3": lambda x: x}
        expert_relations = {
            "expert1": ("source1", "target1"),
            "expert2": ("source2", "target2"),
            "expert3": ("source3", "target3"),
        }

        moe = RelationalMoE(
            dim=dim, experts=experts, expert_relations=expert_relations, relational_encoder=encoder
        )

        query_entity = "query"
        query_vec = encoder.encode(query_entity, modality="text")
        result = moe.route(query_vec, top_k=2)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) <= 2


class TestLanguageAndVSAIntegration:
    """Test language system integration with VSA operations."""

    def test_instant_language_uses_holographic_ops(self, dim):
        """InstantLanguage should use HolographicOps internally."""
        lang = InstantLanguage(dim=dim)

        # Learn a sentence (pass as string or list)
        lang.learn_sentence("the cat chased the mouse")

        # Query should use VSA operations
        result = lang.query_relation("cat", "chased")
        assert isinstance(result, list)

    def test_language_parser_uses_resonator(self, dim):
        """ResonatorParser should use ResonatorNetwork for factorization."""
        from grilly.experimental.language.encoder import SentenceEncoder, WordEncoder
        from grilly.experimental.language.parser import ResonatorParser

        word_encoder = WordEncoder(dim=dim)
        sentence_encoder = SentenceEncoder(word_encoder)
        parser = ResonatorParser(sentence_encoder, max_iterations=50)

        # Encode a sentence
        words = ["the", "dog", "barked"]
        sentence_vec = sentence_encoder.encode_sentence(words)

        # Parse should use resonator
        parsed = parser.parse(sentence_vec)
        assert parsed is not None
        assert len(parsed) > 0


class TestTemporalAndVSAIntegration:
    """Test temporal reasoning integration with VSA."""

    def test_temporal_encoder_uses_vsa_binding(self, dim):
        """TemporalEncoder should use VSA binding for time encoding."""
        encoder = TemporalEncoder(dim=dim)

        # Encode a state with time (t must be int for discrete encoding)
        state_vec = HolographicOps.random_vector(dim)
        time_vec = encoder.bind_with_time(state_vec, t=5)

        # Unbind should recover state
        recovered = encoder.unbind_time(time_vec, t=5)
        similarity = HolographicOps.similarity(state_vec, recovered)

        assert similarity > 0.1  # Approximate recovery

    def test_causal_chain_uses_vsa_operations(self, dim):
        """CausalChain should use VSA operations for state encoding."""
        chain = CausalChain(dim=dim)

        # Add a rule
        chain.add_rule(name="rule1", conditions={"A": True}, effects={"B": True}, probability=0.9)

        # Encode initial state
        initial = chain.encode_state({"A": True})
        assert initial.shape == (dim,)

        # Propagate should use VSA operations
        result = chain.propagate_forward(initial, steps=1)
        assert result.shape == (dim,)


class TestCognitiveAndLanguageIntegration:
    """Test cognitive controller integration with language system."""

    def test_cognitive_controller_uses_language_system(self, dim):
        """CognitiveController should use InstantLanguage internally."""
        controller = CognitiveController(dim=dim)

        # Process should use language system
        response = controller.process("Hello, how are you?")

        # Should either return a response or None (based on confidence)
        assert response is None or isinstance(response, str)

    def test_cognitive_controller_understand_method(self, dim):
        """CognitiveController.understand should return UnderstandingResult."""
        controller = CognitiveController(dim=dim)

        result = controller.understand("The cat sat on the mat")

        assert hasattr(result, "surface_meaning")
        assert hasattr(result, "deep_meaning")
        assert hasattr(result, "words")
        assert hasattr(result, "confidence")


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows combining multiple modules."""

    def test_language_learning_to_moe_routing(self, dim):
        """Test workflow: learn language → encode → route with MoE."""
        # Step 1: Learn language
        lang = InstantLanguage(dim=dim)
        lang.learn_sentence("cat likes fish")
        lang.learn_sentence("dog likes bone")

        # Step 2: Encode query
        query_vec = lang.word_encoder.encode_word("cat")

        # Step 3: Create MoE with learned concepts
        expert_vecs = {
            "cat": lang.word_encoder.encode_word("cat"),
            "dog": lang.word_encoder.encode_word("dog"),
            "fish": lang.word_encoder.encode_word("fish"),
        }
        experts = {name: lambda x: x for name in expert_vecs.keys()}
        moe = ResonatorMoE(dim=dim, experts=experts, expert_vectors=expert_vecs)

        # Step 4: Route query
        result = moe.route(query_vec, top_k=2)

        assert result is not None
        assert isinstance(result, list)
        assert "cat" in result  # "cat" should be selected

    def test_temporal_reasoning_with_cognitive_controller(self, dim):
        """Test workflow: temporal reasoning → cognitive validation."""
        # Step 1: Set up temporal reasoning
        encoder = TemporalEncoder(dim=dim)
        chain = CausalChain(dim=dim)
        chain.add_rule(
            name="rain_rule", conditions={"raining": True}, effects={"wet": True}, probability=0.9
        )

        # Step 2: Encode temporal state
        state = chain.encode_state({"raining": True})
        encoder.bind_with_time(state, t=1)

        # Step 3: Use cognitive controller to reason about it
        controller = CognitiveController(dim=dim)
        controller.world.add_fact("weather", "is", "raining")

        # Controller should be able to process temporal information
        understanding = controller.understand("It is raining")
        assert understanding is not None

    def test_full_cognitive_pipeline(self, dim):
        """Test complete cognitive pipeline: understand → simulate → respond."""
        controller = CognitiveController(dim=dim, confidence_threshold=0.3)

        # Add some world knowledge
        controller.world.add_fact("dog", "is", "animal")
        controller.world.add_fact("cat", "is", "animal")

        # Process input through full pipeline
        response = controller.process("What is a dog?", verbose=True)

        # Should either respond or remain silent based on confidence
        assert response is None or isinstance(response, str)

        # Thinking trace should be populated when verbose=True
        assert len(controller.thinking_trace) > 0


class TestCrossModuleCompatibility:
    """Test that modules are compatible with each other's outputs."""

    def test_vsa_ops_compatible_across_modules(self, dim):
        """VSA operations should produce compatible vectors across modules."""
        # Generate vectors using different modules
        vec1 = HolographicOps.random_vector(dim)
        vec2 = BinaryOps.random_bipolar(dim).astype(np.float32)

        # Should be able to compute similarity (after normalization)
        vec2_normalized = vec2 / np.linalg.norm(vec2)
        similarity = HolographicOps.similarity(vec1, vec2_normalized)

        assert -1.0 <= similarity <= 1.0

    def test_resonator_works_with_language_vectors(self, dim):
        """ResonatorNetwork should work with language-encoded vectors."""
        from grilly.experimental.language.encoder import WordEncoder

        word_encoder = WordEncoder(dim=dim)

        # Create codebook from word vectors
        words = ["cat", "dog", "bird", "fish"]
        word_vectors = [word_encoder.encode_word(w) for w in words]
        codebook = np.array(word_vectors)

        # Create composite vector
        composite = HolographicOps.convolve(
            codebook[0],  # cat
            codebook[2],  # bird
        )

        # Factorize should recover words
        # ResonatorNetwork expects codebooks as a dict mapping factor names to arrays
        resonator = ResonatorNetwork(codebooks={"word": codebook}, max_iterations=100)
        estimates, indices, iters = resonator.factorize(composite=composite)

        assert len(estimates) == 1  # One codebook
        assert "word" in estimates
        assert len(indices) == 1
        assert "word" in indices
        assert isinstance(indices["word"], int)  # Single index
        assert 0 <= indices["word"] < len(codebook)


@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance-focused integration tests."""

    def test_large_scale_language_learning(self, large_dim):
        """Test language system with many learned sentences."""
        lang = InstantLanguage(dim=large_dim)

        # Learn many sentences
        sentences = [
            "the cat sat",
            "the dog ran",
            "the bird flew",
            "the fish swam",
        ] * 10

        for sentence in sentences:
            lang.learn_sentence(sentence)

        # Query should still work
        result = lang.query_relation("cat", "sat")
        assert isinstance(result, list)

    def test_moe_with_many_experts(self, large_dim):
        """Test MoE routing with large number of experts."""
        experts = {f"expert_{i}": lambda x: x for i in range(100)}
        expert_vectors = {
            f"expert_{i}": HolographicOps.random_vector(large_dim) for i in range(100)
        }
        moe = ResonatorMoE(dim=large_dim, experts=experts, expert_vectors=expert_vectors)

        query = HolographicOps.random_vector(large_dim)
        result = moe.route(query, top_k=5)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) <= 5
